import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl
from triton.profiler.mode import Default

# Enable semantic for TTGIR override
pl.enable_semantic("triton")


@triton.jit
def get_topmask_and_fullmask(x):
    tl.static_assert(x.dtype.is_int_unsigned(), "floating-point value must be passed as bits")
    tm: tl.constexpr = 1 << (-1 + x.dtype.primitive_bitwidth)
    fm: tl.constexpr = (1 << x.dtype.primitive_bitwidth) - 1
    tm_arr = tl.full(x.shape, tm, dtype=x.dtype)
    fm_arr = tl.full(x.shape, fm, dtype=x.dtype)
    return tm_arr, fm_arr


@triton.jit
def fpval_to_key(x):
    tm, fm = get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) != 0, fm, tm)


@triton.jit
def key_to_fpval(x):
    tm, fm = get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) == 0, fm, tm)


# stable top-k tie-breaks to value with smaller index
@triton.jit
def indx_to_key(indx, N_EXPTS_PAD: tl.constexpr):
    return N_EXPTS_PAD - indx


@triton.jit
def key_to_indx(indx, N_EXPTS_PAD: tl.constexpr):
    return N_EXPTS_PAD - indx


@triton.jit
def streaming_topk(X, stride_xm, n_expts_tot, offs_m, mask_m, N_EXPTS_PAD: tl.constexpr, N_EXPTS_ACT: tl.constexpr,
                   BLOCK_N: tl.constexpr):
    pl.enter_scope("kernel_start")
    pl.enter_scope("streaming_topk_setup_phase")
    x_nbits: tl.constexpr = X.dtype.element_ty.primitive_bitwidth
    x_utype: tl.constexpr = tl.dtype(f"uint{x_nbits}")
    if x_nbits < 16:
        # this ensures that we leave at least 16 bits for expert index
        # even if the input dtype is smaller than 16 bits:
        y_nbits: tl.constexpr = 32
    else:
        y_nbits: tl.constexpr = x_nbits * 2
    x_ultype: tl.constexpr = tl.dtype(f"uint{y_nbits}")
    x_dtype: tl.constexpr = X.dtype.element_ty

    # subtract 1 from loop iterations because we peel the first (masked) iteration:
    loop_iterations: tl.constexpr = N_EXPTS_PAD // BLOCK_N - 1
    offs_x_n = loop_iterations * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_x_n[None, :] < n_expts_tot
    pl.exit_scope("streaming_topk_setup_phase")

    # first iteration:
    pl.enter_scope("streaming_topk_first_iteration")
    X_ptrs = X + offs_m[:, None] * stride_xm + offs_x_n[None, :]
    x = tl.load(X_ptrs, mask=(mask_m & mask_n), other=float("-inf"))
    x = fpval_to_key(x.to(x_utype, bitcast=True))
    x = (x.to(x_ultype) << 16) | indx_to_key(offs_x_n, N_EXPTS_PAD)[None, :]
    acc = tl.topk(x, N_EXPTS_ACT, dim=1)
    pl.exit_scope("streaming_topk_first_iteration")
    pl.enter_scope("streaming_topk_subsequent_iteration")
    # subsequent iterations:
    for _i in (tl.static_range if loop_iterations <= 4 else range)(loop_iterations):
        acc = tl.bitonic_merge(acc)  # ensure sorted ascending for the merge
        X_ptrs -= BLOCK_N
        offs_x_n -= BLOCK_N
        x = tl.load(X_ptrs, mask=mask_m, other=float("-inf"))
        x = fpval_to_key(x.to(x_utype, bitcast=True))
        x = (x.to(x_ultype) << 16) | indx_to_key(offs_x_n, N_EXPTS_PAD)[None, :]
        acc = tl.maximum(acc, tl.topk(x, N_EXPTS_ACT, dim=1))
    pl.exit_scope("streaming_topk_subsequent_iteration")

    pl.enter_scope("streaming_topk_final_phase")
    # rotate expert index into upper 16 bits:
    # 0000vvvvvvvviiii --> iiii0000vvvvvvvv
    acc = (acc << (y_nbits - 16)) | (acc >> 16)
    # sort in ascending order of expert (descending order of key)
    acc = tl.sort(acc, dim=1, descending=True)
    # iiii0000vvvvvvvv --> 0000iiii:
    y_indices_raw = (acc >> (y_nbits - 16)).to(tl.uint32)
    y_indices = key_to_indx(y_indices_raw, N_EXPTS_PAD)
    # iiii0000vvvvvvvv --> vvvvvvvv:
    y_values_raw = acc.to(x_utype)
    y_values = key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)
    pl.exit_scope("streaming_topk_final_phase")
    pl.exit_scope("kernel_start")

    return y_values, y_indices


@triton.jit
def _topk_forward(X, stride_xm,  # inputs
                  Yv, Yi, stride_ym,  # topk values/indices
                  USE_PROVIDED_INDX: tl.constexpr, Bits, stride_rm: tl.constexpr, stride_rn: tl.constexpr,  # bitmatrix
                  n_rows, n_expts_tot,  # shape
                  S, BLOCK_S: tl.constexpr, s_blocks,  # thing to memset
                  APPLY_SOFTMAX: tl.constexpr,  # constant
                  BLOCK_M: tl.constexpr, N_EXPTS_PAD: tl.constexpr, N_EXPTS_ACT: tl.constexpr, BLOCK_N: tl.constexpr):

    # Setup phase
    pl.enter_scope("setup_phase")
    pid = tl.program_id(0)
    if isinstance(n_rows, tl.tensor) and n_rows.dtype.is_ptr():
        n_rows = tl.load(n_rows)

    if pid < s_blocks:
        tl.store(S + BLOCK_S * pid + tl.arange(0, BLOCK_S), tl.zeros([BLOCK_S], tl.int32))

    if pid * BLOCK_M >= n_rows:
        return

    tl.static_assert(BLOCK_N % 32 == 0)
    tl.static_assert(N_EXPTS_PAD % BLOCK_N == 0)
    x_dtype: tl.constexpr = X.dtype.element_ty

    # load logits
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_y_n = tl.arange(0, N_EXPTS_ACT)
    mask_m = offs_m[:, None] < n_rows
    pl.exit_scope("setup_phase")
    
    # Compute phase
    pl.enter_scope("compute_phase")
    if USE_PROVIDED_INDX:
        Yi_ptrs = Yi + offs_m[:, None] * stride_ym + offs_y_n[None, :]
        y_indices = tl.load(Yi_ptrs, mask=mask_m)
        Xv_ptrs = X + offs_m[:, None] * stride_xm + y_indices
        y_values = tl.load(Xv_ptrs, mask=mask_m)
    else:
        pl.enter_scope("streaming_topk")
        y_values, y_indices = streaming_topk(X, stride_xm, n_expts_tot, offs_m, mask_m,  #
                                             N_EXPTS_PAD, N_EXPTS_ACT, BLOCK_N)
        pl.exit_scope("streaming_topk")

    # normalize selected values
    if APPLY_SOFTMAX:
        y_values = tl.softmax(y_values.to(tl.float32), dim=1, keep_dims=True).to(x_dtype)
    pl.exit_scope("compute_phase")

    # Output phase
    pl.enter_scope("output_phase")
    # write back
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    tl.store(Yv_ptrs, y_values, mask=mask_m)
    if not USE_PROVIDED_INDX:
        Yi_ptrs = Yi + offs_m[:, None] * stride_ym + offs_y_n[None, :]
        tl.store(Yi_ptrs, y_indices, mask=mask_m)

    # pack into bitmatrix
    y_div = y_indices // 32
    y_rem = y_indices % 32
    loop_iterations = N_EXPTS_PAD // BLOCK_N
    for i in range(loop_iterations):
        offs_r_n = tl.arange(0, BLOCK_N // 32) + i * (BLOCK_N // 32)
        y2 = tl.where(y_div[:, :, None] == offs_r_n[None, None, :], (1 << y_rem)[:, :, None], 0)
        r = tl.reduce_or(y2, axis=1)
        BitsPtrs = Bits + offs_m[:, None] * stride_rm + offs_r_n[None, :] * stride_rn
        tl.store(BitsPtrs, r, mask=mask_m)
    pl.exit_scope("output_phase")


def benchmark_topk_original(M, N_experts, K, use_cuda_event: bool = False):
    """Simple benchmark function for original topk kernel"""
    import torch
    
    # Create input data
    X = torch.randn((M, N_experts), device="cuda", dtype=torch.float16)
    
    # Output tensors
    Yv = torch.empty((M, K), device="cuda", dtype=torch.float16)
    Yi = torch.empty((M, K), device="cuda", dtype=torch.int32)
    
    # Additional required tensors for the original kernel
    Bits = torch.zeros((M, N_experts // 32), device="cuda", dtype=torch.int32)
    S = torch.zeros((1024,), device="cuda", dtype=torch.int32)  # Scratch space
    
    # Launch kernel with simplified parameters
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']),)
    
    # Note: This is a simplified call - the original has many more parameters
    # For profiling purposes, we use basic configuration
    N_PAD = triton.next_power_of_2(N_experts)
    
    try:
        # warm up
        for _ in range(5):
            _topk_forward[(1,)](
                X, X.stride(0),
                Yv, Yi, Yv.stride(0),
                False,  # USE_PROVIDED_INDX
                Bits, Bits.stride(0), Bits.stride(1),
                M, N_experts,
                S, 1024, 1,  # scratch space config
                True,  # APPLY_SOFTMAX
                32,  # BLOCK_M
                N_PAD,  # N_EXPTS_PAD
                K,  # N_EXPTS_ACT
                128,  # BLOCK_N
            )
        if use_cuda_event:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(1_000_000)
            start_event.record()
        _topk_forward[(1,)](
                X, X.stride(0),
                Yv, Yi, Yv.stride(0),
                False,  # USE_PROVIDED_INDX
                Bits, Bits.stride(0), Bits.stride(1),
                M, N_experts,
                S, 1024, 1,  # scratch space config
                True,  # APPLY_SOFTMAX
                32,  # BLOCK_M
                N_PAD,  # N_EXPTS_PAD
                K,  # N_EXPTS_ACT
                128,  # BLOCK_N
            )
        if use_cuda_event:
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            print(f"Outside topk elapsed time by cuda event: {elapsed_time} ms")

        print("TopK kernel executed successfully")
        return Yv, Yi
    except Exception as e:
        print(f"TopK kernel failed: {e}")
        return None, None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--M", type=int, default=8192, help="Batch size")
    parser.add_argument("--N", type=int, default=1024, help="Number of experts")
    parser.add_argument("--K", type=int, default=16, help="Top-K value")
    parser.add_argument("--data", type=str, default="tree", choices=["tree", "trace"], help="data to collect with Proton")
    parser.add_argument("--buffer-size", type=int, default=512, help="Proton buffer size")
    parser.add_argument("--use-cuda-event", action="store_true", help="Enable cudaEvent time measurement")

    args = parser.parse_args()
    
    M, N, K = args.M, args.N, args.K
    
    if args.profile:
        proton_mode = Default(buffer_size=args.buffer_size)
        proton.start("topk_original_instrumented", backend="instrumentation", hook="triton", data=args.data, mode=proton_mode)
        values, indices = benchmark_topk_original(M, N, K, use_cuda_event=args.use_cuda_event)
        proton.finalize()
        print(f"Profiled original TopK {M}x{N} (K={K})")
    else:
        values, indices = benchmark_topk_original(M, N, K, use_cuda_event=True)
        print(f"Ran original TopK {M}x{N} (K={K})")
