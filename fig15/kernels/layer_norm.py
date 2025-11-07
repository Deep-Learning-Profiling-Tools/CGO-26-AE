import torch
import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl
from triton.profiler.mode import Default
import argparse
from pathlib import Path
from utils import (
    extract_kernel_time_from_hatchet,
    log_cupti_profile_time,
    log_cuda_event_time,
    set_profile_enabled,
)

# Enable semantic for TTGIR override
pl.enable_semantic("triton")

DEVICE = triton.runtime.driver.active.get_active_torch_device()

_CUPTI_KERNEL_PATTERNS = {
    "forward": r"instrumented_layer_norm_fwd_kernel",
    "backward": r"instrumented_layer_norm_bwd_kernel",
}


@triton.jit
def instrumented_layer_norm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    """Instrumented kernel for layer normalization forward pass with intra-kernel profiling."""
    
    # Record kernel start
    pl.enter_scope("kernel_start")
    
    # Setup phase - map program ID to row and set up pointers
    pl.enter_scope("setup")
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    pl.exit_scope("setup")
    
    # Main compute loop
    pl.enter_scope("main_compute_loop")
    
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    
    # Store mean and rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Store output
        tl.store(Y + cols, y, mask=mask)
    
    pl.exit_scope("main_compute_loop")
    # Record kernel end
    pl.exit_scope("kernel_start")


@triton.jit
def instrumented_layer_norm_bwd_dx_kernel(DX,  # pointer to the input gradient
                                         DY,  # pointer to the output gradient
                                         DW,  # pointer to the partial sum of weights gradient
                                         DB,  # pointer to the partial sum of biases gradient
                                         X,  # pointer to the input
                                         W,  # pointer to the weights
                                         Mean,  # pointer to the mean
                                         Rstd,  # pointer to the 1/std
                                         Lock,  # pointer to the lock
                                         stride,  # how much to increase the pointer when moving by 1 row
                                         N,  # number of columns in X
                                         GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """Instrumented kernel for layer normalization backward pass with intra-kernel profiling."""
    
    # Record kernel start
    pl.enter_scope("kernel_start")
    
    # Setup phase - map program ID and set up pointers
    pl.enter_scope("setup")
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    pl.exit_scope("setup")
    
    # Main compute loop
    pl.enter_scope("main_compute_loop")
    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    
    # Compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    
    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    
    # Atomic operations for parallel reduction
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    
    # Barrier and release lock
    tl.debug_barrier()
    tl.atomic_xchg(Lock, 0)
    pl.exit_scope("main_compute_loop")
    
    # Output phase - store dx
    pl.enter_scope("output_phase")
    tl.store(DX + cols, dx, mask=mask)
    pl.exit_scope("output_phase")
    
    # Record kernel end
    pl.exit_scope("kernel_start")


@triton.jit
def instrumented_layer_norm_bwd_dwdb_kernel(DW,  # pointer to the partial sum of weights gradient
                                           DB,  # pointer to the partial sum of biases gradient
                                           FINAL_DW,  # pointer to the weights gradient
                                           FINAL_DB,  # pointer to the biases gradient
                                           M,  # GROUP_SIZE_M
                                           N,  # number of columns
                                           BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """Instrumented kernel for final reduction of weight and bias gradients."""
    
    # Record kernel start
    pl.enter_scope("kernel_start")
    
    # Setup phase
    pl.enter_scope("setup")
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    pl.exit_scope("setup")
    
    # Main compute loop - iterate through rows to sum partial sums
    pl.enter_scope("main_compute_loop")
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    
    # Sum across rows
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    pl.exit_scope("main_compute_loop")
    
    # Output phase - store final results
    pl.enter_scope("output_phase")
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)
    pl.exit_scope("output_phase")
    
    # Record kernel end
    pl.exit_scope("kernel_start")


class InstrumentedLayerNorm(torch.autograd.Function):
    """Instrumented LayerNorm autograd function with Proton profiling."""

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps, use_cuda_event: bool = False):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        
        # enqueue kernel
        if use_cuda_event:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(1_000_000)
            start_event.record()

        instrumented_layer_norm_fwd_kernel[(M, )](
            x_arg, y, weight, bias, mean, rstd,
            x_arg.stride(0), N, eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        
        if use_cuda_event:
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            log_cuda_event_time("layer_norm", elapsed_time)
            print(f"Outside layer_norm elapsed time by cuda event: {elapsed_time} ms")
    
        
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, v = ctx.saved_tensors
        # heuristics for amount of parallel reduction stream for DW/DB
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256
        
        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        
        # enqueue kernel using forward pass heuristics
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        instrumented_layer_norm_bwd_dx_kernel[(M, )](
            dx, dy, _dw, _db, x, w, m, v, locks,
            x_arg.stride(0), N,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=ctx.num_warps)
        
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )
        # accumulate partial sums in separate kernel
        instrumented_layer_norm_bwd_dwdb_kernel[grid](
            _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128, num_ctas=1)
        return dx, None, dw, db, None


# instrumented_layer_norm = InstrumentedLayerNorm.apply
def instrumented_layer_norm(x, normalized_shape, weight, bias, eps, use_cuda_event: bool = False):
    # Pass positionally into .apply (no kwargs!)
    return InstrumentedLayerNorm.apply(x, normalized_shape, weight, bias, eps, use_cuda_event)


def benchmark_instrumented_layer_norm(M, N, dtype=torch.float16, eps=1e-5, device=DEVICE, mode='forward', use_cuda_event: bool = False):
    """Benchmark instrumented layer norm with cycle measurement."""
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    x.requires_grad_(True)
    
    if mode == 'forward':
        with proton.scope(f"layer_norm_fwd_{M}_{N}"):
            # Warmup
            for _ in range(5):
                _ = instrumented_layer_norm(x, w_shape, weight, bias, eps)

            # Actual benchmark
            result = instrumented_layer_norm(x, w_shape, weight, bias, eps, use_cuda_event=use_cuda_event)
    else:
        # Backward mode
        dy = .1 * torch.randn_like(x)
        
        with proton.scope(f"layer_norm_bwd_{M}_{N}"):
            # Warmup forward pass
            y = instrumented_layer_norm(x, w_shape, weight, bias, eps)
            
            # Warmup backward
            for _ in range(5):
                y.backward(dy, retain_graph=True)
                x.grad = None
                weight.grad = None
                bias.grad = None
            
            # Actual benchmark
            y.backward(dy, retain_graph=True)
            result = y
    
    return result


def test_layer_norm_correctness(M, N, dtype, eps=1e-5, device=DEVICE):
    """Test correctness against PyTorch implementation."""
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    
    # forward pass
    y_tri = instrumented_layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None
    
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
    
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0), "Forward pass results don't match"
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0), "Input gradients don't match"
    assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0), "Bias gradients don't match"
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0), "Weight gradients don't match"
    print("Correctness test passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable timing profiling by Proton cupti backend")
    parser.add_argument("--instrument", action="store_true", help="Enable intra-kernel instrumentation profiling to get cycles (can run with cupti)")
    parser.add_argument("--rows", type=int, default=16384, help="Number of rows")
    parser.add_argument("--cols", type=int, default=8192, help="Number of columns")
    parser.add_argument("--mode", choices=["forward", "backward"], default="forward", help="Benchmark mode")
    parser.add_argument("--test", action="store_true", help="Run correctness test")
    parser.add_argument("--data", type=str, default="tree", choices=["tree", "trace"], help="data to collect with Proton instrumentation backend")
    parser.add_argument("--buffer-size", type=int, default=512, help="Proton buffer size")
    parser.add_argument("--use-cuda-event", action="store_true", help="Enable cudaEvent time measurement")

    args = parser.parse_args()
    set_profile_enabled(args.instrument)
    
    M, N = args.rows, args.cols
    
    if args.test:
        test_layer_norm_correctness(M, N, torch.float16)
        exit(0)
    
    sessions = []
    cupti_profile_name = None
    if args.profile:
        cupti_profile_name = f"layer_norm_cupti_wInstrument{args.instrument}"
        cupti_session = proton.start(
            cupti_profile_name, backend="cupti", hook="triton", data="tree"
        )
        sessions.append(cupti_session)
    if args.instrument:
        proton_mode = Default(buffer_size=args.buffer_size)
        instrument_session = proton.start(
            f"layer_norm_{args.mode}_instrumented",
            backend="instrumentation",
            hook="triton",
            data=args.data,
            mode=proton_mode,
        )
        sessions.append(instrument_session)

    result = benchmark_instrumented_layer_norm(
        M, N, mode=args.mode, use_cuda_event=args.use_cuda_event if args.instrument else True
    )

    for session in reversed(sessions):
        proton.finalize(session)

    if args.profile and cupti_profile_name:
        profile_path = Path(f"{cupti_profile_name}.hatchet")
        try:
            pattern = _CUPTI_KERNEL_PATTERNS.get(args.mode, r"instrumented_layer_norm")
            kernel_time_ns = extract_kernel_time_from_hatchet(profile_path, pattern)
            log_cupti_profile_time(f"layer_norm_{args.mode}", args.instrument, kernel_time_ns)
        except Exception as exc:
            print(f"Failed to log CUPTI timing from {profile_path}: {exc}")

    print(f"Completed layer norm {args.mode} pass {M}x{N} (cupti={args.profile}, instrument={args.instrument})")
