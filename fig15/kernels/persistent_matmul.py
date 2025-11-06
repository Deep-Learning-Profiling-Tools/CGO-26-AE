import torch
import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl
from triton.profiler.mode import Default
from typing import NamedTuple
import argparse

# Enable semantic for TTGIR override
pl.enable_semantic("triton")

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
    M, N, K = args["M"], args["N"], args["K"]
    bytes_per_elem = args["c_ptr"].element_size()
    return {
        "name": f"persistent_matmul_kernel [M={M}, N={N}, K={K}]",
        "flops": 2.0 * M * N * K,
        "bytes": bytes_per_elem * (M * K + N * K + M * N),
    }


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(launch_metadata=metadata_fn)
def persistent_matmul_kernel(a_ptr, b_ptr, c_ptr,
                            M, N, K,
                            stride_am, stride_ak,
                            stride_bk, stride_bn,
                            stride_cm, stride_cn,
                            BLOCK_SIZE_M: tl.constexpr,
                            BLOCK_SIZE_N: tl.constexpr,
                            BLOCK_SIZE_K: tl.constexpr,
                            GROUP_SIZE_M: tl.constexpr,
                            NUM_SMS: tl.constexpr,
                            ):
    pl.enter_scope("kernel_start")
    # Setup phase
    # pl.enter_scope("setup_phase")
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # pl.exit_scope("setup_phase")

    # Main compute loop - persistent kernel loops over multiple tiles
    pl.enter_scope("compute_phase")
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        
        c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)
    pl.exit_scope("compute_phase")
    pl.exit_scope("kernel_start")

def persistent_matmul(a, b, use_cuda_event:bool = False):
    # Check constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Launch kernel with limited grid to enable persistence
    def grid(META):
        return (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    
    if use_cuda_event:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(1_000_000)
        start_event.record()

    persistent_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), 
        c.stride(0), c.stride(1),
        NUM_SMS=NUM_SMS,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
    )
    if use_cuda_event:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        print(f"Outside persistent matmul elapsed time by cuda event: {elapsed_time} ms")

    return c


def benchmark_persistent_matmul(M, N, K, use_cuda_event: bool = False):
    # Create test matrices
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    
    # Test correctness
    torch_result = torch.matmul(a, b)
    # warm-up
    for _ in range(5):
        triton_result = persistent_matmul(a, b, use_cuda_event=False)
    
    triton_result = persistent_matmul(a, b, use_cuda_event=use_cuda_event)    
    if torch.allclose(triton_result, torch_result, atol=1e-2, rtol=1e-2):
        print("Correctness verified!")
    else:
        print("Warning: Results don't match!")
    
    return triton_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--M", type=int, default=4096, help="Matrix dimension M")
    parser.add_argument("--N", type=int, default=4096, help="Matrix dimension N") 
    parser.add_argument("--K", type=int, default=4096, help="Matrix dimension K")
    parser.add_argument("--data", type=str, default="tree", choices=["tree", "trace"], help="data to collect with Proton")
    parser.add_argument("--buffer-size", type=int, default=512, help="Proton buffer size")

    args = parser.parse_args()
    
    M, N, K = args.M, args.N, args.K
    
    if args.profile:
        proton_mode = Default(buffer_size=args.buffer_size)
        proton.start("persistent_matmul_instrumented", backend="instrumentation", hook="triton", data=args.data, mode=proton_mode)
        result = benchmark_persistent_matmul(M, N, K)
        proton.finalize()
        print(f"Profiled instrumented persistent matmul {M}x{N}x{K}")
    else:
        result = benchmark_persistent_matmul(M, N, K, use_cuda_event=True)
        print(f"Ran instrumented persistent matmul {M}x{N}x{K}")