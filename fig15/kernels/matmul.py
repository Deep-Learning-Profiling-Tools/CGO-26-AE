import torch
import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl
from triton.profiler.mode import Default
from typing import NamedTuple
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


def unpack_grid(grid):
    if len(grid) == 1:
        return grid[0], 1, 1
    if len(grid) == 2:
        return grid[0], grid[1], 1
    if len(grid) == 3:
        return grid[0], grid[1], grid[2]


def metadata_fn(
    grid: tuple,
    metadata: NamedTuple,
    args: dict,
):
    grid_x, grid_y, grid_z = unpack_grid(grid)
    num_warps = metadata.num_warps
    num_stages = metadata.num_stages
    cluster_x, cluster_y, cluster_z = unpack_grid((metadata.num_ctas, ))
    shared_memory = metadata.shared
    M, K = args["a_ptr"].shape
    K, N = args["b_ptr"].shape
    return {
        "name":
        f"matmul_<grid:{grid_x}x{grid_y}x{grid_z}>_<cluster:{cluster_x}x{cluster_y}x{cluster_z}>_<warps:{num_warps}>_<shared:{shared_memory}>_<stages:{num_stages}>",
        "flops": 2 * M * N * K,
        "bytes": (M * N + N * K + K * M) * args["a_ptr"].element_size(),
    }


@triton.jit(launch_metadata=metadata_fn)
def instrumented_matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
):
    """Instrumented kernel for computing the matmul C = A x B with intra-kernel profiling."""
    
    # Record kernel start
    pl.enter_scope("kernel_start")
    
    # Map program ids to blocks - measure setup time
    pl.enter_scope("setup")
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    pl.exit_scope("setup")

    # Pointer arithmetic - measure pointer setup time
    pl.enter_scope("pointer_setup")
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    pl.exit_scope("pointer_setup")

    # Initialize accumulator
    pl.enter_scope("accumulator_init")
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    pl.exit_scope("accumulator_init")

    # Main computation loop with detailed instrumentation
    pl.enter_scope("compute_loop")
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    pl.exit_scope("compute_loop")

    # Activation function (if any)
    if ACTIVATION == "leaky_relu":
        pl.enter_scope("activation")
        accumulator = leaky_relu(accumulator)
        pl.exit_scope("activation")

    # Output conversion and storage
    pl.enter_scope("output_phase")
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
    pl.exit_scope("output_phase")
    
    # Record kernel end
    pl.exit_scope("kernel_start")


@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)


def instrumented_matmul(a, b, activation="", use_cuda_event: bool = False):
    """Wrapper function for instrumented matmul."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    if use_cuda_event:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(1_000_000)
        start_event.record()
    instrumented_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 256,
        BLOCK_SIZE_K = 32,
        GROUP_SIZE_M = 8,
    )
    if use_cuda_event:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        log_cuda_event_time("matmul", elapsed_time)
        print(f"Outside matmul elapsed time by cuda event: {elapsed_time} ms")
    return c


def benchmark_instrumented_triton(M, N, K, device="cuda", use_cuda_event: bool = False):
    """Benchmark instrumented Triton kernel with cycle measurement."""
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    
    with proton.scope(f"triton_matmul_{M}_{N}_{K}"):
        # Warmup
        for _ in range(5):
            _ = instrumented_matmul(a, b)

        # Actual benchmark
        result = instrumented_matmul(a, b, use_cuda_event=use_cuda_event)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable timing profiling by Proton cupti backend")
    parser.add_argument("--instrument", action="store_true", help="Enable intra-kernel instrumentation profiling to get cycles (can run with cupti)")
    parser.add_argument("--matrix-size", type=int, default=4096, help="Matrix size")
    parser.add_argument("--data", type=str, default="tree", choices=["tree", "trace"], help="data to collect with Proton instrumentation backend")
    parser.add_argument("--buffer-size", type=int, default=512, help="Proton buffer size")
    parser.add_argument("--use-cuda-event", action="store_true", help="Enable cudaEvent time measurement")

    args = parser.parse_args()
    set_profile_enabled(args.instrument)

    M = N = K = args.matrix_size
    
    sessions = []
    cupti_profile_name = None
    if args.profile:
        cupti_profile_name = f"matmul_cupti_wInstrument{args.instrument}"
        cupti_session = proton.start(
            cupti_profile_name, backend="cupti", hook="triton", data="tree"
        )
        sessions.append(cupti_session)
    if args.instrument:
        proton_mode = Default(buffer_size=args.buffer_size)
        instrument_session = proton.start(
            "matmul_instrumented",
            backend="instrumentation",
            hook="triton",
            data=args.data,
            mode=proton_mode,
        )
        sessions.append(instrument_session)

    result = benchmark_instrumented_triton(
        M, N, K, use_cuda_event=args.use_cuda_event if args.instrument else True
    )

    for session in reversed(sessions):
        proton.finalize(session)

    if args.profile and cupti_profile_name:
        profile_path = Path(f"{cupti_profile_name}.hatchet")
        try:
            kernel_time_ns = extract_kernel_time_from_hatchet(profile_path, r"matmul_<grid:")
            log_cupti_profile_time("matmul", args.instrument, kernel_time_ns)
        except Exception as exc:
            print(f"Failed to log CUPTI timing from {profile_path}: {exc}")

    print(f"Completed Triton matmul {M}x{N}x{K} (cupti={args.profile}, instrument={args.instrument})")
