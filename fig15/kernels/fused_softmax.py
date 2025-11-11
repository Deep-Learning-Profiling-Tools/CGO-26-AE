import torch
import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl
from triton.profiler.mode import Default
from triton.runtime import driver
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


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')


@triton.jit
def instrumented_softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, 
                               BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    """Instrumented kernel for computing fused softmax with intra-kernel profiling."""
    
    # Record kernel start
    pl.enter_scope("kernel_start")
    
    # Setup phase - program ID calculation and loop initialization
    pl.enter_scope("setup")
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    pl.exit_scope("setup")
    
    # Main compute loop
    pl.enter_scope("main_compute_loop")
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # Memory addressing setup
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        
        # Load input row
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        
        # Compute softmax: subtract max, exp, normalize
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        
        # Store result
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
    pl.exit_scope("main_compute_loop")
    # Record kernel end
    pl.exit_scope("kernel_start")


def instrumented_softmax(x, use_cuda_event: bool = False):
    """Wrapper function for instrumented softmax."""
    n_rows, n_cols = x.shape
    
    # Get device properties for optimization
    properties = driver.active.utils.get_device_properties(DEVICE.index)
    NUM_SM = properties["multiprocessor_count"]
    NUM_REGS = properties["max_num_regs"]
    SIZE_SMEM = properties["max_shared_mem"]
    WARP_SIZE = properties["warpSize"]
    
    # The block size of each loop iteration is the smallest power of two greater than the number of columns
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Optimization parameters
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    
    # Allocate output
    y = torch.empty_like(x)
    
    # Pre-compile kernel to get register usage and compute thread occupancy
    kernel = instrumented_softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, 
                                               BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages, 
                                               num_warps=num_warps, grid=(1, ))
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    
    # Calculate occupancy
    if is_hip():
        NUM_GPRS = NUM_REGS
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2
        MAX_NUM_THREADS = properties["max_threads_per_sm"]
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, n_rows)
    
    # Launch kernel
    if use_cuda_event:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(1_000_000)
        start_event.record()

    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    
    if use_cuda_event:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        log_cuda_event_time("fused_softmax", elapsed_time)
        print(f"Outside swiglu elapsed time by cuda event: {elapsed_time} ms")

    return y


def benchmark_instrumented_softmax(M, N, device="cuda", use_cuda_event: bool = False):
    """Benchmark instrumented softmax kernel with cycle measurement."""
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(5):
        _ = instrumented_softmax(x)
    with proton.scope(f"softmax_{M}_{N}"):
        # Actual benchmark
        result = instrumented_softmax(x, use_cuda_event)

    return result


def naive_softmax(x):
    """Compute row-wise softmax using native pytorch for comparison."""
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable timing profiling by Proton cupti backend")
    parser.add_argument("--instrument", action="store_true", help="Enable intra-kernel instrumentation profiling to get cycles (can run with cupti)")
    parser.add_argument("--rows", type=int, default=16384, help="Number of rows")
    parser.add_argument("--cols", type=int, default=8192, help="Number of columns")
    parser.add_argument("--data", type=str, default="tree", choices=["tree", "trace"], help="data to collect with Proton instrumentation backend")
    parser.add_argument("--buffer-size", type=int, default=512, help="Proton buffer size")
    parser.add_argument("--use-cuda-event", action="store_true", help="Enable cudaEvent time measurement")
    args = parser.parse_args()
    
    set_profile_enabled(args.instrument)

    M, N = args.rows, args.cols
    
    sessions = []
    cupti_profile_name = None
    if args.profile:
        cupti_profile_name = f"softmax_cupti_wInstrument{args.instrument}"
        cupti_session = proton.start(
            cupti_profile_name, hook="triton", data="tree"
        )
        sessions.append(cupti_session)
    if args.instrument:
        proton_mode = Default(buffer_size=args.buffer_size)
        instrument_session = proton.start(
            "softmax_instrumented",
            backend="instrumentation",
            hook="triton",
            data=args.data,
            mode=proton_mode,
        )
        sessions.append(instrument_session)

    result = benchmark_instrumented_softmax(
        M, N, use_cuda_event=args.use_cuda_event if args.instrument else True
    )

    for session in reversed(sessions):
        proton.finalize(session)

    if args.profile and cupti_profile_name:
        profile_path = Path(f"{cupti_profile_name}.hatchet")
        try:
            kernel_time_ns = extract_kernel_time_from_hatchet(profile_path, r"instrumented_softmax_kernel")
            log_cupti_profile_time("fused_softmax", args.instrument, kernel_time_ns)
        except Exception as exc:
            print(f"Failed to log CUPTI timing from {profile_path}: {exc}")

    print(f"Completed instrumented softmax {M}x{N} (cupti={args.profile}, instrument={args.instrument})")
    
    # Verify correctness
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    y_instrumented = instrumented_softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_instrumented, y_torch, atol=1e-5), "Results do not match!"
    print("Correctness verified!")
