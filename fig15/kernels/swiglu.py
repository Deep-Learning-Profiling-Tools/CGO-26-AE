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

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
    M, N = args["M"], args["N"]
    bytes_per_elem = args["x_ptr"].element_size()
    return {
        "name": f"swiglu_kernel [M={M}, N={N}]",
        "flops": M * N * 8,  # Approximate FLOPs for SwiGLU
        "bytes": bytes_per_elem * (M * N * 2 + M * N),  # Input (2x) + output
    }


@triton.jit(launch_metadata=metadata_fn)
def swiglu_kernel(
    x_ptr,  # Input tensor pointer (M, 2*N) - contains both gate and linear projections
    output_ptr,  # Output tensor pointer (M, N)
    M,  # Sequence length
    N,  # Hidden dimension
    stride_xm, stride_xn,  # Input strides
    stride_om, stride_on,  # Output strides
    beta: tl.constexpr,  # SwiGLU beta parameter (typically 1.0)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):  
    pl.enter_scope("kernel_start")
    # Setup phase
    pl.enter_scope("setup_phase")
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Input tensor contains [gate_proj, up_proj] concatenated along last dimension
    # gate_proj: x[:, :N], up_proj: x[:, N:]
    gate_offs = offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    up_offs = offs_m[:, None] * stride_xm + (offs_n[None, :] + N) * stride_xn
    pl.exit_scope("setup_phase")
    
    # Compute phase
    pl.enter_scope("compute_phase")
    # Load gate and up projections
    mask = mask_m[:, None] & mask_n[None, :]
    gate = tl.load(x_ptr + gate_offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + up_offs, mask=mask, other=0.0).to(tl.float32)
    
    # SwiGLU computation: gate * silu(up)
    # silu(x) = x * sigmoid(beta * x) = x / (1 + exp(-beta * x))
    sigmoid_input = beta * up
    sigmoid_output = 1.0 / (1.0 + tl.exp(-sigmoid_input))
    silu_output = up * sigmoid_output
    
    # Final SwiGLU output
    output = gate * silu_output
    pl.exit_scope("compute_phase")
    
    # Output phase
    pl.enter_scope("output_phase")
    # Store result
    output_offs = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(output_ptr + output_offs, output.to(tl.float16), mask=mask)
    pl.exit_scope("output_phase")
    pl.exit_scope("kernel_start")


def swiglu(x, beta=1.0, use_cuda_event: bool = False):
    """
    SwiGLU activation function
    
    Args:
        x: Input tensor of shape (M, 2*N) where the last dimension contains
           [gate_proj, up_proj] concatenated
        beta: SwiGLU beta parameter (default: 1.0)
    
    Returns:
        Output tensor of shape (M, N)
    """
    M, hidden_2N = x.shape
    assert hidden_2N % 2 == 0, "Last dimension must be even (contains gate and up projections)"
    N = hidden_2N // 2
    
    # Allocate output
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Launch kernel
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    
    if use_cuda_event:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(1_000_000)
        start_event.record()
    
    swiglu_kernel[grid](
        x, output,
        M, N,
        x.stride(0), x.stride(1),
        output.stride(0), output.stride(1),
        beta,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
    )
    if use_cuda_event:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        log_cuda_event_time("swiglu", elapsed_time)
        print(f"Outside swiglu elapsed time by cuda event: {elapsed_time} ms")
    
    return output


def benchmark_swiglu(M, N, beta=1.0, use_cuda_event: bool = False):
    """Benchmark SwiGLU with specified parameters"""
    # Create input tensor: [gate_proj, up_proj] concatenated
    x = torch.randn((M, 2 * N), device=DEVICE, dtype=torch.float16)
    # warm up
    for _ in range(5):
        _ = swiglu(x, beta, use_cuda_event=False) # always disable cuda event for warm up

    # Run SwiGLU
    output_triton = swiglu(x, beta, use_cuda_event=use_cuda_event)
    
    # Verify with PyTorch implementation
    gate_proj = x[:, :N]
    up_proj = x[:, N:]
    silu = torch.nn.functional.silu(up_proj)
    output_torch = gate_proj * silu
    
    if torch.allclose(output_triton, output_torch, atol=1e-2, rtol=1e-2):
        print("Correctness verified!")
    else:
        print("Warning: Results don't match!")
        max_diff = torch.max(torch.abs(output_triton - output_torch))
        print(f"Max difference: {max_diff}")
    
    return output_triton


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable timing profiling by Proton cupti backend")
    parser.add_argument("--instrument", action="store_true", help="Enable intra-kernel instrumentation profiling to get cycles (can run with cupti)")
    parser.add_argument("--M", type=int, default=16384, help="Sequence length")
    parser.add_argument("--N", type=int, default=8192, help="Hidden dimension")
    parser.add_argument("--beta", type=float, default=1.0, help="SwiGLU beta parameter")
    parser.add_argument("--data", type=str, default="tree", choices=["tree", "trace"], help="data to collect with Proton instrumentation backend")
    parser.add_argument("--buffer-size", type=int, default=512, help="Proton buffer size")
    parser.add_argument("--use-cuda-event", action="store_true", help="Enable cudaEvent time measurement")
    args = parser.parse_args()
    set_profile_enabled(args.instrument)
    
    M, N, beta = args.M, args.N, args.beta
    
    sessions = []
    cupti_profile_name = None
    if args.profile:
        cupti_profile_name = f"swiglu_cupti_wInstrument{args.instrument}"
        cupti_session = proton.start(
            cupti_profile_name, backend="cupti", hook="triton", data="tree"
        )
        sessions.append(cupti_session)
    if args.instrument:
        proton_mode = Default(buffer_size=args.buffer_size)
        instrument_session = proton.start(
            "swiglu_instrumented",
            backend="instrumentation",
            hook="triton",
            data=args.data,
            mode=proton_mode,
        )
        sessions.append(instrument_session)

    result = benchmark_swiglu(
        M, N, beta, use_cuda_event=args.use_cuda_event if args.instrument else True
    )

    for session in reversed(sessions):
        proton.finalize(session)

    if args.profile and cupti_profile_name:
        profile_path = Path(f"{cupti_profile_name}.hatchet")
        try:
            kernel_time_ns = extract_kernel_time_from_hatchet(profile_path, r"swiglu_kernel")
            log_cupti_profile_time("swiglu", args.instrument, kernel_time_ns)
        except Exception as exc:
            print(f"Failed to log CUPTI timing from {profile_path}: {exc}")

    print(f"Completed SwiGLU {M}x{N} (beta={beta}) (cupti={args.profile}, instrument={args.instrument})")
