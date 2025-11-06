import torch
import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl
from triton.profiler.mode import Default
import argparse

# Enable semantic for TTGIR override
pl.enable_semantic("triton")

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def instrumented_seeded_dropout_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    """Instrumented kernel for computing seeded dropout with intra-kernel profiling."""
    
    # Record kernel start
    pl.enter_scope("kernel_start")
    
    # Setup phase - compute memory offsets and program ID
    pl.enter_scope("setup")
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    pl.exit_scope("setup")
    
    # Main compute loop
    pl.enter_scope("main_compute_loop")
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Generate random numbers and compute dropout mask
    random = tl.rand(seed, offsets)
    x_keep = random > p
    
    # Apply dropout: zero out elements or scale by 1/(1-p)
    output = tl.where(x_keep, x / (1 - p), 0.0)
    pl.exit_scope("main_compute_loop")
    
    # Output phase - store results
    pl.enter_scope("output_phase")
    tl.store(output_ptr + offsets, output, mask=mask)
    pl.exit_scope("output_phase")
    
    # Record kernel end
    pl.exit_scope("kernel_start")


def instrumented_seeded_dropout(x, p, seed, use_cuda_event: bool = False):
    """Wrapper function for instrumented seeded dropout."""
    output = torch.empty_like(x)
    assert x.is_contiguous(), "Input tensor must be contiguous"
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    # Launch kernel
    if use_cuda_event:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(1_000_000)
        start_event.record()
    instrumented_seeded_dropout_kernel[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    if use_cuda_event:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        print(f"Outside seeded dropout elapsed time by cuda event: {elapsed_time} ms")
    return output


@triton.jit  
def instrumented_dropout_kernel(
    x_ptr,
    x_keep_ptr, 
    output_ptr,
    n_elements,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    """Instrumented kernel for traditional dropout with explicit mask."""
    
    # Record kernel start
    pl.enter_scope("kernel_start")
    
    # Setup phase - compute offsets and masks
    pl.enter_scope("setup")
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    pl.exit_scope("setup")
    
    # Main compute loop
    pl.enter_scope("main_compute_loop")
    # Load data and mask
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    
    # Apply dropout
    output = tl.where(x_keep, x / (1 - p), 0.0)
    pl.exit_scope("main_compute_loop")
    
    # Output phase - store results  
    pl.enter_scope("output_phase")
    tl.store(output_ptr + offsets, output, mask=mask)
    pl.exit_scope("output_phase")
    
    # Record kernel end
    pl.exit_scope("kernel_start")


def instrumented_dropout(x, x_keep, p):
    """Wrapper function for instrumented traditional dropout."""
    output = torch.empty_like(x)
    assert x.is_contiguous(), "Input tensor must be contiguous"
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    # Launch kernel
    instrumented_dropout_kernel[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


def benchmark_instrumented_dropout(n_elements, p=0.5, seed=123, device="cuda", dropout_type="seeded", use_cuda_event: bool = False):
    """Benchmark instrumented dropout kernels with cycle measurement."""
    x = torch.randn(size=(n_elements,), device=device, dtype=torch.float32)
    
    if dropout_type == "seeded":
        with proton.scope(f"seeded_dropout_{n_elements}"):
            # Warmup
            for _ in range(5):
                _ = instrumented_seeded_dropout(x, p, seed)
            # Actual benchmark
            result = instrumented_seeded_dropout(x, p, seed, use_cuda_event=use_cuda_event)

    else:
        # Traditional dropout with explicit mask
        x_keep = (torch.rand(size=(n_elements,), device=device) > p).to(torch.int32)
        
        with proton.scope(f"traditional_dropout_{n_elements}"):
            # Warmup  
            for _ in range(5):
                _ = instrumented_dropout(x, x_keep, p)
            
            # Actual benchmark
            result = instrumented_dropout(x, x_keep, p)
    
    return result


def naive_dropout(x, p, seed=None):
    """Naive PyTorch dropout implementation for comparison."""
    if seed is not None:
        torch.manual_seed(seed)
    
    mask = torch.rand_like(x) > p
    return torch.where(mask, x / (1 - p), 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--n-elements", type=int, default=50000000, help="Number of elements")
    parser.add_argument("--dropout-prob", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--dropout-type", choices=["seeded", "traditional"], default="seeded",
                        help="Type of dropout to benchmark")
    parser.add_argument("--data", type=str, default="tree", choices=["tree", "trace"], help="data to collect with Proton")
    parser.add_argument("--buffer-size", type=int, default=512, help="Proton buffer size")

    args = parser.parse_args()
    
    n_elements = args.n_elements
    p = args.dropout_prob
    seed = args.seed
    dropout_type = args.dropout_type
    
    if args.profile:
        proton_mode = Default(buffer_size=args.buffer_size)
        proton.start(f"dropout_{dropout_type}_instrumented", backend="instrumentation", hook="triton", data=args.data, mode=proton_mode)
        result = benchmark_instrumented_dropout(n_elements, p, seed, dropout_type=dropout_type)
        proton.finalize()
        print(f"Profiled instrumented {dropout_type} dropout {n_elements} elements")
    else:
        result = benchmark_instrumented_dropout(n_elements, p, seed, dropout_type=dropout_type, use_cuda_event=True)
        print(f"Ran instrumented {dropout_type} dropout {n_elements} elements")
    
    # Verify correctness for seeded dropout
    if dropout_type == "seeded":
        x = torch.randn(size=(10,), device=DEVICE)
        output1 = instrumented_seeded_dropout(x, p=0.5, seed=123)
        output2 = instrumented_seeded_dropout(x, p=0.5, seed=123)
        output3 = instrumented_seeded_dropout(x, p=0.5, seed=456)
        
        # Same seed should produce same output
        assert torch.allclose(output1, output2), "Same seed should produce same output"
        # Different seed should produce different output (with high probability)
        assert not torch.allclose(output1, output3), "Different seeds should produce different outputs"
        print("Seeded dropout correctness verified!")
    else:
        print("Traditional dropout executed successfully!")