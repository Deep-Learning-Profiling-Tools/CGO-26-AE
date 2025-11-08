import torch
import triton
import triton.profiler as proton
import triton.profiler.language as pl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma_init,
    warpgroup_mma,
    warpgroup_mma_wait,
)
import argparse


def is_hopper():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 9


@gluon.jit
def instrumented_matmul_kernel(a_desc, b_desc, c_desc, num_warps: gl.constexpr):
    """Instrumented Gluon kernel for computing matmul with intra-kernel profiling."""
    
    # Record kernel start
    pl.enter_scope("gluon_kernel_start")
    
    # Get block dimensions
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    # Memory allocation phase
    pl.enter_scope("memory_allocation")
    a_smem = gl.allocate_shared_memory(dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, b_desc.block_type.shape, b_desc.layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    pl.exit_scope("memory_allocation")

    # Program ID setup
    pl.enter_scope("program_setup")
    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    pl.exit_scope("program_setup")

    # MMA layout setup
    pl.enter_scope("mma_layout_setup")
    warps_per_cta: gl.constexpr = get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps)
    m: gl.constexpr = 16
    k: gl.constexpr = 256 // dtype.primitive_bitwidth
    n: gl.constexpr = get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps)
    
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )
    
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
    phase = 0
    pl.exit_scope("mma_layout_setup")

    # Main computation loop
    pl.enter_scope("main_compute_loop")
    for k_iter in range(0, K, BLOCK_K):
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k_iter], bar, a_smem)
        tma.async_copy_global_to_shared(b_desc, [k_iter, off_n], bar, b_smem)
        mbarrier.wait(bar, phase=phase)
        phase ^= 1
        acc = warpgroup_mma(a_smem, b_smem, acc, is_async=True)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc, ))
    pl.exit_scope("main_compute_loop")

    # Memory cleanup and output
    pl.enter_scope("output_phase")
    mbarrier.invalidate(bar)
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)
    pl.exit_scope("output_phase")
    
    # Record kernel end
    pl.exit_scope("gluon_kernel_start")


@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    """Calculate optimal warps per CTA configuration."""
    warps_per_cta = [4, 1]
    m = 16
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        if BLOCK_M > m * warps_per_cta[0]:
            warps_per_cta[0] *= 2
        else:
            warps_per_cta[1] *= 2
    return warps_per_cta


@gluon.constexpr_function
def get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps):
    """Calculate optimal instruction shape N."""
    m = 16
    mReps = triton.cdiv(BLOCK_M, m)
    nReps = triton.cdiv(num_warps, mReps)
    maxN = max(BLOCK_N // nReps, 8)
    n = 256
    while n > maxN or BLOCK_N % n != 0:
        n -= 8
    assert n >= 8, "expected to find a valid n"
    return n


def instrumented_gluon_matmul(A, B, C, BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=4):
    """Wrapper function for instrumented Gluon matmul."""
    if not is_hopper():
        raise RuntimeError("Gluon WGMMA requires a Hopper GPU")
    
    M, N = C.shape
    
    # Setup tensor descriptors with NVMMA shared layouts
    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    instrumented_matmul_kernel[grid](a_desc, b_desc, c_desc, num_warps=num_warps)
    

def benchmark_instrumented_gluon(M, N, K, device="cuda"):
    """Benchmark instrumented Gluon kernel with cycle measurement."""
    if not is_hopper():
        print("Skipping Gluon benchmark - requires Hopper GPU")
        return None
        
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)
    C = torch.empty(M, N, device=device, dtype=torch.float16)
    
    # Choose block sizes based on matrix dimensions
    BLOCK_M = min(64, M)
    BLOCK_N = min(128, N) 
    BLOCK_K = min(64, K)
    num_warps = 4
    
    with proton.scope(f"gluon_matmul_{M}_{N}_{K}"):
        # Warmup
        for _ in range(5):
            try:
                instrumented_gluon_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
            except Exception as e:
                print(f"Gluon kernel failed during warmup: {e}")
                return None
        
        # Actual benchmark
        try:
            instrumented_gluon_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
        except Exception as e:
            print(f"Gluon kernel failed during benchmark: {e}")
            return None
    
    return C


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--matrix-size", type=int, default=512, help="Matrix size")
    args = parser.parse_args()
    
    M = N = K = args.matrix_size
    
    if not is_hopper():
        print("This benchmark requires a Hopper NVIDIA GPU")
        exit(1)
    
    if args.profile:
        proton.start("gluon_instrumented", backend="instrumentation", hook="triton")
        result = benchmark_instrumented_gluon(M, N, K)
        proton.finalize()
        if result is not None:
            print(f"Profiled instrumented Gluon matmul {M}x{N}x{K}")
        else:
            print("Gluon benchmark failed")
    else:
        result = benchmark_instrumented_gluon(M, N, K)
        if result is not None:
            print(f"Ran instrumented Gluon matmul {M}x{N}x{K}")
        else:
            print("Gluon benchmark failed")