import argparse
from pathlib import Path

import torch
import triton
import triton.profiler as proton
import triton.profiler.language as pl
from triton.profiler.mode import Default

from triton_kernels.matmul_ogs import (
    matmul_ogs,
)
from triton_kernels.matmul_ogs_details import opt_flags as opt_flags

from utils import (
    extract_kernel_time_from_hatchet,
    log_cupti_profile_time,
    set_profile_enabled,
)

DEVICE = triton.runtime.driver.active.get_active_torch_device()
CUPTI_KERNEL_PATTERN = r"matmul_ogs"
pl.enable_semantic("triton")


def benchmark_matmul_ogs(
    m_tokens: int,
    n: int,
    k: int,
):
    x = torch.randn(m_tokens, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.float32)

    _ = matmul_ogs(
        x,
        w,
        bias,
    )
    torch.cuda.synchronize()

    with proton.scope(
        f"matmul_ogs_{m_tokens}_{n}_{k}"
    ):
        for _ in range(3):
            matmul_ogs(
                x,
                w,
                bias,
            )
        torch.cuda.synchronize()
        result = matmul_ogs(
            x,
            w,
            bias,
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable CUPTI profiling")
    parser.add_argument(
        "--instrument",
        action="store_true",
        help="Enable Triton instrumentation profiling",
    )
    parser.add_argument("--tokens", type=int, default=8192, help="Number of input tokens (fixed benchmark case)")
    parser.add_argument("--n", type=int, default=512, help="Output dimension (fixed benchmark case)")
    parser.add_argument("--k", type=int, default=128, help="Hidden dimension (fixed benchmark case)")
    parser.add_argument(
        "--data",
        type=str,
        default="tree",
        choices=["tree", "trace"],
        help="Instrumentation backend data format",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=512,
        help="Instrumentation backend buffer size",
    )
    args = parser.parse_args()
    set_profile_enabled(args.instrument)

    sessions = []
    cupti_profile_name = None
    if args.profile:
        cupti_profile_name = f"matmul_ogs_cupti_wInstrument{args.instrument}"
        cupti_session = proton.start(cupti_profile_name, hook="triton", data="tree")
        sessions.append(cupti_session)
    if args.instrument:
        proton_mode = Default(buffer_size=args.buffer_size)
        instrument_session = proton.start(
            "matmul_ogs_instrumented",
            backend="instrumentation",
            hook="triton",
            data=args.data,
            mode=proton_mode,
        )
        sessions.append(instrument_session)

    try:
        result = benchmark_matmul_ogs(
            args.tokens,
            args.n,
            args.k,
        )
    finally:
        for session in reversed(sessions):
            proton.finalize(session)

    if args.profile and cupti_profile_name:
        profile_path = Path(f"{cupti_profile_name}.hatchet")
        try:
            kernel_time_ns = extract_kernel_time_from_hatchet(
                profile_path, CUPTI_KERNEL_PATTERN
            )
            log_cupti_profile_time("matmul_ogs", args.instrument, kernel_time_ns)
        except Exception as exc:
            print(f"Failed to log CUPTI timing from {profile_path}: {exc}")

    print(
        "Completed matmul_ogs "
        f"(tokens={args.tokens}, n={args.n}, k={args.k}, "
        f"(cupti={args.profile}, instrument={args.instrument})"
    )
    return result


if __name__ == "__main__":
    main()
