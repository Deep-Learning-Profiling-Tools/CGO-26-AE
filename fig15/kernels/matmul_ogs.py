import argparse
import inspect
from pathlib import Path

import torch
import triton
import triton.profiler as proton
import triton.profiler.language as pl
from triton.profiler.mode import Default

from triton_kernels.matmul_ogs import (
    GatherIndx,
    PrecisionConfig,
    RoutingData,
    ScatterIndx,
    matmul_ogs,
    specializations,
)
from triton_kernels.matmul_ogs_details import _matmul_ogs as matmul_ogs_kernels
import triton_kernels.matmul_ogs_details.opt_flags as opt_flags

from utils import (
    extract_kernel_time_from_hatchet,
    log_cupti_profile_time,
    log_cuda_event_time,
    set_profile_enabled,
)

DEVICE = triton.runtime.driver.active.get_active_torch_device()
CUPTI_KERNEL_PATTERN = r"matmul_ogs"
pl.enable_semantic("triton")


def _instrument_matmul_ogs_kernel():
    import importlib.util
    import tempfile
    import atexit

    base_kernel = matmul_ogs_kernels._matmul_ogs
    src = inspect.getsource(base_kernel.fn)
    src = src.replace("def _matmul_ogs(", "def _instrumented_matmul_ogs(")
    src = src.replace(
        "    tl.assume(grid_n >= 0)\n\n",
        "    tl.assume(grid_n >= 0)\n\n"
        "    pl.enter_scope(\"kernel_start\")\n"
        "    pl.enter_scope(\"setup_phase\")\n\n",
    )
    src = src.replace(
        "    for ki in range(k_tiles):",
        "    pl.exit_scope(\"setup_phase\")\n"
        "    pl.enter_scope(\"compute_loop\")\n"
        "    for ki in range(k_tiles):",
    )
    src = src.replace(
        "    if OutAcc is not None:",
        "    pl.exit_scope(\"compute_loop\")\n\n"
        "    if OutAcc is not None:",
    )
    if not src.rstrip().endswith("pl.exit_scope(\"kernel_start\")"):
        src = src.rstrip() + "\n    pl.exit_scope(\"kernel_start\")\n"

    module_header = (
        "import triton\n"
        "import triton.language as tl\n"
        "import triton.profiler.language as pl\n"
        "import triton_kernels.matmul_ogs_details._matmul_ogs as base_mod\n"
        "for _name in dir(base_mod):\n"
        "    globals()[_name] = getattr(base_mod, _name)\n"
        "del _name\n"
        "del base_mod\n"
        'pl.enable_semantic(\"triton\")\n'
    )
    with tempfile.NamedTemporaryFile("w", suffix="_instrumented_matmul_ogs.py", delete=False) as handle:
        handle.write(module_header + src + "\n")
        module_path = Path(handle.name)

    atexit.register(lambda: module_path.unlink(missing_ok=True))

    spec = importlib.util.spec_from_file_location(
        "_instrumented_matmul_ogs_impl", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    instrumented = module._instrumented_matmul_ogs
    matmul_ogs_kernels._matmul_ogs = instrumented
    specializations.kernels = [
        (name, instrumented if name == "_matmul_ogs" else kernel)
        for name, kernel in specializations.kernels
    ]
    specializations._modules = dict()


_instrument_matmul_ogs_kernel()


_TEST_HELPERS = None


def _load_test_helpers():
    import importlib.util
    global _TEST_HELPERS
    if _TEST_HELPERS is not None:
        return _TEST_HELPERS
    root = Path(__file__).resolve().parents[2]
    tests_path = root / "submodules" / "triton" / "python" / "triton_kernels" / "tests" / "test_matmul.py"
    spec = importlib.util.spec_from_file_location("tk_test_matmul_module", tests_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _TEST_HELPERS = module
    return module


def _prepare_known_case(
    device: torch.device,
    m: int,
    n: int,
    k: int,
    n_expts_tot: int,
    n_expts_act: int,
):
    # expected = (320, 400, 400, 8, 4)
    # if (m, n, k, n_expts_tot, n_expts_act) != expected:
    #     raise ValueError(
    #         f"This harness currently supports the fixed configuration {expected}; "
    #         f"got {(m, n, k, n_expts_tot, n_expts_act)}."
    #     )
    mode = "ragged"
    do_gather = False
    do_scatter = False
    has_y_gammas = False
    inner_expt_opt = None
    split_k = 1
    block_m = 128
    is_persistent = False

    opt_flags.reset_opt_flags_constraints()
    opt_flags.update_opt_flags_constraints(
        {
            "block_m": block_m,
            "split_k": split_k,
            "is_persistent": is_persistent,
            "epilogue_subtile": None,
        }
    )

    helpers = _load_test_helpers()

    precision_config = helpers.init_precision(
        torch.float16,
        act_use_flexpoint=False,
        weight_dtype=torch.float16,
        weight_mxfp=False,
        mode=mode,
        n_expts_tot=n_expts_tot,
        expt_is_inner=inner_expt_opt is not None,
        device=device,
    )

    torch.manual_seed(0)
    m_routed, routing_data, gather_indx, scatter_indx = helpers.init_routing_data(
        m, n_expts_tot, n_expts_act, do_gather, do_scatter, device=device
    )
    m = m_routed  # Align with routing metadata produced by helper.

    x, w, bias, _, _ = helpers.init_compute_data(
        m,
        n,
        k,
        routing_data,
        gather_indx,
        scatter_indx,
        n_expts_tot,
        n_expts_act,
        mode,
        torch.float16,
        torch.float16,
        has_y_gammas,
        requires_grad=False,
        device=device,
        inner_expt_opt=inner_expt_opt,
    )
    return x, w, bias, routing_data, gather_indx, scatter_indx, precision_config


def _run_matmul_ogs(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    routing_data: RoutingData,
    gather_indx: GatherIndx,
    scatter_indx: ScatterIndx,
    precision_config: PrecisionConfig,
    use_cuda_event: bool,
):
    if use_cuda_event:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(1_000_000)
        start_event.record()
    result = matmul_ogs(
        x,
        w,
        bias,
        routing_data=routing_data,
        gather_indx=gather_indx,
        scatter_indx=scatter_indx,
        precision_config=precision_config,
    )
    if use_cuda_event:
        end_event.record()
    torch.cuda.synchronize()
    if use_cuda_event:
        elapsed_ms = start_event.elapsed_time(end_event)
        log_cuda_event_time("matmul_ogs", elapsed_ms)
        print(f"Outside matmul_ogs elapsed time by cuda event: {elapsed_ms:.6f} ms")
    return result


def benchmark_matmul_ogs(
    m_tokens: int,
    n: int,
    k: int,
    n_expts_tot: int,
    n_expts_act: int,
    *,
    device: torch.device = torch.device(DEVICE),
    use_cuda_event: bool = False,
):
    (
        x,
        w,
        bias,
        routing_data,
        gather_indx,
        scatter_indx,
        precision_config,
    ) = _prepare_known_case(
        device, m_tokens, n, k, n_expts_tot, n_expts_act
    )

    try:
        _ = matmul_ogs(
            x,
            w,
            bias,
            routing_data=routing_data,
            gather_indx=gather_indx,
            scatter_indx=scatter_indx,
            precision_config=precision_config,
        )
        torch.cuda.synchronize()

        with proton.scope(
            f"matmul_ogs_{m_tokens}_{n}_{k}_{n_expts_tot}_{n_expts_act}"
        ):
            for _ in range(3):
                matmul_ogs(
                    x,
                    w,
                    bias,
                    routing_data=routing_data,
                    gather_indx=gather_indx,
                    scatter_indx=scatter_indx,
                    precision_config=precision_config,
                )
            torch.cuda.synchronize()
            result = _run_matmul_ogs(
                x,
                w,
                bias,
                routing_data,
                gather_indx,
                scatter_indx,
                precision_config,
                use_cuda_event=use_cuda_event,
            )
    finally:
        opt_flags.reset_opt_flags_constraints()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable CUPTI profiling")
    parser.add_argument(
        "--instrument",
        action="store_true",
        help="Enable Triton instrumentation profiling",
    )
    parser.add_argument("--tokens", type=int, default=1024, help="Number of input tokens (fixed benchmark case)")
    parser.add_argument("--n", type=int, default=512, help="Output dimension (fixed benchmark case)")
    parser.add_argument("--k", type=int, default=128, help="Hidden dimension (fixed benchmark case)")
    parser.add_argument("--n-experts", type=int, default=64, help="Total experts (fixed benchmark case)")
    parser.add_argument("--n-active", type=int, default=4, help="Experts per token (fixed benchmark case)")
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
    parser.add_argument(
        "--use-cuda-event",
        action="store_true",
        help="Record CUDA event timing for the profiled run",
    )
    args = parser.parse_args()
    set_profile_enabled(args.instrument)

    # expected = (320, 400, 400, 8, 4)
    # if (args.tokens, args.n, args.k, args.n_experts, args.n_active) != expected:
    #     parser.error(
    #         f"Benchmark currently supports only tokens={expected[0]}, n={expected[1]}, "
    #         f"k={expected[2]}, n-experts={expected[3]}, n-active={expected[4]}."
    #     )

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
            args.n_experts,
            args.n_active,
            use_cuda_event=args.use_cuda_event if args.instrument else True,
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
        f"experts={args.n_experts}, active={args.n_active}) "
        f"(cupti={args.profile}, instrument={args.instrument})"
    )
    return result


if __name__ == "__main__":
    main()
