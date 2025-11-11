import os
import json
import csv
import re
import subprocess
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple
import torch

_CUDA_EVENT_CSV_PATH = Path(__file__).with_name("cuda_event_timings.csv")
_CUDA_EVENT_CSV_LOCK = Lock()
_PROFILE_ENABLED = False
# Support platform-specific CSV output via environment variable
_CUPTI_CSV_FILENAME = os.environ.get("CUPTI_TIMING_CSV_FILENAME", "cupti_profile_timings.csv")
_CUPTI_TIMING_CSV_PATH = Path(__file__).with_name(_CUPTI_CSV_FILENAME)
_CUPTI_TIMING_CSV_LOCK = Lock()


def set_profile_enabled(enabled: bool) -> None:
    """Update global profiling state so timing logs tag runs consistently."""
    global _PROFILE_ENABLED
    _PROFILE_ENABLED = bool(enabled)


def is_profile_enabled() -> bool:
    """Return whether profiling/instrumentation was requested at the CLI."""
    return _PROFILE_ENABLED


def log_cuda_event_time(kernel_name: str, elapsed_ms: float, instrumented: Optional[bool] = None) -> None:
    """Persist CUDA event timings to a CSV for cross-kernel comparisons."""
    effective_instrumented = is_profile_enabled() if instrumented is None else instrumented
    record = [kernel_name, "instrumented" if effective_instrumented else "baseline", f"{elapsed_ms:.6f}"]
    with _CUDA_EVENT_CSV_LOCK:
        file_exists = _CUDA_EVENT_CSV_PATH.exists()
        _CUDA_EVENT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _CUDA_EVENT_CSV_PATH.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if not file_exists:
                writer.writerow(["kernel", "configuration", "elapsed_ms"])
            writer.writerow(record)


def log_cupti_profile_time(kernel_name: str, instrumented: bool, time_ns: float) -> None:
    """Append CUPTI-derived kernel timings into a CSV for later analysis."""
    record = [
        kernel_name,
        "instrumented" if instrumented else "baseline",
        f"{time_ns:.0f}",
        f"{time_ns / 1e6:.6f}",
    ]
    with _CUPTI_TIMING_CSV_LOCK:
        file_exists = _CUPTI_TIMING_CSV_PATH.exists()
        _CUPTI_TIMING_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _CUPTI_TIMING_CSV_PATH.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if not file_exists:
                writer.writerow(["kernel", "configuration", "time_ns", "time_ms"])
            writer.writerow(record)


def _extract_time_values(metrics: Dict) -> List[float]:
    values = []
    for key, value in (metrics or {}).items():
        if not isinstance(value, (int, float)):
            continue
        key_lower = key.lower()
        if "/s" in key_lower:
            continue
        if ("time" in key_lower or "duration" in key_lower) and any(unit in key_lower for unit in ("ns", "us", "ms", "s")):
            multiplier = 1.0
            if "ns" in key_lower:
                multiplier = 1.0
            elif "us" in key_lower:
                multiplier = 1e3
            elif "ms" in key_lower:
                multiplier = 1e6
            elif "s" in key_lower:
                multiplier = 1e9
            values.append(float(value) * multiplier)
    return values


def _collect_time_metrics(node: Dict, totals: List[float]) -> None:
    totals.extend(_extract_time_values(node.get("metrics")))
    for child in node.get("children", []):
        _collect_time_metrics(child, totals)


def extract_total_time_from_hatchet(path: Path) -> float:
    """Return the largest time-like metric (in nanoseconds) found in a hatchet JSON profile."""
    with path.open() as handle:
        data = json.load(handle)

    totals: List[float] = []
    for root in data if isinstance(data, list) else [data]:
        _collect_time_metrics(root, totals)

    if not totals:
        raise ValueError(f"No timing metrics found in hatchet profile: {path}")
    return max(totals)


def summarize_cupti_profiles(profile_dir: Path, output_csv: Path) -> Path:
    """Aggregate total timings from CUPTI hatchet profiles into a CSV."""
    profile_dir = Path(profile_dir)
    output_csv = Path(output_csv)
    hatchet_files = sorted(profile_dir.glob("*_cupti_wInstrument*.hatchet"))

    if not hatchet_files:
        raise FileNotFoundError(f"No CUPTI hatchet files found in {profile_dir}")

    rows = []
    pattern = re.compile(r"^(?P<kernel>.+)_cupti_wInstrument(?P<instrument>True|False)$")

    for path in hatchet_files:
        match = pattern.match(path.stem)
        if not match:
            continue
        kernel = match.group("kernel")
        instrumented = match.group("instrument") == "True"
        total_ns = extract_total_time_from_hatchet(path)
        rows.append(
            {
                "kernel": kernel,
                "instrumented": instrumented,
                "time_ns": f"{total_ns:.0f}",
                "time_ms": f"{total_ns / 1e6:.6f}",
            }
        )

    if not rows:
        raise ValueError(f"No CUPTI hatchet files matched expected naming in {profile_dir}")

    rows.sort(key=lambda r: (r["kernel"], r["instrumented"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kernel", "instrumented", "time_ns", "time_ms"])
        writer.writeheader()
        writer.writerows(rows)

    return output_csv


def extract_kernel_time_from_hatchet(path: Path, pattern: str) -> float:
    """Sum timing metrics for frames whose name matches the provided regex pattern."""
    regex = re.compile(pattern)

    with path.open() as handle:
        data = json.load(handle)

    totals: List[float] = []

    def visit(node: Dict) -> None:
        frame = node.get("frame") or {}
        name = frame.get("name", "")
        if regex.search(name):
            totals.extend(_extract_time_values(node.get("metrics")))
        for child in node.get("children", []):
            visit(child)

    for root in data if isinstance(data, list) else [data]:
        visit(root)

    if not totals:
        raise ValueError(f"No matching kernel '{pattern}' found in {path}")
    return sum(totals)


def is_hatchet_available():
    """Check if hatchet is available for profile analysis."""
    try:
        subprocess.run(["proton-viewer", "-h"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_proton_viewer(profile_path: str, metrics: List[str] = None, sorted_output: bool = False) -> Optional[str]:
    """Run proton-viewer on a profile and return the output."""
    if not os.path.exists(profile_path):
        print(f"Profile not found: {profile_path}")
        return None
    
    if not is_hatchet_available():
        print("proton-viewer not available. Install with: pip install llnl-hatchet")
        return None
    
    cmd = ["proton-viewer"]
    
    if metrics:
        cmd.extend(["-m", ",".join(metrics)])
    
    if sorted_output:
        cmd.append("--print-sorted")
    
    cmd.append(profile_path)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running proton-viewer: {e}")
        print(f"stderr: {e.stderr}")
        return None


def analyze_profile(profile_path: str, verbose: bool = True) -> Dict:
    """Analyze a profile and extract key metrics."""
    analysis = {
        "profile_path": profile_path,
        "exists": os.path.exists(profile_path),
        "metrics": {},
        "summary": {},
    }
    
    if not analysis["exists"]:
        if verbose:
            print(f"Profile not found: {profile_path}")
        return analysis
    
    # Try to get basic profile info with cycles and time
    metrics = ["time/ns", "cycles"]
    output = run_proton_viewer(profile_path, metrics)
    
    if output:
        analysis["raw_output"] = output
        
        # Parse basic statistics from output
        lines = output.strip().split('\n')
        for line in lines:
            if "kernel_start" in line and "cycles" in line:
                # Extract cycles information
                parts = line.split()
                try:
                    cycles_idx = next(i for i, part in enumerate(parts) if "cycles" in part)
                    if cycles_idx > 0:
                        cycles_value = parts[cycles_idx - 1]
                        analysis["metrics"]["total_cycles"] = cycles_value
                except (ValueError, StopIteration):
                    pass
            
            if "time/ns" in line:
                # Extract timing information
                parts = line.split()
                try:
                    time_idx = next(i for i, part in enumerate(parts) if "/" in part and "ns" in part)
                    if time_idx > 0:
                        time_value = parts[time_idx - 1]
                        analysis["metrics"]["total_time_ns"] = time_value
                except (ValueError, StopIteration):
                    pass
    
    # Get sorted output for detailed breakdown
    sorted_output = run_proton_viewer(profile_path, metrics, sorted_output=True)
    if sorted_output:
        analysis["sorted_output"] = sorted_output
    
    if verbose:
        print(f"Profile analysis for: {profile_path}")
        print(f"  Exists: {analysis['exists']}")
        print(f"  Metrics: {analysis['metrics']}")
    
    return analysis


def compare_profiles(triton_profile: str, gluon_profile: str = None) -> Dict:
    """Compare profiles between different implementations."""
    comparison = {
        "triton": analyze_profile(triton_profile, verbose=False) if triton_profile else None,
        "gluon": analyze_profile(gluon_profile, verbose=False) if gluon_profile else None,
        "comparison": {}
    }
    
    if comparison["triton"] and comparison["gluon"]:
        triton_metrics = comparison["triton"]["metrics"]
        gluon_metrics = comparison["gluon"]["metrics"]
        
        # Compare cycles if available
        if "total_cycles" in triton_metrics and "total_cycles" in gluon_metrics:
            try:
                triton_cycles = float(triton_metrics["total_cycles"].replace(",", ""))
                gluon_cycles = float(gluon_metrics["total_cycles"].replace(",", ""))
                comparison["comparison"]["cycles_ratio"] = triton_cycles / gluon_cycles
            except ValueError:
                pass
        
        # Compare timing if available
        if "total_time_ns" in triton_metrics and "total_time_ns" in gluon_metrics:
            try:
                triton_time = float(triton_metrics["total_time_ns"].replace(",", ""))
                gluon_time = float(gluon_metrics["total_time_ns"].replace(",", ""))
                comparison["comparison"]["time_ratio"] = triton_time / gluon_time
            except ValueError:
                pass
    
    return comparison


def extract_kernel_phases(profile_path: str) -> Dict[str, Dict]:
    """Extract timing information for different kernel phases."""
    phases = {}
    
    if not os.path.exists(profile_path):
        return phases
    
    # Get detailed output with phase breakdown
    output = run_proton_viewer(profile_path, ["time/ns", "cycles"])
    
    if not output:
        return phases
    
    lines = output.strip().split('\n')
    current_phase = None
    
    for line in lines:
        line = line.strip()
        
        # Look for phase markers
        phase_markers = [
            "setup", "pointer_setup", "accumulator_init", 
            "compute_loop", "load_phase", "compute_phase", 
            "ptr_advance", "activation", "output_phase",
            "memory_allocation", "program_setup", "mma_layout_setup",
            "tma_load_phase", "wgmma_phase"
        ]
        
        for marker in phase_markers:
            if marker in line.lower():
                current_phase = marker
                phases[marker] = {"line": line, "metrics": {}}
                
                # Try to extract metrics from the line
                parts = line.split()
                for i, part in enumerate(parts):
                    if "ns" in part and i > 0:
                        try:
                            phases[marker]["metrics"]["time_ns"] = float(parts[i-1].replace(",", ""))
                        except ValueError:
                            pass
                    elif "cycles" in part and i > 0:
                        try:
                            phases[marker]["metrics"]["cycles"] = float(parts[i-1].replace(",", ""))
                        except ValueError:
                            pass
                break
    
    return phases


def generate_summary_report(profile_dir: str, matrix_sizes: List[int]) -> str:
    """Generate a summary report for all profiles."""
    report = []
    report.append("=== Intra-Kernel Profiling Summary Report ===\n")
    
    for size in matrix_sizes:
        report.append(f"Matrix Size: {size}x{size}x{size}")
        report.append("-" * 40)
        
        triton_profile = os.path.join(profile_dir, f"triton_{size}x{size}x{size}.hatchet")
        gluon_profile = os.path.join(profile_dir, f"gluon_{size}x{size}x{size}.hatchet")
        
        triton_exists = os.path.exists(triton_profile)
        gluon_exists = os.path.exists(gluon_profile)
        
        report.append(f"  Triton Profile: {'✓' if triton_exists else '✗'}")
        report.append(f"  Gluon Profile:  {'✓' if gluon_exists else '✗'}")
        
        if triton_exists:
            triton_analysis = analyze_profile(triton_profile, verbose=False)
            if triton_analysis["metrics"]:
                report.append(f"  Triton Metrics: {triton_analysis['metrics']}")
        
        if gluon_exists:
            gluon_analysis = analyze_profile(gluon_profile, verbose=False)
            if gluon_analysis["metrics"]:
                report.append(f"  Gluon Metrics:  {gluon_analysis['metrics']}")
        
        if triton_exists and gluon_exists:
            comparison = compare_profiles(triton_profile, gluon_profile)
            if comparison["comparison"]:
                report.append(f"  Comparison: {comparison['comparison']}")
        
        report.append("")
    
    return "\n".join(report)


def get_gpu_info() -> Dict[str, str]:
    """Get GPU information for the report."""
    info = {}
    try:
        info["name"] = torch.cuda.get_device_name()
        info["capability"] = str(torch.cuda.get_device_capability())
        info["memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        info["sm_count"] = str(torch.cuda.get_device_properties(0).multi_processor_count)
    except Exception as e:
        info["error"] = str(e)
    
    return info


def save_analysis_json(analysis_data: Dict, output_path: str):
    """Save analysis data to JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        print(f"Analysis saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save analysis: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Proton profiles")
    parser.add_argument("profile_path", help="Path to the profile file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--phases", action="store_true", help="Extract kernel phases")
    
    args = parser.parse_args()
    
    analysis = analyze_profile(args.profile_path, args.verbose)
    
    if args.phases:
        phases = extract_kernel_phases(args.profile_path)
        print("\nKernel Phases:")
        for phase, data in phases.items():
            print(f"  {phase}: {data['metrics']}")
    
    print("\nFull Analysis:")
    print(json.dumps(analysis, indent=2, default=str))
