#!/usr/bin/env python3
"""
Experimental script for fig13, 14.
Runs each model with different profiling options and collects results into a table.
"""

import os
import subprocess
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import argparse
import shutil
import glob
import time

# Model names mapping (file -> display name)
MODEL_MAPPING = {
    "llama3_1_8b.py": "Llama-3.1-8B",
    "llama3_70b_bnb_4bit.py": "Llama-3.3-70B",
    "gemma3_4b.py": "Gemma3-4B",
    "qwen3_30b_a3b.py": "Qwen3-30B-A3B",
    "gpt_oss_20b.py": "GPT-OSS-20B",
    "mistral_7b_instruct.py": "Mistral-7B",
    "phi_3_5_mini_instruct.py": "Phi-3.5-Mini",
    "smollm2_1_7b_instruct.py": "SmolLM2.1-7B",
    "tinyllama_chat.py": "TinyLlama-Chat",
    "zephyr_sft.py": "Zephyr-SFT"
}

# Profiling options
PROFILING_OPTIONS = [
    ("baseline", []),
    ("torch", ["--profiling", "torch"]),
    ("proton", ["--profiling", "proton"]),
    ("nsys", [])  # Special case, will be handled differently
]


def run_command(cmd: List[str], output_file: str, use_nsys: bool = False, dry_run: bool = False) -> bool:
    """Run a command and save output to file."""
    if use_nsys:
        # Prepend nsys command
        nsys_cmd = [
            "nsys", "profile",
            "--output", output_file.replace(".log", ".nsys-rep"),
            "--force-overwrite", "true",
            "--trace", "cuda"
        ]
        cmd = nsys_cmd + cmd

    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {output_file}")

    if dry_run:
        print("  [DRY RUN - not executed]")
        return True

    try:
        with open(output_file, "w") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
        return result.returncode == 0
    except Exception as e:
        print(f"  Error running command: {e}")
        with open(output_file, "a") as f:
            f.write(f"\n\n[ERROR: {e}]\n")
        return False


def extract_training_time(log_file: str, dry_run: bool = False) -> Optional[float]:
    """Extract training time from log file."""
    if dry_run:
        return 0.0

    try:
        with open(log_file, "r") as f:
            content = f.read()

        # Look for pattern: "Training loop completed in XX.XX seconds"
        pattern = r"Training loop completed in ([\d.]+) seconds"
        match = re.search(pattern, content)

        if match:
            return float(match.group(1))
        else:
            print(f"  Could not find training time in {log_file}")
            return None
    except Exception as e:
        print(f"  Error reading {log_file}: {e}")
        return None


def clean_experiment_files():
    """Clean up experiment files and directories."""
    script_dir = Path(__file__).parent

    print("Cleaning up experiment files...")

    # Files to remove (with glob patterns)
    file_patterns = [
        "*_standalone.json",
        "*_standalone.hatchet"
    ]

    # Directories to remove
    directories = [
        "experiment_outputs_nv",
        "unsloth_compiled_cache",
        "unsloth_training_checkpoints",
        "_unsloth_sentencepiece_temp",
        "gpt_oss_20b_outputs",
        "llama3_70b_outputs",
        "mistral_7b_outputs",
        "phi_3_5_mini_outputs",
        "tinyllama_chat_outputs",
        "smollm2_1_7b_outputs",
        "zephyr_sft_outputs"
    ]

    # Remove files matching patterns
    for pattern in file_patterns:
        matches = glob.glob(str(script_dir / pattern))
        for file_path in matches:
            try:
                os.remove(file_path)
                print(f"  Removed file: {file_path}")
            except Exception as e:
                print(f"  Error removing {file_path}: {e}")

    # Remove directories
    for dir_name in directories:
        dir_path = script_dir / dir_name
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"  Removed directory: {dir_path}")
            except Exception as e:
                print(f"  Error removing {dir_path}: {e}")
        else:
            print(f"  Directory not found (skipping): {dir_path}")

    print("\nCleanup completed!")


def get_file_size(file_path: Path) -> Optional[int]:
    """Get file size in bytes. Returns None if file doesn't exist."""
    try:
        if file_path.exists():
            return file_path.stat().st_size
        else:
            return None
    except Exception as e:
        print(f"  Error getting file size for {file_path}: {e}")
        return None


def run_model_experiment(model_file: str, model_runners_dir: Path, output_dir: Path, dry_run: bool = False) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[int]]]:
    """Run all experiments for a single model and measure profiling file sizes."""
    model_name = MODEL_MAPPING.get(model_file, model_file.replace(".py", ""))
    model_base_name = model_file.replace('.py', '')
    print(f"\n{'='*60}")
    print(f"Running experiments for {model_name}")
    print(f"{'='*60}")

    model_path = model_runners_dir / model_file
    results = {}
    file_sizes = {}

    script_dir = output_dir.parent

    # Warmup run
    print("\n[Warmup run]")
    warmup_log = output_dir / f"{model_base_name}_warmup.log"
    run_command(["python", str(model_path)], str(warmup_log), dry_run=dry_run)

    # Run each profiling option
    for option_name, option_args in PROFILING_OPTIONS:
        print(f"\n[Running with {option_name}]")

        output_file = output_dir / f"{model_base_name}_{option_name}.log"

        if option_name == "nsys":
            # Special handling for nsys
            success = run_command(
                ["python", str(model_path)],
                str(output_file),
                use_nsys=True,
                dry_run=dry_run
            )
        else:
            # Regular run
            cmd = ["python", str(model_path)] + option_args
            success = run_command(cmd, str(output_file), dry_run=dry_run)

        if success:
            time_val = extract_training_time(str(output_file), dry_run=dry_run)
            results[option_name] = time_val
            if time_val:
                print(f"  Training time: {time_val:.2f} seconds")
        else:
            results[option_name] = None
            print(f"  Execution failed")

        # Measure profiling file size
        if not dry_run and option_name in ["torch", "proton", "nsys"]:
            profiling_file = None
            if option_name == "torch":
                profiling_file = script_dir / f"{model_base_name}_trace_standalone.json"
            elif option_name == "proton":
                profiling_file = script_dir / f"{model_base_name}_standalone.hatchet"
            elif option_name == "nsys":
                profiling_file = output_dir / f"{model_base_name}_nsys.nsys-rep"

            if profiling_file:
                file_size = get_file_size(profiling_file)
                file_sizes[option_name] = file_size
                if file_size is not None:
                    print(f"  Profiling file size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
                else:
                    print(f"  Warning: Profiling file not found: {profiling_file}")

    return results, file_sizes


def print_table(results: Dict[str, Dict[str, Optional[float]]]):
    """Print results table."""
    # Headers
    headers = ["Name", "gh200-baseline", "gh200-torch", "gh200-proton", "gh200-nsys"]
    col_widths = [max(20, max(len(name) for name in results.keys()) + 2)]
    col_widths.extend([15] * 4)

    # Print header
    print("\n" + "="*sum(col_widths))
    header_line = ""
    for i, header in enumerate(headers):
        header_line += f"{header:<{col_widths[i]}}"
    print(header_line)
    print("-"*sum(col_widths))

    # Print data
    for model_name in results:
        row = f"{model_name:<{col_widths[0]}}"
        for option in ["baseline", "torch", "proton", "nsys"]:
            value = results[model_name].get(option)
            col_idx = headers.index(f'gh200-{option}')
            if value is not None:
                row += f"{value:<{col_widths[col_idx]}.2f}"
            else:
                row += f"{'N/A':<{col_widths[col_idx]}}"
        print(row)

    print("="*sum(col_widths))


def generate_python_dict(results: Dict[str, Dict[str, Optional[float]]]):
    """Generate Python dictionary format."""
    print("\n\nPython Dictionary Format:")
    print("="*60)
    print("raw = [")

    for model_file, display_name in MODEL_MAPPING.items():
        if display_name in results:
            row = f'    ["{display_name}"'
            for option in ["baseline", "torch", "proton", "nsys"]:
                value = results[display_name].get(option)
                if value is not None:
                    row += f", {value:.2f}"
                else:
                    row += ", np.nan"
            row += "],"
            print(row)

    print("]")


def main():
    """Main function."""
    # Record start time
    start_time = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run model experiments with different profiling options")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing them")
    parser.add_argument("--clean", action="store_true", help="Clean up experiment files and directories")
    args = parser.parse_args()

    # Handle clean command
    if args.clean:
        clean_experiment_files()
        return

    # Setup paths
    script_dir = Path(__file__).parent
    model_runners_dir = script_dir / "model_runners"
    output_dir = script_dir / "experiment_outputs_nv"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Get list of models
    model_files = sorted([f for f in os.listdir(model_runners_dir) if f.endswith(".py")])

    print(f"Found {len(model_files)} models to test")
    print(f"Output directory: {output_dir}")

    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN MODE - Commands will be displayed but not executed")
        print("="*60)

    # Run experiments
    all_results = {}
    all_file_sizes = {}

    for model_file in model_files:
        model_name = MODEL_MAPPING.get(model_file, model_file.replace(".py", ""))
        results, file_sizes = run_model_experiment(model_file, model_runners_dir, output_dir, dry_run=args.dry_run)
        all_results[model_name] = results
        all_file_sizes[model_name] = file_sizes

    # Print results
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)

    if not args.dry_run:
        print_table(all_results)
        generate_python_dict(all_results)

        # Save results to JSON
        results_file = output_dir / "experiment_results_nv.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

        # Save file sizes to JSON
        file_sizes_file = output_dir / "experiment_results_size_nv.json"
        with open(file_sizes_file, "w") as f:
            json.dump(all_file_sizes, f, indent=2)
        print(f"File sizes saved to: {file_sizes_file}")
    else:
        print("\nDry run completed - no results to display")

    # Record end time and calculate total duration
    end_time = time.time()
    total_duration = end_time - start_time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = total_duration % 60

    print("\n" + "="*60)
    print("EXPERIMENT DURATION")
    print("="*60)
    print(f"Total time: {hours:02d}:{minutes:02d}:{seconds:06.3f}")
    print(f"Total seconds: {total_duration:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()