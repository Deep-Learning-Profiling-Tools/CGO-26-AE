import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# Adjust this to where your logs live *relative to where you run the script*
BASE_DIR = "logs/gpt-oss-x2"

# Map "nice" labels to directory names
MODES = {
    "TP=1, EP=8": "fp8x-fp8w-TP1-EP8",
    "TP=2, EP=4": "fp8x-fp8w-TP2-EP4",
    "TP=4, EP=2": "fp8x-fp8w-TP4-EP2",
    "TP=8, EP=1": "fp8x-fp8w-TP8-EP1",
}

def read_mode_data(mode_dir):
    """
    Read all *.csv files in mode_dir.
    Each CSV has columns: x,flops,bytes,time_ns

    Returns:
      data_by_x: dict[x] -> list of TFLOPs (one per rank/file)
    """
    data_by_x = defaultdict(list)

    # Look only at CSV files in this directory (ignore subdirs like "0/", "1/", etc.)
    for fname in os.listdir(mode_dir):
        if not fname.endswith(".csv"):
            continue

        path = os.path.join(mode_dir, fname)
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = int(row["x"])
                flops = float(row["flops"])
                time_ns = float(row["time_ns"])

                # TFLOPs = (FLOPs / ns) * 1e-3
                # (1 FLOP/ns = 1e-3 TFLOPs)
                tflops = (flops / time_ns) * 1e-3

                data_by_x[x].append(tflops)

    return data_by_x


def compute_imbalance(data_by_x):
    """
    Given data_by_x: x -> list of TFLOPs (one per rank),
    compute imbalance(x) = 1 - min / max for each x.

    Returns:
      xs_sorted: np.array of x
      imbalances: np.array of imbalance values in the same order
    """
    xs = []
    imbalances = []

    for x, tflops_list in data_by_x.items():
        if len(tflops_list) < 2:
            # Not enough ranks to define an imbalance
            continue
        tflops_arr = np.array(tflops_list)
        t_min = tflops_arr.min()
        t_max = tflops_arr.max()
        if t_max <= 0:
            # Avoid division by zero / nonsense
            continue
        imbalance = 1.0 - (t_min / t_max)
        xs.append(x)
        imbalances.append(imbalance)

    # Sort by x
    xs = np.array(xs)
    imbalances = np.array(imbalances)
    order = np.argsort(xs)
    return xs[order], imbalances[order]


def main():
    plt.figure(figsize=(6, 4))

    for label, subdir in MODES.items():
        mode_dir = os.path.join(BASE_DIR, subdir)
        if not os.path.isdir(mode_dir):
            print(f"Warning: directory not found: {mode_dir}")
            continue

        data_by_x = read_mode_data(mode_dir)
        xs, imbalances = compute_imbalance(data_by_x)

        if len(xs) == 0:
            print(f"Warning: no data for mode {label}")
            continue

        plt.plot(xs, imbalances, marker="o", label=label)

    plt.xlabel("x (e.g., sequence length or problem size)")
    plt.ylabel("FLOP imbalance (1 - min/max TFLOPs)")
    plt.title("FLOP Imbalance Across Ranks for Different Parallelism Modes")
    plt.ylim(0, 1)  # imbalance is in [0, 1]
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("flop_imbalance_vs_x.png", dpi=200)
    # Or show interactively:
    # plt.show()


if __name__ == "__main__":
    main()
