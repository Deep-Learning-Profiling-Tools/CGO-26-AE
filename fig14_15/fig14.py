# Generate the ordered PDF chart per user's requested series order.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import json
from pathlib import Path
import os

# Data
# Original hardcoded data (commented out)
# raw = [
#     ["Llama-3.1-8B", 44.69, 50.89, 55.37, 55.33, 51.54, 53.9, 53.38, 54.56],
#     ["Llama-3.3-70B", 249.96, 426.41, 313.85, 319.96, 272.91, 315.16, 280.67, 292.27],
#     ["Gemma3-4B", 58.05, 75.37, 70.2, 72.54, 59.35, 64.31, 63.48, 64.66],
#     ["Qwen3-30B-A3B", 208.11, 282.77, 272.45, 281.11, 150.0, 187.29, 181.49, 204.91],
#     ["GPT-OSS-20B", 270.93, 515.86, 369.41, 358.96, 157.42, 242.06, 194.0, 205.54],
#     ["Mistral-7B", 44.85, 79.38, 54.67, 57.8, 53.47, 58.42, 55.57, 56.35],
#     ["Phi-3.5-Mini", 38.18, 71.57, 51.54, 55.92, np.nan, np.nan, np.nan, np.nan],
#     ["SmolLM2.1-7B", 33.6, 53.89, 39.88, 41.22, 37.46, 45.45, 38.95, 41.2],
#     ["TinyLlama-Chat", 32.97, 54.89, 39.22, 39.91, 36.42, 44.09, 37.97, 39.52],
#     ["Zephyr-SFT", 46.3, 80.06, 55.83, 59.57, 54.03, 61.27, 55.54, 57.3],
# ]

# Load data from JSON files
script_dir = Path(__file__).parent
nv_json_path = script_dir / "experiment_outputs_nv" / "experiment_results_nv.json"
amd_json_path = script_dir / "experiment_outputs_amd" / "experiment_results_amd.json"

# Model names in order
model_names = [
    "Llama-3.1-8B",
    "Llama-3.3-70B",
    "Gemma3-4B",
    "Qwen3-30B-A3B",
    "GPT-OSS-20B",
    "Mistral-7B",
    "Phi-3.5-Mini",
    "SmolLM2.1-7B",
    "TinyLlama-Chat",
    "Zephyr-SFT",
]

# Initialize data structure
raw = []

# Check which JSON files exist
has_nv_data = nv_json_path.exists()
has_amd_data = amd_json_path.exists()

# Load JSON data
nv_data = {}
amd_data = {}

if has_nv_data:
    with open(nv_json_path, "r") as f:
        nv_data = json.load(f)
    print(f"Loaded NVIDIA data from {nv_json_path}")

if has_amd_data:
    with open(amd_json_path, "r") as f:
        amd_data = json.load(f)
    print(f"Loaded AMD data from {amd_json_path}")

# Build raw data
for model_name in model_names:
    row = [model_name]

    # GH200 data (NVIDIA)
    if has_nv_data and model_name in nv_data:
        row.append(nv_data[model_name].get("baseline", np.nan))
        row.append(nv_data[model_name].get("torch", np.nan))
        row.append(nv_data[model_name].get("proton", np.nan))
        row.append(nv_data[model_name].get("nsys", np.nan))
    else:
        row.extend([0, 0, 0, 0])

    # MI300 data (AMD)
    if has_amd_data and model_name in amd_data:
        row.append(amd_data[model_name].get("baseline", np.nan))
        row.append(amd_data[model_name].get("torch", np.nan))
        row.append(amd_data[model_name].get("proton", np.nan))
        row.append(amd_data[model_name].get("rocprof", np.nan))
    else:
        row.extend([0, 0, 0, 0])

    raw.append(row)

cols = [
    "Name",
    "gh200-baseline", "gh200-torch", "gh200-proton", "gh200-nsys",
    "mi300-baseline", "mi300-torch", "mi300-proton", "mi300-rocprof",
]
df = pd.DataFrame(raw, columns=cols).set_index("Name")

# Overhead vs baseline
over_gh200 = df[["gh200-proton", "gh200-torch", "gh200-nsys"]].div(df["gh200-baseline"], axis=0)
over_mi300 = df[["mi300-proton", "mi300-torch", "mi300-rocprof"]].div(df["mi300-baseline"], axis=0)

labels = df.index.tolist()
x = np.arange(len(labels))
width = 0.12

# Create broken axes with gap from 0.05 to 0.95
fig = plt.figure(figsize=(18, 2.4))  # Reduced height by 40% (4 * 0.6 = 2.4)
bax = brokenaxes(ylims=((0, 0.05), (0.95, 2.0)), hspace=.05, d=0.0035, fig=fig, despine=False)

# Offsets in the exact requested left-to-right order:
# 1) proton-gh200, 2) torch-gh200, 3) nsys-gh200,
# 4) proton-mi300, 5) torch-mi300, 6) rocprof-mi300
offsets = [-2.5*width, -1.5*width, -0.5*width, 0.5*width, 1.5*width, 2.5*width]

# Define common colors for each tool type (RGB values normalized to 0-1)
proton_color = (124/255, 166/255, 215/255)
torch_color = (247/255, 213/255, 134/255)
nsys_rocprof_color = (209/255, 108/255, 107/255)

# Plot bars in the correct left-to-right order:
# GH200 bars (left side)
bar1 = bax.bar(x + offsets[0], over_gh200["gh200-proton"].values, width, color=proton_color, label="TritonProf-GH200")
bar2 = bax.bar(x + offsets[1], over_gh200["gh200-torch"].values,  width, color=torch_color, label="TritonProf-MI300X")
bar3 = bax.bar(x + offsets[2], over_gh200["gh200-nsys"].values,   width, color=nsys_rocprof_color, label="TorchProf-GH200")

# MI300 bars (right side)
bar4 = bax.bar(x + offsets[3], over_mi300["mi300-proton"].values, width, color=proton_color, hatch="//", edgecolor="white", label="TorchProf-MI300X")
bar5 = bax.bar(x + offsets[4], over_mi300["mi300-torch"].values,  width, color=torch_color, hatch="//", edgecolor="white", label="nsys-GH200")
bar6 = bax.bar(x + offsets[5], over_mi300["mi300-rocprof"].values,width, color=nsys_rocprof_color, hatch="//", edgecolor="white", label="rocprof-MI300X")

# Baseline line
bax.axhline(1.0, linestyle="--", linewidth=1)

# --- Force display of all category labels: manually draw only on bottom panel ---
# 1) Turn off real x ticks and labels on both panels
for ax in bax.axs:
    ax.set_xticks([])                     # No real ticks
    ax.set_xticklabels([])                # No automatic labels
    ax.tick_params(axis='x', which='both', length=0)

# 2) Draw labels manually on bottom panel using text (won't be auto-hidden by brokenaxes)
bot = bax.axs[-1]
y_lab = -0.16  # Move down a bit to avoid overlap with axis line; adjust as needed
for xi, lab in zip(x, labels):
    bot.text(xi, y_lab, lab,
             transform=bot.get_xaxis_transform(),  # x uses data coordinates, y uses axis coordinates
             ha='center', va='top', fontsize=12, color='black')


bax.set_ylabel("Overhead", fontsize=18)
bax.tick_params(axis='y', labelsize=14)
# Create legend with correct ordering
# matplotlib fills column-by-column with ncol=3
# Position mapping: [0] [2] [4]
#                   [1] [3] [5]
# We want: gh200-proton, gh200-torch, gh200-nsys (top row)
#          mi300-proton, mi300-torch, mi300-rocprof (bottom row)
# So order should be: gh200-proton, mi300-proton, gh200-torch, mi300-torch, gh200-nsys, mi300-rocprof

from matplotlib.patches import Patch

# Create legend entries in the correct order for column-wise filling with ncol=3
# matplotlib fills column-by-column: [0] [2] [4]
#                                    [1] [3] [5]
legend_elements = [
    Patch(facecolor=proton_color),
    Patch(facecolor=proton_color, hatch='//', edgecolor='white'),
    Patch(facecolor=torch_color),
    Patch(facecolor=torch_color, hatch='//', edgecolor='white'),
    Patch(facecolor=nsys_rocprof_color),
    Patch(facecolor=nsys_rocprof_color, hatch='//', edgecolor='white')
]

bax.legend(handles=legend_elements, ncol=3, bbox_to_anchor=(0.5, 1.25), loc="center", frameon=False, fontsize=12)
bax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

# OOR marker (phi on mi300)
for i, name in enumerate(labels):
    if np.isnan(df.loc[name, "mi300-baseline"]):
        bax.text(x[i] + 1.5*width + 0.10, 0.04, "Out-Of-\nResources", rotation=0, ha="center", va="bottom", fontsize=9)

# X-limits - reduce margins on both sides
bax.set_xlim(-0.5, len(labels) - 0.5)

pdf_path = "fig14.pdf"
fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.close(fig)

