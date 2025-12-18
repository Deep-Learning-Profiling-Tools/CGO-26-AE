# Log-scale heatmap for model artifact sizes (MB)
# - Parses K/M/G into MB (K = 1/1024 MB, G = 1024 MB)
# - Uses log10(MB) for color mapping
# - Shows original labels in each cell; marks "out-of-resources" as ×

import re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
from pathlib import Path

# 1) Raw data (paste your table here)
# Original hardcoded data (commented out)
# raw = [
#     ["Llama-3.1-8B","160K","1.10G","57.0M","132K","518M","318M"],
#     ["Llama-3.3-70B","176K","19.0G","313M","136K","18.0G","1.90G"],
#     ["Gemma3-4B","272K","1.30G","65.0M","232K","521M","408M"],
#     ["Qwen3-30B-A3B","224K","9.80G","375M","164K","659M","3.50G"],
#     ["GPT-OSS-20B","400K","41.0G","529M","220K","26.0G","5.80G"],
#     ["Mistral-7B","160K","3.70G","58.0M","176K","3.40G","326M"],
#     ["Phi-3.5-Mini","120K","5.00G","66.0M","out-of-resources","out-of-resources","out-of-resources"],
#     ["SmolLM2.1-7B","160K","2.50G","38.0M","156K","2.50G","232M"],
#     ["TinyLlama-Chat","160K","2.50G","37.0M","160K","2.50G","224M"],
#     ["Zephyr-SFT","164K","3.70G","58.0M","168K","3.40G","327M"],
# ]

# Load data from JSON files
script_dir = Path(__file__).parent
nv_json_path = script_dir / "experiment_outputs_nv" / "experiment_results_size_nv.json"
amd_json_path = script_dir / "experiment_outputs_amd" / "experiment_results_size_amd.json"

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

def bytes_to_human_readable(bytes_val):
    """Convert bytes to human readable format (K/M/G)"""
    if bytes_val is None or bytes_val == 0:
        return "0"

    # Determine the appropriate unit
    if bytes_val < 1024 * 1024:  # Less than 1 MB
        value = bytes_val / 1024
        if value < 10:
            return f"{value:.2f}K"
        elif value < 100:
            return f"{value:.1f}K"
        else:
            return f"{int(round(value))}K"
    elif bytes_val < 1024 * 1024 * 1024:  # Less than 1 GB
        value = bytes_val / (1024 * 1024)
        if value < 10:
            return f"{value:.2f}M"
        elif value < 100:
            return f"{value:.1f}M"
        else:
            return f"{int(round(value))}M"
    else:  # GB or larger
        value = bytes_val / (1024 * 1024 * 1024)
        if value < 10:
            return f"{value:.2f}G"
        elif value < 100:
            return f"{value:.1f}G"
        else:
            return f"{int(round(value))}G"

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
raw = []
for model_name in model_names:
    row = [model_name]

    # GH200 data (NVIDIA) - proton, torch, nsys
    if has_nv_data and model_name in nv_data:
        proton_size = nv_data[model_name].get("proton")
        torch_size = nv_data[model_name].get("torch")
        nsys_size = nv_data[model_name].get("nsys")

        row.append(bytes_to_human_readable(proton_size) if proton_size else "0")
        row.append(bytes_to_human_readable(torch_size) if torch_size else "0")
        row.append(bytes_to_human_readable(nsys_size) if nsys_size else "0")
    else:
        row.extend(["0", "0", "0"])

    # MI300 data (AMD) - proton, torch, rocprof
    if has_amd_data and model_name in amd_data:
        proton_size = amd_data[model_name].get("proton")
        torch_size = amd_data[model_name].get("torch")
        rocprof_size = amd_data[model_name].get("rocprof")

        row.append(bytes_to_human_readable(proton_size) if proton_size else "0")
        row.append(bytes_to_human_readable(torch_size) if torch_size else "0")
        row.append(bytes_to_human_readable(rocprof_size) if rocprof_size else "0")
    else:
        row.extend(["0", "0", "0"])

    raw.append(row)

cols = ["Name","TritonProf-GH200","TorchProf-GH200","nsys-GH200",
        "TritonProf-MI300X","TorchProf-MI300X","rocprof-MI300X"]
df_raw = pd.DataFrame(raw, columns=cols).set_index("Name")

# 2) Convert strings to MB (K/M/G -> MB). "out-of-resources" -> NaN.
def to_mb(s: str) -> float:
    if s is None:
        return math.nan
    s = str(s).strip()
    if "out-of-resources" in s.lower() or s.lower() in ("na","nan",""):
        return math.nan
    # Updated pattern to handle KB, MB, GB
    m = re.match(r'([0-9.]+)\s*([KMG]B)', s, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        unit = m.group(2).upper()
        factor = {'KB': 1/1024, 'MB': 1, 'GB': 1024}[unit]
        return v * factor
    # Legacy support for single letter units
    m = re.match(r'([0-9.]+)\s*([kmg])', s, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        unit = m.group(2).lower()
        factor = {'k': 1/1024, 'm': 1, 'g': 1024}[unit]
        return v * factor
    # If it's a bare number, treat it as MB
    try:
        return float(s)
    except:
        return math.nan

df_mb = df_raw.applymap(to_mb)

# Keep original order - no sorting
# geom = np.exp(np.log(df_mb).replace(-np.inf, np.nan).mean(axis=1, skipna=True))
# df_raw = df_raw.loc[geom.sort_values().index]
# df_mb  = df_mb.loc[df_raw.index]

# 3) Build a log10 heatmap (MB)
fig, ax = plt.subplots(figsize=(14, 6))  # Wider and shorter

# Create custom colormap with the specified RGB colors
colors = [
    (253/255, 235/255, 208/255),  # Light peach/cream (smallest values)
    (247/255, 202/255, 201/255),  # Light pink
    (247/255, 82/255, 112/255),   # Medium red/pink
    (220/255, 20/255, 60/255)     # Deep red/crimson (largest values)
]
n_bins = 100
custom_cmap = LinearSegmentedColormap.from_list('custom_red', colors, N=n_bins)

# Use masked array so NaNs are not drawn - transpose the data
data_log = np.log10(df_mb.to_numpy().T)  # Transpose here
masked = np.ma.masked_invalid(data_log)

im = ax.imshow(masked, aspect='auto', cmap=custom_cmap, vmin=-1.5, vmax=5.5)  # Custom red gradient: larger values darker red, range from ~30KB to ~300GB
ax.set_xticks(range(df_mb.shape[0]))  # Now using rows count for x-axis
ax.set_xticklabels(df_mb.index, rotation=30, ha='right', fontsize=18, color='black')  # Row labels on x-axis
ax.set_yticks(range(df_mb.shape[1]))  # Now using columns count for y-axis
y_labels = [label for label in df_mb.columns]  # No line breaks for any labels
ax.set_yticklabels(y_labels, fontsize=18, rotation=0, ha='right', va='center')  # Column labels on y-axis horizontal

# Colorbar shows log10(MB) with custom labels
cbar = plt.colorbar(im, ax=ax)
# Set custom tick positions and labels
# log10(100/1024) ≈ -1.01 for 100K, log10(10) = 1 for 10M, log10(1024) ≈ 3.01 for 1G, log10(102400) ≈ 5.01 for 100G
cbar.set_ticks([-1.01, 1, 3.01, 5.01])
cbar.set_ticklabels(['100K', '10M', '1G', '100G'])
cbar.ax.tick_params(labelsize=18)

# 4) Annotate each cell with the original label; mark OOR as "×"
# Note: Now we need to transpose the coordinates
for i in range(df_raw.shape[0]):
    for j in range(df_raw.shape[1]):
        s = str(df_raw.iloc[i, j])
        # Set text color to white for TorchProf rows (1 and 4) - now they are rows not columns
        # Out-Of-Resources always stays black
        if "out-of-resources" in s.lower():
            # Add gray background rectangle for Out-Of-Resources cells
            from matplotlib.patches import Rectangle
            rect = Rectangle((i-0.5, j-0.5), 1, 1, facecolor='lightgray', edgecolor='none', zorder=1)
            ax.add_patch(rect)
            # Text on top without X marks
            ax.text(i, j, "Out-Of-\nResources", ha='center', va='center', fontsize=13, color='black', zorder=3)  # Swapped i and j
        else:
            # Force specific values to be black or white
            if s in ['518M', '521M', '659M']:
                text_color = 'black'
            elif s in ['1.90G', '3.50G', '5.80G']:
                text_color = 'white'
            else:
                text_color = 'white' if j in [1, 4] else 'black'  # j still refers to original column indices
            ax.text(i, j, s, ha='center', va='center', fontsize=18, color=text_color)  # Swapped i and j

plt.tight_layout()

# Uncomment to save as PDF
plt.savefig("fig15.pdf", bbox_inches="tight")
