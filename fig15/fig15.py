# Generate kernel overhead chart following e2e_overhead.py format
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from brokenaxes import brokenaxes
from pathlib import Path

# Original hardcoded data (commented out)
# raw = [
#     ["fused_softmax", 1.040776595, 1.1007877],
#     ["grouped_gemm", 1.004727857, 1.005781726],
#     ["layer_norm", 1.172689386, 1.160086201],
#     ["low_memory_dropout", 1.130898257, 1.119103734],
#     ["matmul", 1.016415291, 1.068560898],
#     ["swiglu", 1.041481268, 1.070833197],
#     ["topk", 1.078303451, 1.021686332],
#     ["persistent_matmul", 1.003418845, 1.009570271],
#     ["matmal_ogs", 0.9909863181, 1.024854553],
#     ["fused_attention", 1.014032246, 1.33776454],
# ]

# Kernel name mapping: raw name -> CSV name
kernel_name_mapping = {
    "fused_softmax": "fused_softmax",
    "grouped_gemm": "grouped_gemm",
    "layer_norm": "layer_norm_forward",
    "low_memory_dropout": "seeded_dropout",
    "matmul": "matmul",
    "swiglu": "swiglu",
    "topk": "topk",
    "persistent_matmul": "persistent_matmul",
    "matmul_ogs": "matmul_ogs",
    "fused_attention": "fused_attention",
}

# Define kernel names in the order they should appear
kernel_names = [
    "fused_softmax",
    "grouped_gemm",
    "layer_norm",
    "low_memory_dropout",
    "matmul",
    "swiglu",
    "topk",
    "persistent_matmul",
    "matmul_ogs",
    "fused_attention",
]

# Load CSV files
script_dir = Path(__file__).parent
nv_csv_path = script_dir / "kernels" / "cupti_profile_timings_nv.csv"
amd_csv_path = script_dir / "kernels" / "cupti_profile_timings_amd.csv"

# Check which CSV files exist
has_nv_data = nv_csv_path.exists()
has_amd_data = amd_csv_path.exists()

def calculate_overhead_from_csv(csv_path):
    """Calculate overhead from CSV file: instrumented/baseline"""
    df_csv = pd.read_csv(csv_path)
    overheads = {}

    for kernel in df_csv['kernel'].unique():
        kernel_data = df_csv[df_csv['kernel'] == kernel]
        baseline = kernel_data[kernel_data['configuration'] == 'baseline']['time_ns']
        instrumented = kernel_data[kernel_data['configuration'] == 'instrumented']['time_ns']

        if not baseline.empty and not instrumented.empty:
            overhead = instrumented.values[0] / baseline.values[0]
            overheads[kernel] = overhead

    return overheads

# Load overhead data
nv_overheads = {}
amd_overheads = {}

if has_nv_data:
    nv_overheads = calculate_overhead_from_csv(nv_csv_path)
    print(f"Loaded NVIDIA data from {nv_csv_path}")

if has_amd_data:
    amd_overheads = calculate_overhead_from_csv(amd_csv_path)
    print(f"Loaded AMD data from {amd_csv_path}")

# Build raw data
raw = []
for kernel_name in kernel_names:
    csv_kernel_name = kernel_name_mapping.get(kernel_name)

    # Get GH200 overhead
    if has_nv_data and csv_kernel_name and csv_kernel_name in nv_overheads:
        gh200_overhead = nv_overheads[csv_kernel_name]
    else:
        gh200_overhead = 0

    # Get MI300X overhead
    if has_amd_data and csv_kernel_name and csv_kernel_name in amd_overheads:
        mi300_overhead = amd_overheads[csv_kernel_name]
    else:
        mi300_overhead = 0

    raw.append([kernel_name, gh200_overhead, mi300_overhead])

cols = ["Name", "gh200-overheads", "mi300-overheads"]
df = pd.DataFrame(raw, columns=cols).set_index("Name")

# Print overhead data in table format
print("\n" + "="*70)
print(f"{'Kernel Names':<25} {'gh200-overhead':<20} {'mi300x-overhead':<20}")
print("="*70)
for kernel_name in kernel_names:
    gh200_val = df.loc[kernel_name, "gh200-overheads"]
    mi300_val = df.loc[kernel_name, "mi300-overheads"]
    print(f"{kernel_name:<25} {gh200_val:<20.10f} {mi300_val:<20.10f}")
print("="*70 + "\n")

labels = df.index.tolist()
x = np.arange(len(labels))
width = 0.35

# Create broken axes with gap from 0.05 to 0.95
fig = plt.figure(figsize=(9, 2.4))
bax = brokenaxes(ylims=((0, 0.025), (0.975, 1.4)), hspace=.05, d=0.007, fig=fig, despine=False)

# Define different colors for GH200 and MI300X
gh200_color = (119/255, 170/255, 221/255)  # Light blue
mi300_color = (233/255, 203/255, 240/255)  # Light purple

# Offsets for side-by-side bars
offsets = [-width/2, width/2]

# GH200 bar (solid color, no hatch)
bar1 = bax.bar(x + offsets[0], df["gh200-overheads"].values, width, color=gh200_color, label="TritonProf-GH200")

# MI300 bar (different color with // hatch pattern)
bar2 = bax.bar(x + offsets[1], df["mi300-overheads"].values, width, color=mi300_color, hatch="//", edgecolor="white", label="TritonProf-MI300X")

# Baseline line at 1.0
bax.axhline(1.0, linestyle="--", linewidth=1)

# --- Force display of all category labels: manually draw only on bottom panel ---
# 1) Turn off real x ticks and labels on both panels
for ax in bax.axs:
    ax.set_xticks([])                     # No real ticks
    ax.set_xticklabels([])                # No automatic labels
    ax.tick_params(axis='x', which='both', length=0)

# 2) Draw labels manually on bottom panel using text (won't be auto-hidden by brokenaxes)
bot = bax.axs[-1]
y_lab = -0.08  # Adjust vertical position
for xi, lab in zip(x, labels):
    bot.text(xi, y_lab, lab,
             transform=bot.get_xaxis_transform(),  # x uses data coordinates, y uses axis coordinates
             ha='right', va='top', fontsize=12, color='black', rotation=30)

bax.set_ylabel("Overhead", fontsize=18)
bax.tick_params(axis='y', labelsize=12)

# Create legend with correct ordering for row-wise layout
legend_elements = [
    Patch(facecolor=gh200_color),
    Patch(facecolor=mi300_color, hatch='//', edgecolor='white'),
]

bax.legend(handles=legend_elements, ncol=2, bbox_to_anchor=(0.5, 1.15), loc="center", frameon=False, fontsize=12)
bax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

# X-limits - reduce margins on both sides
bax.set_xlim(-0.5, len(labels) - 0.5)

pdf_path = "fig15.pdf"
fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.close(fig)

