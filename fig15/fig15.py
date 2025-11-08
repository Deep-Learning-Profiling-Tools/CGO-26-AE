# Generate kernel overhead chart following e2e_overhead.py format
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from brokenaxes import brokenaxes

# Data
raw = [
    ["fused_softmax", 1.040776595, 1.1007877],
    ["grouped_gemm", 1.004727857, 1.005781726],
    ["layer_norm", 1.172689386, 1.160086201],
    ["low_memory_dropout", 1.130898257, 1.119103734],
    ["matmul", 1.016415291, 1.068560898],
    ["swiglu", 1.041481268, 1.070833197],
    ["topk", 1.078303451, 1.021686332],
    ["persistent_matmul", 1.003418845, 1.009570271],
    ["matmal_ogs", 0.9909863181, 1.024854553],
    ["fused_attention", 1.014032246, 1.33776454],
]

cols = ["Name", "gh200-overheads", "mi300-overheads"]
df = pd.DataFrame(raw, columns=cols).set_index("Name")

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

