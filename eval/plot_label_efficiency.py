"""
Figure 1: Label Efficiency Curve
Weighted cost/300 vs number of labels.
Shows ARIA as zero-label bridge and LR-Balanced as post-supervision solution.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# From eval/results/lr_balanced_label_efficiency.json / paper Table 4
labels_n = [0, 10, 25, 50, 100, 200, 500, 1000, 6481]
lr_wtd   = [None, 695, 582, 470, 444, 379, 353, 353, 314]  # LR-Balanced Wtd/300
lr_fm    = [None, 80.5, 81.6, 78.1, 77.4, 75.6, 76.2, 79.2, 79.2]  # FM accuracy

ARIA_WTD   = 795
GPT4O_WTD  = 1029
VIABILITY  = 70.0  # FM% threshold

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

x_vals  = [n for n, w in zip(labels_n, lr_wtd) if w is not None]
y_vals  = [w for w in lr_wtd if w is not None]

# GPT-4o baseline
ax.axhline(GPT4O_WTD, color="#d62728", linewidth=1.4, linestyle="--", zorder=1)
ax.text(6600, GPT4O_WTD + 12, "GPT-4o baseline (1,029)", color="#d62728",
        fontsize=8.5, ha="right")

# ARIA flatline — extends from 0
ax.axhline(ARIA_WTD, color="#1f77b4", linewidth=1.6, linestyle="-.", zorder=2)
ax.text(6600, ARIA_WTD + 12, "ARIA (795, zero labels)", color="#1f77b4",
        fontsize=8.5, ha="right")

# LR-Balanced curve
ax.plot(x_vals, y_vals, color="#2ca02c", linewidth=2.2, marker="o",
        markersize=5.5, zorder=3, label="LR-Balanced")

# ARIA operating point at n=0 — draw explicit segment from 0 to first LR point
ax.plot([0, x_vals[0]], [ARIA_WTD, ARIA_WTD], color="#1f77b4",
        linewidth=1.6, linestyle="-.", zorder=2)
ax.scatter([0], [ARIA_WTD], color="#1f77b4", s=70, zorder=5, marker="o")

# Annotate crossover at n=10
ax.annotate("Beats ARIA\nat 10 labels",
            xy=(10, 695), xytext=(25, 750),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.1),
            fontsize=8, color="#555")

# Shade improvement region
ax.fill_between(x_vals, [ARIA_WTD]*len(x_vals), y_vals,
                color="#2ca02c", alpha=0.08, zorder=0)

# FM viability note (secondary axis annotation)
ax.text(10, 310, "All points FM ≥ 70% (operationally viable)",
        fontsize=7.5, color="#555", style="italic")

ax.set_xscale("symlog", linthresh=1)
ax.set_xticks([0, 10, 25, 50, 100, 200, 500, 1000, 6481])
ax.set_xticklabels(["0\n(ARIA)", "10", "25", "50", "100", "200", "500", "1k", "full"],
                   fontsize=8)
ax.set_xlabel("Number of Labelled Training Examples", fontsize=10)
ax.set_ylabel("Weighted Cost / 300 Complaints ↓", fontsize=10)
ax.set_title("Label Efficiency: ARIA vs LR-Balanced\n"
             "ARIA serves as zero-label bridge; supervised solution available at 10 labels",
             fontsize=10)
ax.set_ylim(270, 1100)
ax.set_xlim(-0.5, 7500)
ax.yaxis.grid(True, linestyle=":", alpha=0.5)
ax.set_axisbelow(True)

green_patch = mpatches.Patch(color="#2ca02c", label="LR-Balanced (class-weighted LR)")
blue_line   = mpatches.Patch(color="#1f77b4", label="ARIA (zero labels)")
red_line    = mpatches.Patch(color="#d62728", label="GPT-4o baseline")
ax.legend(handles=[green_patch, blue_line, red_line], fontsize=8.5,
          loc="upper right", framealpha=0.9)

plt.tight_layout()
out = "eval/results/fig1_label_efficiency.pdf"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"Saved: {out.replace('.pdf', '.png')}")
