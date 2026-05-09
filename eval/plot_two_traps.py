"""
Figure 2: Two-Traps Scatter Plot
FM accuracy (x) vs Project accuracy (y), bubble = weighted cost.
Visualizes the accuracy trap and cost trap failure modes.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data from paper Table 2 ───────────────────────────────────────────────────
systems = [
    # (label,              FM%,   Proj%,  Wtd/300, color,       marker)
    ("GPT-4o\nText-only",  91.6,  13.2,   1029,    "#d62728",   "s"),
    ("GPT-4o\nCost-prompt",19.1,  88.4,   1101,    "#ff7f0e",   "^"),
    ("GPT-4o\nE[cost] CoT", 0.0,  100.0,  1860,    "#9467bd",   "D"),
    ("ARIA",               75.5,  35.8,    795,    "#1f77b4",   "o"),
]
# Note: GPT-4o cost-prompt and E[cost] CoT numbers — use actuals from paper
# GPT-4o text-only: FM=91.6%, Proj=13.2%, Wtd=1029
# GPT-4o E[cost] CoT: FM=0.0%, routes 100% to Project, Wtd ~1860 (82.7% worse)
# ARIA: FM=75.5%, Proj=35.8%, Wtd=795

fig, ax = plt.subplots(figsize=(7, 5))

# ── Trap quadrant shading ─────────────────────────────────────────────────────
# Accuracy trap: high FM (>80%), low Project (<40%) → top-left
ax.fill_between([80, 100], [0, 0], [40, 40], color="#d62728", alpha=0.07, zorder=0)
ax.text(83, 5, "Accuracy\nTrap", color="#d62728", fontsize=8.5, alpha=0.8)

# Cost trap: low FM (<70%), high Project (>70%) → bottom-right area
ax.fill_between([0, 70], [70, 70], [100, 100], color="#ff7f0e", alpha=0.07, zorder=0)
ax.text(3, 88, "Cost\nTrap", color="#ff7f0e", fontsize=8.5, alpha=0.8)

# Viability boundary: FM=70%
ax.axvline(70, color="#555", linewidth=1.1, linestyle=":", alpha=0.7)
ax.text(70.5, 2, "FM≥70%\nthreshold", color="#555", fontsize=7.5, alpha=0.8)

# ── Per-system label offsets (hand-tuned to avoid overlap) ───────────────────
label_offsets = {
    "GPT-4o\nText-only":   (2,  -10),
    "GPT-4o\nCost-prompt": (2,    3),
    "GPT-4o\nE[cost] CoT": (2,    3),
    "ARIA":                (2,    3),
}

for label, fm, proj, wtd, color, marker in systems:
    size = (wtd / 300) * 80
    ax.scatter(fm, proj, s=size, color=color, marker=marker,
               zorder=5, edgecolors="white", linewidth=0.8)
    dx, dy = label_offsets.get(label, (2, 3))
    ax.annotate(label, (fm, proj),
                xytext=(fm + dx, proj + dy),
                fontsize=8.5, color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.6))

# ── ARIA callout ──────────────────────────────────────────────────────────────
aria_fm, aria_proj = 75.5, 35.8
ax.annotate("← only operationally\n   viable system",
            xy=(aria_fm, aria_proj), xytext=(78, 20),
            fontsize=8, color="#1f77b4", style="italic",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0))

ax.set_xlabel("FM (Maintenance) Accuracy  →", fontsize=10)
ax.set_ylabel("Project (Structural) Accuracy  →", fontsize=10)
ax.set_title("Two Failure Modes of LLM Complaint Routing\n"
             "Bubble size ∝ weighted cost; FM≥70% required for operational viability",
             fontsize=10)
ax.set_xlim(0, 105)
ax.set_ylim(0, 105)
ax.xaxis.grid(True, linestyle=":", alpha=0.4)
ax.yaxis.grid(True, linestyle=":", alpha=0.4)
ax.set_axisbelow(True)

# Legend for bubble size — move to upper right to avoid accuracy-trap label
for wtd_ref, lbl in [(300, "Wtd=300"), (800, "Wtd=800"), (1500, "Wtd=1500")]:
    ax.scatter([], [], s=(wtd_ref/300)*80, color="gray", alpha=0.5, label=lbl)
ax.legend(title="Weighted cost/300", fontsize=8, title_fontsize=8,
          loc="upper right", framealpha=0.85)

plt.tight_layout()
out = "eval/results/fig2_two_traps.pdf"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"Saved: {out.replace('.pdf', '.png')}")
