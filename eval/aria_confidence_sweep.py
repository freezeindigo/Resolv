"""
ARIA confidence-threshold sweep — overlay curve for Figure 2.

The aria_confidence column in paper_eval_ambiguous_9259.csv is categorical
(low / medium / high). This is the only inference-time confidence signal
ARIA exposes from its multi-tier output. We map the categorical levels to
representative numeric scores and sweep a decision threshold tau in
[0.05, 0.95] step 0.05 with a confidence-gated escalation policy:

    pred(tau) = "Project" if (aria_label == "Project" AND
                             score(aria_confidence) >= tau)
                else "FM"

Interpretation: at low tau, all of ARIA's Project routings are honored
(maximum cost reduction, lowest FM accuracy). At high tau, only the most
confident Project routings are honored (FM accuracy approaches GPT-4o
text-only at the cost of less expected-cost reduction). The sweep traces
the (FM accuracy, cost reduction) frontier reachable from ARIA's
zero-shot output without re-running inference.

Because aria_confidence has only three levels, the sweep collapses into a
small number of distinct operating points by construction. We report all
of them and clearly mark the all-trust point (tau <= score(low)) as ARIA's
deployed operating point.

Inputs:
    eval/results/paper_eval_ambiguous_9259.csv  (ARIA predictions)
    eval/results/paper_eval_baseline_9259.csv   (GPT-4o text-only baseline)

Outputs:
    eval/results/aria_threshold_sweep.csv
    eval/results/threshold_sweep_with_aria.pdf
    eval/results/threshold_sweep_with_aria.png
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARIA_CSV = Path("eval/results/paper_eval_ambiguous_9259.csv")
BASELINE_CSV = Path("eval/results/paper_eval_baseline_9259.csv")
LR_SWEEP_CSV = Path("eval/results/threshold_sweep.csv")

OUT_CSV = Path("eval/results/aria_threshold_sweep.csv")
OUT_PDF = Path("eval/results/threshold_sweep_with_aria.pdf")
OUT_PNG = Path("eval/results/threshold_sweep_with_aria.png")


# Cost matrix (paper Table 2, headline).
W_HIGH_COST = 10  # Project routed to FM
W_LOW_COST = 1    # FM routed to Project

# Mapping from categorical aria_confidence to a representative numeric score.
# Chosen so the sweep over [0.05, 0.95] step 0.05 produces three honest
# operating regimes (trust all / drop low / drop low+medium / drop all):
#   low    -> 0.30 (dropped when tau > 0.30)
#   medium -> 0.60 (dropped when tau > 0.60)
#   high   -> 0.90 (dropped when tau > 0.90)
CONF_SCORE = {"low": 0.30, "medium": 0.60, "high": 0.90}

FM_VIABILITY_CUTOFF = 70.0  # percent


def load_predictions() -> List[Dict[str, str]]:
    aria_rows = list(csv.DictReader(open(ARIA_CSV)))
    baseline_rows = {r["ticket_id"]: r for r in csv.DictReader(open(BASELINE_CSV))}
    paired = []
    for r in aria_rows:
        b = baseline_rows.get(r["ticket_id"])
        if b is None:
            continue
        paired.append({
            "ticket_id": r["ticket_id"],
            "crm_label": r["crm_label"],
            "aria_label": r["aria_label"],
            "aria_confidence": (r["aria_confidence"] or "").strip().lower(),
            "gpt4o_label": b["gpt4o_label"],
        })
    return paired


def cost(rows, predictions: List[str]) -> Dict[str, float]:
    n_fm = sum(1 for r in rows if r["crm_label"] == "FM")
    n_proj = sum(1 for r in rows if r["crm_label"] == "Project")
    hc = sum(1 for r, p in zip(rows, predictions) if r["crm_label"] == "Project" and p == "FM")
    lc = sum(1 for r, p in zip(rows, predictions) if r["crm_label"] == "FM" and p == "Project")
    fm_correct = sum(1 for r, p in zip(rows, predictions) if r["crm_label"] == "FM" and p == "FM")
    proj_correct = sum(1 for r, p in zip(rows, predictions) if r["crm_label"] == "Project" and p == "Project")
    fm_acc = 100.0 * fm_correct / n_fm if n_fm else 0.0
    proj_acc = 100.0 * proj_correct / n_proj if n_proj else 0.0
    weighted = hc * W_HIGH_COST + lc * W_LOW_COST
    return {
        "fm_acc": fm_acc,
        "proj_acc": proj_acc,
        "hc": hc,
        "lc": lc,
        "weighted_cost": weighted,
        "n_fm": n_fm,
        "n_proj": n_proj,
    }


def main():
    rows = load_predictions()
    print(f"Paired rows: {len(rows)}")

    # GPT-4o reference cost on the same population.
    gpt_pred = [r["gpt4o_label"] for r in rows]
    gpt_metrics = cost(rows, gpt_pred)
    gpt_cost_total = gpt_metrics["weighted_cost"]
    print(f"GPT-4o text-only on same population: FM={gpt_metrics['fm_acc']:.2f}% "
          f"Proj={gpt_metrics['proj_acc']:.2f}% wtd_cost={gpt_cost_total}")

    # ARIA all-trust (deployed) operating point.
    aria_pred_all = [r["aria_label"] for r in rows]
    aria_all_metrics = cost(rows, aria_pred_all)
    print(f"ARIA all-trust operating point: FM={aria_all_metrics['fm_acc']:.2f}% "
          f"Proj={aria_all_metrics['proj_acc']:.2f}% wtd_cost={aria_all_metrics['weighted_cost']} "
          f"cost_red={(gpt_cost_total - aria_all_metrics['weighted_cost'])/gpt_cost_total*100:.2f}%")

    # Sweep tau in [0.05, 0.95] step 0.05.
    thresholds = np.round(np.arange(0.05, 0.95 + 1e-9, 0.05), 3)
    sweep: List[Dict[str, float]] = []
    for tau in thresholds:
        pred = []
        for r in rows:
            score = CONF_SCORE.get(r["aria_confidence"], 0.0)
            if r["aria_label"] == "Project" and score >= tau:
                pred.append("Project")
            else:
                pred.append("FM")
        m = cost(rows, pred)
        cr = (gpt_cost_total - m["weighted_cost"]) / gpt_cost_total * 100.0 if gpt_cost_total else 0.0
        sweep.append({
            "tau": float(tau),
            "fm_acc": round(m["fm_acc"], 2),
            "proj_acc": round(m["proj_acc"], 2),
            "hc": int(m["hc"]),
            "lc": int(m["lc"]),
            "weighted_cost": int(m["weighted_cost"]),
            "gpt4o_cost": int(gpt_cost_total),
            "cost_reduction_pct": round(cr, 2),
        })

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)
    print(f"Saved: {OUT_CSV}")

    # Distinct operating points (collapse identical tau steps).
    distinct = []
    seen = set()
    for r in sweep:
        key = (r["fm_acc"], r["proj_acc"], r["weighted_cost"])
        if key not in seen:
            seen.add(key)
            distinct.append(r)
    print("\nDistinct ARIA operating points (along confidence sweep):")
    print(f"{'tau':>6} {'FM_acc':>8} {'Proj_acc':>9} {'HC':>5} {'LC':>5} {'wtd':>6} {'cost_red%':>10}")
    for r in distinct:
        print(f"{r['tau']:>6.2f} {r['fm_acc']:>7.2f}% {r['proj_acc']:>8.2f}% "
              f"{r['hc']:>5d} {r['lc']:>5d} {r['weighted_cost']:>6d} {r['cost_reduction_pct']:>9.2f}%")

    # Load LR sweep for overlay.
    lr_rows = list(csv.DictReader(open(LR_SWEEP_CSV)))
    lr_fm = np.array([float(r["fm_acc"]) for r in lr_rows])
    lr_cr = np.array([float(r["cost_reduction_pct"]) for r in lr_rows])

    # Build ARIA curve from sweep (sorted by FM accuracy).
    aria_fm = np.array([r["fm_acc"] for r in sweep])
    aria_cr = np.array([r["cost_reduction_pct"] for r in sweep])
    order = np.argsort(aria_fm)
    aria_fm_sorted = aria_fm[order]
    aria_cr_sorted = aria_cr[order]

    # ARIA deployed operating point.
    aria_op_fm = aria_all_metrics["fm_acc"]
    aria_op_cr = (gpt_cost_total - aria_all_metrics["weighted_cost"]) / gpt_cost_total * 100.0

    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.axvspan(FM_VIABILITY_CUTOFF, 102, color="green", alpha=0.06, zorder=0)
    ax.axvline(FM_VIABILITY_CUTOFF, color="green", linestyle="--", alpha=0.7,
               label=f"FM viability ($\\geq${int(FM_VIABILITY_CUTOFF)}%)")
    ax.plot(lr_fm, lr_cr, "-o", color="#1f77b4", markersize=4,
            label="LR-Balanced threshold sweep (labels req.)")
    ax.plot(aria_fm_sorted, aria_cr_sorted, "-s", color="#d62728", markersize=5,
            linewidth=1.6, alpha=0.9, label="ARIA confidence-gated sweep (zero-shot)")
    ax.scatter([aria_op_fm], [aria_op_cr], color="#2ca02c", marker="*", s=220,
               zorder=6, edgecolors="black", linewidths=0.6,
               label=f"ARIA deployed point ({aria_op_fm:.1f}%, {aria_op_cr:.1f}%)")
    ax.axhline(0, color="grey", linestyle=":", alpha=0.5)

    ax.set_xlim(5, 102)
    ax.set_ylim(-3, 95)
    ax.set_xlabel("FM Accuracy (%)")
    ax.set_ylabel("Cost Reduction vs GPT-4o (%)")
    ax.set_title("Threshold sweep: LR (with labels) vs ARIA (zero-shot)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=180)
    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {OUT_PNG}")

    # Emit a TikZ-friendly coordinate dump for the deduplicated frontier
    # so we can paste it directly into the .tex file.
    print("\nTikZ coordinates (distinct ARIA frontier, sorted by FM acc):")
    distinct_sorted = sorted(distinct, key=lambda r: r["fm_acc"])
    coords = " ".join(f"({r['fm_acc']:.2f}, {r['cost_reduction_pct']:.2f})" for r in distinct_sorted)
    print(coords)


if __name__ == "__main__":
    main()
