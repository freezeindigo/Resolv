"""
Label Efficiency Experiment for ARIA paper.

Research question: At what labeled-data volume does supervised LR-CostWeighted
surpass ARIA's zero-shot cost reduction? This characterizes the "zero-shot advantage
regime" — the core novel finding that repositions the LR comparison.

What it does:
1. Fetches ambiguous complaints from DB (same categories as ARIA eval).
2. Trains LR-CostWeighted on increasing label counts: 10, 25, 50, 100, 200, 500, full.
3. Evaluates each on the held-out test set (30% stratified).
4. Plots LR cost reduction curve vs. ARIA's flat zero-shot line.
5. Finds the crossover N: "ARIA dominates until ~N labeled examples."

Output:
  eval/results/label_efficiency_results.json   — raw numbers for paper
  eval/results/label_efficiency_curve.png      — Figure 2 for paper
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import psycopg2
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

SEED = 42
DB_NAME = "resolv"

# Must match ARIA's ambiguous categories
AMBIGUOUS_CATEGORIES = [
    "Plumbing", "Leakage", "Seepage", "Carpentary", "Civil Work", "Mason", "Civil",
]

# ARIA full result from paper (zero-shot, no labeled data)
ARIA_COST_REDUCTION_PCT = 27.3   # from paper Table 2
ARIA_PROJ_ACC = 39.3             # from paper Table 2
GPT4O_WEIGHTED_COST = 1085       # from paper Table 2

# Label counts to sweep
LABEL_COUNTS = [10, 25, 50, 100, 200, 500, 1000, "full"]


def fetch_rows() -> list[dict]:
    cats_lower = {c.lower() for c in AMBIGUOUS_CATEGORIES}
    conn = psycopg2.connect(dbname=DB_NAME)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT complaint_title, category, priority, issue_type
        FROM complaints
        WHERE complaint_title IS NOT NULL
          AND issue_type IS NOT NULL
          AND category IS NOT NULL
        """
    )
    out = []
    for title, category, priority, issue_type in cur.fetchall():
        if (category or "").strip().lower() not in cats_lower:
            continue
        own = (issue_type or "").strip().lower()
        if "project" in own:
            y = 1
        elif "fm" in own or "facility" in own:
            y = 0
        else:
            continue
        out.append({
            "title": str(title),
            "category": (category or "").strip(),
            "priority": str(priority) if priority else "Unknown",
            "label": y,
        })
    cur.close()
    conn.close()
    return out


def build_features(train_rows, test_rows):
    tfidf = TfidfVectorizer(max_features=500)
    ohe = OneHotEncoder(handle_unknown="ignore")

    X_tr_text = tfidf.fit_transform([r["title"] for r in train_rows])
    X_te_text = tfidf.transform([r["title"] for r in test_rows])

    tr_meta = np.array([[r["category"], r["priority"]] for r in train_rows], dtype=object)
    te_meta = np.array([[r["category"], r["priority"]] for r in test_rows], dtype=object)
    X_tr_meta = ohe.fit_transform(tr_meta)
    X_te_meta = ohe.transform(te_meta)

    X_train = hstack([X_tr_text, X_tr_meta], format="csr")
    X_test = hstack([X_te_text, X_te_meta], format="csr")
    y_train = np.array([r["label"] for r in train_rows])
    y_test = np.array([r["label"] for r in test_rows])
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred, cost_weight=10):
    n = len(y_true)
    proj_mask = y_true == 1
    fm_mask = y_true == 0
    proj_acc = float((y_pred[proj_mask] == 1).mean()) if proj_mask.any() else 0.0
    fm_acc = float((y_pred[fm_mask] == 0).mean()) if fm_mask.any() else 0.0
    hc = int(((y_true == 1) & (y_pred == 0)).sum())   # Project misrouted to FM
    lc = int(((y_true == 0) & (y_pred == 1)).sum())   # FM misrouted to Project
    wtd = hc * cost_weight + lc * 1

    # Normalize to per-300 (same as paper)
    scale = 300 / n
    wtd_300 = wtd * scale
    hc_300 = hc * scale
    lc_300 = lc * scale

    cost_red = (GPT4O_WEIGHTED_COST - wtd_300) / GPT4O_WEIGHTED_COST * 100
    return {
        "proj_acc": proj_acc,
        "fm_acc": fm_acc,
        "hc_300": round(hc_300, 1),
        "lc_300": round(lc_300, 1),
        "wtd_300": round(wtd_300, 1),
        "cost_reduction_pct": round(cost_red, 1),
    }


def run_label_sweep(train_rows, test_rows, label_counts):
    X_train_full, X_test, y_train_full, y_test = build_features(train_rows, test_rows)
    results = []

    for n_labels in label_counts:
        if n_labels == "full":
            n = len(train_rows)
            X_tr = X_train_full
            y_tr = y_train_full
        else:
            n = int(n_labels)
            if n >= len(train_rows):
                n = len(train_rows)
                X_tr = X_train_full
                y_tr = y_train_full
            else:
                # Stratified subsample
                _, idx = train_test_split(
                    np.arange(len(train_rows)), test_size=n, stratify=y_train_full,
                    random_state=SEED,
                )
                X_tr = X_train_full[idx]
                y_tr = y_train_full[idx]

        model = LogisticRegression(
            class_weight={0: 1, 1: 10}, max_iter=1000, random_state=SEED
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_test)
        m = compute_metrics(y_test, y_pred)
        m["n_labels"] = n
        results.append(m)
        label_str = "full" if n_labels == "full" else str(n_labels)
        print(
            f"  n={label_str:>5}: Proj Acc={m['proj_acc']*100:5.1f}%  "
            f"Cost Red={m['cost_reduction_pct']:+6.1f}%  "
            f"Wtd/300={m['wtd_300']:6.0f}"
        )

    return results


def find_crossover(results, aria_cost_reduction):
    """Find the N at which LR first exceeds ARIA's cost reduction."""
    for r in results:
        if r["cost_reduction_pct"] >= aria_cost_reduction:
            return r["n_labels"]
    return None


def plot_curve(results, aria_cost_reduction, out_path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
    except ImportError:
        print("matplotlib not available — skipping plot. Install with: pip install matplotlib")
        return

    ns = [r["n_labels"] for r in results]
    cost_reds = [r["cost_reduction_pct"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 6))

    # LR-CostWeighted curve
    ax.plot(ns, cost_reds, "o-", color="#c0392b", linewidth=2, markersize=7,
            label="LR-CostWeighted (supervised)", zorder=5)

    # ARIA flat line
    ax.axhline(y=aria_cost_reduction, color="#27ae60", linewidth=2,
               linestyle="--", label=f"ARIA-Full (zero-shot, {aria_cost_reduction}%)", zorder=4)

    # GPT-4o baseline (0%)
    ax.axhline(y=0, color="#e67e22", linewidth=1.5, linestyle=":",
               label="GPT-4o text-only (0%)", alpha=0.7, zorder=3)

    # Crossover annotation
    crossover_n = find_crossover(results, aria_cost_reduction)
    if crossover_n is not None:
        ax.axvline(x=crossover_n, color="gray", linewidth=1, linestyle=":", alpha=0.6)
        ax.annotate(
            f"Crossover ≈ {crossover_n} labels",
            xy=(crossover_n, aria_cost_reduction),
            xytext=(crossover_n * 1.15, aria_cost_reduction - 8),
            fontsize=9, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Labeled Training Examples", fontsize=12)
    ax.set_ylabel("Expected Cost Reduction vs. GPT-4o (%)", fontsize=12)
    ax.set_title(
        "Figure 2: Label Efficiency — Zero-Shot ARIA vs. Supervised LR-CostWeighted\n"
        "ARIA provides the best zero-shot baseline; LR surpasses ARIA only with sufficient labeled data",
        fontsize=10.5, fontweight="bold"
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(8, max(ns) * 1.5)

    # Shade "ARIA advantage zone"
    xs = [r["n_labels"] for r in results if r["cost_reduction_pct"] < aria_cost_reduction]
    if xs:
        ax.axvspan(8, max(xs), alpha=0.07, color="#27ae60",
                   label="Zero-shot advantage zone")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    plt.close()


def main():
    import os
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    print("Fetching data from DB...")
    rows = fetch_rows()
    print(f"Total rows: {len(rows)}")
    labels = [r["label"] for r in rows]
    n_proj = sum(labels)
    n_fm = len(labels) - n_proj
    print(f"  FM={n_fm}, Project={n_proj}, ratio={n_proj/len(labels)*100:.1f}% Project")

    train_rows, test_rows = train_test_split(
        rows, test_size=0.30, random_state=SEED,
        stratify=[r["label"] for r in rows],
    )
    print(f"Train: {len(train_rows)}, Test: {len(test_rows)}")
    print()
    print("Sweeping label counts...")
    print(f"{'N':>7}  {'Proj Acc':>10}  {'Cost Red':>10}  {'Wtd/300':>8}")

    # Resolve "full" to an actual number before sweep
    actual_counts = []
    for n in LABEL_COUNTS:
        if n == "full":
            actual_counts.append("full")
        elif int(n) < len(train_rows):
            actual_counts.append(n)
        else:
            actual_counts.append("full")

    results = run_label_sweep(train_rows, test_rows, actual_counts)

    crossover = find_crossover(results, ARIA_COST_REDUCTION_PCT)
    print()
    if crossover:
        print(f"CROSSOVER: LR first matches ARIA cost reduction at n={crossover} labeled examples")
        print(f"=> ARIA dominates in the zero-shot regime (0 labels) up to ~{crossover} examples")
    else:
        print("LR does not reach ARIA cost reduction in tested range — ARIA is stronger")

    # Save JSON
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "aria_cost_reduction_pct": ARIA_COST_REDUCTION_PCT,
        "aria_proj_acc_pct": ARIA_PROJ_ACC,
        "gpt4o_weighted_cost_baseline": GPT4O_WEIGHTED_COST,
        "crossover_n_labels": crossover,
        "test_size": len(test_rows),
        "train_size_full": len(train_rows),
        "lr_sweep": results,
    }
    json_path = out_dir / "label_efficiency_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")

    # Plot
    plot_curve(results, ARIA_COST_REDUCTION_PCT, out_dir / "label_efficiency_curve.png")

    # Print table for paper
    print()
    print("Paper table:")
    print(f"{'N Labels':>10} | {'Proj Acc':>9} | {'FM Acc':>7} | {'Wtd/300':>8} | {'vs GPT-4o':>10}")
    print("-" * 60)
    # ARIA row first
    print(f"{'ARIA (0)':>10} | {ARIA_PROJ_ACC:>8.1f}% | {'72.5':>6}% | {'789':>8} | {'-27.3%':>10}")
    for r in results:
        n_str = str(r["n_labels"])
        print(
            f"{n_str:>10} | {r['proj_acc']*100:>8.1f}% | {r['fm_acc']*100:>6.1f}% | "
            f"{r['wtd_300']:>8.0f} | {r['cost_reduction_pct']:>+9.1f}%"
        )


if __name__ == "__main__":
    main()
