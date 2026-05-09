"""
LR-Balanced Label Efficiency — the missing comparison.

The existing label_efficiency.py only tests LR-CostWeighted, which ALWAYS
violates the FM≥70% operational constraint regardless of label count.
LR-Balanced IS operationally viable (81.1% FM at full training set) and
achieves 70.6% cost reduction at FM≥70% (Figure 2 / threshold sweep).

This is the dangerous comparison the paper has not made:
  "How many labels does LR-Balanced need before it Pareto-dominates ARIA?"
  (i.e., beats ARIA on cost reduction AND maintains FM≥70%)

If the answer is 200+ labels, ARIA's pre-supervision window is substantial.
If the answer is 50 labels, the window is narrow and the contribution is weak.

Output:
  eval/results/lr_balanced_label_efficiency.json
  eval/results/lr_balanced_label_efficiency.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import psycopg2
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder

SEED = 42
DB_NAME = "resolv"
AMBIGUOUS_CATEGORIES = [
    "Plumbing", "Leakage", "Seepage", "Carpentary", "Civil Work", "Mason", "Civil",
]
ARIA_COST_REDUCTION_PCT = 22.7   # paper v4 verified number
ARIA_FM_ACC = 75.5               # paper v4
ARIA_PROJ_ACC = 35.8             # paper v4
ARIA_WTD_300 = 795               # paper v4
GPT4O_WTD_300 = 1029             # paper v4 baseline
FM_VIABILITY_THRESHOLD = 70.0    # operational constraint

LABEL_COUNTS = [10, 25, 50, 100, 200, 500, 1000, "full"]
N_SEEDS = 5   # average over multiple random subsamples for stability at low N


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


def build_vectorizers(train_rows):
    tfidf = TfidfVectorizer(max_features=500)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    tfidf.fit([r["title"] for r in train_rows])
    meta = np.array([[r["category"], r["priority"]] for r in train_rows], dtype=object)
    ohe.fit(meta)
    return tfidf, ohe


def featurize(rows, tfidf, ohe):
    X_text = tfidf.transform([r["title"] for r in rows])
    meta = np.array([[r["category"], r["priority"]] for r in rows], dtype=object)
    X_meta = ohe.transform(meta)
    return hstack([X_text, X_meta], format="csr")


def compute_metrics(y_true, y_pred, cost_weight=10) -> dict:
    n = len(y_true)
    proj_mask = y_true == 1
    fm_mask = y_true == 0
    proj_acc = float((y_pred[proj_mask] == 1).mean()) if proj_mask.any() else 0.0
    fm_acc = float((y_pred[fm_mask] == 0).mean()) if fm_mask.any() else 0.0
    hc = int(((y_true == 1) & (y_pred == 0)).sum())
    lc = int(((y_true == 0) & (y_pred == 1)).sum())
    wtd = hc * cost_weight + lc
    scale = 300 / n
    wtd_300 = wtd * scale
    cost_red = (GPT4O_WTD_300 - wtd_300) / GPT4O_WTD_300 * 100
    return {
        "proj_acc": round(proj_acc * 100, 1),
        "fm_acc": round(fm_acc * 100, 1),
        "hc_300": round(hc * scale, 1),
        "lc_300": round(lc * scale, 1),
        "wtd_300": round(wtd_300, 1),
        "cost_reduction_pct": round(cost_red, 1),
    }


def run_balanced_sweep(train_rows, test_rows) -> list[dict]:
    """
    LR-Balanced: class_weight='balanced' (equal weight per class regardless of label count).
    Averaged over N_SEEDS random subsamples at each label count for stability.
    """
    y_train_all = np.array([r["label"] for r in train_rows])
    y_test = np.array([r["label"] for r in test_rows])

    # Fit vectorizers on full training set
    tfidf, ohe = build_vectorizers(train_rows)
    X_train_full = featurize(train_rows, tfidf, ohe)
    X_test = featurize(test_rows, tfidf, ohe)

    results = []
    for n_labels in LABEL_COUNTS:
        if n_labels == "full":
            n_actual = len(train_rows)
            # Single run for full training set
            model = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=SEED
            )
            model.fit(X_train_full, y_train_all)
            y_pred = model.predict(X_test)
            m = compute_metrics(y_test, y_pred)
            m["n_labels"] = n_actual
            m["n_seeds_averaged"] = 1
            m["viable"] = m["fm_acc"] >= FM_VIABILITY_THRESHOLD
        else:
            n_actual = int(n_labels)
            if n_actual >= len(train_rows):
                n_actual = len(train_rows)
                model = LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=SEED
                )
                model.fit(X_train_full, y_train_all)
                y_pred = model.predict(X_test)
                m = compute_metrics(y_test, y_pred)
                m["n_labels"] = n_actual
                m["n_seeds_averaged"] = 1
                m["viable"] = m["fm_acc"] >= FM_VIABILITY_THRESHOLD
            else:
                # Average over N_SEEDS subsamples
                seed_metrics = []
                sss = StratifiedShuffleSplit(
                    n_splits=N_SEEDS, train_size=n_actual, random_state=SEED
                )
                for seed_idx, (idx, _) in enumerate(sss.split(X_train_full, y_train_all)):
                    X_sub = X_train_full[idx]
                    y_sub = y_train_all[idx]
                    model = LogisticRegression(
                        class_weight="balanced", max_iter=1000,
                        random_state=SEED + seed_idx
                    )
                    model.fit(X_sub, y_sub)
                    y_pred = model.predict(X_test)
                    seed_metrics.append(compute_metrics(y_test, y_pred))
                # Average all numeric fields
                m = {}
                for key in seed_metrics[0]:
                    m[key] = round(
                        float(np.mean([sm[key] for sm in seed_metrics])), 1
                    )
                m["n_labels"] = n_actual
                m["n_seeds_averaged"] = N_SEEDS
                m["viable"] = m["fm_acc"] >= FM_VIABILITY_THRESHOLD

        results.append(m)
        viable_str = "VIABLE" if m["viable"] else "unviable"
        print(
            f"  n={str(n_labels):>5}: FM={m['fm_acc']:5.1f}%  "
            f"Proj={m['proj_acc']:5.1f}%  "
            f"Cost={m['cost_reduction_pct']:+6.1f}%  "
            f"Wtd={m['wtd_300']:6.0f}  [{viable_str}]"
        )

    return results


def find_pareto_crossover(results: list[dict]) -> dict:
    """
    Find where LR-Balanced first Pareto-dominates ARIA:
      - FM accuracy >= FM_VIABILITY_THRESHOLD (operationally viable)
      - Cost reduction > ARIA_COST_REDUCTION_PCT (better cost performance)
    """
    for r in results:
        if r["viable"] and r["cost_reduction_pct"] > ARIA_COST_REDUCTION_PCT:
            return r
    return {}


def main():
    print("=" * 65)
    print("LR-Balanced Label Efficiency — vs ARIA operating point")
    print(f"ARIA: FM={ARIA_FM_ACC}%, Cost={ARIA_COST_REDUCTION_PCT:+.1f}%, Wtd={ARIA_WTD_300}")
    print(f"Viability threshold: FM≥{FM_VIABILITY_THRESHOLD}%")
    print("=" * 65)

    print("\nFetching data...")
    rows = fetch_rows()
    y_all = np.array([r["label"] for r in rows])
    n_proj = int(y_all.sum())
    n_fm = len(y_all) - n_proj
    print(f"Total: {len(rows)} (FM={n_fm}, Project={n_proj}, "
          f"ratio={n_proj/len(rows)*100:.1f}% Project)")

    train_rows, test_rows = train_test_split(
        rows, test_size=0.30, random_state=SEED,
        stratify=y_all,
    )
    print(f"Train: {len(train_rows)}, Test: {len(test_rows)}\n")

    print("Running LR-Balanced sweep (averaged over "
          f"{N_SEEDS} seeds at low-N)...")
    print(f"{'N':>7}  {'FM Acc':>8}  {'Proj Acc':>9}  {'Cost Red':>9}  "
          f"{'Wtd/300':>8}  {'Viable?':>8}")
    print("-" * 65)

    results = run_balanced_sweep(train_rows, test_rows)

    crossover = find_pareto_crossover(results)

    print("\n" + "=" * 65)
    print("FINDING")
    print("=" * 65)
    if crossover:
        n_cross = crossover["n_labels"]
        print(f"LR-Balanced first Pareto-dominates ARIA at n={n_cross} labels")
        print(f"  (FM={crossover['fm_acc']}% ≥ {FM_VIABILITY_THRESHOLD}%,  "
              f"Cost={crossover['cost_reduction_pct']:+.1f}% > ARIA {ARIA_COST_REDUCTION_PCT:+.1f}%)")
        if n_cross >= 200:
            print(f"\n  STRONG: Pre-supervision window is {n_cross}+ labels.")
            print("  Paper claim: ARIA is Pareto-dominant for the first "
                  f"~{n_cross} labeled examples.")
        elif n_cross >= 50:
            print(f"\n  MODERATE: Window is ~{n_cross} labels — substantiated but narrow.")
            print("  Revise paper: explicitly state the crossover N in Table 4 caption.")
        else:
            print(f"\n  WEAK: LR-Balanced dominates ARIA at only {n_cross} labels.")
            print("  Paper needs major reframe: ARIA's window is extremely narrow.")
    else:
        print("LR-Balanced NEVER Pareto-dominates ARIA in tested range.")
        print("  (Either FM stays below threshold, or cost stays below ARIA.)")
        print("  This is a VERY strong result — ARIA dominates at all tested scales.")

    # Save
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "aria_cost_reduction_pct": ARIA_COST_REDUCTION_PCT,
        "aria_fm_acc": ARIA_FM_ACC,
        "aria_wtd_300": ARIA_WTD_300,
        "gpt4o_wtd_300": GPT4O_WTD_300,
        "fm_viability_threshold": FM_VIABILITY_THRESHOLD,
        "pareto_crossover": crossover,
        "lr_balanced_sweep": results,
        "n_seeds_per_subsample": N_SEEDS,
    }
    json_path = out_dir / "lr_balanced_label_efficiency.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = out_dir / "lr_balanced_label_efficiency.csv"
    with open(csv_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")

    print("\nPaper Table 4 rows (LR-Balanced):")
    print(f"{'System':>25} | {'Labels':>7} | {'FM Acc':>7} | "
          f"{'Proj Acc':>9} | {'Wtd/300':>8} | {'vs GPT-4o':>10} | {'Viable':>7}")
    print("-" * 85)
    print(f"{'ARIA-Full':>25} | {'0':>7} | {ARIA_FM_ACC:>6.1f}% | "
          f"{ARIA_PROJ_ACC:>8.1f}% | {ARIA_WTD_300:>8.0f} | "
          f"{ARIA_COST_REDUCTION_PCT:>+9.1f}% | {'YES':>7}")
    for r in results:
        n_str = str(r["n_labels"])
        viable = "YES" if r["viable"] else "no"
        print(f"{'LR-Balanced':>25} | {n_str:>7} | {r['fm_acc']:>6.1f}% | "
              f"{r['proj_acc']:>8.1f}% | {r['wtd_300']:>8.0f} | "
              f"{r['cost_reduction_pct']:>+9.1f}% | {viable:>7}")


if __name__ == "__main__":
    main()
