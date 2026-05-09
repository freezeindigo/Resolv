"""
Resolution-Time Validation of ARIA's Contested Predictions.

The ground truth concern: ARIA is evaluated against CRM operator labels,
which are the noisy decisions of the current triage system. When ARIA
disagrees with CRM, we cannot tell from labels alone who is right.

This script uses aging_days as an independent signal:
  - FM complaints resolve quickly (median ~0 days, avg ~10 days)
  - Project complaints age significantly longer (median ~25 days, avg ~70 days)

Three contested groups are compared:
  A. ARIA=Project, CRM=FM  ("ARIA calls structural, operator calls maintenance")
     → If these age like Project, ARIA is correct
  B. ARIA=FM, CRM=Project  ("ARIA calls maintenance, operator calls structural")
     → If these age like FM, ARIA is correct
  C. Both agree (ARIA=CRM)  → control group

If Group A ages like Project and Group B ages like FM, ARIA's contested
calls are directionally correct, validating its divergence from CRM labels
against an outcome independent of labeling convention.

Output:
  eval/results/resolution_time_validation.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import psycopg2

EVAL_CSV = "eval/results/paper_eval_ambiguous_9259.csv"
DB_NAME = "resolv"

# Aging thresholds for classification
AGING_FM_THRESHOLD = 7      # ≤7 days = FM-like resolution
AGING_PROJECT_THRESHOLD = 30  # ≥30 days = Project-like aging


def load_predictions(csv_path: str) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def fetch_aging(ticket_ids: list[str]) -> dict[str, dict]:
    """Fetch aging_days and status for a list of ticket_ids."""
    conn = psycopg2.connect(dbname=DB_NAME)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ticket_id, aging_days, status, closed_date,
               resolution_tat_minutes, issue_type
        FROM complaints
        WHERE ticket_id = ANY(%s)
        """,
        (ticket_ids,)
    )
    result = {}
    for tid, aging, status, closed, tat, issue_type in cur.fetchall():
        result[tid] = {
            "aging_days": aging,
            "status": status,
            "closed": closed is not None,
            "resolution_tat_minutes": tat,
            "issue_type": issue_type,
        }
    cur.close()
    conn.close()
    return result


def aging_stats(aging_values: list[float]) -> dict:
    if not aging_values:
        return {"n": 0}
    arr = np.array(aging_values)
    return {
        "n": len(arr),
        "mean": round(float(arr.mean()), 1),
        "median": round(float(np.median(arr)), 1),
        "p75": round(float(np.percentile(arr, 75)), 1),
        "p90": round(float(np.percentile(arr, 90)), 1),
        "pct_over_7d": round(float((arr > 7).mean() * 100), 1),
        "pct_over_30d": round(float((arr > 30).mean() * 100), 1),
        "pct_over_60d": round(float((arr > 60).mean() * 100), 1),
    }


def main():
    print("=" * 65)
    print("Resolution-Time Validation of ARIA Contested Predictions")
    print("=" * 65)

    # Load ARIA predictions
    preds = load_predictions(EVAL_CSV)
    print(f"\nLoaded {len(preds)} predictions")

    # Segment into four groups
    groups = {
        "aria_proj_crm_fm":    [],  # A: ARIA=Project, CRM=FM  (contested, ARIA claims structural)
        "aria_fm_crm_proj":    [],  # B: ARIA=FM, CRM=Project  (contested, ARIA claims maintenance)
        "both_fm":             [],  # C1: Both agree = FM
        "both_proj":           [],  # C2: Both agree = Project
    }
    for row in preds:
        aria = row["aria_label"]
        crm = row["crm_label"]
        tid = row["ticket_id"]
        if aria == "Project" and crm == "FM":
            groups["aria_proj_crm_fm"].append(tid)
        elif aria == "FM" and crm == "Project":
            groups["aria_fm_crm_proj"].append(tid)
        elif aria == "FM" and crm == "FM":
            groups["both_fm"].append(tid)
        else:
            groups["both_proj"].append(tid)

    print(f"\nGroup sizes:")
    labels = {
        "aria_proj_crm_fm": "A: ARIA=Project, CRM=FM  (ARIA claims structural)",
        "aria_fm_crm_proj": "B: ARIA=FM,      CRM=Project (ARIA claims maintenance)",
        "both_fm":          "C1: Both=FM      (agreement, FM)",
        "both_proj":        "C2: Both=Project (agreement, Project)",
    }
    for k, tids in groups.items():
        print(f"  {labels[k]}: n={len(tids)}")

    # Fetch aging for all tickets
    all_ids = [tid for tids in groups.values() for tid in tids]
    print(f"\nFetching aging data for {len(all_ids)} tickets...")
    aging_data = fetch_aging(all_ids)
    print(f"  Found: {len(aging_data)} / {len(all_ids)}")

    # Compute stats per group
    results = {}
    print("\n" + "=" * 65)
    print("AGING STATISTICS BY GROUP")
    print("=" * 65)

    for k, tids in groups.items():
        aging_vals = [
            aging_data[tid]["aging_days"]
            for tid in tids
            if tid in aging_data and aging_data[tid]["aging_days"] is not None
        ]
        stats = aging_stats(aging_vals)
        results[k] = stats
        print(f"\n{labels[k]}:")
        print(f"  n={stats.get('n',0)}, mean={stats.get('mean','N/A')}d, "
              f"median={stats.get('median','N/A')}d, p90={stats.get('p90','N/A')}d")
        print(f"  >7d: {stats.get('pct_over_7d','N/A')}%  "
              f">30d: {stats.get('pct_over_30d','N/A')}%  "
              f">60d: {stats.get('pct_over_60d','N/A')}%")

    # Verdict — WITHIN-GROUP comparison only
    # The correct test is NOT "does Group A age like Project?"
    # (Project complaints always age longer — that's intrinsic to the work type)
    #
    # The correct test is: WITHIN the CRM=FM pool, do ARIA's contested calls
    # (ARIA=Project) age longer than uncontested FM calls (ARIA=FM)?
    # If yes: ARIA is flagging the structurally ambiguous cases that sit
    # unresolved in FM queues — the outcome-proxy logic per-prediction.
    #
    # Similarly: WITHIN the CRM=Project pool, do ARIA's contested calls
    # (ARIA=FM) age shorter than uncontested Project calls (ARIA=Project)?

    print("\n" + "=" * 65)
    print("VERDICT — within-CRM-label comparison (controls for work-type confound)")
    print("=" * 65)

    a = results["aria_proj_crm_fm"]   # CRM=FM, ARIA=Project
    c1 = results["both_fm"]            # CRM=FM, ARIA=FM  (uncontested FM)
    b = results["aria_fm_crm_proj"]   # CRM=Project, ARIA=FM
    c2 = results["both_proj"]          # CRM=Project, ARIA=Project (uncontested Project)

    if not all(r.get("n", 0) > 0 for r in [a, b, c1, c2]):
        print("Insufficient data for verdict.")
        return

    # Test 1: Within CRM=FM pool, do ARIA's contested calls age longer?
    # (if yes: they're likely misrouted structural cases — ARIA is right)
    a_ages_longer_than_c1 = a["median"] > c1["median"]
    a_pct_diff = round((a["pct_over_30d"] - c1["pct_over_30d"]), 1)

    # Test 2: Within CRM=Project pool, do ARIA's contested calls age shorter?
    # (if yes: they're likely over-escalated FM cases — ARIA is right)
    b_ages_shorter_than_c2 = b["median"] < c2["median"]
    b_pct_diff = round((b["pct_over_30d"] - c2["pct_over_30d"]), 1)

    print(f"\nTest 1 — Within CRM=FM pool:")
    print(f"  Uncontested FM (ARIA=FM):      n={c1['n']}, median={c1['median']}d, "
          f">30d={c1['pct_over_30d']}%")
    print(f"  Contested FM   (ARIA=Project): n={a['n']},  median={a['median']}d, "
          f">30d={a['pct_over_30d']}%")
    print(f"  ARIA-flagged cases age longer: {'YES ✓' if a_ages_longer_than_c1 else 'NO ✗'} "
          f"(+{a_pct_diff}pp in >30d rate)")

    print(f"\nTest 2 — Within CRM=Project pool:")
    print(f"  Uncontested Project (ARIA=Proj): n={c2['n']}, median={c2['median']}d, "
          f">30d={c2['pct_over_30d']}%")
    print(f"  Contested Project   (ARIA=FM):   n={b['n']},  median={b['median']}d, "
          f">30d={b['pct_over_30d']}%")
    print(f"  ARIA-dismissed cases age shorter: {'YES ✓' if b_ages_shorter_than_c2 else 'NO ✗'} "
          f"({b_pct_diff:+.1f}pp in >30d rate)")

    if a_ages_longer_than_c1 and b_ages_shorter_than_c2:
        verdict = (
            f"STRONG: Within the CRM=FM pool, ARIA's contested calls age "
            f"{a['median']}d (median) vs {c1['median']}d for uncontested FM "
            f"(+{a_pct_diff}pp in >30d rate), consistent with misrouted structural "
            f"cases sitting unresolved in FM queues. Within the CRM=Project pool, "
            f"ARIA's dismissed cases age {b['median']}d vs {c2['median']}d for "
            f"uncontested Project ({b_pct_diff:+.1f}pp in >30d rate), consistent "
            f"with over-escalated maintenance cases resolving faster."
        )
    elif a_ages_longer_than_c1:
        verdict = (
            f"PARTIAL: ARIA's contested FM calls (ARIA=Project, CRM=FM) age "
            f"{a['median']}d vs {c1['median']}d for uncontested FM "
            f"(+{a_pct_diff}pp >30d), supporting the misrouting hypothesis."
        )
    elif b_ages_shorter_than_c2:
        verdict = (
            f"PARTIAL: ARIA's dismissed Project calls age shorter than uncontested "
            f"Project ({b['median']}d vs {c2['median']}d)."
        )
    else:
        verdict = "INCONCLUSIVE: Within-group aging patterns do not validate ARIA's contested calls."

    print(f"\n{verdict}")

    # Paper-ready numbers
    print("\n" + "=" * 65)
    print("PAPER-READY NUMBERS")
    print("=" * 65)
    print(f"Within the {c1['n'] + a['n']} CRM=FM complaints:")
    print(f"  Uncontested FM (ARIA agrees): median {c1['median']}d, {c1['pct_over_30d']}% open >30d")
    print(f"  ARIA-flagged structural ({a['n']} cases): median {a['median']}d, "
          f"{a['pct_over_30d']}% open >30d (+{a_pct_diff}pp)")
    print(f"\nWithin the {c2['n'] + b['n']} CRM=Project complaints:")
    print(f"  Uncontested Project (ARIA agrees): median {c2['median']}d, {c2['pct_over_30d']}% open >30d")
    print(f"  ARIA-dismissed maintenance ({b['n']} cases): median {b['median']}d, "
          f"{b['pct_over_30d']}% open >30d ({b_pct_diff:+.1f}pp)")

    # Save
    out = {
        "verdict": verdict,
        "groups": results,
        "within_fm_pool": {
            "uncontested_fm_median_days": c1["median"],
            "aria_contested_median_days": a["median"],
            "aria_contested_pct_over_30d": a["pct_over_30d"],
            "uncontested_pct_over_30d": c1["pct_over_30d"],
            "aria_flags_age_longer": a_ages_longer_than_c1,
        },
        "within_project_pool": {
            "uncontested_proj_median_days": c2["median"],
            "aria_dismissed_median_days": b["median"],
            "aria_dismissed_pct_over_30d": b["pct_over_30d"],
            "uncontested_pct_over_30d": c2["pct_over_30d"],
            "aria_dismissals_age_shorter": b_ages_shorter_than_c2,
        },
    }
    out_path = Path("eval/results/resolution_time_validation.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
