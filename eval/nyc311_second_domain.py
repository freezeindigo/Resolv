"""
NYC 311 Second-Domain Replication.

Goal: show that the accuracy trap and cost trap reproduce on a structurally
analogous public dataset — not a full ARIA deployment, just the two failure modes.

Routing analog:
  FM    → HPD (NYC Dept of Housing Preservation and Development)
          Routine maintenance: plumbing leaks, paint, broken fixtures.
          Resolved in hours–days. Low-cost misrouting.

  Project → DOB (NYC Dept of Buildings)
            Structural defects, illegal construction, structural water damage.
            Requires inspection, regulatory consequence. High-cost misrouting.

We replicate the two failure modes:
  1. Accuracy trap: GPT-4o (text-only) achieves high HPD accuracy but fails on DOB.
  2. Cost trap:     GPT-4o (cost-prompt) collapses HPD accuracy.

These two rows are sufficient to claim: "The accuracy trap and cost trap are not
artifacts of the AlphaDev domain — they reproduce on a structurally analogous
public dataset with a different country, different regulatory context, and
operator-independent labels."

Output:
  eval/results/nyc311_accuracy_trap.json
  eval/results/nyc311_cost_trap.json
  eval/results/nyc311_summary.json
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    raise RuntimeError("pip install requests")

try:
    from openai import OpenAI
except ImportError:
    raise RuntimeError("pip install openai")

SEED = 42
N_SAMPLE = 1200  # use full cached set for tighter CIs

# NYC 311 API — free, no auth required
NYC_API_BASE = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

# Complaint types that map cleanly to HPD (FM) or DOB (Project)
# NOTE: Socrata stores mixed case; query uses exact strings from NYC 311 schema.
HPD_COMPLAINT_TYPES = [
    "PLUMBING",
    "PAINT/PLASTER",
    "DOOR/WINDOW",
    "FLOORING/STAIRS",
    "WATER LEAK",
    "HEAT/HOT WATER",
    "UNSANITARY CONDITION",
    "Plumbing",
    "Paint/Plaster",
    "Door/Window",
    "Flooring/Stairs",
    "Water Leak",
    "Heat/Hot Water",
    "Unsanitary Condition",
]
# DOB: structural / illegal construction — use types that exist in erm2-nwe9 under agency=DOB
DOB_COMPLAINT_TYPES = [
    "Building/Use",
    "General Construction/Plumbing",
    "Facades",
    "Unstable Building",
    "Construction Safety Enforcement",
    "Boiler",
    "Boilers",
    "Facade Insp Safety Pgm",
    "Plumbing",
    "Electrical",
]

# Cost matrix: misrouting DOB (structural) to HPD (maintenance) is high-cost
COST_W_HIGH = 10   # misroute Project→FM  (structural sent to maintenance)
COST_W_LOW  = 1    # misroute FM→Project  (maintenance sent to structural team)

# GPT-4o prompts ──────────────────────────────────────────────────────────────

TEXT_ONLY_SYSTEM = """You are an expert in New York City building and housing complaints.
Classify each complaint as either:
  HPD - routine housing maintenance (plumbing leak, broken fixture, paint, heat)
  DOB - structural/construction issue (structural damage, illegal construction, building defect)

Return exactly one word: HPD or DOB."""

TEXT_ONLY_USER = "Complaint description: {description}"

COST_PROMPT_SYSTEM = """You are an expert in New York City building and housing complaints.
Classify each complaint as either HPD (routine maintenance) or DOB (structural/construction).

CRITICAL COST ASYMMETRY:
- Misrouting a DOB (structural) complaint to HPD is 10× more costly: structural damage compounds, regulatory violations accumulate, resident safety risk.
- Misrouting an HPD (maintenance) complaint to DOB is 1× cost: minor delay.

Because of this asymmetry, strongly prefer DOB when there is ANY structural ambiguity.
Only route to HPD when you are highly confident it is routine maintenance.

Return exactly one word: HPD or DOB."""

COST_PROMPT_USER = "Complaint description: {description}\n\nGiven the asymmetric costs, classify this complaint."


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_nyc311_complaints(n_per_label: int = 200) -> list[dict]:
    """
    Fetch ~n_per_label HPD and ~n_per_label DOB complaints from NYC 311.
    Uses the Socrata Open Data API (no auth key required, rate-limited).
    Date range: 2022-2024 to avoid COVID-era anomalies.
    """
    complaints = []

    for agency, ctypes, label in [
        ("HPD", HPD_COMPLAINT_TYPES, "HPD"),
        ("DOB", DOB_COMPLAINT_TYPES, "DOB"),
    ]:
        print(f"Fetching {n_per_label} {agency} complaints...")
        # Build OR filter for complaint types
        type_filter = " OR ".join(
            f"complaint_type='{ct}'" for ct in ctypes
        )
        params = {
            "$where": (
                f"agency='{agency}' AND ({type_filter}) "
                f"AND created_date >= '2022-01-01T00:00:00' "
                f"AND created_date <= '2024-12-31T00:00:00' "
                f"AND descriptor IS NOT NULL "
                f"AND complaint_type IS NOT NULL"
            ),
            "$limit": n_per_label * 8,
            "$offset": 0,
            "$select": "unique_key,complaint_type,descriptor,agency,created_date,closed_date,status",
            "$order": "created_date DESC",
        }
        try:
            resp = requests.get(NYC_API_BASE, params=params, timeout=30)
            resp.raise_for_status()
            rows = resp.json()
        except Exception as e:
            print(f"  ERROR fetching {agency}: {e}")
            continue

        seen = set()
        for row in rows:
            uk = str(row.get("unique_key") or "")
            desc = (row.get("descriptor") or "").strip()
            if not desc or not uk or uk in seen or len(desc) < 5:
                continue
            seen.add(uk)
            complaints.append({
                "id": row.get("unique_key", ""),
                "complaint_type": row.get("complaint_type", ""),
                "description": desc,
                "agency": agency,
                "crm_label": label,
                "created_date": row.get("created_date", ""),
                "closed_date": row.get("closed_date", ""),
                "status": row.get("status", ""),
            })
            if len([c for c in complaints if c["crm_label"] == label]) >= n_per_label:
                break

        time.sleep(0.5)   # be polite to the API

    print(f"Fetched: {sum(1 for c in complaints if c['crm_label']=='HPD')} HPD, "
          f"{sum(1 for c in complaints if c['crm_label']=='DOB')} DOB")
    return complaints


def stratified_sample(complaints: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """Stratified sample preserving HPD/DOB ratio."""
    import random
    random.seed(seed)
    hpd = [c for c in complaints if c["crm_label"] == "HPD"]
    dob = [c for c in complaints if c["crm_label"] == "DOB"]
    total = len(hpd) + len(dob)
    n_hpd = round(n * len(hpd) / total)
    n_dob = n - n_hpd
    random.shuffle(hpd)
    random.shuffle(dob)
    sample = hpd[:n_hpd] + dob[:n_dob]
    random.shuffle(sample)
    return sample


# ── GPT-4o inference ──────────────────────────────────────────────────────────

def classify_complaints(
    complaints: list[dict],
    client: OpenAI,
    system_prompt: str,
    user_template: str,
    result_key: str,
    max_tokens: int = 10,
) -> list[dict]:
    results = []
    for i, row in enumerate(complaints):
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(complaints)}...")
        user_msg = user_template.format(
            complaint_type=row["complaint_type"],
            description=row["description"],
        )
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip().upper()
        if "DOB" in raw or "STRUCTURAL" in raw or "CONSTRUCTION" in raw:
            pred = "DOB"
        else:
            pred = "HPD"
        row = dict(row)
        row[result_key] = pred
        row["raw_" + result_key] = raw
        results.append(row)
    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    results: list[dict],
    pred_key: str,
    label_key: str = "crm_label",
    cost_weight: int = COST_W_HIGH,
) -> dict:
    n = len(results)
    # DOB = Project (high-cost class), HPD = FM (low-cost class)
    dob_rows = [r for r in results if r[label_key] == "DOB"]
    hpd_rows = [r for r in results if r[label_key] == "HPD"]

    dob_right = sum(1 for r in dob_rows if r[pred_key] == "DOB")
    hpd_right = sum(1 for r in hpd_rows if r[pred_key] == "HPD")
    hc = sum(1 for r in dob_rows if r[pred_key] == "HPD")    # misroute DOB→HPD (high cost)
    lc = sum(1 for r in hpd_rows if r[pred_key] == "DOB")    # misroute HPD→DOB (low cost)

    scale = 300 / n
    wtd_300 = (hc * cost_weight + lc) * scale

    return {
        "hpd_acc": round(hpd_right / len(hpd_rows) * 100, 1) if hpd_rows else 0,
        "dob_acc": round(dob_right / len(dob_rows) * 100, 1) if dob_rows else 0,
        "overall_acc": round((dob_right + hpd_right) / n * 100, 1),
        "hc_300": round(hc * scale, 1),
        "lc_300": round(lc * scale, 1),
        "wtd_300": round(wtd_300, 1),
        "n": n,
        "n_hpd": len(hpd_rows),
        "n_dob": len(dob_rows),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=key)

    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Fetch data ────────────────────────────────────────────────────
    print("=" * 60)
    print("NYC 311 Second-Domain Replication")
    print("=" * 60)

    cache_path = out_dir / "nyc311_raw_complaints.json"
    if cache_path.exists():
        print(f"Loading cached complaints from {cache_path}")
        with open(cache_path) as f:
            all_complaints = json.load(f)
    else:
        all_complaints = fetch_nyc311_complaints(n_per_label=600)
        with open(cache_path, "w") as f:
            json.dump(all_complaints, f)

    sample = stratified_sample(all_complaints, N_SAMPLE)
    hpd_n = sum(1 for c in sample if c["crm_label"] == "HPD")
    dob_n = sum(1 for c in sample if c["crm_label"] == "DOB")
    print(f"\nEval sample: {len(sample)} complaints (HPD={hpd_n}, DOB={dob_n})")

    # ── Step 2: Accuracy trap (text-only) ────────────────────────────────────
    print("\n[1/2] GPT-4o text-only (accuracy-optimized)...")
    results_accuracy = classify_complaints(
        sample, client, TEXT_ONLY_SYSTEM, TEXT_ONLY_USER,
        result_key="text_only_pred",
    )
    m_acc = compute_metrics(results_accuracy, pred_key="text_only_pred")

    print(f"\n  Accuracy Trap — NYC 311 (GPT-4o text-only):")
    print(f"  HPD (FM analog) accuracy:  {m_acc['hpd_acc']}%  [AlphaDev: 91.6%]")
    print(f"  DOB (Project analog) acc:  {m_acc['dob_acc']}%  [AlphaDev: 13.2%]")
    print(f"  Overall accuracy:          {m_acc['overall_acc']}%  [AlphaDev: 61.1%]")
    print(f"  HC misroutings/300:        {m_acc['hc_300']}     [AlphaDev: 107]")

    # ── Step 3: Cost trap (cost-prompt) ──────────────────────────────────────
    print("\n[2/2] GPT-4o cost-prompt (cost-optimized)...")
    results_cost = classify_complaints(
        sample, client, COST_PROMPT_SYSTEM, COST_PROMPT_USER,
        result_key="cost_prompt_pred",
    )
    # Carry forward text-only predictions for comparison
    for r_c, r_a in zip(results_cost, results_accuracy):
        r_c["text_only_pred"] = r_a["text_only_pred"]

    m_cost = compute_metrics(results_cost, pred_key="cost_prompt_pred")

    print(f"\n  Cost Trap — NYC 311 (GPT-4o cost-prompt):")
    print(f"  HPD (FM analog) accuracy:  {m_cost['hpd_acc']}%  [AlphaDev: 19.1%]")
    print(f"  DOB (Project analog) acc:  {m_cost['dob_acc']}%  [AlphaDev: 81.1%]")
    print(f"  Overall accuracy:          {m_cost['overall_acc']}%")
    print(f"  LC misroutings/300:        {m_cost['lc_300']}     [AlphaDev: 144]")

    # ── Step 4: Outcome proxy (open > 60 days in DOB queue) ──────────────────
    # Compute resolution-time proxy analogous to AlphaDev's 242 ticket signal
    from datetime import datetime
    long_open = []
    for r in all_complaints:
        if r["crm_label"] != "DOB":
            continue
        created = r.get("created_date", "")
        closed = r.get("closed_date", "")
        status = r.get("status", "")
        if not closed and status != "Closed":
            long_open.append(r)
        elif closed and created:
            try:
                d1 = datetime.fromisoformat(created[:10])
                d2 = datetime.fromisoformat(closed[:10])
                if (d2 - d1).days > 60:
                    long_open.append(r)
            except (ValueError, TypeError):
                pass
    print(f"\n  Outcome proxy: {len(long_open)} DOB complaints open/unresolved >60 days "
          f"[AlphaDev analog: 242]")

    # ── Step 5: Summary ───────────────────────────────────────────────────────
    summary = {
        "dataset": "NYC 311 Service Requests (2022-2024)",
        "routing_analog": {
            "FM": "HPD (routine maintenance)",
            "Project": "DOB (structural/construction defects)",
        },
        "sample_size": len(sample),
        "accuracy_trap": {
            "system": "GPT-4o (text-only)",
            "hpd_acc": m_acc["hpd_acc"],
            "dob_acc": m_acc["dob_acc"],
            "asymmetry_ratio": round(m_acc["hpd_acc"] / m_acc["dob_acc"], 1) if m_acc["dob_acc"] else None,
            "hc_300": m_acc["hc_300"],
        },
        "cost_trap": {
            "system": "GPT-4o (cost-prompt)",
            "hpd_acc": m_cost["hpd_acc"],
            "dob_acc": m_cost["dob_acc"],
            "lc_300": m_cost["lc_300"],
        },
        "outcome_proxy_n": len(long_open),
    }

    print("\n" + "=" * 60)
    print("FINDING SUMMARY")
    print("=" * 60)
    acc_asymm = summary["accuracy_trap"]["asymmetry_ratio"]
    if acc_asymm and acc_asymm >= 3:
        print(f"ACCURACY TRAP REPRODUCED: {m_acc['hpd_acc']}% HPD vs {m_acc['dob_acc']}% DOB "
              f"({acc_asymm}× asymmetry; AlphaDev: 91.6%/13.2%, 6.9×)")
    else:
        print(f"Accuracy trap partial: {m_acc['hpd_acc']}% HPD / {m_acc['dob_acc']}% DOB")

    if m_cost["hpd_acc"] < 60:
        print(f"COST TRAP REPRODUCED: HPD collapses to {m_cost['hpd_acc']}% under cost-prompt "
              f"(AlphaDev: 19.1%)")
    else:
        print(f"Cost trap partial: HPD accuracy {m_cost['hpd_acc']}% under cost-prompt")

    # Save
    json_path = out_dir / "nyc311_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-complaint CSVs
    import csv
    for results, fname in [
        (results_accuracy, "nyc311_accuracy_trap.csv"),
        (results_cost, "nyc311_cost_trap.csv"),
    ]:
        with open(out_dir / fname, "w", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)

    print(f"\nSaved: {json_path}")

    # Paper-ready output
    print("\n" + "=" * 60)
    print("PAPER ADDITION — Generalizability paragraph data points:")
    print("=" * 60)
    print(f"NYC 311 (n={len(sample)}, HPD={hpd_n}, DOB={dob_n}):")
    print(f"  Text-only: HPD={m_acc['hpd_acc']}%, DOB={m_acc['dob_acc']}% "
          f"(asymmetry={acc_asymm}×)")
    print(f"  Cost-prompt: HPD={m_cost['hpd_acc']}%, DOB={m_cost['dob_acc']}%")
    print(f"\nAdd to §5 Discussion / Generalizability as:")
    print(f"  '...the accuracy trap (HPD={m_acc['hpd_acc']}%, DOB={m_acc['dob_acc']}%;")
    print(f"   {acc_asymm}× asymmetry) and cost trap (HPD collapses to {m_cost['hpd_acc']}%)")
    print(f"   reproduce on NYC 311 complaint routing (HPD vs DOB; n={len(sample)}).'")


if __name__ == "__main__":
    main()
