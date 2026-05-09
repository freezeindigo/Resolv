"""
CFPB Second Domain — Two-Traps Replication.

Tests whether the accuracy trap and cost trap reproduce on CFPB consumer
complaint data, using:
  - Mortgage complaints  → "Mortgage Specialist" (high-cost class, like Project)
  - Credit card / Checking or savings → "General Service" (low-cost class, like FM)

Cost asymmetry mirrors the paper:
  W[route=General, true=Mortgage] = 10  (regulatory consequence, escalation delay)
  W[route=Mortgage, true=General] = 1   (minor over-escalation)

Two baselines:
  1. Text-only: GPT-4o routes on complaint narrative alone, no cost framing
     → Expect accuracy trap: General class dominates, Mortgage recall collapses
  2. Cost-prompt: GPT-4o given 10× cost asymmetry
     → Expect cost trap: routes everything to Mortgage, General class collapses

Output:
  eval/results/cfpb_second_domain_results.json
  eval/results/cfpb_text_only_results.csv
  eval/results/cfpb_cost_prompt_results.csv
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import re
import time
import zipfile
from pathlib import Path

import psycopg2

try:
    from openai import OpenAI, RateLimitError
except ImportError:
    raise RuntimeError("pip install openai")

SEED = 42
ZIP_PATH = os.path.expanduser("~/Downloads/complaints.csv.zip")
N_SAMPLE = 500      # per class (balanced)
MIN_NARRATIVE_LEN = 50

# Complaint routing analog
HIGH_COST_LABEL = "Mortgage"         # like Project: misrouting has regulatory consequence
LOW_COST_LABEL  = "General"          # like FM: Credit card + Checking/savings

MORTGAGE_PRODUCTS = {"mortgage"}
GENERAL_PRODUCTS  = {"credit card", "checking or savings account",
                     "credit card or prepaid card"}

# ── Prompts ───────────────────────────────────────────────────────────────────

TEXT_ONLY_SYSTEM = """You are a financial complaint routing system.

Route each consumer complaint to one of two teams:
- Mortgage Specialist: handles mortgage loans, HELOC, home equity, refinancing, foreclosure
- General Service: handles credit cards, checking accounts, savings accounts, prepaid cards

Respond with JSON only: {"decision": "Mortgage Specialist" or "General Service", "reasoning": "<one sentence>"}"""

TEXT_ONLY_USER = "Consumer complaint: {narrative}"

COST_PROMPT_SYSTEM = """You are a financial complaint routing system.

Route each consumer complaint to one of two teams:
- Mortgage Specialist: handles mortgage loans, HELOC, home equity, refinancing, foreclosure
- General Service: handles credit cards, checking accounts, savings accounts, prepaid cards

## Cost Matrix
Sending a Mortgage complaint to General Service: cost = 10
  (Consequence: regulatory escalation delayed, potential CFPB violation, legal liability)
Sending a General Service complaint to Mortgage Specialist: cost = 1
  (Consequence: minor misdirection, easily corrected)

When uncertain, prefer Mortgage Specialist — the asymmetric cost justifies routing up.

Respond with JSON only: {"decision": "Mortgage Specialist" or "General Service", "reasoning": "<one sentence>"}"""

COST_PROMPT_USER = "Consumer complaint: {narrative}"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_cfpb_sample(zip_path: str, n_per_class: int = N_SAMPLE) -> list[dict]:
    """Stream-read the CFPB zip and extract balanced Mortgage vs General samples."""
    random.seed(SEED)
    print(f"Opening {zip_path}...")

    mortgage_rows = []
    general_rows  = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        csv_name = next(n for n in names if n.endswith(".csv"))
        print(f"  Reading {csv_name} ...")

        with zf.open(csv_name) as raw:
            text_stream = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            reader = csv.DictReader(text_stream)
            for i, row in enumerate(reader):
                if i % 500_000 == 0 and i > 0:
                    print(f"    scanned {i:,} rows  "
                          f"mortgage={len(mortgage_rows):,}  general={len(general_rows):,}")
                # Stop early if we have enough candidates (10× buffer)
                if len(mortgage_rows) >= n_per_class * 10 and len(general_rows) >= n_per_class * 10:
                    break

                product  = (row.get("Product") or "").strip().lower()
                narrative = (row.get("Consumer complaint narrative") or "").strip()
                if len(narrative) < MIN_NARRATIVE_LEN:
                    continue

                if product in MORTGAGE_PRODUCTS:
                    mortgage_rows.append({"narrative": narrative, "product": row["Product"],
                                          "issue": row.get("Issue","")})
                elif product in GENERAL_PRODUCTS:
                    general_rows.append({"narrative": narrative, "product": row["Product"],
                                         "issue": row.get("Issue","")})

    print(f"  Candidates: Mortgage={len(mortgage_rows):,}, General={len(general_rows):,}")

    random.shuffle(mortgage_rows)
    random.shuffle(general_rows)
    take = min(n_per_class, len(mortgage_rows), len(general_rows))
    print(f"  Sampling {take} per class (seed={SEED})")

    sampled = []
    for row in mortgage_rows[:take]:
        sampled.append({**row, "true_label": HIGH_COST_LABEL})
    for row in general_rows[:take]:
        sampled.append({**row, "true_label": LOW_COST_LABEL})

    random.shuffle(sampled)
    return sampled


# ── Inference ─────────────────────────────────────────────────────────────────

def parse_response(raw: str) -> dict:
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            decision_raw = str(parsed.get("decision", "")).lower()
            if "mortgage" in decision_raw:
                decision = HIGH_COST_LABEL
            else:
                decision = LOW_COST_LABEL
            return {"decision": decision, "reasoning": str(parsed.get("reasoning", "")),
                    "parse_ok": True}
        except (json.JSONDecodeError, ValueError):
            pass
    if "mortgage" in raw.lower():
        return {"decision": HIGH_COST_LABEL, "reasoning": raw[:200], "parse_ok": False}
    return {"decision": LOW_COST_LABEL, "reasoning": raw[:200], "parse_ok": False}


def run_baseline(complaints: list[dict], client: OpenAI,
                 system_prompt: str, user_template: str,
                 label: str) -> list[dict]:
    results = []
    parse_failures = 0
    print(f"\nRunning {label} ({len(complaints)} complaints)...")
    for i, row in enumerate(complaints):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(complaints)}  parse_failures={parse_failures}")
        user_msg = user_template.format(narrative=row["narrative"][:1000])
        resp = None
        for attempt in range(10):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0,
                    max_tokens=150,
                    response_format={"type": "json_object"},
                )
                break
            except RateLimitError:
                wait = min(90, 3 + attempt * 5)
                print(f"    rate limit at row {i+1}, sleeping {wait}s...")
                time.sleep(wait)
        if resp is None:
            raise RuntimeError("OpenAI rate limit: exhausted retries")
        time.sleep(0.12)
        raw = (resp.choices[0].message.content or "").strip()
        parsed = parse_response(raw)
        if not parsed["parse_ok"]:
            parse_failures += 1
        results.append({
            "true_label":  row["true_label"],
            "product":     row["product"],
            "decision":    parsed["decision"],
            "reasoning":   parsed["reasoning"],
            "parse_ok":    parsed["parse_ok"],
            "narrative":   row["narrative"][:300],
        })
    print(f"  Parse failures: {parse_failures}/{len(complaints)}")
    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict], cost_weight: int = 10,
                    baseline_wtd: float | None = None) -> dict:
    n = len(results)
    mort_rows  = [r for r in results if r["true_label"] == HIGH_COST_LABEL]
    gen_rows   = [r for r in results if r["true_label"] == LOW_COST_LABEL]

    mort_right = sum(1 for r in mort_rows if r["decision"] == HIGH_COST_LABEL)
    gen_right  = sum(1 for r in gen_rows  if r["decision"] == LOW_COST_LABEL)

    hc = sum(1 for r in mort_rows if r["decision"] == LOW_COST_LABEL)   # high-cost errors
    lc = sum(1 for r in gen_rows  if r["decision"] == HIGH_COST_LABEL)  # low-cost errors
    wtd = hc * cost_weight + lc

    scale = 300 / n
    wtd_300 = wtd * scale

    overall_acc = (mort_right + gen_right) / n * 100
    mort_acc    = mort_right / len(mort_rows) * 100 if mort_rows else 0
    gen_acc     = gen_right  / len(gen_rows)  * 100 if gen_rows  else 0

    cost_red = None
    if baseline_wtd is not None and baseline_wtd > 0:
        cost_red = round((baseline_wtd - wtd_300) / baseline_wtd * 100, 1)

    return {
        "n": n,
        "overall_acc": round(overall_acc, 1),
        "mortgage_acc": round(mort_acc, 1),
        "general_acc":  round(gen_acc,  1),
        "hc_errors": hc,
        "lc_errors": lc,
        "hc_300": round(hc * scale, 1),
        "lc_300": round(lc * scale, 1),
        "wtd_300": round(wtd_300, 1),
        "cost_reduction_pct": cost_red,
        "parse_failures": sum(1 for r in results if not r["parse_ok"]),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def print_metrics(label: str, m: dict):
    print(f"\n{'='*60}")
    print(f"RESULTS — {label}")
    print(f"{'='*60}")
    print(f"  Overall accuracy:   {m['overall_acc']}%")
    print(f"  Mortgage accuracy:  {m['mortgage_acc']}%   (HIGH-COST class — expect trap)")
    print(f"  General accuracy:   {m['general_acc']}%   (LOW-COST class)")
    print(f"  HC errors/300:      {m['hc_300']}")
    print(f"  LC errors/300:      {m['lc_300']}")
    print(f"  Wtd cost/300:       {m['wtd_300']}")
    if m["cost_reduction_pct"] is not None:
        print(f"  Cost reduction:     {m['cost_reduction_pct']:+.1f}%  (vs text-only baseline)")
    print(f"  Parse failures:     {m['parse_failures']}/{m['n']}")


def save_csv(results: list[dict], path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "true_label", "product", "decision", "reasoning", "parse_ok", "narrative"
        ])
        writer.writeheader()
        writer.writerows(results)


def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=key)

    # Load data
    complaints = load_cfpb_sample(ZIP_PATH, N_SAMPLE)
    n_mort = sum(1 for r in complaints if r["true_label"] == HIGH_COST_LABEL)
    n_gen  = len(complaints) - n_mort
    print(f"\nSample: {len(complaints)} total  Mortgage={n_mort}  General={n_gen}")

    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Baseline 1: Text-only ────────────────────────────────────────────────
    text_results = run_baseline(complaints, client,
                                TEXT_ONLY_SYSTEM, TEXT_ONLY_USER,
                                "GPT-4o Text-Only")
    text_m = compute_metrics(text_results, baseline_wtd=None)
    print_metrics("GPT-4o Text-Only", text_m)

    # Check accuracy trap
    print(f"\n  ACCURACY TRAP CHECK:")
    if text_m["mortgage_acc"] < 50:
        print(f"  ✓ REPRODUCED — Mortgage (high-cost) recall={text_m['mortgage_acc']}% "
              f"(<50% — model defaults to General majority class)")
    else:
        print(f"  ✗ NOT reproduced — Mortgage accuracy={text_m['mortgage_acc']}%")

    save_csv(text_results, out_dir / "cfpb_text_only_results.csv")

    # ── Baseline 2: Cost-prompt ──────────────────────────────────────────────
    cost_results = run_baseline(complaints, client,
                                COST_PROMPT_SYSTEM, COST_PROMPT_USER,
                                "GPT-4o Cost-Prompt")
    cost_m = compute_metrics(cost_results, baseline_wtd=text_m["wtd_300"])
    print_metrics("GPT-4o Cost-Prompt", cost_m)

    # Check cost trap
    print(f"\n  COST TRAP CHECK:")
    if cost_m["general_acc"] < 50:
        print(f"  ✓ REPRODUCED — General (low-cost) recall={cost_m['general_acc']}% "
              f"(<50% — cost prompt collapses low-cost class)")
    else:
        print(f"  ✗ NOT reproduced — General accuracy={cost_m['general_acc']}%")

    save_csv(cost_results, out_dir / "cfpb_cost_prompt_results.csv")

    # ── Summary ───────────────────────────────────────────────────────────────
    accuracy_trap = text_m["mortgage_acc"] < 50
    cost_trap     = cost_m["general_acc"] < 50
    both_traps    = accuracy_trap and cost_trap

    print(f"\n{'='*60}")
    print("PAPER-READY VERDICT")
    print(f"{'='*60}")
    print(f"  Accuracy trap (text-only, Mortgage recall):  "
          f"{'REPRODUCED ✓' if accuracy_trap else 'not reproduced ✗'}  "
          f"Mortgage={text_m['mortgage_acc']}%")
    print(f"  Cost trap (cost-prompt, General collapse):   "
          f"{'REPRODUCED ✓' if cost_trap else 'not reproduced ✗'}  "
          f"General={cost_m['general_acc']}%")

    if both_traps:
        print("\n  BOTH TRAPS REPRODUCED — paper-ready second domain confirmed.")

    print(f"\nPaper Table / Generalizability row:")
    print(f"  Text-only:   Mortgage={text_m['mortgage_acc']}%, General={text_m['general_acc']}%")
    print(f"  Cost-prompt: Mortgage={cost_m['mortgage_acc']}%, General={cost_m['general_acc']}%")

    # Save JSON
    out = {
        "domain": "CFPB",
        "n_per_class": N_SAMPLE,
        "high_cost_class": HIGH_COST_LABEL,
        "low_cost_class":  LOW_COST_LABEL,
        "text_only":   text_m,
        "cost_prompt": cost_m,
        "accuracy_trap_reproduced": accuracy_trap,
        "cost_trap_reproduced":     cost_trap,
        "both_traps_reproduced":    both_traps,
    }
    json_path = out_dir / "cfpb_second_domain_results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {out_dir/'cfpb_text_only_results.csv'}")
    print(f"Saved: {out_dir/'cfpb_cost_prompt_results.csv'}")


if __name__ == "__main__":
    main()
