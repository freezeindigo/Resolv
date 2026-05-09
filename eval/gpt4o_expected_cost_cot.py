"""
GPT-4o Expected-Cost Chain-of-Thought Baseline.

Reviewer concern: "A more sophisticated version: give GPT-4o the full
decision-theoretic setup, the cost matrix, and ask it to compute
argmin_o sum_j P(j|x) * W[o,j] explicitly in chain-of-thought."

This is architecturally distinct from the existing cost-prompt baseline
(cost_aware_prompting.py), which only instructs GPT-4o to 'prefer Project
under uncertainty.' This baseline asks GPT-4o to:
  1. Estimate P(FM|x) and P(Project|x) explicitly
  2. Compute expected cost for each routing option
  3. Return the routing that minimises expected cost

If ARIA still beats this baseline, the architectural contribution is proven:
ARIA's hypothesis-scoped posterior estimation is better than GPT-4o's
self-estimated posteriors even when the decision rule is identical.

Output:
  eval/results/gpt4o_expected_cost_cot_results.json
  eval/results/gpt4o_expected_cost_cot_results.csv
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path

import psycopg2

try:
    from openai import OpenAI
    from openai import RateLimitError
except ImportError:
    raise RuntimeError("pip install openai")

SEED = 42
DB_NAME = "resolv"
AMBIGUOUS_CATEGORIES = [
    "Plumbing", "Leakage", "Seepage", "Carpentary", "Civil Work", "Mason", "Civil",
]
GPT4O_WEIGHTED_COST_BASELINE = 1029  # normalised per-300, matching paper Table 2

# ── System prompt: full decision-theoretic setup ─────────────────────────────

SYSTEM_PROMPT = """You are a complaint routing decision system for a large Indian residential developer.

Your task is to assign each complaint to the correct team using expected-cost minimisation.

## Teams
- FM (Facilities Management): on-site maintenance team. Handles routine repairs.
- Project: developer warranty and structural defects team. Handles structural failures, installation defects, and warranty claims.

## Cost Matrix
Routing a complaint that is truly a Project issue to FM:  W[FM, Project] = 10
  (Consequence: warranty voided, structural damage compounds, developer liability.)
Routing a complaint that is truly an FM issue to Project: W[Project, FM] = 1
  (Consequence: minor delay, slight disruption to warranty team.)

## Decision Rule
Given a complaint x, estimate:
  P(FM | x)       = your probability that the true ownership is FM
  P(Project | x)  = your probability that the true ownership is Project

Compute expected cost for each routing decision:
  E[cost | route=FM]      = P(Project | x) × W[FM, Project] = P(Project | x) × 10
  E[cost | route=Project] = P(FM | x)      × W[Project, FM] = P(FM | x)      × 1

Choose the routing with lower expected cost:
  decision = argmin { E[cost | FM], E[cost | Project] }

## Output format (JSON only — no other text)
{
  "p_fm": <float 0-1>,
  "p_project": <float 0-1>,
  "e_cost_fm": <float>,
  "e_cost_project": <float>,
  "decision": "FM" or "Project",
  "reasoning": "<one sentence: what signal drove your posterior estimate>"
}"""

USER_TEMPLATE = """Category: {category}
Complaint: {complaint}

Estimate posteriors, compute expected costs, and return the decision JSON."""


# ── Data loading ──────────────────────────────────────────────────────────────

def fetch_eval_complaints(n: int = 300) -> list[dict]:
    """Fetch the same stratified 300 complaints used in ARIA eval (seed=42)."""
    import random
    random.seed(SEED)

    cats_lower = {c.lower() for c in AMBIGUOUS_CATEGORIES}
    conn = psycopg2.connect(dbname=DB_NAME)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ticket_id, complaint_title, category, issue_type
        FROM complaints
        WHERE complaint_title IS NOT NULL
          AND issue_type IS NOT NULL
          AND category IS NOT NULL
        """
    )
    rows = []
    for tid, title, category, issue_type in cur.fetchall():
        if (category or "").strip().lower() not in cats_lower:
            continue
        own = (issue_type or "").strip().lower()
        if "project" in own:
            crm = "Project"
        elif "fm" in own or "facility" in own:
            crm = "FM"
        else:
            continue
        rows.append({"ticket_id": tid, "complaint_title": title,
                     "category": category, "crm_label": crm})
    cur.close()
    conn.close()

    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    total = len(rows)
    sampled = []
    remainders = []
    for cat, cat_rows in by_cat.items():
        proportion = len(cat_rows) / total
        exact = n * proportion
        take = int(exact)
        remainders.append((exact - take, cat))
        cat_copy = cat_rows[:]
        random.shuffle(cat_copy)
        sampled.extend(cat_copy[:take])
        by_cat[cat] = cat_copy[take:]

    remaining = n - len(sampled)
    for _, cat in sorted(remainders, reverse=True):
        if remaining == 0:
            break
        if by_cat[cat]:
            sampled.append(by_cat[cat].pop(0))
            remaining -= 1

    random.shuffle(sampled)
    return sampled[:n]


# ── Inference ─────────────────────────────────────────────────────────────────

def parse_response(raw: str) -> dict:
    """Parse JSON from GPT-4o response; fall back to regex extraction."""
    # Try to extract JSON block (model may wrap in markdown)
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            decision = parsed.get("decision", "FM")
            if isinstance(decision, str) and "project" in decision.lower():
                decision = "Project"
            else:
                decision = "FM"
            return {
                "decision": decision,
                "p_fm": float(parsed.get("p_fm", 0.5)),
                "p_project": float(parsed.get("p_project", 0.5)),
                "e_cost_fm": float(parsed.get("e_cost_fm", 5.0)),
                "e_cost_project": float(parsed.get("e_cost_project", 0.5)),
                "reasoning": str(parsed.get("reasoning", "")),
                "parse_ok": True,
            }
        except (json.JSONDecodeError, ValueError):
            pass
    # Fallback: look for "FM" or "Project" in raw text
    if "project" in raw.lower():
        return {"decision": "Project", "parse_ok": False, "p_fm": 0.5,
                "p_project": 0.5, "e_cost_fm": 5.0, "e_cost_project": 0.5,
                "reasoning": raw[:200]}
    return {"decision": "FM", "parse_ok": False, "p_fm": 0.5,
            "p_project": 0.5, "e_cost_fm": 5.0, "e_cost_project": 0.5,
            "reasoning": raw[:200]}


def run_baseline(complaints: list[dict], client: OpenAI) -> list[dict]:
    results = []
    parse_failures = 0
    for i, row in enumerate(complaints):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(complaints)}...  parse_failures={parse_failures}")
        user_msg = USER_TEMPLATE.format(
            category=row["category"],
            complaint=row["complaint_title"],
        )
        resp = None
        for attempt in range(10):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0,
                    max_tokens=200,
                    response_format={"type": "json_object"},
                )
                break
            except RateLimitError:
                wait = min(90, 3 + attempt * 5)
                print(f"    rate limit at row {i+1}, sleeping {wait}s...")
                time.sleep(wait)
        if resp is None:
            raise RuntimeError("OpenAI rate limit: exhausted retries")
        time.sleep(0.15)
        raw = (resp.choices[0].message.content or "").strip()
        parsed = parse_response(raw)
        if not parsed["parse_ok"]:
            parse_failures += 1
        results.append({
            "ticket_id": row["ticket_id"],
            "complaint_title": row["complaint_title"],
            "category": row["category"],
            "crm_label": row["crm_label"],
            "ecost_decision": parsed["decision"],
            "p_fm": parsed["p_fm"],
            "p_project": parsed["p_project"],
            "e_cost_fm": parsed["e_cost_fm"],
            "e_cost_project": parsed["e_cost_project"],
            "reasoning": parsed["reasoning"],
            "parse_ok": parsed["parse_ok"],
            "raw_response": raw,
            "agreed": parsed["decision"] == row["crm_label"],
        })
    print(f"  Parse failures: {parse_failures}/{len(complaints)}")
    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict], cost_weight: int = 10) -> dict:
    n = len(results)
    proj_rows = [r for r in results if r["crm_label"] == "Project"]
    fm_rows = [r for r in results if r["crm_label"] == "FM"]

    proj_right = sum(1 for r in proj_rows if r["ecost_decision"] == "Project")
    fm_right = sum(1 for r in fm_rows if r["ecost_decision"] == "FM")
    hc = sum(1 for r in proj_rows if r["ecost_decision"] == "FM")   # high-cost errors
    lc = sum(1 for r in fm_rows if r["ecost_decision"] == "Project")  # low-cost errors

    wtd = hc * cost_weight + lc
    scale = 300 / n
    wtd_300 = wtd * scale

    overall = sum(1 for r in results if r["agreed"]) / n
    proj_acc = proj_right / len(proj_rows) if proj_rows else 0
    fm_acc = fm_right / len(fm_rows) if fm_rows else 0
    cost_red = (GPT4O_WEIGHTED_COST_BASELINE - wtd_300) / GPT4O_WEIGHTED_COST_BASELINE * 100

    return {
        "overall_acc": round(overall * 100, 1),
        "proj_acc": round(proj_acc * 100, 1),
        "fm_acc": round(fm_acc * 100, 1),
        "hc_300": round(hc * scale, 1),
        "lc_300": round(lc * scale, 1),
        "wtd_300": round(wtd_300, 1),
        "cost_reduction_pct": round(cost_red, 1),
        "n": n,
        "parse_failures": sum(1 for r in results if not r["parse_ok"]),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=key)

    print("Fetching 300-complaint eval sample (seed=42)...")
    complaints = fetch_eval_complaints(300)
    print(f"Fetched {len(complaints)} complaints")
    proj_n = sum(1 for r in complaints if r["crm_label"] == "Project")
    fm_n = len(complaints) - proj_n
    print(f"  FM={fm_n}, Project={proj_n}")

    print("\nRunning GPT-4o with explicit expected-cost chain-of-thought...")
    results = run_baseline(complaints, client)

    metrics = compute_metrics(results)
    print("\nRESULTS — GPT-4o (expected-cost CoT):")
    print(f"  Overall accuracy:    {metrics['overall_acc']}%")
    print(f"  Project accuracy:    {metrics['proj_acc']}%   (paper: GPT-4o=13.2%, ARIA=35.8%)")
    print(f"  FM accuracy:         {metrics['fm_acc']}%   (paper: GPT-4o=91.6%, ARIA=75.5%)")
    print(f"  HC errors/300:       {metrics['hc_300']}   (paper: GPT-4o=107, ARIA=75)")
    print(f"  LC errors/300:       {metrics['lc_300']}   (paper: GPT-4o=15, ARIA=45)")
    print(f"  Wtd cost/300:        {metrics['wtd_300']}  (paper: GPT-4o=1029, ARIA=795)")
    print(f"  Cost reduction:      {metrics['cost_reduction_pct']:+.1f}%  (ARIA: -22.7%)")
    print(f"  Parse failures:      {metrics['parse_failures']}/300")

    print()
    aria_cost = 795
    if metrics["wtd_300"] > aria_cost:
        print("FINDING: ARIA achieves lower expected cost than GPT-4o (expected-cost CoT).")
        print("=> ARIA's hypothesis-scoped posterior estimation is superior to GPT-4o")
        print("   self-estimated posteriors even when the decision rule is identical.")
        print("=> The architectural contribution is proven.")
    elif abs(metrics["wtd_300"] - aria_cost) < 30:
        print("FINDING: GPT-4o (expected-cost CoT) is near-equivalent to ARIA.")
        print("=> The decision rule, not hypothesis scoping, drives ARIA's gain.")
        print("=> Revise the architectural contribution claim in the paper.")
    else:
        print("FINDING: GPT-4o (expected-cost CoT) outperforms ARIA.")
        print("=> ARIA's contribution is in the pre-supervision zero-shot regime only.")
        print("=> Revise claims about architectural superiority.")

    # Check FM viability
    if metrics["fm_acc"] < 70:
        print(f"\nNOTE: FM accuracy {metrics['fm_acc']}% < 70% viability threshold.")
        print("=> GPT-4o (expected-cost CoT) is operationally unviable (same failure mode")
        print("   as the existing cost-prompt baseline). ARIA remains the only viable zero-shot point.")

    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "gpt4o_expected_cost_cot_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ticket_id", "complaint_title", "category", "crm_label",
            "ecost_decision", "p_fm", "p_project", "e_cost_fm", "e_cost_project",
            "reasoning", "parse_ok", "agreed",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    json_path = out_dir / "gpt4o_expected_cost_cot_results.json"
    with open(json_path, "w") as f:
        json.dump({"metrics": metrics, "sample_size": len(complaints)}, f, indent=2)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")

    print("\nPaper Table 2 row:")
    print(f"GPT-4o (E[cost] CoT) & None & {metrics['fm_acc']}\\% & {metrics['proj_acc']}\\% & "
          f"{metrics['hc_300']:.0f} & {metrics['lc_300']:.0f} & "
          f"{metrics['wtd_300']:.0f} & {metrics['cost_reduction_pct']:+.1f}\\% \\\\")


if __name__ == "__main__":
    main()
