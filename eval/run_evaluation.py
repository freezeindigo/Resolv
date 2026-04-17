"""
Evaluation framework — run complaints through the pipeline and measure accuracy.

Usage:
    # Smoke test (10 complaints, ~$0.20 API cost)
    python3 eval/run_evaluation.py --sample 10 --tiers 1,2,3

    # Small eval (100 complaints, ~$2 API cost)
    python3 eval/run_evaluation.py --sample 100

    # Full eval (all 17K, ~$150+ API cost — budget separately)
    python3 eval/run_evaluation.py --full

IMPORTANT: Check your Anthropic balance before running with --sample > 50.
Estimated cost: Tier1=~$0, Tier2=~$0.02/complaint, Tier3=~$0.10/complaint
"""

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.resolv_graph import process_complaint
from src.nodes.domain_classifier import classify_domain
from src.nodes.complexity_assessor import assess_complexity


def estimate_cost(n_complaints: int) -> str:
    """Rough cost estimate before running."""
    tier1 = int(n_complaints * 0.20)
    tier2 = int(n_complaints * 0.64)
    tier3 = int(n_complaints * 0.16)
    cost = tier1 * 0 + tier2 * 0.02 + tier3 * 0.10
    return f"~${cost:.2f} (T1:{tier1} T2:{tier2} T3:{tier3})"


async def run_single(row: dict) -> dict:
    t_start = time.monotonic()
    try:
        result = await process_complaint(
            ticket_id=row["ticket_id"],
            complaint_title=row["complaint_title"],
            site_name=row["site_name"] or "unknown",
            tower=row["tower"] or "unknown",
            flat=row["flat"] or "unknown",
        )
        decision = result.get("routing_decision")
        return {
            "ticket_id": row["ticket_id"],
            "complaint_title": row["complaint_title"][:80],
            "raw_category": row["category"],
            "domain": result["domain"],
            "domain_confidence": result["domain_confidence"],
            "tier": result["tier"],
            "tier_reason": result["tier_reason"],
            "action": decision.primary_action if decision else "none",
            "priority": decision.priority if decision else "?",
            "reasoning": decision.reasoning[:200] if decision else "",
            "total_tokens": result["total_tokens"],
            "latency_ms": int((time.monotonic() - t_start) * 1000),
            "error": result.get("error"),
        }
    except Exception as e:
        return {
            "ticket_id": row["ticket_id"],
            "complaint_title": row["complaint_title"][:80],
            "raw_category": row["category"],
            "error": str(e),
            "latency_ms": int((time.monotonic() - t_start) * 1000),
            "total_tokens": 0,
        }


async def run_batch(rows: list, concurrency: int = 5) -> list:
    """Run complaints in batches to control API concurrency."""
    results = []
    sem = asyncio.Semaphore(concurrency)

    async def bounded(row):
        async with sem:
            return await run_single(row)

    tasks = [bounded(row) for row in rows]
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        if (i + 1) % 10 == 0:
            tokens_so_far = sum(r.get("total_tokens", 0) for r in results)
            print(f"  {i+1}/{len(rows)} | tokens so far: {tokens_so_far:,}", end="\r")

    return results


def generate_report(results: list, output_dir: Path):
    from collections import Counter

    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV
    csv_path = output_dir / f"eval_{ts}.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # Summary
    total = len(results)
    errors = sum(1 for r in results if r.get("error"))
    tier_dist = Counter(r.get("tier") for r in results if not r.get("error"))
    domain_dist = Counter(r.get("domain") for r in results if not r.get("error"))
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    avg_latency = sum(r.get("latency_ms", 0) for r in results) / max(total, 1)

    report = {
        "run_timestamp": ts,
        "total_complaints": total,
        "errors": errors,
        "tier_distribution": dict(tier_dist),
        "tier_percentages": {k: f"{v/total*100:.1f}%" for k, v in tier_dist.items()},
        "domain_distribution": dict(domain_dist),
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(total_tokens * 0.000003, 4),
        "avg_latency_ms": round(avg_latency),
        "csv_path": str(csv_path),
    }

    report_path = output_dir / f"report_{ts}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print("EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"Total: {total} | Errors: {errors}")
    print(f"Tier distribution: {report['tier_percentages']}")
    print(f"Total tokens: {total_tokens:,} | Est. cost: ${report['estimated_cost_usd']}")
    print(f"Avg latency: {avg_latency:.0f}ms")
    print(f"Saved: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=10,
                        help="Number of complaints to sample (default: 10)")
    parser.add_argument("--full", action="store_true",
                        help="Run all complaints (expensive — see cost estimate first)")
    parser.add_argument("--tiers", default="1,2,3",
                        help="Which tiers to include (default: 1,2,3)")
    parser.add_argument("--db", default="resolv")
    parser.add_argument("--out", default="eval/results")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    # Fetch complaints
    conn = psycopg2.connect(dbname=args.db)
    cur = conn.cursor()
    if args.full:
        cur.execute("""
            SELECT ticket_id, complaint_title, site_name, tower, flat, category
            FROM complaints WHERE complaint_title IS NOT NULL ORDER BY RANDOM()
        """)
    else:
        cur.execute("""
            SELECT ticket_id, complaint_title, site_name, tower, flat, category
            FROM complaints WHERE complaint_title IS NOT NULL ORDER BY RANDOM() LIMIT %s
        """, (args.sample,))
    rows = [
        {"ticket_id": r[0], "complaint_title": r[1], "site_name": r[2],
         "tower": r[3], "flat": r[4], "category": r[5]}
        for r in cur.fetchall()
    ]
    cur.close()
    conn.close()

    n = len(rows)
    print(f"\nComplaints to process: {n}")
    print(f"Estimated API cost:    {estimate_cost(n)}")
    print(f"Concurrency:           {args.concurrency}")
    print(f"\nProceed? [y/N] ", end="")
    if input().strip().lower() != "y":
        print("Aborted.")
        return

    print(f"\nRunning...")
    results = asyncio.run(run_batch(rows, concurrency=args.concurrency))
    generate_report(results, Path(args.out))


if __name__ == "__main__":
    main()
