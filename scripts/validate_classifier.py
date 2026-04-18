"""
Validate domain classifier and complexity assessor against PostgreSQL complaint samples.

Usage:
    python3 scripts/validate_classifier.py
    python3 scripts/validate_classifier.py --sample 1000 --show 30
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nodes.domain_classifier import classify_domain
from src.nodes.complexity_assessor import assess_complexity


def run(dbname: str, sample_size: int, show_examples: int):
    conn = psycopg2.connect(dbname=dbname)
    cur = conn.cursor()

    cur.execute("""
        SELECT ticket_id, complaint_title, category, priority
        FROM complaints
        WHERE complaint_title IS NOT NULL
        ORDER BY RANDOM()
        LIMIT %s
    """, (sample_size,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    print(f"\nSample size: {len(rows)} complaints\n")

    domain_counts = Counter()
    tier_counts = Counter()
    method_counts = Counter()
    results = []

    for ticket_id, title, raw_category, priority in rows:
        d_result = classify_domain(title)
        t_result = assess_complexity(
            title,
            d_result["domain"],
            domain_confidence=d_result["confidence"],
            domain_method=d_result["method"],
        )

        domain_counts[d_result["domain"]] += 1
        tier_counts[t_result["tier"]] += 1
        method_counts[d_result["method"]] += 1

        results.append({
            "ticket_id": ticket_id,
            "title": title,
            "raw_category": raw_category,
            "domain": d_result["domain"],
            "domain_conf": d_result["confidence"],
            "method": d_result["method"],
            "tier": t_result["tier"],
            "tier_reason": t_result["reason"],
        })

    # --- Distribution report ---
    total = len(results)

    print("=" * 70)
    print("DOMAIN DISTRIBUTION")
    print("=" * 70)
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(count / total * 40)
        print(f"  {domain:<20} {count:>5}  ({count/total*100:5.1f}%)  {bar}")

    print("\n" + "=" * 70)
    print("TIER DISTRIBUTION")
    print("=" * 70)
    targets = {1: 35, 2: 40, 3: 25}
    for tier in [1, 2, 3]:
        count = tier_counts[tier]
        pct = count / total * 100
        target = targets[tier]
        delta = pct - target
        flag = " ✓" if abs(delta) < 8 else f" ← target {target}%"
        print(f"  Tier {tier}: {count:>5}  ({pct:5.1f}%){flag}")

    print("\n" + "=" * 70)
    print("CLASSIFICATION METHOD")
    print("=" * 70)
    for method, count in method_counts.items():
        print(f"  {method:<12} {count:>5}  ({count/total*100:5.1f}%)")

    # --- Confidence distribution ---
    conf_buckets = {"high (≥0.7)": 0, "medium (0.5–0.7)": 0, "low (<0.5)": 0}
    for r in results:
        c = r["domain_conf"]
        if c >= 0.7:
            conf_buckets["high (≥0.7)"] += 1
        elif c >= 0.5:
            conf_buckets["medium (0.5–0.7)"] += 1
        else:
            conf_buckets["low (<0.5)"] += 1

    print("\n" + "=" * 70)
    print("CONFIDENCE DISTRIBUTION")
    print("=" * 70)
    for bucket, count in conf_buckets.items():
        print(f"  {bucket:<20} {count:>5}  ({count/total*100:5.1f}%)")

    # --- Sample examples ---
    print("\n" + "=" * 70)
    print(f"SAMPLE ASSIGNMENTS (showing {show_examples})")
    print("=" * 70)

    # Show mix: some from each tier
    tier1_ex = [r for r in results if r["tier"] == 1][:show_examples // 3]
    tier2_ex = [r for r in results if r["tier"] == 2][:show_examples // 3]
    tier3_ex = [r for r in results if r["tier"] == 3][:show_examples // 3]
    sample_ex = tier1_ex + tier2_ex + tier3_ex
    random.shuffle(sample_ex)

    for r in sample_ex:
        print(f"\n  [{r['ticket_id']}] T{r['tier']} | {r['domain']} ({r['domain_conf']:.2f}) | {r['method']}")
        print(f"  Title:      {r['title'][:90]}")
        print(f"  Raw cat:    {r['raw_category']}")
        print(f"  Tier why:   {r['tier_reason']}")

    # --- Mismatches: where our domain differs from raw category ---
    print("\n" + "=" * 70)
    print("DOMAIN VS RAW CATEGORY — NOTABLE MISMATCHES (sample 15)")
    print("=" * 70)

    # Build simple expected domain from raw category
    cat_to_domain = {
        "plumbing": "water_plumbing", "leakage": "water_plumbing",
        "water": "water_plumbing", "seepage": "structural_civil",
        "electrical": "electrical", "carpentary": "carpentry",
        "carpentry": "carpentry", "civil work": "structural_civil",
        "civil": "structural_civil", "mason": "structural_civil",
        "elevator": "lift_elevator", "ac repair": "hvac",
        "security": "safety_security", "hk": "pest_hygiene",
        "common area": "common_area", "amenities": "common_area",
    }

    mismatches = []
    for r in results:
        cat_norm = (r["raw_category"] or "").strip().lower().rstrip()
        expected = cat_to_domain.get(cat_norm)
        if expected and expected != r["domain"]:
            mismatches.append(r)

    random.shuffle(mismatches)
    for r in mismatches[:15]:
        print(f"\n  [{r['ticket_id']}] assigned={r['domain']} | raw_cat={r['raw_category']}")
        print(f"  Title: {r['title'][:90]}")
        print(f"  Reason (tier): {r['tier_reason']}")

    print(f"\n  Total mismatches: {len(mismatches)}/{total} ({len(mismatches)/total*100:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",      default="resolv")
    parser.add_argument("--sample",  type=int, default=500)
    parser.add_argument("--show",    type=int, default=21)
    args = parser.parse_args()
    run(args.db, args.sample, args.show)


if __name__ == "__main__":
    main()
