#!/usr/bin/env python3
"""
Smoke report for eight diverse complaints: rules snapshot + optional full pipeline.

Rules snapshot (default): classify + assess + Tier 1 default action + ownership.

Full LLM pipeline (optional): set ANTHROPIC_API_KEY and RESOLV_RUN_FULL_PIPELINE=1

Usage:
  PYTHONPATH=. python3 scripts/report_eight_complaints.py
  RESOLV_RUN_FULL_PIPELINE=1 PYTHONPATH=. python3 scripts/report_eight_complaints.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = [
    (
        "Seepage from ceiling in master bedroom, same issue was reported 3 months ago",
        "Godrej Woods",
        "T6",
        "1803",
    ),
    ("Someone put color on my car seat in parking area", "Godrej Retreat", "T3", "504"),
    ("Main door lock not closing properly since possession", "Godrej Se7en", "T1", "1204"),
    ("Flush not working in master bathroom", "Godrej Woods", "T4", "901"),
    ("Burning smell from kitchen switchboard", "Godrej Meridien", "T2", "1502"),
    ("Lift stuck between 4th and 5th floor with people inside", "Godrej Golf Link (Crest)", "V1", "006"),
    ("Wall cracks spreading in bedroom, visible since last monsoon", "Godrej Nurture-Pune", "T3", "1305"),
    ("AC not cooling, was serviced last week but problem came back", "Godrej Woods", "T1", "2801"),
]


def rules_snapshot():
    from src.config.routing_actions import get_tier1_rule_tuple, normalize_primary_action
    from src.nodes.complexity_assessor import assess_complexity
    from src.nodes.domain_classifier import classify_domain
    from src.nodes.ownership import infer_ownership

    print(
        "Full LLM pipeline:",
        "will run after snapshot (RESOLV_RUN_FULL_PIPELINE=1 and ANTHROPIC_API_KEY)"
        if os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("RESOLV_RUN_FULL_PIPELINE")
        else "skipped (export RESOLV_RUN_FULL_PIPELINE=1 to enable)",
    )
    print()

    for i, (title, site, tower, flat) in enumerate(CASES, 1):
        cd = classify_domain(title)
        ev = assess_complexity(title, cd["domain"], cd["confidence"], cd["method"])
        tier = ev["tier"]
        raw_a, *_ = get_tier1_rule_tuple(cd["domain"], title)
        t1_action = normalize_primary_action(raw_a)
        own = infer_ownership(title, domain=cd["domain"], tier=tier, hypothesis_results=None)

        reasoning_stub = (
            f"Unambiguous {cd['domain']} — rule path (snapshot)."
            if tier == 1
            else (ev.get("reason", "") or "")[:150]
        )

        print(f"=== {i} ===")
        print(f"title: {title[:70]}...")
        print(
            f"tier={tier} domain={cd['domain']} domain_conf={cd['confidence']} "
            f"ownership={own} tier1_default_action={t1_action} confidence=n/a(rules)"
        )
        print(f"reasoning[:150]={reasoning_stub[:150]}")
        print()


async def run_pipeline():
    import asyncio

    from src.pipeline.resolv_graph import process_complaint

    for i, (title, site, tower, flat) in enumerate(CASES, 1):
        print(f"--- PIPELINE {i} ---")
        try:
            r = await process_complaint(
                ticket_id=f"REPORT-{i}",
                complaint_title=title,
                site_name=site,
                tower=tower,
                flat=flat,
            )
            dec = r.get("routing_decision")
            if dec:
                print(
                    f"tier={r['tier']} domain={r['domain']} ownership={dec.ownership} "
                    f"action={dec.primary_action} confidence={dec.confidence}"
                )
                print(f"reasoning[:150]={(dec.reasoning or '')[:150]}")
            else:
                print("no routing_decision", r.get("error"))
        except Exception as e:
            print("error:", e)
        print()


def main():
    rules_snapshot()
    if os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("RESOLV_RUN_FULL_PIPELINE"):
        import asyncio

        print("=== FULL PIPELINE (LLM) ===\n")
        asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
