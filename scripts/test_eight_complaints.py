#!/usr/bin/env python3
"""
Smoke-test eight diverse complaints through the full Resolv pipeline.
Requires ANTHROPIC_API_KEY for Tier 2/3 paths.

Usage:
  PYTHONPATH=. python3 scripts/test_eight_complaints.py
"""

from __future__ import annotations

import asyncio
import sys

from src.pipeline.resolv_graph import process_complaint

CASES = [
    (
        "Seepage from ceiling in master bedroom, same issue was reported 3 months ago",
        "Godrej Woods",
        "T6",
        "1803",
    ),
    (
        "Someone put color on my car seat in parking area",
        "Godrej Retreat",
        "T3",
        "504",
    ),
    (
        "Main door lock not closing properly since possession",
        "Godrej Se7en",
        "T1",
        "1204",
    ),
    (
        "Flush not working in master bathroom",
        "Godrej Woods",
        "T4",
        "901",
    ),
    (
        "Burning smell from kitchen switchboard",
        "Godrej Meridien",
        "T2",
        "1502",
    ),
    (
        "Lift stuck between 4th and 5th floor with people inside",
        "Godrej Golf Link (Crest)",
        "V1",
        "006",
    ),
    (
        "Wall cracks spreading in bedroom, visible since last monsoon",
        "Godrej Nurture-Pune",
        "T3",
        "1305",
    ),
    (
        "AC not cooling, was serviced last week but problem came back",
        "Godrej Woods",
        "T1",
        "2801",
    ),
]


async def run_one(i: int, title: str, site: str, tower: str, flat: str) -> None:
    ticket = f"DEMO-{i}"
    state = await process_complaint(
        ticket_id=ticket,
        complaint_title=title,
        site_name=site,
        tower=tower,
        flat=flat,
    )
    d = state.get("routing_decision")
    if not d:
        print(f"{i}. ERROR: no routing_decision — {state.get('error')}")
        return
    conf = d.confidence
    if isinstance(conf, float):
        conf = f"{conf:.2f}"
    reason = (d.reasoning or "")[:150].replace("\n", " ")
    print(
        f"{i}. tier={state.get('tier')} domain={state.get('domain')} "
        f"ownership={getattr(d, 'ownership', '?')} action={d.primary_action} confidence={conf}"
    )
    print(f"   reasoning: {reason}")


async def main() -> None:
    for i, (title, site, tower, flat) in enumerate(CASES, start=1):
        try:
            await run_one(i, title, site, tower, flat)
        except Exception as e:
            print(f"{i}. EXCEPTION: {e}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
