#!/usr/bin/env python3
"""
Sample complaint titles from PostgreSQL and list those that classify as domain "other".

Usage:
  PYTHONPATH=. python3 scripts/report_domain_other.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psycopg2

from src.nodes.domain_classifier import classify_domain


def main():
    try:
        conn = psycopg2.connect(dbname="resolv")
    except Exception as e:
        print("Could not connect to PostgreSQL (dbname=resolv):", e)
        return

    cur = conn.cursor()
    cur.execute(
        """
        SELECT complaint_title FROM complaints
        WHERE complaint_title IS NOT NULL AND TRIM(complaint_title) <> ''
        ORDER BY created_date DESC NULLS LAST
        LIMIT 5000
        """
    )
    rows = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()

    other_titles = []
    for title in rows:
        d = classify_domain(title)
        if d["domain"] == "other":
            other_titles.append((title, d.get("confidence", 0)))

    print(f"Scanned {len(rows)} titles; {len(other_titles)} classified as 'other' (rules).")
    print("Sample (up to 50):")
    for title, conf in other_titles[:50]:
        print(f"  [{conf:.2f}] {title[:120]}")


if __name__ == "__main__":
    main()
