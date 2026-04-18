#!/usr/bin/env python3
"""
Print complaint titles from PostgreSQL that the rule classifier maps to domain "other"
(sample up to N rows from DB — tune LIMIT in query).

Usage:
  PYTHONPATH=. python3 scripts/sample_other_domain_titles.py
"""

import psycopg2

from src.nodes.domain_classifier import classify_domain


def main() -> None:
    conn = psycopg2.connect(dbname="resolv")
    cur = conn.cursor()
    cur.execute(
        """
        SELECT complaint_title FROM complaints
        WHERE complaint_title IS NOT NULL AND TRIM(complaint_title) <> ''
        ORDER BY random()
        LIMIT 3000
        """
    )
    rows = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()

    other_titles = [t for t in rows if classify_domain(t)["domain"] == "other"]
    print(f"Sampled {len(rows)} titles; {len(other_titles)} classified as 'other' (showing up to 50):\n")
    for t in other_titles[:50]:
        print(t)


if __name__ == "__main__":
    main()
