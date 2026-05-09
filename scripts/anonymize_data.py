#!/usr/bin/env python3
"""
Apply a legacy-site → anonymized-site mapping to PostgreSQL (complaints + flat_adjacency).

The mapping JSON is intentionally **not** stored in git (proprietary labels). Pass a path:

  python3 scripts/anonymize_data.py --map /path/to/mapping.json --db resolv

`mapping.json` format: { \"Legacy Site Name\": \"Riverside Heights\", ... }

If --map is omitted, the script exits successfully with a short message (no DB changes).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psycopg2


def run_mapping(map_path: Path, dbname: str) -> int:
    with open(map_path, encoding="utf-8") as f:
        mapping: dict[str, str] = json.load(f)

    try:
        conn = psycopg2.connect(dbname=dbname)
    except Exception as e:
        print(f"Could not connect to PostgreSQL (dbname={dbname}): {e}", file=sys.stderr)
        return 1

    conn.autocommit = False
    cur = conn.cursor()

    total_c = 0
    total_a = 0

    print("Updating complaints.site_name …")
    for old_name, new_name in sorted(mapping.items(), key=lambda x: x[0]):
        cur.execute(
            """
            UPDATE complaints
            SET site_name = %s
            WHERE LOWER(TRIM(site_name)) = LOWER(TRIM(%s))
            """,
            (new_name, old_name),
        )
        n = cur.rowcount
        if n:
            print(f"  complaints: updated {n} rows")
            total_c += n

    print("Updating flat_adjacency.site_name …")
    for old_name, new_name in sorted(mapping.items(), key=lambda x: x[0]):
        cur.execute(
            """
            UPDATE flat_adjacency
            SET site_name = %s
            WHERE LOWER(TRIM(site_name)) = LOWER(TRIM(%s))
            """,
            (new_name, old_name),
        )
        n = cur.rowcount
        if n:
            print(f"  flat_adjacency: updated {n} rows")
            total_a += n

    conn.commit()
    cur.close()
    conn.close()

    print(f"\nDone. complaints rows updated: {total_c}")
    print(f"Done. flat_adjacency rows updated: {total_a}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Anonymize site_name in PostgreSQL")
    parser.add_argument(
        "--map",
        type=Path,
        default=None,
        help="Path to legacy→anonymized JSON mapping (not committed; proprietary)",
    )
    parser.add_argument("--db", default="resolv", help="PostgreSQL database name")
    args = parser.parse_args()

    if args.map is None:
        print(
            "No --map provided. Skipping (mapping files with real site labels are not stored in git).",
            file=sys.stderr,
        )
        return 0

    if not args.map.is_file():
        print(f"Mapping file not found: {args.map}", file=sys.stderr)
        return 1

    return run_mapping(args.map, args.db)


if __name__ == "__main__":
    raise SystemExit(main())
