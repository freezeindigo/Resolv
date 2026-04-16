"""
Build flat_adjacency table from complaint data.

Flat numbering conventions found in Godrej data:
  1. Pure numeric:       "1001" → floor=10, unit=01  |  "204" → floor=2, unit=04
  2. Wing-prefixed:      "A204" → wing=A, floor=2, unit=04  |  "C501" → wing=C, floor=5, unit=01
  3. Tower-dash:         "T1-1201" → floor=12, unit=01
  4. Non-residential:    "Facilities", "clubhouse", etc. → skipped

Adjacency definition:
  - above_flat  : same tower, same unit suffix, floor+1
  - below_flat  : same tower, same unit suffix, floor-1
  - lateral_flats: same tower, same floor, all other units

Usage:
    python3 scripts/build_adjacency.py
    python3 scripts/build_adjacency.py --db resolv
"""

import argparse
import re
from collections import defaultdict

import psycopg2
import psycopg2.extras


# Minimum floor number we treat as residential (skip floor 0 edge cases)
MIN_FLOOR = 1
MAX_FLOOR = 60


def parse_flat(flat_str: str):
    """
    Returns (floor: int, unit: str, canonical: str) or None if not parseable.

    Flat numbering conventions:
      "1001"   → floor=10, unit="01"
      "204"    → floor=2,  unit="04"
      "A204"   → floor=2,  unit="04"
      "C-1002" → floor=10, unit="02"
      "T1-1201"→ floor=12, unit="01"

    Strategy: extract all digit runs, take the LAST one that is 3-4 digits.
    This correctly handles alphanumeric prefixes like "T1-", "A", "C-".
    """
    s = flat_str.strip()

    # Skip obvious non-residential values
    non_residential = {
        "common", "common area", "facilities", "clubhouse",
        "office", "reception", "lobby", "basement", "terrace",
        "parking", "amenities", "guard", "security", "gym", "pool",
        "offices",
    }
    if s.lower() in non_residential or not any(c.isdigit() for c in s):
        return None

    # Find all digit sequences; take the rightmost one that is 3-4 digits
    digit_runs = re.findall(r'\d+', s)
    num = None
    for run in reversed(digit_runs):
        if len(run) in (3, 4):
            num = run
            break

    if num is None:
        return None

    if len(num) == 3:
        floor = int(num[0])
        unit = num[1:]
    else:  # 4 digits
        floor = int(num[:2])
        unit = num[2:]

    if MIN_FLOOR <= floor <= MAX_FLOOR:
        canonical = f"{floor:02d}{unit}"
        return floor, unit, canonical

    return None


def build_adjacency(dbname: str):
    conn = psycopg2.connect(dbname=dbname)
    cur = conn.cursor()

    # Fetch all distinct (site_name, tower, flat) combinations
    cur.execute("""
        SELECT DISTINCT site_name, tower, flat
        FROM complaints
        WHERE site_name IS NOT NULL
          AND tower IS NOT NULL
          AND flat IS NOT NULL
        ORDER BY site_name, tower, flat
    """)
    rows = cur.fetchall()
    print(f"Distinct (site, tower, flat) combinations: {len(rows)}")

    # Group by (site_name, tower) → list of (flat, floor, unit)
    # Key: (site_name, tower)  Value: list of (flat_str, floor, unit)
    tower_flats = defaultdict(list)
    skipped = 0
    for site, tower, flat in rows:
        parsed = parse_flat(flat)
        if parsed is None:
            skipped += 1
            continue
        floor, unit, canonical = parsed
        tower_flats[(site, tower)].append((flat, floor, unit))

    print(f"  Parseable flats: {sum(len(v) for v in tower_flats.values())}, skipped: {skipped}")

    # Build adjacency records
    records = []
    for (site, tower), flat_list in tower_flats.items():
        # Index by floor and by unit for quick lookup
        by_floor = defaultdict(list)   # floor → [flat_str, ...]
        by_unit = defaultdict(list)    # unit  → [(floor, flat_str), ...]

        for flat_str, floor, unit in flat_list:
            by_floor[floor].append(flat_str)
            by_unit[unit].append((floor, flat_str))

        for flat_str, floor, unit in flat_list:
            # above: same unit, floor+1
            above = None
            for f, fs in by_unit[unit]:
                if f == floor + 1:
                    above = fs
                    break

            # below: same unit, floor-1
            below = None
            for f, fs in by_unit[unit]:
                if f == floor - 1:
                    below = fs
                    break

            # lateral: same floor, different flat
            lateral = [fs for fs in by_floor[floor] if fs != flat_str]

            records.append({
                "site_name":     site,
                "tower":         tower,
                "flat":          flat_str,
                "above_flat":    above,
                "below_flat":    below,
                "lateral_flats": lateral if lateral else None,
            })

    print(f"  Adjacency records to insert: {len(records)}")

    # Truncate and reload (idempotent)
    cur.execute("TRUNCATE TABLE flat_adjacency")

    psycopg2.extras.execute_batch(
        cur,
        """
        INSERT INTO flat_adjacency (site_name, tower, flat, above_flat, below_flat, lateral_flats)
        VALUES (%(site_name)s, %(tower)s, %(flat)s, %(above_flat)s, %(below_flat)s, %(lateral_flats)s)
        """,
        records,
        page_size=500,
    )

    conn.commit()
    cur.close()
    conn.close()
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="resolv")
    args = parser.parse_args()
    build_adjacency(args.db)


if __name__ == "__main__":
    main()
