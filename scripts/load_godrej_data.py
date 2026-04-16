"""
ETL: Load Godrej complaint data from Excel into PostgreSQL.

Usage:
    python3 scripts/load_godrej_data.py
    python3 scripts/load_godrej_data.py --file data/godrej_complaints.xlsx --db resolv
"""

import argparse
import json
import re
import sys
from datetime import datetime, date, time

import openpyxl
import psycopg2
import psycopg2.extras

SHEET_NAME = "Pan India"

# Column indices (0-based) in the Pan India sheet
COL = {
    "site_name":               0,
    "zone":                    1,
    "ticket_id":               2,
    "created_date":            3,
    "created_time":            4,
    "complaint_title":         5,
    "status":                  6,
    "category":                7,
    "issue_type":              8,   # "Issue Related To (FM/Project)"
    "sub_category":            9,
    "created_by":              10,
    "tower":                   11,
    "flat":                    12,
    "aging_text":              13,
    "aging_days":              14,
    "aging_hours":             15,
    "aging_minutes":           16,
    "priority":                17,
    "response_tat_minutes":    18,
    "resolution_tat_minutes":  19,
    "response_tat_breached":   20,
    "resolution_tat_breached": 21,
    "closed_date":             22,
    "closure_time":            23,
}

HEADERS = [
    "Site Name", "Zone", "Ticket ID", "Created Date", "Created Time",
    "Complaint Title", "Status", "Category", "Issue Related To (FM/Project)",
    "Sub Category", "Created By", "Tower", "Flat", "Aging",
    "Day", "Hours", "Minutes ", "Priority",
    "Response TAT (Min)", "Resolution TAT (Min)",
    "Response TAT Breached", "Resolution TAT Breached",
    "Closed on", "Closure Time",
]


def normalize_priority(val):
    if not val or not str(val).strip():
        return None
    v = str(val).strip().upper()
    if re.match(r'^P[1-5]$', v):
        return v
    return None


def normalize_bool(val):
    if val is None:
        return None
    v = str(val).strip().lower()
    if v in ("yes", "true", "1"):
        return True
    if v in ("no", "false", "0"):
        return False
    return None


def normalize_category(val):
    if not val:
        return None
    return str(val).strip()


def combine_datetime(d, t):
    """Combine a date/datetime and a time object into a single datetime."""
    if d is None:
        return None
    if isinstance(d, datetime):
        base = d
    elif isinstance(d, date):
        base = datetime(d.year, d.month, d.day)
    else:
        return None

    if isinstance(t, time):
        return base.replace(hour=t.hour, minute=t.minute, second=t.second)
    return base


def row_to_record(row):
    """Convert a raw Excel row tuple to a complaint dict."""
    def g(key):
        return row[COL[key]]

    ticket_id = g("ticket_id")
    if not ticket_id:
        return None
    ticket_id = str(ticket_id).strip()

    complaint_title = g("complaint_title")
    if not complaint_title:
        return None
    complaint_title = str(complaint_title).strip()

    created_dt = combine_datetime(g("created_date"), g("created_time"))
    closed_dt = combine_datetime(g("closed_date"), g("closure_time"))

    aging_days = g("aging_days")
    try:
        aging_days = int(aging_days) if aging_days is not None else None
    except (ValueError, TypeError):
        aging_days = None

    response_tat = g("response_tat_minutes")
    try:
        response_tat = int(response_tat) if response_tat is not None else None
    except (ValueError, TypeError):
        response_tat = None

    resolution_tat = g("resolution_tat_minutes")
    try:
        resolution_tat = int(resolution_tat) if resolution_tat is not None else None
    except (ValueError, TypeError):
        resolution_tat = None

    # Build raw_data from all columns
    raw = {}
    for key, idx in COL.items():
        val = row[idx]
        if isinstance(val, (datetime, date)):
            raw[key] = val.isoformat()
        elif isinstance(val, time):
            raw[key] = val.isoformat()
        else:
            raw[key] = val

    return {
        "ticket_id":               ticket_id,
        "site_name":               str(g("site_name")).strip() if g("site_name") else None,
        "zone":                    str(g("zone")).strip() if g("zone") else None,
        "created_date":            created_dt,
        "complaint_title":         complaint_title,
        "status":                  str(g("status")).strip() if g("status") else None,
        "category":                normalize_category(g("category")),
        "sub_category":            normalize_category(g("sub_category")),
        "issue_type":              str(g("issue_type")).strip() if g("issue_type") else None,
        "created_by":              str(g("created_by")).strip() if g("created_by") else None,
        "tower":                   str(g("tower")).strip() if g("tower") else None,
        "flat":                    str(g("flat")).strip() if g("flat") else None,
        "aging_days":              aging_days,
        "priority":                normalize_priority(g("priority")),
        "response_tat_minutes":    response_tat,
        "resolution_tat_minutes":  resolution_tat,
        "response_tat_breached":   normalize_bool(g("response_tat_breached")),
        "resolution_tat_breached": normalize_bool(g("resolution_tat_breached")),
        "closed_date":             closed_dt,
        "raw_data":                json.dumps(raw, default=str),
    }


INSERT_SQL = """
INSERT INTO complaints (
    ticket_id, site_name, zone, created_date, complaint_title,
    status, category, sub_category, issue_type, created_by,
    tower, flat, aging_days, priority,
    response_tat_minutes, resolution_tat_minutes,
    response_tat_breached, resolution_tat_breached,
    closed_date, raw_data
) VALUES (
    %(ticket_id)s, %(site_name)s, %(zone)s, %(created_date)s, %(complaint_title)s,
    %(status)s, %(category)s, %(sub_category)s, %(issue_type)s, %(created_by)s,
    %(tower)s, %(flat)s, %(aging_days)s, %(priority)s,
    %(response_tat_minutes)s, %(resolution_tat_minutes)s,
    %(response_tat_breached)s, %(resolution_tat_breached)s,
    %(closed_date)s, %(raw_data)s
)
ON CONFLICT (ticket_id) DO NOTHING
"""


def load(xlsx_path: str, dbname: str):
    print(f"Opening {xlsx_path} ...")
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb[SHEET_NAME]

    rows = list(ws.iter_rows(min_row=2, values_only=True))
    wb.close()
    print(f"  {len(rows)} data rows read from sheet '{SHEET_NAME}'")

    records = []
    skipped = 0
    for row in rows:
        rec = row_to_record(row)
        if rec is None:
            skipped += 1
        else:
            records.append(rec)

    print(f"  {len(records)} valid records, {skipped} skipped (no ticket_id or title)")

    conn = psycopg2.connect(dbname=dbname)
    conn.autocommit = False
    cur = conn.cursor()

    batch_size = 500
    inserted = 0
    conflicts = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        for rec in batch:
            cur.execute(INSERT_SQL, rec)
            if cur.rowcount == 1:
                inserted += 1
            else:
                conflicts += 1

        conn.commit()
        print(f"  Progress: {min(i + batch_size, len(records))}/{len(records)}", end="\r")

    cur.close()
    conn.close()

    print(f"\nDone. Inserted: {inserted}, Skipped (conflict): {conflicts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/godrej_complaints.xlsx")
    parser.add_argument("--db",   default="resolv")
    args = parser.parse_args()
    load(args.file, args.db)


if __name__ == "__main__":
    main()
