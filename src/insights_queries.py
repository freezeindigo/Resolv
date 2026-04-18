"""
PostgreSQL-backed analytics for /insights/* endpoints.
Uses human-assigned `category` as a stand-in for "domain" where ML domain is not persisted.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import psycopg2

DB = "resolv"


def _conn():
    return psycopg2.connect(dbname=DB)


def get_summary() -> Dict[str, Any]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM complaints")
    total = cur.fetchone()[0]

    cur.execute(
        """
        SELECT COUNT(*) FROM complaints
        WHERE LOWER(COALESCE(TRIM(status), '')) NOT LIKE '%clos%'
           OR (closed_date IS NULL AND (status IS NULL OR status = ''))
        """
    )
    open_cnt = cur.fetchone()[0]

    cur.execute(
        """
        SELECT AVG(aging_days), percentile_cont(0.5) WITHIN GROUP (ORDER BY aging_days)
        FROM complaints
        WHERE aging_days IS NOT NULL
          AND LOWER(COALESCE(TRIM(status), '')) NOT LIKE '%clos%'
        """
    )
    row = cur.fetchone()
    avg_age = float(row[0]) if row and row[0] is not None else None
    med_age = float(row[1]) if row and row[1] is not None else None

    cur.execute(
        """
        SELECT AVG(resolution_tat_minutes) FROM complaints
        WHERE resolution_tat_minutes IS NOT NULL AND resolution_tat_minutes > 0
        """
    )
    r = cur.fetchone()
    avg_res_min = float(r[0]) if r and r[0] is not None else None

    cur.execute("SELECT COUNT(DISTINCT site_name) FROM complaints WHERE site_name IS NOT NULL")
    site_n = cur.fetchone()[0]

    cur.close()
    conn.close()

    # Tier projection: computed separately in main via classifier sample
    return {
        "total_complaints": total,
        "open_complaints": open_cnt,
        "avg_aging_days": round(avg_age, 1) if avg_age is not None else None,
        "median_aging_days": round(med_age, 1) if med_age is not None else None,
        "avg_resolution_minutes": round(avg_res_min, 0) if avg_res_min is not None else None,
        "distinct_sites": site_n,
    }


def get_hotspots(limit: int = 20) -> List[Dict[str, Any]]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT site_name, tower, flat, COUNT(*) AS cnt,
               COUNT(DISTINCT NULLIF(TRIM(category), '')) AS cat_n
        FROM complaints
        WHERE site_name IS NOT NULL
        GROUP BY site_name, tower, flat
        ORDER BY cnt DESC
        LIMIT %s
        """,
        (limit,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    out = []
    for site_name, tower, flat, cnt, cat_n in rows:
        if cnt > 100:
            rec = "Comprehensive inspection needed"
        elif cnt > 30:
            rec = "Recurring issue investigation"
        else:
            rec = "Monitor — elevated ticket volume"

        out.append(
            {
                "site_name": site_name,
                "tower": tower or "",
                "flat": flat or "",
                "total_complaints": cnt,
                "unique_categories": int(cat_n or 0),
                "recommendation": rec,
                "highlight_severe": cnt > 50,  # UI: red row when >50
            }
        )
    return out


def get_domains_heatmap() -> Dict[str, Any]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT site_name, COALESCE(NULLIF(TRIM(category), ''), 'Unknown') AS cat, COUNT(*)::int
        FROM complaints
        WHERE site_name IS NOT NULL
        GROUP BY site_name, category
        ORDER BY site_name, cat
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    sites = sorted({r[0] for r in rows})
    cats = sorted({r[1] for r in rows})
    matrix: Dict[str, Dict[str, int]] = defaultdict(dict)
    for site, cat, n in rows:
        matrix[site][cat] = n
    return {"sites": sites, "categories": cats, "counts": {s: dict(matrix[s]) for s in sites}}


def get_recurrence() -> Dict[str, Any]:
    """Flats with a follow-up complaint in the same category within 90 days (pairwise)."""
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c1.category, COUNT(DISTINCT (c1.site_name, c1.tower, c1.flat))::int
        FROM complaints c1
        INNER JOIN complaints c2
          ON c1.site_name = c2.site_name
         AND COALESCE(c1.tower, '') = COALESCE(c2.tower, '')
         AND COALESCE(c1.flat, '') = COALESCE(c2.flat, '')
         AND c1.category IS NOT NULL
         AND c1.category = c2.category
         AND c1.id < c2.id
         AND c1.created_date IS NOT NULL
         AND c2.created_date IS NOT NULL
         AND c2.created_date > c1.created_date
         AND c2.created_date <= c1.created_date + interval '90 days'
        GROUP BY c1.category
        """
    )
    recurring_flats_by_cat = {r[0]: r[1] for r in cur.fetchall()}

    cur.execute(
        """
        SELECT category, COUNT(DISTINCT (site_name, tower, flat))::int
        FROM complaints
        WHERE category IS NOT NULL
        GROUP BY category
        """
    )
    flats_by_cat = {r[0]: r[1] for r in cur.fetchall()}
    cur.close()
    conn.close()

    rates = {}
    for cat, n_flat in flats_by_cat.items():
        rec = recurring_flats_by_cat.get(cat, 0)
        rates[cat] = round(100.0 * rec / n_flat, 1) if n_flat else 0.0

    return {
        "recurrence_rate_by_category_pct": rates,
        "recurring_flats_by_category": recurring_flats_by_cat,
        "distinct_flats_by_category": flats_by_cat,
    }


def get_aging_buckets() -> Dict[str, int]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
          CASE
            WHEN aging_days IS NULL THEN 'unknown'
            WHEN aging_days < 30 THEN '<30'
            WHEN aging_days < 60 THEN '30-60'
            WHEN aging_days < 120 THEN '60-120'
            WHEN aging_days <= 200 THEN '120-200'
            ELSE '>200'
          END AS bucket,
          COUNT(*)::int
        FROM complaints
        WHERE aging_days IS NOT NULL
          AND LOWER(COALESCE(TRIM(status), '')) NOT LIKE '%clos%'
        GROUP BY 1
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {r[0]: r[1] for r in rows}


def get_taxonomy_chaos(sample_limit: int = 8000) -> Dict[str, Any]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT complaint_title, COALESCE(NULLIF(TRIM(category), ''), 'Unknown')
        FROM complaints
        WHERE complaint_title IS NOT NULL
        LIMIT %s
        """,
        (sample_limit,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    word_to_cats: Dict[str, set] = defaultdict(set)
    token_re = re.compile(r"[a-zA-Z][a-zA-Z\-]{2,}")
    for title, cat in rows:
        for m in token_re.finditer((title or "").lower()):
            w = m.group(0)
            if len(w) < 4 and w not in ("ac", "mcb"):
                continue
            word_to_cats[w].add(cat)

    chaos = [{"keyword": w, "distinct_categories": len(cats), "categories": sorted(cats)[:12]}
             for w, cats in word_to_cats.items() if len(cats) >= 5]
    chaos.sort(key=lambda x: -x["distinct_categories"])
    return {"keywords": chaos[:40], "sample_size": len(rows)}


def tier_projection_sample(max_rows: int = 1200) -> Dict[str, int]:
    from src.nodes.domain_classifier import classify_domain
    from src.nodes.complexity_assessor import assess_complexity

    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT complaint_title FROM complaints
        WHERE complaint_title IS NOT NULL AND TRIM(complaint_title) <> ''
        ORDER BY RANDOM()
        LIMIT %s
        """,
        (max_rows,),
    )
    titles = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()

    dist = Counter()
    for title in titles:
        d = classify_domain(title)
        t = assess_complexity(
            title,
            d["domain"],
            domain_confidence=d["confidence"],
            domain_method=d["method"],
        )
        dist[t["tier"]] += 1
    return {f"T{k}": dist.get(k, 0) for k in (1, 2, 3)}
