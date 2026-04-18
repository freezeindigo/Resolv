"""
Context Assembler — Pipeline Node (deterministic, no LLM)

Runs three parallel async DB queries to retrieve all context
needed for hypothesis agents before any LLM call is made.

Returns a ContextPackage with:
  - flat_history:     last 365 days of complaints from this flat
  - adjacent_history: last 90 days from above/below/lateral flats
  - building_pattern: complaint category counts by floor in this tower, last 90 days

Usage (async):
    from src.nodes.context_assembler import assemble_context
    ctx = await assemble_context(site_name, tower, flat)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import asyncpg

DB_DSN = "postgresql://localhost/resolv"
FLAT_HISTORY_DAYS = 365
ADJACENT_HISTORY_DAYS = 90
BUILDING_PATTERN_DAYS = 90
FLAT_HISTORY_LIMIT = 20
ADJACENT_HISTORY_LIMIT = 15

# Indicative building ages for hypothesis trigger context (years, rough estimates)
_SITE_BUILDING_AGE_ESTIMATE: dict[str, float] = {
    "godrej woods": 12.0,
    "godrej meridien": 10.0,
    "godrej nurture-pune": 7.0,
    "godrej se7en": 8.0,
    "godrej retreat": 11.0,
    "godrej golf link (crest)": 10.0,
    "godrej golf link crest": 10.0,
    "godrej urban park": 9.0,
    "godrej rks": 6.0,
}


def _estimate_building_age_years(site_name: str) -> Optional[float]:
    key = site_name.strip().lower()
    return _SITE_BUILDING_AGE_ESTIMATE.get(key)


@dataclass
class ComplaintSummary:
    ticket_id: str
    created_date: Optional[datetime]
    complaint_title: str
    category: Optional[str]
    status: Optional[str]
    priority: Optional[str]
    resolution_tat_minutes: Optional[int]
    closed_date: Optional[datetime]
    flat: Optional[str]
    tower: Optional[str]


@dataclass
class FloorPattern:
    floor: str
    category: str
    count: int


@dataclass
class ContextPackage:
    site_name: str
    tower: str
    flat: str
    flat_history: List[ComplaintSummary] = field(default_factory=list)
    adjacent_history: List[ComplaintSummary] = field(default_factory=list)
    building_pattern: List[FloorPattern] = field(default_factory=list)
    adjacency_info: dict = field(default_factory=dict)  # above/below/lateral flats
    retrieval_ms: int = 0
    # Rough estimate for trigger logic (years); None if unknown
    building_age_years: Optional[float] = None

    def to_prompt_context(self) -> str:
        """Serialise context into a compact string suitable for LLM prompt inclusion."""
        lines = []

        lines.append(f"=== CONTEXT FOR {self.site_name} / {self.tower} / Flat {self.flat} ===\n")

        # Flat history
        if self.flat_history:
            lines.append(f"FLAT HISTORY (last {FLAT_HISTORY_DAYS} days, {len(self.flat_history)} complaints):")
            for c in self.flat_history:
                date_str = c.created_date.strftime("%Y-%m-%d") if c.created_date else "unknown"
                lines.append(
                    f"  [{date_str}] {c.category or 'unknown'} | {c.status or '?'} | "
                    f"TAT {c.resolution_tat_minutes or '?'} min | {c.complaint_title[:80]}"
                )
        else:
            lines.append("FLAT HISTORY: none in last 365 days (first complaint or new resident)")

        lines.append("")

        # Adjacent history
        adj = self.adjacency_info
        above = adj.get("above_flat")
        below = adj.get("below_flat")
        lateral = adj.get("lateral_flats") or []
        lines.append(
            f"ADJACENT FLATS: above={above or 'unknown'}, "
            f"below={below or 'unknown'}, lateral={lateral}"
        )
        if self.adjacent_history:
            lines.append(f"ADJACENT COMPLAINTS (last {ADJACENT_HISTORY_DAYS} days, {len(self.adjacent_history)}):")
            for c in self.adjacent_history:
                date_str = c.created_date.strftime("%Y-%m-%d") if c.created_date else "unknown"
                lines.append(
                    f"  [{date_str}] Flat {c.flat} | {c.category or 'unknown'} | "
                    f"{c.status or '?'} | {c.complaint_title[:80]}"
                )
        else:
            lines.append("ADJACENT COMPLAINTS: none in last 90 days")

        lines.append("")

        # Building pattern
        if self.building_pattern:
            lines.append(f"BUILDING PATTERN — {self.tower} (last {BUILDING_PATTERN_DAYS} days by floor):")
            for fp in self.building_pattern[:20]:
                lines.append(f"  Floor {fp.floor:<4} | {fp.category:<20} | {fp.count} complaints")
        else:
            lines.append("BUILDING PATTERN: no data")

        return "\n".join(lines)


async def _get_flat_history(conn: asyncpg.Connection, site_name: str, flat: str) -> List[ComplaintSummary]:
    since = datetime.now() - timedelta(days=FLAT_HISTORY_DAYS)
    rows = await conn.fetch("""
        SELECT ticket_id, created_date, complaint_title, category,
               status, priority, resolution_tat_minutes, closed_date, flat, tower
        FROM complaints
        WHERE site_name = $1
          AND flat = $2
          AND created_date > $3
        ORDER BY created_date DESC
        LIMIT $4
    """, site_name, flat, since, FLAT_HISTORY_LIMIT)
    return [ComplaintSummary(**dict(r)) for r in rows]


async def _get_adjacent_history(
    conn: asyncpg.Connection,
    site_name: str,
    tower: str,
    flat: str,
) -> tuple:
    """Returns (adjacency_info dict, list of ComplaintSummary)."""
    # Get adjacency info
    adj_row = await conn.fetchrow("""
        SELECT above_flat, below_flat, lateral_flats
        FROM flat_adjacency
        WHERE site_name = $1 AND tower = $2 AND flat = $3
    """, site_name, tower, flat)

    if not adj_row:
        return {}, []

    adjacency_info = {
        "above_flat": adj_row["above_flat"],
        "below_flat": adj_row["below_flat"],
        "lateral_flats": list(adj_row["lateral_flats"]) if adj_row["lateral_flats"] else [],
    }

    # Collect all adjacent flat ids
    adjacent_flats = []
    if adj_row["above_flat"]:
        adjacent_flats.append(adj_row["above_flat"])
    if adj_row["below_flat"]:
        adjacent_flats.append(adj_row["below_flat"])
    if adj_row["lateral_flats"]:
        adjacent_flats.extend(adj_row["lateral_flats"])

    if not adjacent_flats:
        return adjacency_info, []

    since = datetime.now() - timedelta(days=ADJACENT_HISTORY_DAYS)
    rows = await conn.fetch("""
        SELECT ticket_id, created_date, complaint_title, category,
               status, priority, resolution_tat_minutes, closed_date, flat, tower
        FROM complaints
        WHERE site_name = $1
          AND flat = ANY($2::varchar[])
          AND created_date > $3
        ORDER BY created_date DESC
        LIMIT $4
    """, site_name, adjacent_flats, since, ADJACENT_HISTORY_LIMIT)

    return adjacency_info, [ComplaintSummary(**dict(r)) for r in rows]


async def _get_building_pattern(
    conn: asyncpg.Connection,
    site_name: str,
    tower: str,
) -> List[FloorPattern]:
    since = datetime.now() - timedelta(days=BUILDING_PATTERN_DAYS)
    rows = await conn.fetch("""
        SELECT
            SUBSTRING(flat FROM '^\d+') AS floor,
            category,
            COUNT(*) AS count
        FROM complaints
        WHERE site_name = $1
          AND tower = $2
          AND created_date > $3
          AND flat ~ '^\d'
          AND category IS NOT NULL
        GROUP BY floor, category
        ORDER BY count DESC
        LIMIT 40
    """, site_name, tower, since)
    return [FloorPattern(floor=r["floor"] or "?", category=r["category"], count=r["count"]) for r in rows]


async def assemble_context(site_name: str, tower: str, flat: str) -> ContextPackage:
    """
    Assemble full complaint context for a given flat.
    Runs three DB queries in parallel using a connection pool.
    """
    t_start = asyncio.get_event_loop().time()

    # Pool of 3 connections — one per parallel query
    pool = await asyncpg.create_pool(DB_DSN, min_size=3, max_size=3)
    try:
        async with pool.acquire() as c1, pool.acquire() as c2, pool.acquire() as c3:
            flat_hist, (adj_info, adj_hist), bldg_pattern = await asyncio.gather(
                _get_flat_history(c1, site_name, flat),
                _get_adjacent_history(c2, site_name, tower, flat),
                _get_building_pattern(c3, site_name, tower),
            )
    finally:
        await pool.close()

    elapsed_ms = int((asyncio.get_event_loop().time() - t_start) * 1000)

    return ContextPackage(
        site_name=site_name,
        tower=tower,
        flat=flat,
        flat_history=flat_hist,
        adjacent_history=adj_hist,
        building_pattern=bldg_pattern,
        adjacency_info=adj_info,
        retrieval_ms=elapsed_ms,
        building_age_years=_estimate_building_age_years(site_name),
    )
