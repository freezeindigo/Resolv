"""
Resolv FastAPI — complaint intake and reasoning trace endpoints.
"""

import asyncio
from datetime import datetime
from typing import Optional

import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline.resolv_graph import process_complaint
from src.memory.pattern_state import get_active_clusters, ingest_complaint

app = FastAPI(title="Resolv.AI", version="0.1.0")


# ── Request / Response models ──────────────────────────────────────────────

class ComplaintRequest(BaseModel):
    complaint_title: str
    site_name: str
    tower: str
    flat: str
    ticket_id: Optional[str] = None
    priority_requested: Optional[str] = None


class RoutingResponse(BaseModel):
    ticket_id: str
    tier: int
    domain: str
    domain_confidence: float
    primary_action: str
    vendor_skill_level: str
    priority: str
    sla_hours: int
    secondary_action: Optional[str]
    reasoning: str
    confidence: str
    escalation_trigger: str
    total_tokens: int
    total_latency_ms: int


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/complaints", response_model=RoutingResponse)
async def submit_complaint(req: ComplaintRequest):
    """Submit a complaint and get back routing decision + reasoning trace."""
    ticket_id = req.ticket_id or f"RESOLV-{int(datetime.now().timestamp())}"

    result = await process_complaint(
        ticket_id=ticket_id,
        complaint_title=req.complaint_title,
        site_name=req.site_name,
        tower=req.tower,
        flat=req.flat,
        priority_requested=req.priority_requested,
    )

    decision = result.get("routing_decision")
    if not decision:
        raise HTTPException(status_code=500, detail="Routing decision not produced")

    # Ingest into pattern state for future cluster detection
    from src.nodes.domain_classifier import classify_domain
    from src.nodes.complexity_assessor import assess_complexity
    from src.memory.pattern_state import _floor_from_flat
    floor = _floor_from_flat(req.flat) or 0
    ingest_complaint(
        site_name=req.site_name,
        tower=req.tower,
        flat=req.flat,
        floor=floor,
        category=result.get("domain", "other"),
        ticket_id=ticket_id,
    )

    return RoutingResponse(
        ticket_id=ticket_id,
        tier=result["tier"],
        domain=result["domain"],
        domain_confidence=result["domain_confidence"],
        primary_action=decision.primary_action,
        vendor_skill_level=decision.vendor_skill_level,
        priority=decision.priority,
        sla_hours=decision.sla_hours,
        secondary_action=decision.secondary_action,
        reasoning=decision.reasoning,
        confidence=decision.confidence,
        escalation_trigger=decision.escalation_trigger,
        total_tokens=result["total_tokens"],
        total_latency_ms=result["total_latency_ms"],
    )


@app.get("/complaints/{ticket_id}")
async def get_complaint(ticket_id: str):
    """Get full audit trace for a processed complaint."""
    # TODO: fetch from audit table in Phase 2
    return {"ticket_id": ticket_id, "message": "Audit persistence coming in Phase 2"}


@app.get("/complaints/stats")
async def get_stats():
    """Processing stats and tier distribution."""
    conn = psycopg2.connect(dbname="resolv")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM complaints")
    total = cur.fetchone()[0]
    cur.execute("SELECT status, COUNT(*) FROM complaints GROUP BY status")
    statuses = dict(cur.fetchall())
    cur.execute("SELECT category, COUNT(*) FROM complaints GROUP BY category ORDER BY COUNT(*) DESC LIMIT 10")
    top_categories = [{"category": r[0], "count": r[1]} for r in cur.fetchall()]
    cur.close()
    conn.close()
    return {"total_complaints": total, "statuses": statuses, "top_categories": top_categories}


@app.get("/clusters/active")
async def get_active_clusters_endpoint(
    site_name: str,
    tower: str,
    domain: str = "water_plumbing",
):
    """Get currently active complaint clusters for a building."""
    signal = get_active_clusters(site_name, tower, domain)
    return {
        "site_name": site_name,
        "tower": tower,
        "domain": domain,
        "cluster_count": len(signal.active_clusters),
        "has_stack_pattern": signal.has_stack_pattern,
        "building_complaint_count": signal.building_complaint_count,
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "complaint_count": c.complaint_count,
                "spatial_pattern": c.spatial_pattern,
                "temporal_pattern": c.temporal_pattern,
                "dominant_category": c.dominant_category,
                "floors_affected": c.floors_affected,
                "confidence": c.confidence,
            }
            for c in signal.active_clusters
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "resolv"}
