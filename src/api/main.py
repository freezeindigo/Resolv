"""
Resolv FastAPI — complaint intake and reasoning trace endpoints.
"""

import csv
import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import psycopg2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.pipeline.resolv_graph import process_complaint
from src.memory.pattern_state import get_active_clusters, ingest_complaint
from src.nodes.domain_classifier import classify_domain
from src.nodes.complexity_assessor import assess_complexity
from src import insights_queries

app = FastAPI(title="Resolv.AI", version="0.1.0")
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")


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
    tier_reason: Optional[str] = None
    domain: str
    domain_confidence: float
    primary_action: str
    vendor_skill_level: str
    priority: str
    sla_hours: int
    secondary_action: Optional[str]
    reasoning: str
    confidence: Union[str, float, None] = None
    escalation_trigger: str
    total_tokens: int
    total_latency_ms: int
    hypothesis_triggers: Optional[Dict[str, Any]] = None
    ownership: str = "FM"
    judge_verdict: Optional[str] = None
    judge_reason: Optional[str] = None
    human_review_queued: bool = False
    original_primary_action: Optional[str] = None
    original_reasoning: Optional[str] = None
    rag_sources_used: List[Dict[str, Any]] = Field(default_factory=list)
    rag_enhanced: bool = False


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

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

    conf = result["routing_decision"].confidence
    if isinstance(conf, (int, float)):
        if conf > 0.8:
            conf = "high"
        elif conf > 0.5:
            conf = "medium"
        else:
            conf = "low"

    audit = result.get("audit_log") or {}
    jv = audit.get("judge_verdict") or {}
    verdict = jv.get("verdict")
    if verdict not in ("approve", "flag", "override"):
        verdict = None
    orig_snap = jv.get("original_decision") or {}
    hr = jv.get("human_review") or {}

    ctx = result.get("context")
    rag_sources: List[Dict[str, Any]] = []
    rag_enhanced = False
    if ctx is not None:
        rag_sources = list(getattr(ctx, "rag_sources_used", None) or [])
        rag_enhanced = bool(rag_sources)

    return RoutingResponse(
        ticket_id=ticket_id,
        tier=result["tier"],
        tier_reason=result.get("tier_reason"),
        domain=result["domain"],
        domain_confidence=result["domain_confidence"],
        primary_action=decision.primary_action,
        vendor_skill_level=decision.vendor_skill_level,
        priority=decision.priority,
        sla_hours=decision.sla_hours,
        secondary_action=decision.secondary_action,
        reasoning=decision.reasoning,
        confidence=conf,
        escalation_trigger=decision.escalation_trigger,
        total_tokens=result["total_tokens"],
        total_latency_ms=result["total_latency_ms"],
        hypothesis_triggers=audit.get("hypothesis_triggers"),
        ownership=getattr(decision, "ownership", None) or "FM",
        judge_verdict=verdict,
        judge_reason=jv.get("reason"),
        human_review_queued=bool(hr.get("queued")),
        original_primary_action=orig_snap.get("primary_action") if verdict == "override" else None,
        original_reasoning=orig_snap.get("reasoning") if verdict == "override" else None,
        rag_sources_used=rag_sources,
        rag_enhanced=rag_enhanced,
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


# ── Bulk analysis (rule-based, no LLM) ────────────────────────────────────


def _estimate_llm_cost_usd_tier(tier: int, n: int) -> float:
    """Rough marginal $/complaint by tier (same order of magnitude as eval notes)."""
    per = {1: 0.0, 2: 0.02, 3: 0.10}
    return round(n * per.get(tier, 0.02), 2)


@app.post("/bulk-analyze")
async def bulk_analyze(file: UploadFile = File(...)):
    """
    CSV columns: complaint_text (or complaint_title), site_name, tower, flat.
    Runs domain_classifier + complexity_assessor only (no LLM).
    """
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="Empty CSV")

    rows_out: List[Dict[str, Any]] = []
    tier_counts = {1: 0, 2: 0, 3: 0}
    domain_counts: Dict[str, int] = {}

    for row in reader:
        title = (row.get("complaint_text") or row.get("complaint_title") or "").strip()
        site = (row.get("site_name") or "").strip()
        tower = (row.get("tower") or "").strip()
        flat = (row.get("flat") or "").strip()
        if not title:
            continue

        d = classify_domain(title)
        t = assess_complexity(
            title,
            d["domain"],
            domain_confidence=d["confidence"],
            domain_method=d["method"],
        )
        tier = t["tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        dom = d["domain"]
        domain_counts[dom] = domain_counts.get(dom, 0) + 1

        rows_out.append(
            {
                "complaint_text": title[:500],
                "site_name": site,
                "tower": tower,
                "flat": flat,
                "domain": dom,
                "domain_confidence": d["confidence"],
                "tier": tier,
                "tier_reason": t["reason"],
            }
        )

    n = len(rows_out)
    est = _estimate_llm_cost_usd_tier(1, tier_counts[1]) + _estimate_llm_cost_usd_tier(
        2, tier_counts[2]
    ) + _estimate_llm_cost_usd_tier(3, tier_counts[3])

    return {
        "row_count": n,
        "tier_distribution": {f"T{k}": tier_counts.get(k, 0) for k in (1, 2, 3)},
        "domain_distribution": domain_counts,
        "estimated_full_llm_cost_usd": round(est, 2),
        "rows": rows_out,
    }


# ── Insights (PostgreSQL) ─────────────────────────────────────────────────


@app.get("/insights/summary")
async def insights_summary():
    data = insights_queries.get_summary()
    try:
        data["tier_distribution_projection"] = insights_queries.tier_projection_sample(500)
    except Exception as e:
        data["tier_distribution_projection"] = {}
        data["tier_projection_error"] = str(e)
    open_n = data.get("open_complaints") or 0
    data["projected_monthly_savings_inr"] = round(open_n * 0.28 * 900, 0)
    return data


@app.get("/insights/hotspots")
async def insights_hotspots():
    return {"hotspots": insights_queries.get_hotspots(20)}


@app.get("/insights/domains")
async def insights_domains():
    return insights_queries.get_domains_heatmap()


@app.get("/insights/recurrence")
async def insights_recurrence():
    return insights_queries.get_recurrence()


@app.get("/insights/aging")
async def insights_aging():
    return {"buckets": insights_queries.get_aging_buckets()}


@app.get("/insights/taxonomy")
async def insights_taxonomy():
    return insights_queries.get_taxonomy_chaos()


@app.get("/insights/ownership")
async def insights_ownership():
    return insights_queries.get_ownership_split()


@app.get("/insights/multitrade")
async def insights_multitrade():
    return insights_queries.get_multitrade_patterns(25)
