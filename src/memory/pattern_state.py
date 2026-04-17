"""
Pattern State — Redis sliding-window complaint clustering.

On each new complaint: update sorted sets for spatial-temporal aggregation.
On query: run DBSCAN over recent complaints in a building to detect clusters.

No LLM. Pure deterministic clustering.
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import redis
from sklearn.cluster import DBSCAN

REDIS_URL = "redis://localhost:6379"
WINDOW_SECONDS = 90 * 24 * 3600   # 90-day sliding window
CLUSTER_KEY = "resolv:cluster:{site}:{tower}"
COMPLAINT_KEY = "resolv:complaints:{site}:{tower}"


@dataclass
class ClusterSignal:
    cluster_id: str
    complaint_count: int
    spatial_pattern: str       # "vertical_stack" | "floor_range" | "scattered"
    temporal_pattern: str      # "last_24h" | "last_7d" | "last_30d"
    dominant_category: str
    floors_affected: List[int]
    confidence: float


@dataclass
class PatternSignal:
    active_clusters: List[ClusterSignal]
    has_stack_pattern: bool
    building_complaint_count: int   # in last 90 days


def _get_redis() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)


def ingest_complaint(
    site_name: str,
    tower: str,
    flat: str,
    floor: int,
    category: str,
    ticket_id: str,
    timestamp: Optional[float] = None,
):
    """
    Add a complaint to the Redis sliding window for its building.
    Call this whenever a new complaint is processed.
    """
    if timestamp is None:
        timestamp = time.time()

    r = _get_redis()
    key = COMPLAINT_KEY.format(site=site_name, tower=tower)

    payload = json.dumps({
        "ticket_id": ticket_id,
        "flat": flat,
        "floor": floor,
        "category": category,
        "ts": timestamp,
    })

    # Sorted set: score = timestamp, member = payload
    r.zadd(key, {payload: timestamp})

    # Trim to window
    cutoff = timestamp - WINDOW_SECONDS
    r.zremrangebyscore(key, "-inf", cutoff)
    r.expire(key, WINDOW_SECONDS + 3600)


def _floor_from_flat(flat: str) -> Optional[int]:
    """Extract floor number from flat string. Returns None if unparseable."""
    import re
    digit_runs = re.findall(r"\d+", flat)
    for run in reversed(digit_runs):
        if len(run) in (3, 4):
            return int(run[:2]) if len(run) == 4 else int(run[0])
    return None


def get_active_clusters(
    site_name: str,
    tower: str,
    domain: str,
    since_days: int = 90,
) -> PatternSignal:
    """
    Query active complaint clusters for a building using DBSCAN.
    Returns PatternSignal with cluster info relevant to the given domain.
    """
    r = _get_redis()
    key = COMPLAINT_KEY.format(site=site_name, tower=tower)

    since_ts = time.time() - (since_days * 24 * 3600)
    raw = r.zrangebyscore(key, since_ts, "+inf", withscores=True)

    if not raw:
        return PatternSignal(active_clusters=[], has_stack_pattern=False, building_complaint_count=0)

    records = []
    for member, score in raw:
        try:
            rec = json.loads(member)
            rec["ts"] = score
            records.append(rec)
        except json.JSONDecodeError:
            continue

    total = len(records)

    # Filter to relevant domain (loosely — category contains domain keywords)
    domain_keywords = {
        "water_plumbing": ["plumb", "leakage", "water", "seepage", "drain"],
        "electrical": ["electr", "power", "mcb", "inverter"],
        "structural_civil": ["civil", "crack", "seepage", "mason"],
        "carpentry": ["door", "window", "carpentar"],
        "hvac": ["ac", "hvac", "cool"],
        "lift_elevator": ["lift", "elevator"],
    }
    keywords = domain_keywords.get(domain, [])

    def matches_domain(rec):
        cat = (rec.get("category") or "").lower()
        return any(k in cat for k in keywords) if keywords else True

    domain_records = [r for r in records if matches_domain(r)]

    if len(domain_records) < 2:
        return PatternSignal(active_clusters=[], has_stack_pattern=False, building_complaint_count=total)

    # Build feature matrix: [floor, time_bucket]
    # time_bucket: days ago (0 = today, 90 = oldest)
    now = time.time()
    features = []
    for rec in domain_records:
        floor = rec.get("floor") or _floor_from_flat(rec.get("flat", "")) or 0
        days_ago = (now - rec["ts"]) / (24 * 3600)
        features.append([floor, days_ago])

    X = np.array(features)

    # DBSCAN: eps=3 floors or 7 days, min_samples=2
    db = DBSCAN(eps=3.5, min_samples=2, metric="euclidean")
    labels = db.fit_predict(X)

    clusters = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # noise

    for label in unique_labels:
        mask = labels == label
        cluster_records = [domain_records[i] for i, m in enumerate(mask) if m]
        cluster_floors = sorted(set(
            r.get("floor") or _floor_from_flat(r.get("flat", "")) or 0
            for r in cluster_records
        ))
        cluster_floors = [f for f in cluster_floors if f > 0]

        # Classify spatial pattern
        if len(cluster_floors) >= 2:
            floor_range = max(cluster_floors) - min(cluster_floors)
            gaps = [cluster_floors[i+1] - cluster_floors[i] for i in range(len(cluster_floors)-1)]
            if all(g >= 3 for g in gaps):
                spatial = "vertical_stack"   # same unit, every N floors
            elif floor_range <= 3:
                spatial = "floor_range"
            else:
                spatial = "scattered"
        else:
            spatial = "scattered"

        # Temporal pattern
        ages = [(now - r["ts"]) / 3600 for r in cluster_records]
        min_age_h = min(ages)
        if min_age_h <= 24:
            temporal = "last_24h"
        elif min_age_h <= 168:
            temporal = "last_7d"
        else:
            temporal = "last_30d"

        # Dominant category
        cats = [r.get("category", "unknown") for r in cluster_records]
        dominant = max(set(cats), key=cats.count)

        confidence = min(0.95, 0.4 + len(cluster_records) * 0.1)

        clusters.append(ClusterSignal(
            cluster_id=f"{site_name}:{tower}:{label}",
            complaint_count=len(cluster_records),
            spatial_pattern=spatial,
            temporal_pattern=temporal,
            dominant_category=dominant,
            floors_affected=cluster_floors,
            confidence=confidence,
        ))

    has_stack = any(c.spatial_pattern == "vertical_stack" for c in clusters)

    return PatternSignal(
        active_clusters=clusters,
        has_stack_pattern=has_stack,
        building_complaint_count=total,
    )
