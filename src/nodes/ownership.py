"""
FM vs Project ownership — rules + Tier 3 winning hypothesis signal.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence


def _norm(s: str) -> str:
    return (s or "").lower().strip()


def _ownership_from_hypothesis_id(hid: str) -> Optional[str]:
    """Return 'Project', 'FM', or None if unknown."""
    h = _norm(hid).replace(" ", "_")
    if not h:
        return None

    project_ids = (
        "structural_seepage",
        "waterproofing_failure",
        "installation_defect",
        "settlement_movement",
    )
    if h in project_ids:
        return "Project"

    fm_ids = (
        "pipe_failure",
        "mechanical_wear",
        "environmental",
        "equipment_specific",
        "safety_hazard",
        "internal_wiring",
        "external_supply",
        "hvac_condensate",
    )
    if h in fm_ids:
        return "FM"

    if "defect" in h or "installation" in h:
        return "Project"
    if "structural" in h or "waterproof" in h or "settlement" in h:
        return "Project"
    if "wear" in h and "structural" not in h:
        return "FM"
    if "failure" in h:
        if any(x in h for x in ("structural", "waterproof", "settlement", "installation", "seepage")):
            return "Project"
        return "FM"
    return None


def _project_text_signals(title: str) -> bool:
    t = _norm(title)
    markers = (
        "since possession",
        "since handover",
        "at possession",
        "post possession",
        "handover",
        "builder",
        "construction",
        "developer",
        "pending from builder",
        "incomplete work",
        "snag",
        "snagging",
        "dlp",
        "defect liability",
        "pre-possession",
        "rera",
        "new construction",
        "carpet area",
        "fitout by project",
        "project team",
        "construction defect",
    )
    return any(m in t for m in markers)


def _fm_ops_text_signals(title: str) -> bool:
    t = _norm(title)
    markers = (
        "not working",
        "broken",
        "stopped",
        "choked",
        "jammed",
        "not closing",
        "not opening",
        "not cooling",
        "stuck",
        "tripped",
    )
    return any(m in t for m in markers)


def infer_ownership(
    complaint_title: str,
    *,
    domain: Optional[str] = None,
    tier: Optional[int] = None,
    hypothesis_results: Optional[Sequence[Any]] = None,
) -> str:
    """
    Determine FM vs Project ownership.

    - safety_security is always FM (security / access incidents are FM ops).
    - Tier 3: winning hypothesis (highest adjusted_score) drives ownership when mappable.
    - Tier 2: extra project vs FM text signals.
    - Tier 1/2: baseline project markers + domain heuristics.
    """
    t = _norm(complaint_title)
    domain_l = _norm(domain or "")

    if domain_l == "safety_security":
        return "FM"

    # Tier 3 — winning hypothesis
    if tier == 3 and hypothesis_results:
        results = list(hypothesis_results)
        if results:
            sorted_r = sorted(
                results,
                key=lambda x: float(getattr(x, "adjusted_score", 0.0)),
                reverse=True,
            )
            winner = sorted_r[0]
            hid = getattr(winner, "hypothesis_id", "") or ""
            mapped = _ownership_from_hypothesis_id(hid)
            if mapped:
                return mapped

    # Tier 2 — text signals (possession / ops failure)
    if tier == 2:
        if "since possession" in t or "since handover" in t:
            return "Project"
        if _project_text_signals(t) and not _fm_ops_text_signals(t):
            return "Project"
        if _fm_ops_text_signals(t) and not _project_text_signals(t):
            return "FM"

    # Baseline: strong project markers
    if _project_text_signals(t):
        return "Project"

    # Domain hint: structural / civil often project-adjacent when DLP-era
    if domain_l in ("structural_civil",) and any(
        x in t for x in ("possession", "handover", "builder", "developer", "dlp")
    ):
        return "Project"

    return "FM"
