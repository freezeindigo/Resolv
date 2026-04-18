"""Rule-based FM vs Project ownership hint (no LLM)."""


def infer_ownership(complaint_title: str) -> str:
    t = (complaint_title or "").lower()
    project_markers = (
        "snag",
        "snagging",
        "handover",
        "possession",
        "dlp",
        "defect liability",
        "pre-possession",
        "rera",
        "developer",
        "new construction",
        "carpet area",
        "fitout by project",
        "project team",
        "construction defect",
    )
    if any(m in t for m in project_markers):
        return "Project"
    return "FM"
