"""
Canonical primary_action vocabulary for Tier 1 rule routing and Tier 2 JSON output.
Every pipeline path must use only these strings (after normalization).
"""

from __future__ import annotations

from typing import Optional, Tuple

ALLOWED_PRIMARY_ACTIONS = frozenset(
    {
        "send_plumber",
        "send_electrician",
        "send_carpenter",
        "send_structural_team",
        "send_hvac_tech",
        "send_pest_control",
        "assign_security_team",
        "assign_housekeeping",
        "assign_lift_operator",
        "assign_fm_manager",
        "escalate_project_team",
        "immediate_emergency",
    }
)

# Legacy / LLM drift → canonical
LEGACY_ACTION_ALIASES: dict[str, str] = {
    "send_civil_team": "send_structural_team",
    "send_lift_engineer": "assign_lift_operator",
    "send_security": "assign_security_team",
    "send_maintenance": "assign_fm_manager",
    "send_housekeeping": "assign_housekeeping",
    "route_to_external_vendor": "assign_fm_manager",
    "route_to_external": "assign_fm_manager",
    "reassign_domain": "assign_fm_manager",
    "investigate_further": "assign_fm_manager",
    "send_plummer": "send_plumber",
    "send_electrican": "send_electrician",
    "assign_security": "assign_security_team",
    "assign_lift": "assign_lift_operator",
    "escalate_to_project": "escalate_project_team",
    "emergency_dispatch": "immediate_emergency",
}

# Tier 1 domain → (primary_action, vendor_skill, default_priority, sla_hours)
TIER1_DOMAIN_DEFAULTS: dict[str, Tuple[str, str, str, int]] = {
    "water_plumbing": ("send_plumber", "junior", "P2", 24),
    "electrical": ("send_electrician", "junior", "P2", 24),
    "carpentry": ("send_carpenter", "junior", "P3", 48),
    "hvac": ("send_hvac_tech", "junior", "P2", 24),
    "lift_elevator": ("assign_lift_operator", "specialist", "P1", 4),
    "structural_civil": ("send_structural_team", "senior", "P2", 48),
    "safety_security": ("assign_security_team", "senior", "P1", 4),
    "pest_hygiene": ("send_pest_control", "junior", "P3", 48),
    "common_area": ("assign_fm_manager", "junior", "P3", 48),
    "other": ("assign_fm_manager", "senior", "P3", 48),
}


def tier1_life_safety_override(complaint_title: str) -> Optional[Tuple[str, str, str, int]]:
    """Life-safety overrides before domain table."""
    tn = complaint_title.lower()
    if "lift" in tn and "stuck" in tn:
        return ("immediate_emergency", "specialist", "P1", 1)
    if "burning smell" in tn or "electric shock" in tn:
        return ("immediate_emergency", "specialist", "P1", 2)
    return None


def get_tier1_rule_tuple(domain: str, complaint_title: str) -> Tuple[str, str, str, int]:
    spec = tier1_life_safety_override(complaint_title)
    if spec:
        return spec
    return TIER1_DOMAIN_DEFAULTS.get(domain, ("assign_fm_manager", "senior", "P3", 48))


def normalize_primary_action(raw: Optional[str]) -> str:
    if not raw:
        return "assign_fm_manager"
    if isinstance(raw, dict):
        raw = raw.get("action") or raw.get("primary_action")
        if not raw:
            return "assign_fm_manager"
    r = str(raw).strip()
    if r in ALLOWED_PRIMARY_ACTIONS:
        return r
    key = r.lower().replace(" ", "_").replace("-", "_")
    while "__" in key:
        key = key.replace("__", "_")
    if key in ALLOWED_PRIMARY_ACTIONS:
        return key
    if key in LEGACY_ACTION_ALIASES:
        return LEGACY_ACTION_ALIASES[key]
    alnum = "".join(c for c in key if c.isalnum() or c == "_")
    if alnum in LEGACY_ACTION_ALIASES:
        return LEGACY_ACTION_ALIASES[alnum]
    return "assign_fm_manager"


def allowed_actions_json_hint() -> str:
    return ", ".join(sorted(ALLOWED_PRIMARY_ACTIONS))
