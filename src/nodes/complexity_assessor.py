"""
Complexity Assessor — Pipeline Node (deterministic, no LLM)

Assigns Tier 1, 2, or 3 to a complaint based on:
  - Tier 1: matches an unambiguous pattern → rule route, no LLM
  - Tier 3: recurrence signal, ambiguity keywords, safety keywords, or high-cost domain
  - Tier 2: default

Usage:
    from src.nodes.complexity_assessor import assess_complexity
    result = assess_complexity("seepage issue again in master bedroom", "water_plumbing")
    # {"tier": 3, "reason": "recurrence signal: 'again'"}
"""

import re
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config" / "tier_rules.yaml"


@lru_cache(maxsize=1)
def _load_rules():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _matches_any(text_norm: str, patterns: list):
    """Return first matching pattern or None.

    Supports:
    - plain keyword/phrase patterns via case-insensitive contains
    - regex patterns prefixed with ``re:``
    """
    for pattern in patterns:
        if not pattern:
            continue
        if pattern.startswith("re:"):
            regex = pattern[3:].strip()
            if regex and re.search(regex, text_norm):
                return pattern
            continue

        if _normalise(pattern) in text_norm:
            return pattern
    return None


def assess_complexity(
    complaint_title: str,
    domain: str,
    domain_confidence: float = 1.0,
    domain_method: str = "rules",
) -> dict:
    """
    Assign processing tier to a complaint.

    Args:
        complaint_title: raw complaint text
        domain: domain classification result (e.g. "water_plumbing")

    Returns:
        {
            "tier": int,    # 1, 2, or 3
            "reason": str,  # human-readable explanation
        }
    """
    if not complaint_title or not complaint_title.strip():
        return {"tier": 2, "reason": "empty title — default tier"}

    rules = _load_rules()
    title_norm = _normalise(complaint_title)

    # --- TIER 3 CHECKS (evaluated before Tier 1) ---

    # 1. Safety keywords — always Tier 3, highest priority
    match = _matches_any(title_norm, rules.get("safety_keywords", []))
    if match:
        return {"tier": 3, "reason": f"safety keyword: '{match}'"}

    # 2. High-cost domains — always Tier 3
    if domain in rules.get("tier3_domains", []):
        return {"tier": 3, "reason": f"high-cost domain: {domain}"}

    # 3. Recurrence phrases — strong Tier 3 signal
    match = _matches_any(title_norm, rules.get("recurrence_phrases", []))
    if match:
        return {"tier": 3, "reason": f"recurrence signal: '{match}'"}

    # 4. Ambiguity keywords — multi-hypothesis needed
    match = _matches_any(title_norm, rules.get("ambiguity_keywords", []))
    if match:
        return {"tier": 3, "reason": f"ambiguity keyword: '{match}'"}

    # --- TIER 1 CHECKS ---
    # Only reached if no Tier 3 signals found
    tier1_by_domain = rules.get("tier1_patterns", {})

    # Tier 1 requires a clear, deterministic domain mapping.
    # If domain was not confidently mapped, defer to Tier 2 reasoning.
    min_conf = float(rules.get("tier1_min_domain_confidence", 0.6))
    if domain == "other" or domain_method != "rules" or domain_confidence < min_conf:
        return {
            "tier": 2,
            "reason": (
                "domain not unambiguous for Tier 1 "
                f"(domain={domain}, method={domain_method}, confidence={domain_confidence:.3f})"
            ),
        }

    # Check patterns for the classified domain first
    domain_patterns = tier1_by_domain.get(domain, [])
    match = _matches_any(title_norm, domain_patterns)
    if match:
        return {"tier": 1, "reason": f"exact pattern match: '{match}'"}

    # --- TIER 2 DEFAULT ---
    return {"tier": 2, "reason": "no strong signal — default single-agent reasoning"}
