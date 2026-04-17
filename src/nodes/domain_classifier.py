"""
Domain Classifier — Pipeline Node (deterministic, no LLM)

Classifies a complaint title into one of 10 domains using keyword rules
defined in src/config/domain_rules.yaml.

Scoring:
  - Phrase match (multi-word): weight 3
  - Keyword match (single token): weight 1
  - Multiple matches accumulate

Confidence:
  - top_score / (top_score + second_score + 1) normalised to [0, 1]
  - If top_score == 0: return "other" with confidence 0.0
  - If confidence < CONFIDENCE_THRESHOLD: flag for LLM fallback

Usage:
    from src.nodes.domain_classifier import classify_domain
    result = classify_domain("water leakage from ceiling")
    # {"domain": "water_plumbing", "confidence": 0.91, "method": "rules"}
"""

import os
import re
from functools import lru_cache
from pathlib import Path

import yaml

CONFIDENCE_THRESHOLD = 0.5
PHRASE_WEIGHT = 3
KEYWORD_WEIGHT = 1

CONFIG_PATH = Path(__file__).parent.parent / "config" / "domain_rules.yaml"


@lru_cache(maxsize=1)
def _load_rules():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation noise."""
    text = text.lower().strip()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _score_domain(title_norm: str, domain_config: dict) -> int:
    score = 0

    for phrase in domain_config.get("phrase_matches", []):
        if _normalise(phrase) in title_norm:
            score += PHRASE_WEIGHT

    for keyword in domain_config.get("keyword_matches", []):
        kw = _normalise(keyword).strip()
        # Use word-boundary match for short keywords to avoid substring noise
        if len(kw) <= 4:
            if re.search(r'\b' + re.escape(kw) + r'\b', title_norm):
                score += KEYWORD_WEIGHT
        else:
            if kw in title_norm:
                score += KEYWORD_WEIGHT

    return score


def classify_domain(complaint_title: str) -> dict:
    """
    Classify a complaint title into a domain.

    Returns:
        {
            "domain": str,
            "confidence": float,   # 0.0 – 1.0
            "method": str,         # "rules" | "fallback"
            "scores": dict,        # per-domain scores (for debugging)
        }
    """
    if not complaint_title or not complaint_title.strip():
        return {
            "domain": "other",
            "confidence": 0.0,
            "method": "fallback",
            "scores": {},
        }

    rules = _load_rules()
    title_norm = _normalise(complaint_title)

    scores = {}
    for domain, config in rules.items():
        if not isinstance(config, dict):
            continue
        scores[domain] = _score_domain(title_norm, config)

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_domain, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    if top_score == 0:
        return {
            "domain": "other",
            "confidence": 0.0,
            "method": "fallback",
            "scores": scores,
        }

    # Confidence: how dominant is the top domain vs the runner-up
    confidence = top_score / (top_score + second_score + 1)

    if confidence < CONFIDENCE_THRESHOLD:
        # In production: call LLM here. For now, return top domain with low confidence.
        return {
            "domain": top_domain,
            "confidence": round(confidence, 3),
            "method": "fallback",
            "scores": scores,
        }

    return {
        "domain": top_domain,
        "confidence": round(confidence, 3),
        "method": "rules",
        "scores": scores,
    }
