"""
Hypothesis Agent — LLM-powered reasoning unit.

Each agent:
  - Has one isolated system prompt (loaded from .md file)
  - Receives only evidence relevant to its hypothesis (via evidence_filter)
  - Returns a structured HypothesisResult

This module also provides spawn_hypothesis_agents() which reads the
hypothesis_library.yaml and spawns all relevant agents in parallel.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
import yaml

from src.nodes.context_assembler import ContextPackage

LIBRARY_PATH = Path(__file__).parent.parent / "config" / "hypothesis_library.yaml"
PROMPTS_DIR = Path(__file__).parent / "prompts"

# Model config — Sonnet for Tier 3 hypothesis agents
HYPOTHESIS_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
TEMPERATURE = 0.0  # deterministic scoring


@dataclass
class HypothesisResult:
    hypothesis_id: str
    hypothesis_name: str
    domain: str
    likelihood: float           # 0.0 – 1.0
    confidence: str             # "high" | "medium" | "low"
    evidence_for: List[str]
    evidence_against: List[str]
    reasoning: str
    recommended_action: str
    cost_of_error_weight: float
    adjusted_score: float       # likelihood × cost_of_error_weight
    raw_response: Dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    latency_ms: int = 0
    error: Optional[str] = None


@lru_cache(maxsize=1)
def _load_library() -> dict:
    with open(LIBRARY_PATH) as f:
        return yaml.safe_load(f)


def _load_prompt(prompt_path: str) -> str:
    path = Path(prompt_path)
    if not path.exists():
        # Fall back to prompts dir
        path = PROMPTS_DIR / path.name
    with open(path) as f:
        return f.read()


def _filter_evidence(context: ContextPackage, evidence_filter: List[str]) -> str:
    """
    Return only the context sections relevant to this hypothesis.
    Prevents cognitive contamination between hypothesis agents.
    """
    parts = []

    if "flat_history" in evidence_filter and context.flat_history:
        parts.append("=== FLAT COMPLAINT HISTORY (last 365 days) ===")
        for c in context.flat_history:
            date_str = c.created_date.strftime("%Y-%m-%d") if c.created_date else "unknown"
            parts.append(
                f"[{date_str}] {c.category or '?'} | {c.status or '?'} | "
                f"TAT {c.resolution_tat_minutes or '?'} min | {c.complaint_title}"
            )

    if "adjacent_above_flat_history" in evidence_filter or "adjacent_history" in evidence_filter:
        above = context.adjacency_info.get("above_flat")
        below = context.adjacency_info.get("below_flat")
        parts.append(f"\n=== ADJACENT FLATS: above={above}, below={below} ===")
        if context.adjacent_history:
            for c in context.adjacent_history:
                date_str = c.created_date.strftime("%Y-%m-%d") if c.created_date else "unknown"
                parts.append(
                    f"[{date_str}] Flat {c.flat} | {c.category or '?'} | "
                    f"{c.status or '?'} | {c.complaint_title}"
                )
        else:
            parts.append("No adjacent complaints in last 90 days.")

    if "building_pattern" in evidence_filter or "building_seepage_history" in evidence_filter:
        if context.building_pattern:
            parts.append(f"\n=== BUILDING PATTERN — {context.tower} (last 90 days) ===")
            for fp in context.building_pattern[:15]:
                parts.append(f"Floor {fp.floor} | {fp.category} | {fp.count} complaints")
        else:
            parts.append("\n=== BUILDING PATTERN: no data ===")

    return "\n".join(parts) if parts else "No relevant context available for this hypothesis."


def _parse_llm_response(response_text: str, hypothesis_id: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Strip markdown code block if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
    if match:
        response_text = match.group(1)

    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        return {
            "hypothesis": hypothesis_id,
            "likelihood": 0.5,
            "confidence": "low",
            "evidence_for": [],
            "evidence_against": [],
            "reasoning": f"Parse error — raw response: {response_text[:200]}",
            "recommended_action": "investigate_further",
        }


async def run_hypothesis_agent(
    hypothesis_config: dict,
    domain: str,
    complaint_title: str,
    context: ContextPackage,
    client: anthropic.AsyncAnthropic,
) -> HypothesisResult:
    """Run a single hypothesis agent and return its result."""
    import time

    hyp_id = hypothesis_config["id"]
    hyp_name = hypothesis_config["name"]
    cost_weight = hypothesis_config.get("cost_of_error_weight", 1.0)
    evidence_filter = hypothesis_config.get("evidence_filter", [])
    prompt_path = hypothesis_config["prompt_template"]

    system_prompt = _load_prompt(prompt_path)
    filtered_context = _filter_evidence(context, evidence_filter)

    user_message = (
        f"COMPLAINT: {complaint_title}\n\n"
        f"LOCATION: {context.site_name} / {context.tower} / Flat {context.flat}\n\n"
        f"CONTEXT:\n{filtered_context}"
    )

    t_start = time.monotonic()
    try:
        response = await client.messages.create(
            model=HYPOTHESIS_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        latency_ms = int((time.monotonic() - t_start) * 1000)
        tokens = response.usage.input_tokens + response.usage.output_tokens
        parsed = _parse_llm_response(response.content[0].text, hyp_id)

        likelihood = float(parsed.get("likelihood", 0.5))
        likelihood = max(0.0, min(1.0, likelihood))

        return HypothesisResult(
            hypothesis_id=hyp_id,
            hypothesis_name=hyp_name,
            domain=domain,
            likelihood=likelihood,
            confidence=parsed.get("confidence", "low"),
            evidence_for=parsed.get("evidence_for", []),
            evidence_against=parsed.get("evidence_against", []),
            reasoning=parsed.get("reasoning", ""),
            recommended_action=parsed.get("recommended_action", "investigate_further"),
            cost_of_error_weight=cost_weight,
            adjusted_score=round(likelihood * cost_weight, 3),
            raw_response=parsed,
            tokens_used=tokens,
            latency_ms=latency_ms,
        )

    except Exception as e:
        latency_ms = int((time.monotonic() - t_start) * 1000)
        return HypothesisResult(
            hypothesis_id=hyp_id,
            hypothesis_name=hyp_name,
            domain=domain,
            likelihood=0.5,
            confidence="low",
            evidence_for=[],
            evidence_against=[],
            reasoning="",
            recommended_action="investigate_further",
            cost_of_error_weight=cost_weight,
            adjusted_score=0.5 * cost_weight,
            latency_ms=latency_ms,
            error=str(e),
        )


def _should_spawn(hyp_config: dict, complaint_title: str) -> bool:
    """Check trigger conditions for conditional hypotheses."""
    trigger = hyp_config.get("trigger_condition")
    if not trigger:
        return True

    title_lower = complaint_title.lower()
    if trigger == "ceiling_complaint":
        return any(w in title_lower for w in ["ceiling", "ceil", "overhead", "top", "roof"])

    return True


async def spawn_hypothesis_agents(
    domain: str,
    complaint_title: str,
    context: ContextPackage,
    client: anthropic.AsyncAnthropic,
) -> List[HypothesisResult]:
    """
    Load hypothesis library, select agents for this domain,
    and run them all in parallel.
    """
    library = _load_library()
    domain_config = library.get(domain)

    if not domain_config:
        return []

    hypotheses = domain_config.get("hypothesis_types", [])

    # Filter by trigger conditions
    active = [h for h in hypotheses if _should_spawn(h, complaint_title)]

    if not active:
        return []

    # Run all in parallel
    tasks = [
        run_hypothesis_agent(h, domain, complaint_title, context, client)
        for h in active
    ]
    results = await asyncio.gather(*tasks)

    # Sort by adjusted_score descending
    return sorted(results, key=lambda r: r.adjusted_score, reverse=True)
