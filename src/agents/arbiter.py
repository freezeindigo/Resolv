"""
Arbiter Agent — integrates hypothesis scores + pattern signal → final routing decision.
LLM-powered. Uses Claude Opus for highest reasoning quality.
"""

import json
import re
import time
from dataclasses import dataclass
from typing import List, Optional

import anthropic

from src.agents.hypothesis_agent import HypothesisResult
from src.memory.pattern_state import PatternSignal
from src.nodes.ownership import infer_ownership

ARBITER_MODEL = "claude-opus-4-6"
MAX_TOKENS = 1500
TEMPERATURE = 0.0

PROMPT_PATH = "src/agents/prompts/arbiter.md"


@dataclass
class RoutingDecision:
    primary_action: str
    vendor_skill_level: str
    priority: str
    sla_hours: int
    materials_hint: str
    secondary_action: Optional[str]
    routing_basis: str
    confidence: str
    reasoning: str
    escalation_trigger: str
    ownership: str = "FM"  # FM | Project
    tokens_used: int = 0
    latency_ms: int = 0


def _load_prompt() -> str:
    with open(PROMPT_PATH) as f:
        return f.read()


def _format_hypotheses(results: List[HypothesisResult]) -> str:
    lines = ["HYPOTHESIS SCORES (sorted by adjusted score):"]
    for r in sorted(results, key=lambda x: x.adjusted_score, reverse=True):
        lines.append(
            f"  [{r.hypothesis_id}] likelihood={r.likelihood:.2f} "
            f"× cost_weight={r.cost_of_error_weight} "
            f"= adjusted={r.adjusted_score:.2f} | confidence={r.confidence}"
        )
        lines.append(f"    reasoning: {r.reasoning[:120]}")
        lines.append(f"    recommended: {r.recommended_action}")
    return "\n".join(lines)


def _format_pattern(signal: Optional[PatternSignal]) -> str:
    if not signal or not signal.active_clusters:
        return "PATTERN SIGNAL: no active clusters detected"
    lines = [f"PATTERN SIGNAL: {len(signal.active_clusters)} active cluster(s)"]
    for c in signal.active_clusters:
        lines.append(
            f"  [{c.spatial_pattern}] {c.complaint_count} complaints | "
            f"floors {c.floors_affected} | {c.temporal_pattern} | "
            f"dominant: {c.dominant_category} | confidence: {c.confidence:.2f}"
        )
    if signal.has_stack_pattern:
        lines.append("  ⚠ VERTICAL STACK PATTERN DETECTED — systemic cause likely")
    return "\n".join(lines)


def _parse_response(text: str) -> dict:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {
            "primary_action": {"action": "investigate_further", "vendor_skill_level": "senior",
                               "priority": "P2", "sla_hours": 24, "materials_hint": ""},
            "secondary_action": None,
            "routing_basis": "parse error",
            "confidence": "low",
            "reasoning": text[:300],
            "escalation_trigger": "manual review required",
        }


async def run_arbiter(
    complaint_title: str,
    domain: str,
    hypothesis_results: List[HypothesisResult],
    pattern_signal: Optional[PatternSignal],
    client: anthropic.AsyncAnthropic,
    pattern_interpretation: Optional[str] = None,
) -> RoutingDecision:
    system_prompt = _load_prompt()

    interpretation_section = (
        f"\nPATTERN INTERPRETATION:\n{pattern_interpretation}"
        if pattern_interpretation else ""
    )
    user_message = (
        f"COMPLAINT: {complaint_title}\n"
        f"DOMAIN: {domain}\n\n"
        f"{_format_hypotheses(hypothesis_results)}\n\n"
        f"{_format_pattern(pattern_signal)}"
        f"{interpretation_section}"
    )

    t_start = time.monotonic()
    response = await client.messages.create(
        model=ARBITER_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    latency_ms = int((time.monotonic() - t_start) * 1000)
    tokens = response.usage.input_tokens + response.usage.output_tokens

    parsed = _parse_response(response.content[0].text)
    primary = parsed.get("primary_action", {})
    secondary = parsed.get("secondary_action")

    secondary_str = None
    if secondary:
        secondary_str = (
            f"{secondary.get('action','?')} | {secondary.get('priority','?')} | "
            f"SLA {secondary.get('sla_hours','?')}h | {secondary.get('materials_hint','')}"
        )

    return RoutingDecision(
        primary_action=primary.get("action", "investigate_further"),
        vendor_skill_level=primary.get("vendor_skill_level", "senior"),
        priority=primary.get("priority", "P2"),
        sla_hours=primary.get("sla_hours", 24),
        materials_hint=primary.get("materials_hint", ""),
        secondary_action=secondary_str,
        routing_basis=parsed.get("routing_basis", ""),
        confidence=parsed.get("confidence", "low"),
        reasoning=parsed.get("reasoning", ""),
        escalation_trigger=parsed.get("escalation_trigger", ""),
        ownership=infer_ownership(complaint_title),
        tokens_used=tokens,
        latency_ms=latency_ms,
    )
