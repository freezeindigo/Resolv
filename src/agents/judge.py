"""
LLM-as-Judge — validation pass for Tier 2 and Tier 3 routing only (Haiku).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from typing import Any, Dict, Optional

import anthropic

from src.agents.arbiter import RoutingDecision
JUDGE_MODEL = "claude-haiku-4-5-20251001"
PROMPT_PATH = "src/agents/prompts/judge.md"


def routing_decision_snapshot(d: RoutingDecision) -> Dict[str, Any]:
    return {
        "primary_action": d.primary_action,
        "vendor_skill_level": d.vendor_skill_level,
        "priority": d.priority,
        "sla_hours": d.sla_hours,
        "secondary_action": d.secondary_action,
        "routing_basis": d.routing_basis,
        "confidence": d.confidence,
        "reasoning": (d.reasoning or "")[:800],
        "escalation_trigger": d.escalation_trigger,
        "ownership": d.ownership,
    }


def _load_system_prompt() -> str:
    with open(PROMPT_PATH) as f:
        return f.read()


def _parse_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {"verdict": "approve", "reason": "judge parse fallback — approving"}


async def run_judge(
    *,
    complaint_title: str,
    domain: str,
    tier: int,
    decision: RoutingDecision,
    client: anthropic.AsyncAnthropic,
) -> tuple[Dict[str, Any], Optional[RoutingDecision]]:
    """
    Returns (audit_dict, new_decision_or_none).
    audit_dict is merged into audit_log['judge_verdict'].
    """
    system = _load_system_prompt()
    user = json.dumps(
        {
            "tier": tier,
            "complaint": complaint_title,
            "domain": domain,
            "proposed_routing": routing_decision_snapshot(decision),
        },
        indent=2,
    )

    t0 = time.monotonic()
    response = await client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=600,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    tokens = response.usage.input_tokens + response.usage.output_tokens
    raw_text = response.content[0].text
    parsed = _parse_json(raw_text)
    verdict = (parsed.get("verdict") or "approve").lower().strip()
    if verdict not in ("approve", "flag", "override"):
        verdict = "approve"
    reason = str(parsed.get("reason") or "")[:500]

    original = routing_decision_snapshot(decision)
    audit: Dict[str, Any] = {
        "verdict": verdict,
        "reason": reason,
        "original_decision": original,
        "model": JUDGE_MODEL,
        "tokens_used": tokens,
        "latency_ms": latency_ms,
        "human_review": None,
        "override_decision": None,
    }

    new_decision: Optional[RoutingDecision] = None
    if verdict == "override":
        ov = parsed.get("override") or {}
        if isinstance(ov, dict) and ov.get("primary_action"):
            new_decision = replace(
                decision,
                primary_action=str(ov.get("primary_action", decision.primary_action)),
                vendor_skill_level=str(ov.get("vendor_skill_level", decision.vendor_skill_level)),
                priority=str(ov.get("priority", decision.priority)),
                sla_hours=int(ov.get("sla_hours", decision.sla_hours)),
                reasoning=str(ov.get("reasoning", decision.reasoning)),
                escalation_trigger=str(ov.get("escalation_trigger", decision.escalation_trigger)),
                routing_basis=(decision.routing_basis or "") + " | judge_override",
                tokens_used=decision.tokens_used,
                latency_ms=decision.latency_ms,
                ownership=decision.ownership,
            )
            audit["override_decision"] = routing_decision_snapshot(new_decision)
        else:
            audit["verdict"] = "flag"
            audit["reason"] = (reason + " — invalid override payload; flagged").strip()
            verdict = "flag"

    if verdict == "flag":
        audit["human_review"] = {"queued": True, "reason": reason or "flagged by judge"}

    return audit, new_decision
