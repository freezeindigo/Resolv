"""
Resolv LangGraph pipeline — full tier routing graph.

Tier 1: intake → classify → assess → rule_route → execute → audit
Tier 2: + assemble_context → query_patterns → single_reasoning → execute → audit
Tier 3: + spawn_hypotheses → interpret_patterns → arbitrate → execute → audit
"""

import asyncio
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from src.agents.llm_client import llm_call
from src.config.routing_actions import (
    allowed_actions_json_hint,
    get_tier1_rule_tuple,
    normalize_primary_action,
)
from src.nodes.domain_classifier import classify_domain
from src.nodes.complexity_assessor import assess_complexity
from src.nodes.context_assembler import assemble_context, ContextPackage
from src.memory.pattern_state import get_active_clusters, ingest_complaint, PatternSignal
from src.agents.hypothesis_agent import spawn_hypothesis_agents, HypothesisResult
from src.agents.arbiter import run_arbiter, RoutingDecision
from src.agents.judge import run_judge
from src.nodes.ownership import infer_ownership

TIER2_SYSTEM_PROMPT = (
    "You are a facility management routing assistant for high-rise residential towers.\n"
    "Output a single JSON object only (no markdown code fences).\n"
    "Required keys: action, vendor_skill_level, priority, sla_hours, materials_hint, reasoning, confidence, ownership.\n"
    "- action MUST be exactly one of: "
    + allowed_actions_json_hint()
    + "\n"
    "- ownership must be \"FM\" or \"Project\" (FM = facility operations; Project = developer / DLP / handover-era defects).\n"
    "- vendor_skill_level: junior | senior | specialist\n"
    "- priority: P1 | P2 | P3 | P4\n"
    "- confidence: high | medium | low\n"
    "Choose the single best dispatch action matching the DOMAIN and building context.\n"
    "Do not invent action names — only the list above. If unclear, use assign_fm_manager.\n"
)


# ── State ──────────────────────────────────────────────────────────────────

class ResolvState(TypedDict):
    # Input
    complaint_title: str
    ticket_id: str
    site_name: str
    tower: str
    flat: str
    priority_requested: Optional[str]

    # Pipeline outputs
    domain: str
    domain_confidence: float
    domain_method: str
    tier: int
    tier_reason: str
    context: Optional[ContextPackage]
    pattern_signal: Optional[PatternSignal]
    hypothesis_results: List[HypothesisResult]
    pattern_interpretation: Optional[str]
    routing_decision: Optional[RoutingDecision]
    tier2_reasoning: Optional[str]

    # Audit
    audit_log: Dict[str, Any]
    total_tokens: int
    total_latency_ms: int
    error: Optional[str]


# ── Node implementations ───────────────────────────────────────────────────

def node_intake(state: ResolvState) -> ResolvState:
    state["audit_log"]["intake"] = {
        "ticket_id": state["ticket_id"],
        "site": state["site_name"],
        "tower": state["tower"],
        "flat": state["flat"],
    }
    return state


def node_classify_domain(state: ResolvState) -> ResolvState:
    result = classify_domain(state["complaint_title"])
    state["domain"] = result["domain"]
    state["domain_confidence"] = result["confidence"]
    state["domain_method"] = result["method"]
    state["audit_log"]["domain"] = result
    return state


def node_assess_complexity(state: ResolvState) -> ResolvState:
    result = assess_complexity(
        state["complaint_title"],
        state["domain"],
        domain_confidence=state["domain_confidence"],
        domain_method=state["domain_method"],
    )
    state["tier"] = result["tier"]
    state["tier_reason"] = result["reason"]
    state["audit_log"]["tier"] = result
    return state


async def node_assemble_context(state: ResolvState) -> ResolvState:
    ctx = await assemble_context(
        state["site_name"],
        state["tower"],
        state["flat"],
        complaint_title=state["complaint_title"],
        domain=state["domain"],
    )
    state["context"] = ctx
    state["total_latency_ms"] += ctx.retrieval_ms + ctx.rag_retrieval_ms
    state["audit_log"]["context_retrieval_ms"] = ctx.retrieval_ms
    state["audit_log"]["rag_retrieval_ms"] = ctx.rag_retrieval_ms
    if ctx.rag_sources_used:
        state["audit_log"]["rag_sources_used"] = ctx.rag_sources_used
    return state


def node_query_patterns(state: ResolvState) -> ResolvState:
    signal = get_active_clusters(state["site_name"], state["tower"], state["domain"])
    state["pattern_signal"] = signal
    state["audit_log"]["pattern_signal"] = {
        "cluster_count": len(signal.active_clusters),
        "has_stack": signal.has_stack_pattern,
    }

    # Upgrade to Tier 3 if stack pattern detected
    if signal.has_stack_pattern and state["tier"] == 2:
        state["tier"] = 3
        state["tier_reason"] = "upgraded: vertical stack pattern detected"
        state["audit_log"]["tier_upgraded"] = True

    return state


def node_rule_route(state: ResolvState) -> ResolvState:
    """Tier 1: deterministic routing, no LLM."""
    action, skill, priority, sla = get_tier1_rule_tuple(
        state["domain"], state["complaint_title"]
    )
    action = normalize_primary_action(action)
    state["routing_decision"] = RoutingDecision(
        primary_action=action,
        vendor_skill_level=skill,
        priority=state.get("priority_requested") or priority,
        sla_hours=sla,
        materials_hint="",
        secondary_action=None,
        routing_basis=f"Tier 1 rule: {state['tier_reason']}",
        confidence="high",
        reasoning=f"Unambiguous {state['domain']} complaint — rule routed without LLM.",
        escalation_trigger="If vendor cannot resolve, escalate to senior.",
        ownership=infer_ownership(
            state["complaint_title"],
            domain=state["domain"],
            tier=state["tier"],
            hypothesis_results=None,
        ),
    )
    state["audit_log"]["routing"] = {"method": "rule", "action": action}
    return state


async def node_tier2_reasoning(state: ResolvState) -> ResolvState:
    """Tier 2: single LLM reasoning call with context."""
    ctx = state["context"]
    context_str = ctx.to_prompt_context() if ctx else "No context available."

    user_content = (
        f"COMPLAINT: {state['complaint_title']}\n"
        f"DOMAIN: {state['domain']}\n\n"
        f"{context_str}"
    )

    try:
        import json, re

        out = await llm_call("tier2_reasoning", TIER2_SYSTEM_PROMPT, user_content)
        latency_ms = out["latency_ms"]
        tokens = out["tokens"]
        text = out["text"]
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1)
        try:
            parsed = json.loads(text.strip())
        except Exception:
            parsed = {"action": "assign_fm_manager", "vendor_skill_level": "senior",
                      "priority": "P2", "sla_hours": 24, "materials_hint": "",
                      "reasoning": text[:200], "confidence": "low"}

        primary = normalize_primary_action(parsed.get("action"))
        own = infer_ownership(
            state["complaint_title"],
            domain=state["domain"],
            tier=2,
            hypothesis_results=None,
        )
        ol = parsed.get("ownership")
        if isinstance(ol, str) and ol.strip():
            oln = ol.strip().lower()
            if oln == "project":
                own = "Project"
            elif oln in ("fm", "facility", "facilities"):
                own = "FM"
        state["routing_decision"] = RoutingDecision(
            primary_action=primary,
            vendor_skill_level=parsed.get("vendor_skill_level", "senior"),
            priority=parsed.get("priority", "P2"),
            sla_hours=parsed.get("sla_hours", 24),
            materials_hint=parsed.get("materials_hint", ""),
            secondary_action=None,
            routing_basis="Tier 2 single-agent reasoning",
            confidence=parsed.get("confidence", "medium"),
            reasoning=parsed.get("reasoning", ""),
            escalation_trigger="If unresolved in SLA window, escalate to senior.",
            ownership=own,
            tokens_used=tokens,
            latency_ms=latency_ms,
        )
        state["total_tokens"] += tokens
        state["total_latency_ms"] += latency_ms
        state["audit_log"]["tier2_tokens"] = tokens
    except Exception as e:
        state["error"] = str(e)
        state["audit_log"]["tier2_error"] = str(e)
        state = node_rule_route(state)
        state["audit_log"]["routing"]["fallback"] = "tier2_api_failure"

    return state


async def node_spawn_hypotheses(state: ResolvState) -> ResolvState:
    """Tier 3: spawn all hypothesis agents in parallel."""
    try:
        results, trigger_audit = await spawn_hypothesis_agents(
            domain=state["domain"],
            complaint_title=state["complaint_title"],
            context=state["context"],
            pattern_signal=state.get("pattern_signal"),
        )
        state["hypothesis_results"] = results
        state["audit_log"]["hypothesis_triggers"] = trigger_audit
        total_tokens = sum(r.tokens_used for r in results)
        state["total_tokens"] += total_tokens
        state["audit_log"]["hypothesis_tokens"] = total_tokens
        state["audit_log"]["hypothesis_count"] = len(results)
    except Exception as e:
        state["error"] = str(e)
        state["audit_log"]["hypothesis_error"] = str(e)
        state = node_rule_route(state)
        state["audit_log"]["routing"]["fallback"] = "hypothesis_api_failure"

    return state


async def node_interpret_patterns(state: ResolvState) -> ResolvState:
    """Tier 3: pattern interpretation agent — adjusts hypothesis likelihoods based on spatial clusters."""
    # Skip if hypothesis spawning already fell back to rule-route
    if not state["hypothesis_results"]:
        return state

    with open("src/agents/prompts/pattern_interpreter.md") as f:
        system_prompt = f.read()

    signal = state["pattern_signal"]
    if signal and signal.active_clusters:
        cluster_lines = [f"{len(signal.active_clusters)} active cluster(s):"]
        for c in signal.active_clusters:
            cluster_lines.append(
                f"  [{c.spatial_pattern}] {c.complaint_count} complaints | "
                f"floors {c.floors_affected} | {c.temporal_pattern} | "
                f"dominant: {c.dominant_category} | confidence: {c.confidence:.2f}"
            )
        if signal.has_stack_pattern:
            cluster_lines.append("  VERTICAL STACK PATTERN DETECTED")
        cluster_summary = "\n".join(cluster_lines)
    else:
        cluster_summary = "No active clusters."

    hypothesis_lines = ["HYPOTHESIS SCORES:"]
    for r in sorted(state["hypothesis_results"], key=lambda x: x.adjusted_score, reverse=True):
        hypothesis_lines.append(
            f"  {r.hypothesis_id}: likelihood={r.likelihood:.2f} "
            f"adjusted={r.adjusted_score:.2f} confidence={r.confidence}"
        )
    hypothesis_summary = "\n".join(hypothesis_lines)

    user_message = (
        f"LOCATION: {state['site_name']} | Tower {state['tower']} | Flat {state['flat']}\n"
        f"COMPLAINT: {state['complaint_title']}\n\n"
        f"ACTIVE CLUSTERS:\n{cluster_summary}\n\n"
        f"{hypothesis_summary}"
    )

    try:
        out = await llm_call("pattern_interpreter", system_prompt, user_message)
        latency_ms = out["latency_ms"]
        tokens = out["tokens"]
        state["pattern_interpretation"] = out["text"]
        state["total_tokens"] += tokens
        state["total_latency_ms"] += latency_ms
        state["audit_log"]["pattern_interpreter_tokens"] = tokens
    except Exception as e:
        state["pattern_interpretation"] = None
        state["error"] = str(e)
        state["audit_log"]["pattern_interpreter_error"] = str(e)

    return state


async def node_arbitrate(state: ResolvState) -> ResolvState:
    """Tier 3: arbiter integrates all signals into final decision."""
    decision = await run_arbiter(
        complaint_title=state["complaint_title"],
        domain=state["domain"],
        hypothesis_results=state["hypothesis_results"],
        pattern_signal=state["pattern_signal"],
        pattern_interpretation=state.get("pattern_interpretation"),
    )
    state["routing_decision"] = decision
    state["total_tokens"] += decision.tokens_used
    state["audit_log"]["arbiter_tokens"] = decision.tokens_used
    return state


async def node_judge(state: ResolvState) -> ResolvState:
    """Haiku validation for Tier 2 and Tier 3 only; Tier 1 never reaches this node."""
    tier = state["tier"]
    decision = state.get("routing_decision")
    if tier not in (2, 3) or not decision:
        return state

    try:
        audit, new_decision = await run_judge(
            complaint_title=state["complaint_title"],
            domain=state["domain"],
            tier=tier,
            decision=decision,
        )
        state["audit_log"]["judge_verdict"] = audit
        state["total_tokens"] += audit.get("tokens_used", 0)
        state["total_latency_ms"] += audit.get("latency_ms", 0)
        if new_decision is not None:
            state["routing_decision"] = new_decision
    except Exception as e:
        state["audit_log"]["judge_verdict"] = {
            "verdict": "approve",
            "reason": f"judge error — leaving routing unchanged: {e}",
            "error": str(e),
            "tokens_used": 0,
            "latency_ms": 0,
        }

    return state


def node_execute(state: ResolvState) -> ResolvState:
    """Execution layer — stub for MVP. Real dispatch goes here."""
    decision = state["routing_decision"]
    state["audit_log"]["execution"] = {
        "action": decision.primary_action if decision else "none",
        "priority": decision.priority if decision else "P3",
        "stub": True,
    }
    # TODO Phase 2: real vendor dispatch API call
    return state


def node_audit(state: ResolvState) -> ResolvState:
    state["audit_log"]["final_tokens"] = state["total_tokens"]
    state["audit_log"]["final_latency_ms"] = state["total_latency_ms"]
    # TODO Phase 2: persist to PostgreSQL audit table
    return state


# ── Routing logic ──────────────────────────────────────────────────────────

def route_after_assess(state: ResolvState) -> str:
    """Skip context assembly for Tier 1."""
    return "rule_route" if state["tier"] == 1 else "assemble_context"


def route_after_patterns(state: ResolvState) -> str:
    if state["tier"] == 2:
        return "tier2_reasoning"
    return "spawn_hypotheses"  # Tier 3


# ── Graph assembly ─────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(ResolvState)

    g.add_node("intake",            node_intake)
    g.add_node("classify_domain",   node_classify_domain)
    g.add_node("assess_complexity", node_assess_complexity)
    g.add_node("assemble_context",  node_assemble_context)
    g.add_node("query_patterns",    node_query_patterns)
    g.add_node("rule_route",        node_rule_route)
    g.add_node("tier2_reasoning",   node_tier2_reasoning)
    g.add_node("spawn_hypotheses",   node_spawn_hypotheses)
    g.add_node("interpret_patterns", node_interpret_patterns)
    g.add_node("arbitrate",          node_arbitrate)
    g.add_node("judge",              node_judge)
    g.add_node("execute",           node_execute)
    g.add_node("audit",             node_audit)

    g.set_entry_point("intake")
    g.add_edge("intake",            "classify_domain")
    g.add_edge("classify_domain",   "assess_complexity")
    g.add_conditional_edges("assess_complexity", route_after_assess,
                            {"rule_route": "rule_route", "assemble_context": "assemble_context"})
    g.add_edge("assemble_context",  "query_patterns")
    g.add_conditional_edges("query_patterns", route_after_patterns,
                            {"tier2_reasoning": "tier2_reasoning", "spawn_hypotheses": "spawn_hypotheses"})
    g.add_edge("rule_route",        "execute")
    g.add_edge("tier2_reasoning",   "judge")
    g.add_edge("spawn_hypotheses",   "interpret_patterns")
    g.add_edge("interpret_patterns", "arbitrate")
    g.add_edge("arbitrate",         "judge")
    g.add_edge("judge",             "execute")
    g.add_edge("execute",           "audit")
    g.add_edge("audit",             END)

    return g.compile()


RESOLV_GRAPH = None


def get_graph():
    global RESOLV_GRAPH
    if RESOLV_GRAPH is None:
        RESOLV_GRAPH = build_graph()
    return RESOLV_GRAPH


async def process_complaint(
    ticket_id: str,
    complaint_title: str,
    site_name: str,
    tower: str,
    flat: str,
    priority_requested: Optional[str] = None,
) -> ResolvState:
    """Entry point: process a single complaint through the full pipeline."""
    initial_state: ResolvState = {
        "complaint_title": complaint_title,
        "ticket_id": ticket_id,
        "site_name": site_name,
        "tower": tower,
        "flat": flat,
        "priority_requested": priority_requested,
        "domain": "other",
        "domain_confidence": 0.0,
        "domain_method": "fallback",
        "tier": 2,
        "tier_reason": "",
        "context": None,
        "pattern_signal": None,
        "hypothesis_results": [],
        "pattern_interpretation": None,
        "routing_decision": None,
        "tier2_reasoning": None,
        "audit_log": {},
        "total_tokens": 0,
        "total_latency_ms": 0,
        "error": None,
    }

    graph = get_graph()
    result = await graph.ainvoke(initial_state)
    return result
