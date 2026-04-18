"""
Hypothesis Agent — LLM-powered reasoning unit.

spawn_hypothesis_agents() reads hypothesis_library.yaml, applies trigger_conditions,
then runs selected agents in parallel (max 4; minimum 2 for Tier 3 when possible).
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import yaml

from src.memory.pattern_state import PatternSignal
from src.nodes.context_assembler import ContextPackage

LIBRARY_PATH = Path(__file__).parent.parent / "config" / "hypothesis_library.yaml"
PROMPTS_DIR = Path(__file__).parent / "prompts"

HYPOTHESIS_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
TEMPERATURE = 0.0

MAX_HYPOTHESES = 4
MIN_HYPOTHESES = 2


@dataclass
class TriggerContext:
    building_age_years: Optional[float]
    has_recurrence: bool
    has_vertical_stack: bool
    flat_has_ac: bool
    complaint_mentions_ac: bool
    is_monsoon_window: bool


@dataclass
class HypothesisResult:
    hypothesis_id: str
    hypothesis_name: str
    domain: str
    likelihood: float
    confidence: str
    evidence_for: List[str]
    evidence_against: List[str]
    reasoning: str
    recommended_action: str
    cost_of_error_weight: float
    adjusted_score: float
    raw_response: Dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    latency_ms: int = 0
    error: Optional[str] = None


@lru_cache(maxsize=1)
def _load_library() -> dict:
    with open(LIBRARY_PATH) as f:
        return yaml.safe_load(f)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _keyword_hit(title_norm: str, kw: str) -> bool:
    k = _norm(kw)
    if not k:
        return False
    if " " in k:
        return k in title_norm
    if k == "wc":
        return bool(re.search(r"\bwc\b", title_norm))
    if len(k) <= 3:
        return bool(re.search(rf"\b{re.escape(k)}\b", title_norm))
    return k in title_norm


def _suppress_hit(title_norm: str, phrase: str) -> bool:
    p = _norm(phrase)
    if not p:
        return False
    if " " in p:
        return p in title_norm
    return _keyword_hit(title_norm, p)


def _build_trigger_context(
    context: ContextPackage,
    complaint_title: str,
    pattern_signal: Optional[PatternSignal],
) -> TriggerContext:
    title_norm = _norm(complaint_title)
    now = datetime.now()

    recur_re = re.compile(
        r"\b(again|same issue|same problem|months?\s+ago|recurring|repeated|reported)\b",
        re.I,
    )
    has_recurrence_kw = bool(recur_re.search(title_norm))

    hist = context.flat_history or []
    recent_count = 0
    for c in hist:
        if c.created_date:
            if (now - c.created_date).days <= 365:
                recent_count += 1
    has_recurrence_hist = recent_count >= 2
    has_recurrence = has_recurrence_kw or has_recurrence_hist

    has_vertical_stack = bool(pattern_signal and pattern_signal.has_stack_pattern)

    complaint_mentions_ac = bool(
        re.search(r"\b(ac|a/c|air conditioning|hvac|split)\b", title_norm, re.I)
    )
    flat_has_ac = complaint_mentions_ac
    for c in hist:
        t = (c.complaint_title or "").lower()
        cat = (c.category or "").lower()
        if re.search(r"\bac\b|air condition|hvac|split", t):
            flat_has_ac = True
            break
        if "ac" in cat or "ac repair" in cat:
            flat_has_ac = True
            break

    monsoon_kw = "monsoon" in title_norm
    is_monsoon_window = now.month in (6, 7, 8, 9) or monsoon_kw

    age = context.building_age_years

    return TriggerContext(
        building_age_years=age,
        has_recurrence=has_recurrence,
        has_vertical_stack=has_vertical_stack,
        flat_has_ac=flat_has_ac,
        complaint_mentions_ac=complaint_mentions_ac,
        is_monsoon_window=is_monsoon_window,
    )


def _eval_require_context(rc: Optional[dict], tr: TriggerContext) -> bool:
    if not rc:
        return True
    for key, val in rc.items():
        if key == "building_age_gt":
            if tr.building_age_years is None or not (tr.building_age_years > float(val)):
                return False
        elif key == "has_recurrence":
            if bool(val) != tr.has_recurrence:
                return False
        elif key == "has_vertical_stack":
            if bool(val) != tr.has_vertical_stack:
                return False
        elif key == "flat_has_ac":
            if bool(val) != tr.flat_has_ac:
                return False
        elif key == "complaint_mentions_ac":
            if bool(val) != tr.complaint_mentions_ac:
                return False
    return True


def _eval_require_context_or(rco: Optional[dict], tr: TriggerContext) -> bool:
    if not rco:
        return True
    for key, val in rco.items():
        if not val:
            continue
        if key == "flat_has_ac" and tr.flat_has_ac:
            return True
        if key == "complaint_mentions_ac" and tr.complaint_mentions_ac:
            return True
    return False


def _selection_weight(hyp: dict, tr: TriggerContext) -> float:
    tc = hyp.get("trigger_conditions") or {}
    w = float(hyp.get("cost_of_error_weight", 1.0))
    w += float(tc.get("boost_score") or 0)
    hid = hyp["id"]
    eb = tc.get("extra_boost") or {}
    if hid == "structural_seepage" and eb.get("recurrence_or_vertical_stack"):
        if tr.has_recurrence or tr.has_vertical_stack:
            w += float(eb["recurrence_or_vertical_stack"])
    if hid == "waterproofing_failure" and eb.get("monsoon_or_age_gt_7"):
        if tr.is_monsoon_window or (tr.building_age_years and tr.building_age_years > 7):
            w += float(eb["monsoon_or_age_gt_7"])
    if hid == "structural_movement" and eb.get("building_age_gt_5"):
        if tr.building_age_years and tr.building_age_years > 5:
            w += float(eb["building_age_gt_5"])
    return w


def _evaluate_hypothesis_triggers(
    hyp: dict,
    domain: str,
    title_norm: str,
    tr: TriggerContext,
    domain_config: dict,
) -> Tuple[bool, str]:
    """Return (should_spawn, reason)."""
    if domain_config.get("always_all_hypotheses"):
        return True, "safety_security: always_all_hypotheses"

    tc = hyp.get("trigger_conditions") or {}
    hid = hyp["id"]

    if hid == "safety_hazard" and domain == "electrical":
        req = tc.get("require_any") or []
        if any(_keyword_hit(title_norm, k) for k in req):
            return True, "safety_hazard: keyword match (never suppress)"
        return False, "safety_hazard: no safety keywords matched"

    if tc.get("always_spawn_in_domain"):
        return True, f"{hid}: always_spawn_in_domain ({domain})"

    for s in tc.get("suppress") or []:
        if _suppress_hit(title_norm, s):
            return False, f"suppress matched: {s}"

    req_any = tc.get("require_any") or []
    if req_any:
        if not any(_keyword_hit(title_norm, k) for k in req_any):
            return False, "require_any not satisfied"

    if not _eval_require_context(tc.get("require_context"), tr):
        return False, "require_context not satisfied"

    if tc.get("require_context_or"):
        if not _eval_require_context_or(tc.get("require_context_or"), tr):
            return False, "require_context_or not satisfied"

    return True, "triggers satisfied"


def _select_hypotheses_to_run(
    hypotheses: List[dict],
    domain: str,
    domain_config: dict,
    title_norm: str,
    tr: TriggerContext,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Returns (selected_hypothesis_configs, audit)."""
    audit_filtered: List[Dict[str, Any]] = []
    audit_spawn: List[Dict[str, Any]] = []
    passed: List[Tuple[dict, float, str]] = []

    for hyp in hypotheses:
        ok, reason = _evaluate_hypothesis_triggers(hyp, domain, title_norm, tr, domain_config)
        entry = {"id": hyp["id"], "name": hyp.get("name"), "reason": reason}
        if ok:
            w = _selection_weight(hyp, tr)
            passed.append((hyp, w, reason))
            audit_spawn.append({**entry, "selection_weight": w})
        else:
            audit_filtered.append(entry)

    passed.sort(key=lambda x: x[1], reverse=True)
    selected = [p[0] for p in passed]
    selected = selected[:MAX_HYPOTHESES]

    backfill_audit: List[Dict[str, Any]] = []
    if len(selected) < MIN_HYPOTHESES and not domain_config.get("always_all_hypotheses"):
        by_cost = sorted(hypotheses, key=lambda h: float(h.get("cost_of_error_weight", 0)), reverse=True)
        sel_ids = {h["id"] for h in selected}
        for h in by_cost:
            if h["id"] in sel_ids:
                continue
            selected.append(h)
            sel_ids.add(h["id"])
            backfill_audit.append(
                {"id": h["id"], "reason": "minimum Tier-3 breadth: backfill by cost_of_error_weight"}
            )
            if len(selected) >= MIN_HYPOTHESES:
                break

    if len(selected) > MAX_HYPOTHESES:
        selected = sorted(
            selected,
            key=lambda h: float(h.get("cost_of_error_weight", 0)),
            reverse=True,
        )[:MAX_HYPOTHESES]

    if not selected and hypotheses:
        selected = sorted(
            hypotheses,
            key=lambda h: float(h.get("cost_of_error_weight", 0)),
            reverse=True,
        )[: max(MIN_HYPOTHESES, 1)]
        audit_filtered.append(
            {
                "id": "_emergency",
                "name": "",
                "reason": "no triggers matched — fallback to highest cost_of_error_weight",
            }
        )

    return selected, {
        "spawned": audit_spawn,
        "filtered": audit_filtered,
        "backfilled": backfill_audit,
        "selected_ids": [h["id"] for h in selected],
        "max_cap": MAX_HYPOTHESES,
        "min_target": MIN_HYPOTHESES,
    }


def _load_prompt(prompt_path: str) -> str:
    path = Path(prompt_path)
    if not path.exists():
        path = PROMPTS_DIR / path.name
    with open(path) as f:
        return f.read()


def _filter_evidence(context: ContextPackage, evidence_filter: List[str]) -> str:
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


async def spawn_hypothesis_agents(
    domain: str,
    complaint_title: str,
    context: ContextPackage,
    client: anthropic.AsyncAnthropic,
    pattern_signal: Optional[PatternSignal] = None,
) -> Tuple[List[HypothesisResult], Dict[str, Any]]:
    library = _load_library()
    domain_config = library.get(domain)

    if not domain_config:
        return [], {"error": f"no hypothesis library for domain={domain}"}

    hypotheses = domain_config.get("hypothesis_types", [])
    if not hypotheses:
        return [], {"error": "empty hypothesis_types"}

    title_norm = _norm(complaint_title)
    tr = _build_trigger_context(context, complaint_title, pattern_signal)

    selected, audit = _select_hypotheses_to_run(hypotheses, domain, domain_config, title_norm, tr)

    if not selected:
        return [], {**audit, "note": "no hypotheses selected"}

    tasks = [
        run_hypothesis_agent(h, domain, complaint_title, context, client)
        for h in selected
    ]
    results = await asyncio.gather(*tasks)
    results = sorted(results, key=lambda r: r.adjusted_score, reverse=True)
    return results, audit

