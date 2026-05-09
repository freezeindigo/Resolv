"""
Isolation ablation — does hypothesis-scoped evidence filtering matter?

This experiment isolates ARIA's design choice of running each hypothesis agent
on a *filtered* slice of context (its evidence_filter) versus alternatives:

  Condition A — ARIA-Full (current production path)
                Each hypothesis agent receives ONLY the evidence types in its
                evidence_filter. Spawns ~3-4 separate parallel LLM calls.
                Arbiter integrates all hypothesis results.

  Condition B — ARIA-Pooled
                Each hypothesis agent still spawned separately, but each
                receives the FULL context bundle (no filtering). Same arbiter.
                Tests whether the *filter* does work, or whether full context
                helps each agent more.

  Condition C — ARIA-Single-Prompt
                One LLM call lists ALL selected hypotheses and asks the model
                to score every hypothesis simultaneously, with the full
                context. The arbiter then runs over the parsed results.
                Tests whether architectural separation matters at all.

We run all three conditions on the same Tier-3 cases from
eval/results/paper_eval_ambiguous_300.csv and compare:
    - Project accuracy
    - FM accuracy
    - High-cost errors (Project labelled FM)
    - Low-cost errors (FM labelled Project)
    - Weighted cost (10*HC + LC)

Outputs:
    eval/results/isolation_ablation_predictions.csv   per-complaint per-condition
    eval/results/isolation_ablation_summary.json      summary metrics
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

# Make sure repo root is on sys.path so we can import src.* even when invoked
# directly via `python3 eval/isolation_ablation.py`.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import psycopg2

from src.pipeline.resolv_graph import process_complaint
from src.config.model_config import MODEL_CONFIG
from src.agents import hypothesis_agent as ha
from src.agents.hypothesis_agent import (
    HypothesisResult,
    _build_trigger_context,
    _filter_evidence,
    _load_library,
    _norm,
    _parse_llm_response,
    _select_hypotheses_to_run,
    run_hypothesis_agent,
)
from src.agents.llm_client import llm_call
from src.nodes.context_assembler import ContextPackage


SOURCE_CSV = Path("eval/results/paper_eval_ambiguous_300.csv")
OUT_PRED = Path("eval/results/isolation_ablation_300.csv")
OUT_SUMMARY = Path("eval/results/isolation_ablation_300.json")
# Legacy filenames retained for backward compatibility — will be written too if missing.
OUT_PRED_LEGACY = Path("eval/results/isolation_ablation_predictions.csv")
OUT_SUMMARY_LEGACY = Path("eval/results/isolation_ablation_summary.json")
DB_NAME = "resolv"

CONDITIONS = ["aria_full", "aria_pooled", "aria_single"]


def fetch_complaint_meta(ticket_ids: List[str]) -> Dict[str, dict]:
    """Pull site/tower/flat for the listed ticket ids so we can rerun the pipeline."""
    if not ticket_ids:
        return {}
    conn = psycopg2.connect(dbname=DB_NAME)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ticket_id, complaint_title, category, issue_type,
               site_name, tower, flat
        FROM complaints
        WHERE ticket_id = ANY(%s)
        """,
        (ticket_ids,),
    )
    out: Dict[str, dict] = {}
    for tid, title, category, issue_type, site, tower, flat in cur.fetchall():
        out[str(tid)] = {
            "ticket_id": str(tid),
            "complaint_title": title,
            "category": category,
            "issue_type": issue_type,
            "site_name": site or "unknown",
            "tower": tower or "unknown",
            "flat": flat or "unknown",
        }
    cur.close()
    conn.close()
    return out


@contextmanager
def hypothesis_mode(condition: str):
    """Monkey-patch hypothesis_agent functions to switch isolation regime.

    aria_full   : original behavior (filtered evidence, separate calls)
    aria_pooled : same agents, each receives the full evidence bundle
    aria_single : one combined LLM call evaluating all selected hypotheses
    """
    orig_filter = ha._filter_evidence
    orig_spawn = ha.spawn_hypothesis_agents

    def _full_evidence(context: ContextPackage, evidence_filter: List[str]) -> str:
        return orig_filter(
            context,
            ["flat_history", "adjacent_history", "building_pattern"],
        )

    async def _spawn_single_call(
        domain: str,
        complaint_title: str,
        context: ContextPackage,
        pattern_signal=None,
    ):
        """Replacement spawn: one LLM call covering all selected hypotheses."""
        library = _load_library()
        domain_config = library.get(domain) or {}
        hypotheses = domain_config.get("hypothesis_types", [])
        if not hypotheses:
            return [], {"note": "no hypotheses for domain"}
        title_norm = _norm(complaint_title)
        tr = _build_trigger_context(context, complaint_title, pattern_signal)
        selected, audit = _select_hypotheses_to_run(
            hypotheses, domain, domain_config, title_norm, tr
        )
        if not selected:
            return [], audit

        full_ctx = orig_filter(
            context,
            ["flat_history", "adjacent_history", "building_pattern"],
        )

        hyp_block_lines = []
        for h in selected:
            hyp_block_lines.append(
                f"- id: {h['id']}\n  name: {h.get('name','')}\n  description: {h.get('description','')}"
            )
        hyp_block = "\n".join(hyp_block_lines)

        system_prompt = (
            "You are a residential complaint routing analyst.\n"
            "You will be given several hypotheses for the same complaint and the "
            "available context. For EACH hypothesis, output a JSON object with keys:\n"
            "  hypothesis_id, likelihood (0..1), confidence (low|medium|high),\n"
            "  evidence_for (list), evidence_against (list), reasoning (string),\n"
            "  recommended_action (string).\n"
            "Return ONLY a JSON array (no markdown fences) of these objects, "
            "one per hypothesis, in the same order they were given."
        )
        user_message = (
            f"COMPLAINT: {complaint_title}\n\n"
            f"LOCATION: {context.site_name} / {context.tower} / Flat {context.flat}\n\n"
            f"HYPOTHESES TO EVALUATE:\n{hyp_block}\n\n"
            f"FULL CONTEXT (shared across all hypotheses):\n{full_ctx}"
        )

        t0 = time.monotonic()
        try:
            out = await llm_call("hypothesis_agents", system_prompt, user_message)
            text = out["text"]
            tokens = out["tokens"]
            latency_ms = out["latency_ms"]
        except Exception as e:
            results = [
                HypothesisResult(
                    hypothesis_id=h["id"],
                    hypothesis_name=h.get("name", ""),
                    domain=domain,
                    likelihood=0.5,
                    confidence="low",
                    evidence_for=[],
                    evidence_against=[],
                    reasoning="single-prompt LLM failed",
                    recommended_action="investigate_further",
                    cost_of_error_weight=float(h.get("cost_of_error_weight", 1.0)),
                    adjusted_score=0.5 * float(h.get("cost_of_error_weight", 1.0)),
                    latency_ms=int((time.monotonic() - t0) * 1000),
                    error=str(e),
                )
                for h in selected
            ]
            results.sort(key=lambda r: r.adjusted_score, reverse=True)
            return results, {**audit, "single_call_error": str(e)}

        import re as _re

        match = _re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        body = match.group(1) if match else text
        try:
            arr = json.loads(body.strip())
            if isinstance(arr, dict):
                # Some models wrap in a top-level dict.
                arr = arr.get("hypotheses") or arr.get("results") or []
        except json.JSONDecodeError:
            arr = []

        by_id = {}
        if isinstance(arr, list):
            for entry in arr:
                if not isinstance(entry, dict):
                    continue
                hid = entry.get("hypothesis_id") or entry.get("hypothesis") or entry.get("id")
                if hid:
                    by_id[hid] = entry

        results: List[HypothesisResult] = []
        for h in selected:
            hid = h["id"]
            cost_w = float(h.get("cost_of_error_weight", 1.0))
            entry = by_id.get(hid, {})
            try:
                like = float(entry.get("likelihood", 0.5))
            except (TypeError, ValueError):
                like = 0.5
            like = max(0.0, min(1.0, like))
            results.append(
                HypothesisResult(
                    hypothesis_id=hid,
                    hypothesis_name=h.get("name", ""),
                    domain=domain,
                    likelihood=like,
                    confidence=str(entry.get("confidence", "low")),
                    evidence_for=list(entry.get("evidence_for", []) or []),
                    evidence_against=list(entry.get("evidence_against", []) or []),
                    reasoning=str(entry.get("reasoning", "")),
                    recommended_action=str(entry.get("recommended_action", "investigate_further")),
                    cost_of_error_weight=cost_w,
                    adjusted_score=round(like * cost_w, 3),
                    raw_response=entry,
                    tokens_used=int(tokens / max(1, len(selected))),
                    latency_ms=latency_ms,
                )
            )
        results.sort(key=lambda r: r.adjusted_score, reverse=True)
        return results, audit

    try:
        if condition == "aria_full":
            pass
        elif condition == "aria_pooled":
            ha._filter_evidence = _full_evidence
        elif condition == "aria_single":
            ha._filter_evidence = _full_evidence
            ha.spawn_hypothesis_agents = _spawn_single_call
        else:
            raise ValueError(f"unknown condition: {condition}")
        yield
    finally:
        ha._filter_evidence = orig_filter
        ha.spawn_hypothesis_agents = orig_spawn


async def _process_one(row: dict) -> Optional[dict]:
    try:
        result = await process_complaint(
            ticket_id=row["ticket_id"],
            complaint_title=row["complaint_title"],
            site_name=row["site_name"],
            tower=row["tower"],
            flat=row["flat"],
        )
        decision = result.get("routing_decision")
        label = decision.ownership if decision else "FM"
        return {
            "ticket_id": row["ticket_id"],
            "tier_used": result.get("tier"),
            "label": label,
            "confidence": (decision.confidence if decision else "low"),
            "tokens": result.get("total_tokens", 0),
        }
    except Exception as e:
        print(f"[WARN] {row['ticket_id']} failed: {e}", flush=True)
        return {
            "ticket_id": row["ticket_id"],
            "tier_used": None,
            "label": "FM",
            "confidence": "error",
            "tokens": 0,
        }


async def run_condition(rows: List[dict], condition: str, concurrency: int = 4) -> List[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def bounded(r):
        async with sem:
            return await _process_one(r)

    with hypothesis_mode(condition):
        tasks = [asyncio.create_task(bounded(r)) for r in rows]
        out: List[dict] = []
        for i, t in enumerate(asyncio.as_completed(tasks), 1):
            res = await t
            res["condition"] = condition
            out.append(res)
            if i % 25 == 0:
                print(f"  [{condition}] {i}/{len(rows)} done", flush=True)
        return out


def normalize(label: str) -> str:
    s = (label or "").strip().lower()
    if "project" in s:
        return "Project"
    if "fm" in s or "facility" in s:
        return "FM"
    return "FM"


def metrics(predictions: List[dict], crm_by_id: Dict[str, str]) -> dict:
    paired = [
        (p, crm_by_id.get(p["ticket_id"]))
        for p in predictions
        if p["ticket_id"] in crm_by_id
    ]
    proj = [(p, c) for p, c in paired if c == "Project"]
    fm = [(p, c) for p, c in paired if c == "FM"]
    proj_right = sum(1 for p, _ in proj if normalize(p["label"]) == "Project")
    fm_right = sum(1 for p, _ in fm if normalize(p["label"]) == "FM")
    hc = sum(1 for p, _ in proj if normalize(p["label"]) == "FM")
    lc = sum(1 for p, _ in fm if normalize(p["label"]) == "Project")
    n = len(paired)
    n_proj = len(proj)
    n_fm = len(fm)
    return {
        "n": n,
        "n_proj": n_proj,
        "n_fm": n_fm,
        "overall_acc": round((proj_right + fm_right) / n * 100, 2) if n else 0.0,
        "proj_acc": round(proj_right / n_proj * 100, 2) if n_proj else 0.0,
        "fm_acc": round(fm_right / n_fm * 100, 2) if n_fm else 0.0,
        "hc": hc,
        "lc": lc,
        "weighted_cost": hc * 10 + lc * 1,
    }


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=0,
                   help="Cap number of source rows (0=all from CSV)")
    p.add_argument("--tier-3-only", action="store_true",
                   help="Only run on cases that ARIA-Full classified at Tier 3")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--conditions", default=",".join(CONDITIONS),
                   help=f"Comma-separated subset of: {','.join(CONDITIONS)}")
    p.add_argument(
        "--paper-mode",
        action="store_true",
        help="Apply paper-mode model overrides (Groq for tier2/hypothesis/judge, Haiku for arbiter).",
    )
    return p.parse_args()


def apply_paper_mode():
    MODEL_CONFIG["tier2_reasoning"]["provider"] = "groq"
    MODEL_CONFIG["tier2_reasoning"]["model"] = "llama-3.3-70b-versatile"
    MODEL_CONFIG["hypothesis_agents"]["provider"] = "groq"
    MODEL_CONFIG["hypothesis_agents"]["model"] = "llama-3.3-70b-versatile"
    MODEL_CONFIG["arbiter"]["provider"] = "anthropic"
    MODEL_CONFIG["arbiter"]["model"] = "claude-haiku-4-5-20251001"
    MODEL_CONFIG["judge"]["provider"] = "groq"
    MODEL_CONFIG["judge"]["model"] = "llama-3.1-8b-instant"
    print(
        "Paper-mode model override active: "
        "Tier2=Groq, Hypothesis=Groq, Arbiter=Anthropic Haiku, Judge=Groq.",
        flush=True,
    )


def main():
    args = parse_args()

    if args.paper_mode:
        apply_paper_mode()

    if not SOURCE_CSV.exists():
        raise RuntimeError(f"Missing source CSV: {SOURCE_CSV}")

    with open(SOURCE_CSV) as f:
        src_rows = list(csv.DictReader(f))
    print(f"Loaded {len(src_rows)} rows from {SOURCE_CSV}")

    if args.tier_3_only:
        src_rows = [r for r in src_rows if str(r.get("tier_used")) == "3"]
        print(f"Tier 3 only: {len(src_rows)} rows")

    if args.max and args.max < len(src_rows):
        src_rows = src_rows[: args.max]
        print(f"Capped to first {len(src_rows)} rows (--max)")

    ticket_ids = [r["ticket_id"] for r in src_rows]
    crm_by_id = {r["ticket_id"]: normalize(r["crm_label"]) for r in src_rows}

    meta = fetch_complaint_meta(ticket_ids)
    rows = []
    for tid in ticket_ids:
        m = meta.get(tid)
        if not m:
            print(f"[WARN] ticket {tid} missing in DB; skipping", flush=True)
            continue
        rows.append(m)
    print(f"Resolved {len(rows)} complaints from DB")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in CONDITIONS:
            raise ValueError(f"Unknown condition '{c}'. Allowed: {CONDITIONS}")

    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    all_preds: List[dict] = []
    summary: Dict[str, dict] = {}

    for cond in conditions:
        print(f"\n=== Running condition: {cond} (n={len(rows)}) ===", flush=True)
        t0 = time.monotonic()
        preds = asyncio.run(run_condition(rows, cond, concurrency=args.concurrency))
        elapsed = time.monotonic() - t0
        m = metrics(preds, crm_by_id)
        m["elapsed_seconds"] = round(elapsed, 1)
        summary[cond] = m
        all_preds.extend(preds)
        print(f"  {cond}: overall {m['overall_acc']}% | "
              f"FM {m['fm_acc']}% | Proj {m['proj_acc']}% | "
              f"HC {m['hc']} LC {m['lc']} | wtd {m['weighted_cost']} | "
              f"{m['elapsed_seconds']}s", flush=True)

    fields = ["ticket_id", "condition", "tier_used", "label", "confidence", "tokens"]
    for out_path in (OUT_PRED, OUT_PRED_LEGACY):
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for p in all_preds:
                writer.writerow({k: p.get(k, "") for k in fields})
        print(f"\nSaved predictions: {out_path}")

    full = {
        "n_complaints": len(rows),
        "tier_3_only": args.tier_3_only,
        "conditions": summary,
        "cost_matrix": {
            "high_cost_misroute": 10,
            "low_cost_misroute": 1,
            "note": "Project->FM = 10 (warranty voided); FM->Project = 1 (minor delay).",
        },
    }
    for out_path in (OUT_SUMMARY, OUT_SUMMARY_LEGACY):
        with open(out_path, "w") as f:
            json.dump(full, f, indent=2)
        print(f"Saved summary: {out_path}")

    print("\nIsolation ablation summary:")
    print(f"{'condition':<14} {'n':>4} {'overall':>7} {'fm':>6} {'proj':>6} "
          f"{'hc':>4} {'lc':>4} {'wtd':>5}")
    for c, m in summary.items():
        print(
            f"{c:<14} {m['n']:>4} {m['overall_acc']:>6.2f}% {m['fm_acc']:>5.2f}% "
            f"{m['proj_acc']:>5.2f}% {m['hc']:>4} {m['lc']:>4} {m['weighted_cost']:>5}"
        )


if __name__ == "__main__":
    main()
