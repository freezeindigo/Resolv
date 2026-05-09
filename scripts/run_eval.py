"""
Reviewer quick-start evaluation script for ARIA.

Runs ARIA (and optionally GPT-4o baseline) on a JSONL dataset file.
No PostgreSQL or Redis required — uses no-context mode (all DB retrieval bypassed).

Usage:
    python scripts/run_eval.py \\
        --dataset data/sample/synthetic_300.jsonl \\
        --config eval/config.yaml

Outputs:
    eval/results/synthetic_eval_summary.json   ARIA vs GPT-4o metrics
    eval/results/ablation_table.md             Markdown summary table

If OPENAI_API_KEY is not set, GPT-4o baseline is skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from src.pipeline.resolv_graph import process_complaint
import src.nodes.context_assembler as context_assembler
import src.pipeline.resolv_graph as resolv_graph
from src.config.model_config import MODEL_CONFIG


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_eval_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_model_config(cfg: dict) -> None:
    """Apply model overrides from eval/config.yaml."""
    models = cfg.get("model", {})
    if "tier2" in models:
        provider, model_id = _split_model(models["tier2"])
        MODEL_CONFIG["tier2_reasoning"]["provider"] = provider
        MODEL_CONFIG["tier2_reasoning"]["model"] = model_id
    if "hypotheses" in models:
        provider, model_id = _split_model(models["hypotheses"])
        MODEL_CONFIG["hypothesis_agents"]["provider"] = provider
        MODEL_CONFIG["hypothesis_agents"]["model"] = model_id
    if "pattern_interpreter" in models:
        provider, model_id = _split_model(models["pattern_interpreter"])
        MODEL_CONFIG["pattern_interpreter"]["provider"] = provider
        MODEL_CONFIG["pattern_interpreter"]["model"] = model_id
    if "arbiter" in models:
        provider, model_id = _split_model(models["arbiter"])
        MODEL_CONFIG["arbiter"]["provider"] = provider
        MODEL_CONFIG["arbiter"]["model"] = model_id
    if "judge" in models:
        provider, model_id = _split_model(models["judge"])
        MODEL_CONFIG["judge"]["provider"] = provider
        MODEL_CONFIG["judge"]["model"] = model_id


def _split_model(spec: str) -> tuple[str, str]:
    """Split 'groq/llama-3.3-70b-versatile' into ('groq', 'llama-3.3-70b-versatile')."""
    if "/" in spec:
        provider, model_id = spec.split("/", 1)
        return provider, model_id
    # If no slash, try to infer provider
    if "claude" in spec:
        return "anthropic", spec
    if "gpt" in spec:
        return "openai", spec
    return "groq", spec


def normalize_label(value: str) -> str:
    t = (value or "").strip().lower()
    if "project" in t:
        return "Project"
    return "FM"


def parse_openai_label(text: str) -> str:
    t = (text or "").strip().upper()
    if "PROJECT" in t:
        return "Project"
    return "FM"


async def _empty_context(site_name: str, tower: str, flat: str,
                         complaint_title: str = "", domain: str = "other"):
    """No-op context assembler for reviewer mode (no DB needed)."""
    return context_assembler.ContextPackage(
        site_name=site_name, tower=tower, flat=flat,
        flat_history=[], adjacent_history=[], building_pattern=[],
        adjacency_info={}, retrieval_ms=0, rag_sources_used=[],
        audit_context=[], mom_context=[], rag_retrieval_ms=0,
    )


async def run_aria(rows: list[dict], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    out = []

    async def run_one(row: dict) -> dict:
        async with sem:
            try:
                result = await process_complaint(
                    ticket_id=row["ticket_id"],
                    complaint_title=row["complaint_title"],
                    site_name=row.get("site_name", "SITE_A001"),
                    tower=row.get("tower", "T01"),
                    flat=row.get("flat", "T01-0101"),
                )
                decision = result.get("routing_decision")
                aria_label = decision.ownership if decision else "FM"
                crm = normalize_label(row.get("issue_type", "FM"))
                return {
                    "ticket_id": row["ticket_id"],
                    "complaint_text": row["complaint_title"],
                    "category": row.get("category", ""),
                    "crm_label": crm,
                    "aria_label": aria_label,
                    "aria_confidence": (decision.confidence if decision else "low"),
                    "tier_used": result.get("tier"),
                    "aria_reasoning": (decision.reasoning if decision else "")[:200],
                    "agreed": aria_label == crm,
                    "total_tokens": result.get("total_tokens", 0),
                }
            except Exception as e:
                print(f"  [WARN] {row['ticket_id']} failed: {e}", flush=True)
                crm = normalize_label(row.get("issue_type", "FM"))
                return {
                    "ticket_id": row["ticket_id"],
                    "complaint_text": row["complaint_title"],
                    "category": row.get("category", ""),
                    "crm_label": crm,
                    "aria_label": "FM",
                    "aria_confidence": "error",
                    "tier_used": None,
                    "aria_reasoning": f"ERROR: {e}",
                    "agreed": crm == "FM",
                    "total_tokens": 0,
                }

    # Patch context assembler to bypass DB
    orig_resolv = resolv_graph.assemble_context
    orig_ctx = context_assembler.assemble_context
    resolv_graph.assemble_context = _empty_context
    context_assembler.assemble_context = _empty_context
    try:
        tasks = [run_one(r) for r in rows]
        total = len(tasks)
        done = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            out.append(result)
            done += 1
            if done % 50 == 0:
                print(f"  ARIA: {done}/{total} complaints processed", flush=True)
    finally:
        resolv_graph.assemble_context = orig_resolv
        context_assembler.assemble_context = orig_ctx

    return out


def run_gpt4o(rows: list[dict]) -> list[dict]:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai to run GPT-4o baseline")

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return []

    client = OpenAI(api_key=key)
    output = []
    for i, row in enumerate(rows):
        prompt = (
            "Classify this Indian residential maintenance complaint as FM or Project.\n"
            "FM = Facility Management (routine maintenance, operations).\n"
            "Project = Developer / DLP (original construction defects, warranty).\n"
            "Return exactly one word: FM or Project.\n"
            f"Category: {row.get('category', '')}\n"
            f"Complaint: {row['complaint_title']}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        text = (resp.choices[0].message.content or "").strip()
        label = parse_openai_label(text)
        crm = normalize_label(row.get("issue_type", "FM"))
        output.append({
            "ticket_id": row["ticket_id"],
            "crm_label": crm,
            "gpt4o_label": label,
            "agreed": label == crm,
        })
        if (i + 1) % 50 == 0:
            print(f"  GPT-4o: {i+1}/{len(rows)} done", flush=True)
    return output


def row_cost(crm: str, pred: str, ratio: int = 10) -> int:
    if crm == "Project" and pred == "FM":
        return ratio
    if crm == "FM" and pred == "Project":
        return 1
    return 0


def bootstrap_ci(aria_rows: list[dict], baseline_rows: list[dict],
                 n_boot: int = 1000, ratio: int = 10) -> dict:
    base_by_id = {r["ticket_id"]: r for r in baseline_rows}
    paired = [(r, base_by_id[r["ticket_id"]]) for r in aria_rows if r["ticket_id"] in base_by_id]
    n = len(paired)
    if n == 0:
        return {}

    def compute_reduction(sample: list) -> float:
        ac = sum(row_cost(a["crm_label"], a["aria_label"], ratio) for a, _ in sample)
        gc = sum(row_cost(a["crm_label"], b["gpt4o_label"], ratio) for a, b in sample)
        return (gc - ac) / gc * 100 if gc > 0 else 0.0

    rng = random.Random(42)
    reductions = sorted([compute_reduction([paired[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot)])
    point = compute_reduction(paired)
    return {
        "point_estimate": round(point, 2),
        "ci_95_lo": round(reductions[int(0.025 * n_boot)], 2),
        "ci_95_hi": round(reductions[int(0.975 * n_boot)], 2),
        "n_boot": n_boot,
        "n_paired": n,
    }


def write_ablation_table(aria_rows: list[dict], baseline_rows: list[dict],
                         ci: dict, out_path: Path, cost_ratio: int = 10) -> None:
    """Write a markdown summary table reviewers can inspect."""
    base_by_id = {r["ticket_id"]: r for r in baseline_rows}

    def metrics(rows, pred_field):
        proj = [r for r in rows if r["crm_label"] == "Project"]
        fm = [r for r in rows if r["crm_label"] == "FM"]
        p_right = sum(1 for r in proj if r[pred_field] == "Project")
        f_right = sum(1 for r in fm if r[pred_field] == "FM")
        hc = sum(1 for r in proj if r[pred_field] == "FM")
        lc = sum(1 for r in fm if r[pred_field] == "Project")
        cost = sum(row_cost(r["crm_label"], r[pred_field], cost_ratio) for r in rows)
        return {
            "overall_acc": sum(1 for r in rows if r.get("agreed") or r[pred_field] == r["crm_label"]) / len(rows) * 100,
            "proj_acc": p_right / len(proj) * 100 if proj else 0,
            "fm_acc": f_right / len(fm) * 100 if fm else 0,
            "hc": hc, "lc": lc, "cost": cost,
        }

    lines = ["# ARIA vs GPT-4o — Synthetic Sample Evaluation", ""]
    lines.append(f"n = {len(aria_rows)} complaints | Cost ratio = 1:{cost_ratio} (FM:Project)")
    lines.append("")

    lines.append("| Metric | ARIA | GPT-4o |")
    lines.append("|---|---|---|")

    a = metrics(aria_rows, "aria_label")
    paired_base = [base_by_id[r["ticket_id"]] for r in aria_rows if r["ticket_id"] in base_by_id]
    g = metrics(paired_base, "gpt4o_label") if paired_base else None

    def fmt_or_na(val):
        return f"{val:.1f}%" if val is not None else "N/A (no API key)"

    lines.append(f"| Overall accuracy | {a['overall_acc']:.1f}% | {fmt_or_na(g['overall_acc'] if g else None)} |")
    lines.append(f"| FM accuracy | {a['fm_acc']:.1f}% | {fmt_or_na(g['fm_acc'] if g else None)} |")
    lines.append(f"| Project accuracy | {a['proj_acc']:.1f}% | {fmt_or_na(g['proj_acc'] if g else None)} |")
    lines.append(f"| High-cost errors (Project→FM) | {a['hc']} | {g['hc'] if g else 'N/A'} |")
    lines.append(f"| Low-cost errors (FM→Project) | {a['lc']} | {g['lc'] if g else 'N/A'} |")
    lines.append(f"| Weighted cost | {a['cost']} | {g['cost'] if g else 'N/A'} |")
    if ci and g:
        lines.append(f"| Cost reduction vs GPT-4o | {ci.get('point_estimate', '?')}% | — |")
        lines.append(f"| 95% CI | [{ci.get('ci_95_lo')}%, {ci.get('ci_95_hi')}%] | — |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- These results use no-context mode (no flat history, no building adjacency, no RAG retrieval).")
    lines.append("- Absolute numbers differ from paper (n=9,259, with full DB context).")
    lines.append("- Directional expectation: ARIA weighted cost ≤ GPT-4o weighted cost.")
    lines.append("- See `eval/results/full_ambiguous_eval_9259.json` for stored paper results.")
    lines.append(f"- Paper cost reduction: 22.7%, 95% CI [20.6%, 24.7%], n=9,259.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="ARIA reviewer evaluation script")
    parser.add_argument("--dataset", required=True, help="Path to JSONL complaint file")
    parser.add_argument("--config", default="eval/config.yaml", help="Eval config YAML")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--no-gpt", action="store_true", help="Skip GPT-4o baseline")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    config_path = Path(args.config)

    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}")
        sys.exit(1)

    rows = load_jsonl(dataset_path)
    print(f"Loaded {len(rows)} complaints from {dataset_path}")

    cfg = {}
    if config_path.exists():
        cfg = load_eval_config(config_path)
        apply_model_config(cfg)
        print(f"Config loaded from {config_path}")
    else:
        print(f"[WARN] Config not found at {config_path}, using default model_config.py")

    cost_ratio = cfg.get("cost_matrix", {}).get("project_misrouted_as_fm", 10)
    n_boot = args.bootstrap

    print(f"\nRunning ARIA on {len(rows)} complaints (no-context mode, no DB required)...")
    aria_rows = asyncio.run(run_aria(rows, concurrency=args.concurrency))

    baseline_rows = []
    if not args.no_gpt and os.environ.get("OPENAI_API_KEY"):
        print(f"\nRunning GPT-4o baseline on {len(rows)} complaints...")
        baseline_rows = run_gpt4o(rows)
    elif not args.no_gpt:
        print("\nOPENAI_API_KEY not set — skipping GPT-4o baseline.")

    proj_rows = [r for r in aria_rows if r["crm_label"] == "Project"]
    fm_rows = [r for r in aria_rows if r["crm_label"] == "FM"]
    proj_right = sum(1 for r in proj_rows if r["aria_label"] == "Project")
    fm_right = sum(1 for r in fm_rows if r["aria_label"] == "FM")

    aria_cost = sum(row_cost(r["crm_label"], r["aria_label"], cost_ratio) for r in aria_rows)

    print(f"\n--- ARIA Results ---")
    print(f"Overall accuracy: {sum(1 for r in aria_rows if r['agreed'])/len(aria_rows)*100:.1f}%")
    print(f"FM accuracy: {fm_right/len(fm_rows)*100:.1f}% ({fm_right}/{len(fm_rows)})")
    print(f"Project accuracy: {proj_right/len(proj_rows)*100:.1f}% ({proj_right}/{len(proj_rows)})" if proj_rows else "Project accuracy: N/A (no Project labels in sample)")
    print(f"Weighted cost (ratio 1:{cost_ratio}): {aria_cost}")
    print(f"Tier distribution: {dict(Counter(r['tier_used'] for r in aria_rows))}")

    ci = {}
    if baseline_rows:
        gpt_cost = sum(row_cost(r["crm_label"], r["gpt4o_label"], cost_ratio) for r in baseline_rows)
        if gpt_cost > 0:
            cost_red = (gpt_cost - aria_cost) / gpt_cost * 100
            print(f"\n--- GPT-4o vs ARIA ---")
            print(f"GPT-4o weighted cost: {gpt_cost}")
            print(f"Cost reduction: {cost_red:.1f}%")
            if n_boot > 0:
                ci = bootstrap_ci(aria_rows, baseline_rows, n_boot, cost_ratio)
                print(f"Bootstrap 95% CI: [{ci['ci_95_lo']}%, {ci['ci_95_hi']}%]")

    # Save outputs
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": str(dataset_path),
        "n": len(aria_rows),
        "cost_ratio": cost_ratio,
        "aria": {
            "overall_accuracy": round(sum(1 for r in aria_rows if r["agreed"]) / len(aria_rows) * 100, 2),
            "fm_accuracy": round(fm_right / len(fm_rows) * 100, 2) if fm_rows else None,
            "project_accuracy": round(proj_right / len(proj_rows) * 100, 2) if proj_rows else None,
            "weighted_cost": aria_cost,
            "tier_distribution": dict(Counter(r["tier_used"] for r in aria_rows)),
        },
        "gpt4o": None,
        "cost_reduction": None,
    }
    if baseline_rows:
        gpt_proj = [r for r in baseline_rows if r["crm_label"] == "Project"]
        gpt_fm = [r for r in baseline_rows if r["crm_label"] == "FM"]
        gpt_cost = sum(row_cost(r["crm_label"], r["gpt4o_label"], cost_ratio) for r in baseline_rows)
        summary["gpt4o"] = {
            "overall_accuracy": round(sum(1 for r in baseline_rows if r["agreed"]) / len(baseline_rows) * 100, 2),
            "fm_accuracy": round(sum(1 for r in gpt_fm if r["gpt4o_label"] == "FM") / len(gpt_fm) * 100, 2) if gpt_fm else None,
            "project_accuracy": round(sum(1 for r in gpt_proj if r["gpt4o_label"] == "Project") / len(gpt_proj) * 100, 2) if gpt_proj else None,
            "weighted_cost": gpt_cost,
        }
        if gpt_cost > 0:
            summary["cost_reduction"] = {
                "point_estimate_pct": round((gpt_cost - aria_cost) / gpt_cost * 100, 2),
                "bootstrap_ci": ci if ci else None,
            }

    summary_path = out_dir / "synthetic_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {summary_path}")

    table_path = out_dir / "ablation_table.md"
    write_ablation_table(aria_rows, baseline_rows, ci, table_path, cost_ratio)


if __name__ == "__main__":
    main()
