import os
from dotenv import load_dotenv

load_dotenv("/Users/kartheek/resolv/.env")

"""
Paper evaluation harness for ambiguous complaints.

What it does:
1) Pull ambiguous complaints from PostgreSQL complaints table.
2) Stratified proportional sample of N complaints by category.
3) Run ARIA pipeline and write eval/results/paper_eval_ambiguous_<N>.csv
4) Run GPT-4o text-only baseline and write eval/results/paper_eval_baseline_<N>.csv
5) Print summary: ARIA vs CRM agreement, GPT-4o vs CRM agreement, tier distribution, estimated cost.

Default sample size is 300. The script prints an estimate and asks for confirmation before running.
"""

import argparse
import asyncio
import csv
import os
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.resolv_graph import process_complaint
from src.config.model_config import MODEL_CONFIG
import src.nodes.context_assembler as context_assembler
import src.pipeline.resolv_graph as resolv_graph


AMBIGUOUS_CATEGORIES = [
    "Plumbing",
    "Leakage",
    "Seepage",
    "Carpentary",
    "Civil Work",
    "Mason",
    "Civil",
]


def normalize_label(value: str) -> str:
    t = (value or "").strip().lower()
    if "project" in t:
        return "Project"
    if "fm" in t or "facility" in t:
        return "FM"
    return "FM"


def parse_openai_label(text: str) -> str:
    t = (text or "").strip().upper()
    if t.startswith("PROJECT") or "PROJECT" in t:
        return "Project"
    if t.startswith("FM") or "FM" in t:
        return "FM"
    return "FM"


def fetch_ambiguous_rows(dbname: str) -> List[dict]:
    conn = psycopg2.connect(dbname=dbname)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ticket_id, complaint_title, category, issue_type, site_name, tower, flat
        FROM complaints
        WHERE complaint_title IS NOT NULL
          AND category = ANY(%s)
        """,
        (AMBIGUOUS_CATEGORIES,),
    )
    rows = [
        {
            "ticket_id": r[0],
            "complaint_title": r[1],
            "category": r[2],
            "issue_type": r[3],
            "site_name": r[4],
            "tower": r[5],
            "flat": r[6],
        }
        for r in cur.fetchall()
    ]
    cur.close()
    conn.close()
    return rows


def stratified_sample(rows: List[dict], n: int) -> List[dict]:
    import random

    random.seed(42)
    by_cat: Dict[str, List[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    total = len(rows)
    if total <= n:
        random.shuffle(rows)
        return rows

    sampled: List[dict] = []
    remainders = []
    for cat, cat_rows in by_cat.items():
        proportion = len(cat_rows) / total
        exact = n * proportion
        take = int(exact)
        remainders.append((exact - take, cat))
        cat_copy = cat_rows[:]
        random.shuffle(cat_copy)
        sampled.extend(cat_copy[:take])
        by_cat[cat] = cat_copy[take:]

    remaining = n - len(sampled)
    for _, cat in sorted(remainders, reverse=True):
        if remaining == 0:
            break
        if by_cat[cat]:
            sampled.append(by_cat[cat].pop(0))
            remaining -= 1

    if remaining > 0:
        pool = []
        for cat_rows in by_cat.values():
            pool.extend(cat_rows)
        random.shuffle(pool)
        sampled.extend(pool[:remaining])

    random.shuffle(sampled)
    return sampled[:n]


async def run_aria(rows: List[dict], concurrency: int) -> List[dict]:
    sem = asyncio.Semaphore(concurrency)
    out = []

    async def run_one(row: dict) -> dict:
        async with sem:
            result = await process_complaint(
                ticket_id=row["ticket_id"],
                complaint_title=row["complaint_title"],
                site_name=row["site_name"] or "unknown",
                tower=row["tower"] or "unknown",
                flat=row["flat"] or "unknown",
            )
            decision = result.get("routing_decision")
            aria_label = decision.ownership if decision else "FM"
            return {
                "ticket_id": row["ticket_id"],
                "complaint_text": row["complaint_title"],
                "category": row["category"],
                "crm_label": normalize_label(row["issue_type"]),
                "aria_label": aria_label,
                "aria_confidence": (decision.confidence if decision else "low"),
                "tier_used": result.get("tier"),
                "aria_reasoning": (decision.reasoning if decision else ""),
                "agreed": aria_label == normalize_label(row["issue_type"]),
                "total_tokens": result.get("total_tokens", 0),
            }

    tasks = [run_one(r) for r in rows]
    for coro in asyncio.as_completed(tasks):
        out.append(await coro)
    return out


@contextmanager
def aria_eval_overrides(no_context: bool, no_docs: bool):
    """
    Eval-only overrides:
    - no_context: bypass all context retrieval (DB + docs)
    - no_docs: keep DB context but disable doc retrieval
    """
    orig_graph_assemble = resolv_graph.assemble_context
    orig_ctx_assemble = context_assembler.assemble_context
    orig_retrieve = context_assembler.retrieve_for_complaint

    async def _empty_context(site_name: str, tower: str, flat: str, complaint_title: str = "", domain: str = "other"):
        return context_assembler.ContextPackage(
            site_name=site_name,
            tower=tower,
            flat=flat,
            flat_history=[],
            adjacent_history=[],
            building_pattern=[],
            adjacency_info={},
            retrieval_ms=0,
            rag_sources_used=[],
            audit_context=[],
            mom_context=[],
            rag_retrieval_ms=0,
        )

    def _no_docs_retrieve(*args, **kwargs):
        return [], [], [], 0

    try:
        if no_context:
            resolv_graph.assemble_context = _empty_context
            context_assembler.assemble_context = _empty_context
        elif no_docs:
            context_assembler.retrieve_for_complaint = _no_docs_retrieve
        yield
    finally:
        resolv_graph.assemble_context = orig_graph_assemble
        context_assembler.assemble_context = orig_ctx_assemble
        context_assembler.retrieve_for_complaint = orig_retrieve


def run_gpt4o_baseline(rows: List[dict]) -> List[dict]:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "openai package is required for GPT-4o baseline. Install with: python3 -m pip install openai"
        ) from e

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set for GPT-4o baseline.")

    client = OpenAI(api_key=key)
    output = []
    for row in rows:
        prompt = (
            "Classify this Indian residential maintenance complaint as FM or Project.\n"
            "Return exactly one word: FM or Project.\n"
            f"Category: {row['category']}\n"
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
        crm = normalize_label(row["issue_type"])
        output.append(
            {
                "ticket_id": row["ticket_id"],
                "complaint_text": row["complaint_title"],
                "category": row["category"],
                "crm_label": crm,
                "gpt4o_label": label,
                "agreed": label == crm,
                "raw_response": text,
            }
        )
    return output


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="resolv")
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--run", action="store_true", help="Actually run ARIA + GPT-4o after showing estimate.")
    parser.add_argument(
        "--gpt-only",
        action="store_true",
        help="Run only GPT-4o baseline on sampled complaints (skip ARIA entirely).",
    )
    parser.add_argument(
        "--paper-mode",
        action="store_true",
        help="Evaluation-only model override: keep Groq for Tier2/hypothesis/judge and use Haiku arbiter.",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Eval mode: disable all context retrieval (no flat history, no adjacency, no docs).",
    )
    parser.add_argument(
        "--no-docs",
        action="store_true",
        help="Eval mode: keep complaint history context but disable operational document retrieval.",
    )
    args = parser.parse_args()
    if args.no_context and args.no_docs:
        raise RuntimeError("Use only one of --no-context or --no-docs.")

    rows = fetch_ambiguous_rows(args.db)
    sampled = stratified_sample(rows, args.sample)

    print(f"Ambiguous subset size in DB: {len(rows)}")
    print(f"Sampled complaints: {len(sampled)}")
    cat_dist = Counter(r["category"] for r in sampled)
    print(f"Sample category distribution: {dict(cat_dist)}")

    # Budget estimate heuristic requested by user.
    low_est, high_est = 8, 12
    print(f"Estimated cost for {len(sampled)} complaints: ~${low_est}-${high_est} (depends on tier mix and token length).")
    if args.gpt_only:
        print("Mode: --gpt-only (ARIA skipped; OpenAI baseline only).")
    if not args.run and not args.gpt_only:
        print("Dry run only. Re-run with --run to execute full batch.")
        return

    if not args.gpt_only:
        print("Proceed? [y/N] ", end="")
        if input().strip().lower() != "y":
            print("Aborted.")
            return

    original_cfg = None
    if args.paper_mode:
        original_cfg = {
            k: v.copy() if isinstance(v, dict) else v
            for k, v in MODEL_CONFIG.items()
        }
        # Evaluation-only override; production config file is unchanged.
        MODEL_CONFIG["tier2_reasoning"]["provider"] = "groq"
        MODEL_CONFIG["tier2_reasoning"]["model"] = "llama-3.3-70b-versatile"
        MODEL_CONFIG["hypothesis_agents"]["provider"] = "groq"
        MODEL_CONFIG["hypothesis_agents"]["model"] = "llama-3.3-70b-versatile"
        MODEL_CONFIG["arbiter"]["provider"] = "anthropic"
        MODEL_CONFIG["arbiter"]["model"] = "claude-haiku-4-5-20251001"
        MODEL_CONFIG["judge"]["provider"] = "groq"
        MODEL_CONFIG["judge"]["model"] = "llama-3.1-8b-instant"
        print(
            "Paper mode model override active: "
            "Tier2=Groq llama-3.3-70b-versatile, "
            "Hypothesis=Groq llama-3.3-70b-versatile, "
            "Arbiter=Anthropic claude-haiku-4-5-20251001, "
            "Judge=Groq llama-3.1-8b-instant"
        )
        print("Paper mode estimated cost target: ~80% lower than Sonnet/Opus-heavy routing.")
    if args.no_context:
        print("Context mode override active: --no-context (DB context + docs disabled).")
    if args.no_docs:
        print("Context mode override active: --no-docs (DB history on, docs off).")

    aria_rows = []
    baseline_rows = []
    if not args.gpt_only:
        with aria_eval_overrides(args.no_context, args.no_docs):
            aria_rows = asyncio.run(run_aria(sampled, concurrency=args.concurrency))
    baseline_rows = run_gpt4o_baseline(sampled)

    out_dir = Path("eval/results")
    aria_csv = out_dir / f"paper_eval_ambiguous_{len(sampled)}.csv"
    base_csv = out_dir / f"paper_eval_baseline_{len(sampled)}.csv"
    if not args.gpt_only:
        write_csv(aria_csv, aria_rows)
    write_csv(base_csv, baseline_rows)

    aria_agree = sum(1 for r in aria_rows if r["agreed"]) if aria_rows else 0
    base_agree = sum(1 for r in baseline_rows if r["agreed"])
    tier_dist = Counter(r["tier_used"] for r in aria_rows) if aria_rows else Counter()
    total_tokens = sum(int(r.get("total_tokens", 0) or 0) for r in aria_rows) if aria_rows else 0

    print("\nSUMMARY")
    if aria_rows:
        print(f"ARIA vs CRM agreement rate: {aria_agree}/{len(aria_rows)} = {aria_agree/len(aria_rows)*100:.2f}%")
        project_rows = [r for r in aria_rows if r["crm_label"] == "Project"]
        project_right = sum(1 for r in project_rows if r["aria_label"] == "Project")
        high_cost_errors = sum(1 for r in project_rows if r["aria_label"] == "FM")
        project_acc = (project_right / len(project_rows) * 100) if project_rows else 0.0
        print(f"ARIA Project accuracy: {project_right}/{len(project_rows)} = {project_acc:.2f}%")
        print(f"ARIA high-cost errors (CRM=Project, ARIA=FM): {high_cost_errors}")
    else:
        print("ARIA vs CRM agreement rate: skipped (--gpt-only)")
    print(f"GPT-4o vs CRM agreement rate: {base_agree}/{len(baseline_rows)} = {base_agree/len(baseline_rows)*100:.2f}%")
    if aria_rows:
        print(f"Tier distribution: {dict(tier_dist)}")
        print(f"ARIA total tokens: {total_tokens:,}")
        print(f"Saved ARIA CSV: {aria_csv}")
    else:
        print("Tier distribution: skipped (--gpt-only)")
        print("ARIA total tokens: skipped (--gpt-only)")
    print(f"Saved baseline CSV: {base_csv}")

    if original_cfg is not None:
        for k, v in original_cfg.items():
            MODEL_CONFIG[k] = v


if __name__ == "__main__":
    main()
