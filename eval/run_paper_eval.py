import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

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
import json
import os
import random
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

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


async def run_aria(rows: List[dict], concurrency: int, chunk_size: int = 500, checkpoint_path: str = "") -> List[dict]:
    """Process in chunks to avoid memory/event-loop issues at large scale.

    If checkpoint_path is set, saves progress after each chunk so runs can be resumed.
    On restart with the same checkpoint_path, already-processed tickets are skipped.
    """
    # Load checkpoint if it exists
    all_out = []
    done_ids: set = set()
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_out = list(reader)
        done_ids = {r["ticket_id"] for r in all_out}
        print(f"  Resuming from checkpoint: {len(done_ids)} tickets already done", flush=True)

    remaining = [r for r in rows if str(r["ticket_id"]) not in done_ids]
    total = len(rows)
    already_done = len(done_ids)

    for chunk_start in range(0, len(remaining), chunk_size):
        chunk = remaining[chunk_start: chunk_start + chunk_size]
        abs_start = already_done + chunk_start + 1
        abs_end = min(already_done + chunk_start + len(chunk), total)
        print(f"  ARIA chunk {abs_start}–{abs_end}/{total}...", flush=True)
        chunk_out = await _run_aria_chunk(chunk, concurrency)
        all_out.extend(chunk_out)
        if checkpoint_path:
            write_csv(Path(checkpoint_path), all_out)
            print(f"  Checkpoint saved ({len(all_out)} rows): {checkpoint_path}", flush=True)
    return all_out


async def _run_aria_chunk(rows: List[dict], concurrency: int) -> List[dict]:
    sem = asyncio.Semaphore(concurrency)
    out = []

    async def run_one(row: dict) -> dict:
        async with sem:
            try:
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
            except Exception as e:
                print(f"  [WARN] ticket {row['ticket_id']} failed: {e}", flush=True)
                crm = normalize_label(row["issue_type"])
                return {
                    "ticket_id": row["ticket_id"],
                    "complaint_text": row["complaint_title"],
                    "category": row["category"],
                    "crm_label": crm,
                    "aria_label": "FM",
                    "aria_confidence": "error",
                    "tier_used": None,
                    "aria_reasoning": f"ERROR: {e}",
                    "agreed": crm == "FM",
                    "total_tokens": 0,
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


def run_gpt4o_baseline(rows: List[dict], checkpoint_path: str = "") -> List[dict]:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "openai package is required for GPT-4o baseline. Install with: python3 -m pip install openai"
        ) from e

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set for GPT-4o baseline.")

    # Load checkpoint if it exists
    output = []
    done_ids: set = set()
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path, newline="", encoding="utf-8") as f:
            output = list(csv.DictReader(f))
        done_ids = {r["ticket_id"] for r in output}
        print(f"  GPT-4o resuming from checkpoint: {len(done_ids)} already done", flush=True)

    client = OpenAI(api_key=key)
    total = len(rows)
    save_every = 500
    for i, row in enumerate(rows):
        if str(row["ticket_id"]) in done_ids:
            continue
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
        if checkpoint_path and len(output) % save_every == 0:
            write_csv(Path(checkpoint_path), output)
            print(f"  GPT-4o checkpoint saved ({len(output)}/{total})", flush=True)
    if checkpoint_path and output:
        write_csv(Path(checkpoint_path), output)
    return output


def expected_cost(aria_rows: List[dict], baseline_rows: List[dict]) -> Tuple[float, float]:
    """
    Asymmetric cost model from paper:
      - Miss Project (CRM=Project, pred=FM): cost 10
      - False alarm (CRM=FM, pred=Project): cost 1
    Returns (aria_cost, gpt4o_cost) as raw sums.
    Baseline rows keyed by ticket_id for pairing.
    """
    base_by_id = {r["ticket_id"]: r for r in baseline_rows}

    def row_cost(crm: str, pred: str) -> int:
        if crm == "Project" and pred == "FM":
            return 10
        if crm == "FM" and pred == "Project":
            return 1
        return 0

    aria_c = sum(row_cost(r["crm_label"], r["aria_label"]) for r in aria_rows)
    gpt_c = sum(
        row_cost(r["crm_label"], base_by_id[r["ticket_id"]]["gpt4o_label"])
        for r in aria_rows
        if r["ticket_id"] in base_by_id
    )
    return aria_c, gpt_c


def bootstrap_cost_ci(
    aria_rows: List[dict], baseline_rows: List[dict], n_boot: int = 1000
) -> dict:
    """
    Bootstrap 95% CI on cost reduction % (ARIA vs GPT-4o).
    Cost reduction = (gpt_cost - aria_cost) / gpt_cost * 100
    """
    base_by_id = {r["ticket_id"]: r for r in baseline_rows}
    paired = [
        (r, base_by_id[r["ticket_id"]])
        for r in aria_rows
        if r["ticket_id"] in base_by_id
    ]
    n = len(paired)

    def row_cost(crm: str, pred: str) -> int:
        if crm == "Project" and pred == "FM":
            return 10
        if crm == "FM" and pred == "Project":
            return 1
        return 0

    def compute_reduction(sample: list) -> float:
        aria_c = sum(row_cost(a["crm_label"], a["aria_label"]) for a, _ in sample)
        gpt_c = sum(row_cost(a["crm_label"], b["gpt4o_label"]) for a, b in sample)
        if gpt_c == 0:
            return 0.0
        return (gpt_c - aria_c) / gpt_c * 100

    rng = random.Random(42)
    reductions = []
    for _ in range(n_boot):
        sample = [paired[rng.randrange(n)] for _ in range(n)]
        reductions.append(compute_reduction(sample))

    reductions.sort()
    point = compute_reduction(paired)
    lo = reductions[int(0.025 * n_boot)]
    hi = reductions[int(0.975 * n_boot)]
    return {
        "point_estimate": round(point, 2),
        "ci_95_lo": round(lo, 2),
        "ci_95_hi": round(hi, 2),
        "n_boot": n_boot,
        "n_paired": n,
    }


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
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt (for non-interactive/background runs).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="N",
        help="Compute bootstrap 95%% CI on cost reduction using N resamples (e.g. 1000).",
    )
    parser.add_argument(
        "--save-json",
        metavar="PATH",
        default="",
        help="Save full summary (metrics + CI) as JSON to this path.",
    )
    parser.add_argument(
        "--checkpoint",
        metavar="PATH",
        default="",
        help="Save ARIA progress CSV after each chunk; resume from this file if it already exists.",
    )
    parser.add_argument(
        "--gpt-checkpoint",
        metavar="PATH",
        default="",
        help="Save GPT-4o baseline progress CSV every 500 rows; resume from this file if it already exists.",
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

    if not args.gpt_only and not args.yes:
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
            aria_rows = asyncio.run(run_aria(sampled, concurrency=args.concurrency, checkpoint_path=args.checkpoint))
    baseline_rows = run_gpt4o_baseline(sampled, checkpoint_path=args.gpt_checkpoint)

    out_dir = Path("eval/results")
    aria_csv = out_dir / f"paper_eval_ambiguous_{len(sampled)}.csv"
    base_csv = out_dir / f"paper_eval_baseline_{len(sampled)}.csv"
    if not args.gpt_only:
        write_csv(aria_csv, aria_rows)
    write_csv(base_csv, baseline_rows)

    # Recompute agreement from labels (handles "True"/"False" strings if rows loaded from CSV checkpoint)
    aria_agree = sum(1 for r in aria_rows if r["aria_label"] == r["crm_label"]) if aria_rows else 0
    base_agree = sum(1 for r in baseline_rows if r["gpt4o_label"] == r["crm_label"])
    tier_dist = Counter(r["tier_used"] for r in aria_rows) if aria_rows else Counter()
    total_tokens = sum(int(r.get("total_tokens", 0) or 0) for r in aria_rows) if aria_rows else 0

    # Per-class accuracy
    project_rows = [r for r in aria_rows if r["crm_label"] == "Project"] if aria_rows else []
    fm_rows = [r for r in aria_rows if r["crm_label"] == "FM"] if aria_rows else []
    project_right = sum(1 for r in project_rows if r["aria_label"] == "Project")
    fm_right = sum(1 for r in fm_rows if r["aria_label"] == "FM")
    high_cost_errors = sum(1 for r in project_rows if r["aria_label"] == "FM")
    low_cost_errors = sum(1 for r in fm_rows if r["aria_label"] == "Project")
    project_acc = (project_right / len(project_rows) * 100) if project_rows else 0.0
    fm_acc = (fm_right / len(fm_rows) * 100) if fm_rows else 0.0

    # GPT-4o per-class
    gpt_project_rows = [r for r in baseline_rows if r["crm_label"] == "Project"]
    gpt_fm_rows = [r for r in baseline_rows if r["crm_label"] == "FM"]
    gpt_proj_right = sum(1 for r in gpt_project_rows if r["gpt4o_label"] == "Project")
    gpt_fm_right = sum(1 for r in gpt_fm_rows if r["gpt4o_label"] == "FM")
    gpt_proj_acc = (gpt_proj_right / len(gpt_project_rows) * 100) if gpt_project_rows else 0.0
    gpt_fm_acc = (gpt_fm_right / len(gpt_fm_rows) * 100) if gpt_fm_rows else 0.0

    # Expected cost (raw)
    aria_cost_raw, gpt_cost_raw = (0, 0)
    if aria_rows and baseline_rows:
        aria_cost_raw, gpt_cost_raw = expected_cost(aria_rows, baseline_rows)

    print("\nSUMMARY")
    if aria_rows:
        print(f"ARIA vs CRM agreement rate: {aria_agree}/{len(aria_rows)} = {aria_agree/len(aria_rows)*100:.2f}%")
        print(f"ARIA Project accuracy: {project_right}/{len(project_rows)} = {project_acc:.2f}%")
        print(f"ARIA FM accuracy: {fm_right}/{len(fm_rows)} = {fm_acc:.2f}%")
        print(f"ARIA high-cost errors (CRM=Project, ARIA=FM): {high_cost_errors}")
        print(f"ARIA low-cost errors (CRM=FM, ARIA=Project): {low_cost_errors}")
        print(f"ARIA raw expected cost: {aria_cost_raw}")
    else:
        print("ARIA vs CRM agreement rate: skipped (--gpt-only)")
    print(f"GPT-4o vs CRM agreement rate: {base_agree}/{len(baseline_rows)} = {base_agree/len(baseline_rows)*100:.2f}%")
    print(f"GPT-4o Project accuracy: {gpt_proj_right}/{len(gpt_project_rows)} = {gpt_proj_acc:.2f}%")
    print(f"GPT-4o FM accuracy: {gpt_fm_right}/{len(gpt_fm_rows)} = {gpt_fm_acc:.2f}%")
    if aria_rows:
        print(f"GPT-4o raw expected cost: {gpt_cost_raw}")
        if gpt_cost_raw > 0:
            cost_red = (gpt_cost_raw - aria_cost_raw) / gpt_cost_raw * 100
            print(f"Point cost reduction (ARIA vs GPT-4o): {cost_red:.1f}%")

    if aria_rows:
        print(f"Tier distribution: {dict(tier_dist)}")
        print(f"ARIA total tokens: {total_tokens:,}")
        print(f"Saved ARIA CSV: {aria_csv}")
    else:
        print("Tier distribution: skipped (--gpt-only)")
        print("ARIA total tokens: skipped (--gpt-only)")
    print(f"Saved baseline CSV: {base_csv}")

    # Bootstrap CI
    ci_result = {}
    if args.bootstrap > 0 and aria_rows and baseline_rows:
        print(f"\nRunning bootstrap CI (n={args.bootstrap})...")
        ci_result = bootstrap_cost_ci(aria_rows, baseline_rows, args.bootstrap)
        print(
            f"Cost reduction 95% CI: {ci_result['point_estimate']:.1f}% "
            f"[{ci_result['ci_95_lo']:.1f}%, {ci_result['ci_95_hi']:.1f}%]"
        )

    # JSON summary
    if args.save_json:
        summary = {
            "sample_size": len(sampled),
            "aria": {
                "overall_accuracy": round(aria_agree / len(aria_rows) * 100, 2) if aria_rows else None,
                "project_accuracy": round(project_acc, 2),
                "fm_accuracy": round(fm_acc, 2),
                "high_cost_errors": high_cost_errors,
                "low_cost_errors": low_cost_errors,
                "raw_expected_cost": aria_cost_raw,
                "tier_distribution": dict(tier_dist),
                "total_tokens": total_tokens,
            } if aria_rows else None,
            "gpt4o": {
                "overall_accuracy": round(base_agree / len(baseline_rows) * 100, 2),
                "project_accuracy": round(gpt_proj_acc, 2),
                "fm_accuracy": round(gpt_fm_acc, 2),
                "raw_expected_cost": gpt_cost_raw,
            },
            "cost_reduction": {
                "point_estimate_pct": round((gpt_cost_raw - aria_cost_raw) / gpt_cost_raw * 100, 2) if gpt_cost_raw > 0 and aria_rows else None,
                "bootstrap_ci": ci_result if ci_result else None,
            },
        }
        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved JSON summary: {json_path}")

    if original_cfg is not None:
        for k, v in original_cfg.items():
            MODEL_CONFIG[k] = v


if __name__ == "__main__":
    main()
