#!/usr/bin/env python3
"""
Update paper numbers from full-scale eval JSON.

Usage:
    python3 scripts/update_paper_numbers.py eval/results/full_ambiguous_eval_9259.json

Reads the eval JSON, computes normalized-per-300 metrics, and prints:
  1. All numbers to plug into both ARIA_paper_draft_v2.md and ARIA_paper.tex
  2. Exact sed commands / instructions for each substitution

Also optionally writes the patched .md and .tex files in-place if --apply is passed.
"""

import argparse
import json
import sys
from pathlib import Path


PAPER_DIRS = [
    Path(__file__).parent.parent,
    Path(__file__).parent.parent.parent / "vinod-test",
]
DRAFT_FILENAME = "ARIA_paper_draft_v2.md"
TEX_FILENAME = "ARIA_paper.tex"


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_metrics(data: dict) -> dict:
    n = data["sample_size"]
    aria = data["aria"]
    gpt4o = data["gpt4o"]
    cr = data["cost_reduction"]
    ci = cr.get("bootstrap_ci") or {}

    scale = 300 / n

    aria_hc_300 = round(aria["high_cost_errors"] * scale)
    aria_lc_300 = round(aria["low_cost_errors"] * scale)
    aria_wtd_300 = round(aria["raw_expected_cost"] * scale)

    gpt_hc_300 = round((gpt4o["raw_expected_cost"] / 10) * scale)  # HC errors = cost/10 since each HC=10
    # Can't decompose GPT HC/LC separately from raw cost alone; use per-class to recompute
    # HC errors for GPT-4o = Project rows predicted as FM
    # We don't have that split in the JSON, so estimate from overall cost:
    # gpt_raw_cost = HC_errors * 10 + LC_errors * 1
    # gpt_proj_acc tells us Project rows right; we know Project proportion ~35% of 9259
    # Best we can do: report raw cost and note HC/LC breakdown unavailable
    gpt_wtd_300 = round(gpt4o["raw_expected_cost"] * scale)

    point = cr.get("point_estimate_pct") or 0.0
    ci_lo = ci.get("ci_95_lo", 0.0)
    ci_hi = ci.get("ci_95_hi", 0.0)
    ci_str = f"[{ci_lo:.1f}%, {ci_hi:.1f}%]" if ci else "N/A"

    return {
        "n": n,
        "scale": scale,
        # ARIA
        "aria_overall_acc": aria["overall_accuracy"],
        "aria_fm_acc": aria["fm_accuracy"],
        "aria_proj_acc": aria["project_accuracy"],
        "aria_hc_300": aria_hc_300,
        "aria_lc_300": aria_lc_300,
        "aria_wtd_300": aria_wtd_300,
        # GPT-4o
        "gpt_overall_acc": gpt4o["overall_accuracy"],
        "gpt_fm_acc": gpt4o["fm_accuracy"],
        "gpt_proj_acc": gpt4o["project_accuracy"],
        "gpt_wtd_300": gpt_wtd_300,
        # Cost reduction
        "cost_reduction_pct": round(point, 1),
        "ci_str": ci_str,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        # Tier distribution
        "tier_dist": aria.get("tier_distribution", {}),
    }


def print_summary(m: dict) -> None:
    print("=" * 60)
    print(f"EVAL RESULTS (n={m['n']:,})")
    print("=" * 60)
    print(f"\nARIA:")
    print(f"  Overall acc:    {m['aria_overall_acc']:.1f}%  (was 59.0%)")
    print(f"  FM accuracy:    {m['aria_fm_acc']:.1f}%  (was 72.5%)")
    print(f"  Project acc:    {m['aria_proj_acc']:.1f}%  (was 39.3%)")
    print(f"  HC/300:         {m['aria_hc_300']}        (was 74)")
    print(f"  LC/300:         {m['aria_lc_300']}        (was 49)")
    print(f"  Wtd/300:        {m['aria_wtd_300']}       (was 789)")
    print(f"\nGPT-4o text-only:")
    print(f"  Overall acc:    {m['gpt_overall_acc']:.1f}%  (was 59.0%)")
    print(f"  FM accuracy:    {m['gpt_fm_acc']:.1f}%  (was 91.6%)")
    print(f"  Project acc:    {m['gpt_proj_acc']:.1f}%  (was 12.3%)")
    print(f"  Wtd/300:        {m['gpt_wtd_300']}       (was 1085)")
    print(f"\nCost reduction:  {m['cost_reduction_pct']:.1f}%  (was 27.3%)")
    print(f"Bootstrap 95%CI: {m['ci_str']}")
    print(f"Tier dist:       {m['tier_dist']}")


def apply_to_md(draft_path: Path, m: dict) -> None:
    text = draft_path.read_text()

    # Replace [PENDING] markers
    text = text.replace(
        "[FULL DATASET EVAL PENDING: n=9,259]",
        f"(n={m['n']:,}, bootstrap 95% CI {m['ci_str']})"
    )
    text = text.replace(
        "[FULL DATASET EVAL PENDING: n=9,259 with bootstrap 95% CI on cost reduction.]",
        f"Full-dataset eval n={m['n']:,}; bootstrap 95% CI on cost reduction: {m['ci_str']}."
    )
    text = text.replace(
        "Primary eval on 300-complaint stratified sample from 9,727 ambiguous cases. [FULL DATASET EVAL PENDING: n=9,259 with bootstrap 95% CI on cost reduction.]",
        f"Full-dataset eval on {m['n']:,}-complaint stratified sample from 9,727 ambiguous cases. Bootstrap 95% CI on cost reduction: {m['ci_str']}."
    )
    text = text.replace(
        "[RESULTS PENDING]",
        ""
    )

    # Methodology section: replace 300-sample reference
    text = text.replace(
        "**Signal 1 (primary):** ARIA and GPT-4o variants process a stratified 300-complaint sample from the ambiguous subset; agreement with CRM labels computed. [FULL DATASET EVAL PENDING: n=9,259]",
        f"**Signal 1 (primary):** ARIA and GPT-4o variants processed a stratified {m['n']:,}-complaint sample from the ambiguous subset; agreement with CRM labels computed."
    )

    # Core numbers — replace old → new
    replacements = [
        # ARIA
        ("72.5% FM accuracy", f"{m['aria_fm_acc']:.1f}% FM accuracy"),
        ("72.5%** FM accuracy", f"{m['aria_fm_acc']:.1f}%** FM accuracy"),
        ("39.3%** Project accuracy", f"{m['aria_proj_acc']:.1f}%** Project accuracy"),
        ("39.3% Project accuracy", f"{m['aria_proj_acc']:.1f}% Project accuracy"),
        ("39.3%** |", f"{m['aria_proj_acc']:.1f}%** |"),
        ("72.5%** |", f"{m['aria_fm_acc']:.1f}%** |"),
        ("**74**", f"**{m['aria_hc_300']}**"),
        ("**49**", f"**{m['aria_lc_300']}**"),
        ("**789**", f"**{m['aria_wtd_300']}**"),
        ("**−27.3%**", f"**−{m['cost_reduction_pct']:.1f}%**"),
        ("27.3% cost reduction", f"{m['cost_reduction_pct']:.1f}% cost reduction"),
        ("27.3% with zero", f"{m['cost_reduction_pct']:.1f}% with zero"),
        ("ARIA's 27.3%", f"ARIA's {m['cost_reduction_pct']:.1f}%"),
        ("cost by 27.3%", f"cost by {m['cost_reduction_pct']:.1f}%"),
        ("cost 27.3%", f"cost {m['cost_reduction_pct']:.1f}%"),
        # GPT-4o
        ("91.6% FM accuracy", f"{m['gpt_fm_acc']:.1f}% FM accuracy"),
        ("91.6%** accuracy", f"{m['gpt_fm_acc']:.1f}%** accuracy"),
        ("91.6% accuracy", f"{m['gpt_fm_acc']:.1f}% accuracy"),
        ("12.3% Project accuracy", f"{m['gpt_proj_acc']:.1f}% Project accuracy"),
        ("12.3%** |", f"{m['gpt_proj_acc']:.1f}%** |"),
        ("only 12.3%", f"only {m['gpt_proj_acc']:.1f}%"),
        ("but only 11.5%", f"but only {m['gpt_proj_acc']:.1f}%"),  # abstract uses 11.5%
        ("107 & 15 & 1,085 &", f"N/A & N/A & {m['gpt_wtd_300']} &"),  # GPT HC/LC not in JSON
    ]
    for old, new in replacements:
        if old != new:
            text = text.replace(old, new)

    # Add CI to abstract
    ci_note = f" (95% CI: {m['ci_str']})" if m['ci_str'] != "N/A" else ""
    text = text.replace(
        f"{m['cost_reduction_pct']:.1f}% cost reduction over GPT-4o while maintaining",
        f"{m['cost_reduction_pct']:.1f}% cost reduction{ci_note} over GPT-4o while maintaining",
    )

    draft_path.write_text(text)
    print(f"  Patched: {draft_path}")


def apply_to_tex(tex_path: Path, m: dict) -> None:
    text = tex_path.read_text()

    replacements_tex = [
        ("72.5\\%", f"{m['aria_fm_acc']:.1f}\\%"),
        ("39.3\\%", f"{m['aria_proj_acc']:.1f}\\%"),
        ("91.6\\%", f"{m['gpt_fm_acc']:.1f}\\%"),
        ("12.3\\%", f"{m['gpt_proj_acc']:.1f}\\%"),
        ("27.3\\%", f"{m['cost_reduction_pct']:.1f}\\%"),
        ("−27.3\\%", f"−{m['cost_reduction_pct']:.1f}\\%"),
        ("& 107 & 15 & 1,085 &", f"& N/A & N/A & {m['gpt_wtd_300']} &"),
        ("& 74} & 49} & 789}", f"& {m['aria_hc_300']}" + "} & " + f"{m['aria_lc_300']}" + "} & " + f"{m['aria_wtd_300']}" + "}"),
    ]
    for old, new in replacements_tex:
        if old != new:
            text = text.replace(old, new)

    tex_path.write_text(text)
    print(f"  Patched: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", help="Path to eval results JSON (e.g. eval/results/full_ambiguous_eval_9259.json)")
    parser.add_argument("--apply", action="store_true", help="Write changes to paper files in-place (both resolv/ and vinod-test/)")
    args = parser.parse_args()

    data = load_json(args.json_path)
    m = compute_metrics(data)
    print_summary(m)

    if args.apply:
        print("\nApplying changes...")
        for base in PAPER_DIRS:
            md = base / DRAFT_FILENAME
            tex = base / TEX_FILENAME
            if md.exists():
                apply_to_md(md, m)
            if tex.exists():
                apply_to_tex(tex, m)
        print("\nDone. Rebuild PDF:")
        print("  cd <latex-build-dir> && pdflatex ARIA_paper.tex && pdflatex ARIA_paper.tex")
    else:
        print("\nDry run. Pass --apply to write changes to paper files.")
        print("  python3 scripts/update_paper_numbers.py eval/results/full_ambiguous_eval_9259.json --apply")


if __name__ == "__main__":
    main()
