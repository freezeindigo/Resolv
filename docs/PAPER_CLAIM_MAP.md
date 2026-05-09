# Paper Claim Map — ARIA / CIKM '26

Every quantitative claim from the paper, mapped to the code path that produces it.
NDA-only claims are flagged; stored results are referenced where available.

---

## §1 Introduction / Abstract

| Claim | Value | Code path | Public? |
|---|---|---|---|
| Production complaint corpus size | 17,098 complaints | `scripts/load_complaints_xlsx.py` ingests `data/godrej_complaints.xlsx` | NDA-only |
| Ambiguous subset evaluated | n = 9,259 | `eval/run_paper_eval.py:fetch_ambiguous_rows()`, stored in `full_ambiguous_eval_9259.json` | NDA-only |
| Cost reduction vs GPT-4o | 22.7% | `eval/run_paper_eval.py:expected_cost()`, `bootstrap_cost_ci()` | NDA-only (stored result: `eval/results/full_ambiguous_eval_9259.json`) |
| 95% CI on cost reduction | [20.6%, 24.7%] | `eval/run_paper_eval.py:bootstrap_cost_ci()`, seed=42, n_boot=1000 | NDA-only (stored) |

---

## §2 Problem Formulation / Cost Model

| Claim | Value | Code path | Public? |
|---|---|---|---|
| Asymmetric cost ratio | FM:Project = 1:10 | `data/cost_matrix.yaml`, `eval/run_paper_eval.py:row_cost()` | Yes |
| High-cost error definition | CRM=Project, pred=FM → cost 10 | `eval/run_paper_eval.py:row_cost()` | Yes |
| Low-cost error definition | CRM=FM, pred=Project → cost 1 | `eval/run_paper_eval.py:row_cost()` | Yes |
| Outcome asymmetry (TAT) | Project: 71-day median, FM: 10-day median | `eval/resolution_time_validation.py` → `eval/results/resolution_time_validation.json` | NDA-only |
| 4.3× outcome asymmetry | 71 / 10 = 7.1× raw; 4.3× after normalization | `eval/resolution_time_validation.py` | NDA-only |

---

## §3 System Design

| Claim | Value | Code path | Public? |
|---|---|---|---|
| 3-tier pipeline | Deterministic → single-pass → multi-agent | `src/pipeline/resolv_graph.py:build_graph()` | Yes |
| 24+ hypothesis agents | 27 defined across 7 domains | `src/config/hypothesis_library.yaml` (27 `id:` entries) | Yes |
| Evidence isolation per hypothesis | Each agent filtered to its `evidence_filter` | `src/agents/hypothesis_agent.py` | Yes |
| Cost-weighted arbiter | Multiplies likelihood × `cost_of_error_weight` | `src/agents/arbiter.py` | Yes |
| Groq llama-3.3-70b for Tier 2 + hypothesis agents | Model ID | `src/config/model_config.py`, `--paper-mode` in `eval/run_paper_eval.py` | Yes |
| Anthropic claude-haiku for arbiter (paper eval) | Model ID | `eval/run_paper_eval.py:--paper-mode` | Yes |

---

## §4 Experimental Setup

| Claim | Value | Code path | Public? |
|---|---|---|---|
| Ambiguous category list (6) | Leakage, Seepage, Civil Work, Carpentry, Plumbing, Mason/Civil | `eval/run_paper_eval.py:AMBIGUOUS_CATEGORIES` | Yes |
| Leakage — % Project | 46.7% | Derived from `data/godrej_complaints.xlsx` via DB | NDA-only |
| Seepage — % Project | 47.7% | Same | NDA-only |
| Civil Work — % Project | 27.8% | Same | NDA-only |
| Carpentry — % Project | 33.1% | Same | NDA-only |
| Plumbing — % Project | 23.5% | Same | NDA-only |
| Mason/Civil/Other — % Project | 45–47% | Same | NDA-only |
| Overall FM/Project split | 59.3% FM / 40.7% Project | `eval/results/full_ambiguous_eval_9259.json` (n_fm/n_proj counts) | NDA-only (stored) |
| Stratified sample seed | 42 | `eval/run_paper_eval.py:stratified_sample()` | Yes |
| Bootstrap seed | 42 | `eval/run_paper_eval.py:bootstrap_cost_ci()` | Yes |
| Bootstrap resamples | 1,000 | `--bootstrap 1000` | Yes |

---

## §5 Results — Headline numbers

| Claim | Value | Code path | Public? |
|---|---|---|---|
| ARIA overall accuracy | 60.0% | `eval/results/full_ambiguous_eval_9259.json:aria.overall_accuracy` | NDA-only (stored) |
| ARIA FM accuracy | 75.5% (75.45% precise) | `eval/results/full_ambiguous_eval_9259.json:aria.fm_accuracy` | NDA-only (stored) |
| ARIA Project accuracy | 35.8% (35.76% precise) | `eval/results/full_ambiguous_eval_9259.json:aria.project_accuracy` | NDA-only (stored) |
| ARIA high-cost errors | 2,316 | `eval/results/full_ambiguous_eval_9259.json:aria.high_cost_errors` | NDA-only (stored) |
| ARIA low-cost errors | 1,388 | `eval/results/full_ambiguous_eval_9259.json:aria.low_cost_errors` | NDA-only (stored) |
| ARIA raw expected cost | 24,548 | `eval/results/full_ambiguous_eval_9259.json:aria.raw_expected_cost` | NDA-only (stored) |
| GPT-4o overall accuracy | 61.1% (61.08% precise) | `eval/results/full_ambiguous_eval_9259.json:gpt4o.overall_accuracy` | NDA-only (stored) |
| GPT-4o FM accuracy | 91.6% | `eval/results/full_ambiguous_eval_9259.json:gpt4o.fm_accuracy` | NDA-only (stored) |
| GPT-4o Project accuracy | 13.2% | `eval/results/full_ambiguous_eval_9259.json:gpt4o.project_accuracy` | NDA-only (stored) |
| GPT-4o raw expected cost | 31,765 | `eval/results/full_ambiguous_eval_9259.json:gpt4o.raw_expected_cost` | NDA-only (stored) |
| Cost reduction point estimate | 22.7% (22.72% precise) | `eval/results/full_ambiguous_eval_9259.json:cost_reduction.point_estimate_pct` | NDA-only (stored) |
| Cost reduction 95% CI lower | 20.6% (20.61%) | `eval/results/full_ambiguous_eval_9259.json:cost_reduction.bootstrap_ci.ci_95_lo` | NDA-only (stored) |
| Cost reduction 95% CI upper | 24.7% (24.71%) | `eval/results/full_ambiguous_eval_9259.json:cost_reduction.bootstrap_ci.ci_95_hi` | NDA-only (stored) |

---

## §5 Results — Isolation ablation (Table 3)

All stored in `eval/results/isolation_ablation_summary.json` (n=300 subset).

| Claim | ARIA-Full | ARIA-Pooled | ARIA-Single | Public? |
|---|---|---|---|---|
| Overall accuracy | 60.0% | 59.67% | 59.33% | Partial (stored 300-row result) |
| Project accuracy | 40.16% | 39.34% | 39.34% | Partial (stored) |
| FM accuracy | 73.6% | 73.6% | 73.03% | Partial (stored) |
| High-cost errors | 73 | 74 | 74 | Partial (stored) |
| Low-cost errors | 47 | 47 | 48 | Partial (stored) |
| Weighted cost | 777 | 787 | 788 | Partial (stored) |

To rerun ablation on public repo (requires 300-row eval CSV from proprietary corpus or synthetic stand-in):
```bash
python eval/isolation_ablation.py
```

---

## §5 Results — Cost sensitivity

Generated by `eval/cost_sensitivity.py` from stored 300-row CSVs.
Stored result: `eval/results/cost_sensitivity_results.json`.

Claim: ARIA maintains cost advantage over GPT-4o across all stress-test ratios from 2× to 50×.

---

## §6 Validation / Generalizability

| Claim | Code path | Public? |
|---|---|---|
| TAT validation (71-day vs 10-day) | `eval/resolution_time_validation.py` | NDA-only |
| CFPB second-domain results | `eval/cfpb_second_domain.py` | Yes (public CFPB dataset) |
| NYC311 second-domain results | `eval/nyc311_second_domain.py` | Yes (public NYC311 dataset) |

---

## How to verify stored results are authentic

The stored JSON files (`eval/results/*.json`) are unmodified outputs of the eval harness. To verify:

1. Check git log for `eval/results/full_ambiguous_eval_9259.json` — committed in the final paper eval run.
2. Cross-check numbers against the paper: every value in the JSON matches the paper to 2 decimal places (pre-rounding).
3. For reviewers requiring full corpus verification: contact via EasyChair to arrange NDA access and a screen-share session running `eval/run_paper_eval.py` on the production DB.
