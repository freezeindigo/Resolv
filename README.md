# Resolv / ARIA — Cost-Sensitive Complaint Routing under Asymmetric Risk

**ARIA** (Adaptive Routing via Inference Arbitration) is a cost-aware system for routing residential facility-management complaints to the correct responsible party — FM (Facility Management) or Project (developer / DLP). It combines deterministic rules, single-pass LLM reasoning, and a parallel multi-agent deliberative tier in which each hypothesis agent sees only evidence relevant to its hypothesis, and a cost-weighted arbiter integrates the results. Evaluated on 17,098 production complaints from a major Indian residential developer, ARIA achieves a **22.7% reduction in weighted routing cost** relative to a zero-shot GPT-4o baseline, with **75.5% FM accuracy** and **35.8% Project accuracy** on the ambiguous subset (n=9,259), under a 1:10 asymmetric cost model (missing a Project complaint costs 10× more than a false Project alarm).

---

## Paper

Companion to: **Kolanupaka, K. (2026). Cost-Sensitive Decision Making with LLMs under Asymmetric Risk: A Multi-Agent Approach to Complaint Routing. CIKM '26 Applied Research Track. [link TBD on acceptance]**

---

## Reproducibility map

| Paper Section | Claim | Code / File | Reproducible from public repo? |
|---|---|---|---|
| §2 Problem & Cost Model | FM:Project cost asymmetry = 1:10 | `data/cost_matrix.yaml`, `eval/run_paper_eval.py:341` | Yes |
| §2 Cost Model | Stress tests at 2×–50× | `eval/cost_sensitivity.py`, `data/cost_matrix.yaml` | Yes (synthetic data) |
| §3 System Design | 3-tier pipeline architecture | `src/pipeline/resolv_graph.py` | Yes |
| §3 System Design | 24+ hypothesis agents across 7 domains | `src/config/hypothesis_library.yaml`, `src/agents/prompts/` | Yes |
| §3 System Design | Evidence isolation per hypothesis agent | `src/agents/hypothesis_agent.py` | Yes |
| §4 Experiment Setup | Ambiguous category list (Leakage, Seepage, Civil Work, Carpentry, Plumbing, Mason/Civil) | `eval/run_paper_eval.py:AMBIGUOUS_CATEGORIES` | Yes |
| §4 Table 1 | Category-level FM/Project split percentages | `eval/run_paper_eval.py`, `scripts/run_eval.py` | Partial (distributions visible in synthetic sample; exact counts need full corpus) |
| §4 Eval | GPT-4o text-only baseline | `eval/run_paper_eval.py:run_gpt4o_baseline()` | Yes (needs OPENAI_API_KEY) |
| §5 Results | 22.7% cost reduction, 95% CI [20.6%, 24.7%] | `eval/results/full_ambiguous_eval_9259.json` | NDA-only (stored result; full corpus needed to rerun) |
| §5 Results | 75.5% FM accuracy, 35.8% Project accuracy | `eval/results/full_ambiguous_eval_9259.json` | NDA-only (stored result) |
| §5 Results | GPT-4o: 91.6% FM acc, 13.2% Project acc | `eval/results/full_ambiguous_eval_9259.json` | NDA-only (stored result) |
| §5 Table 3 | Isolation ablation (ARIA-Full vs Pooled vs Single-Prompt) | `eval/isolation_ablation.py`, `eval/results/isolation_ablation_summary.json` | Partial (300-row result stored; full rerun needs corpus) |
| §5 Figure | Cost sensitivity curve (2×–50× stress test) | `eval/cost_sensitivity.py`, `eval/results/cost_sensitivity_results.json` | Partial (stored; full rerun needs corpus) |
| §6 TAT validation | 71-day Project vs 10-day FM resolution time | `eval/resolution_time_validation.py`, `eval/results/resolution_time_validation.json` | NDA-only |
| §6 Generalizability | CFPB and NYC311 second-domain results | `eval/cfpb_second_domain.py`, `eval/nyc311_second_domain.py` | Yes (public datasets) |

---

## What's in the repo

```
src/
  pipeline/resolv_graph.py      LangGraph 3-tier pipeline (deterministic → single-pass → multi-agent)
  agents/hypothesis_agent.py    Parallel hypothesis agent spawner with evidence filtering
  agents/arbiter.py             Cost-weighted arbiter (integrates all hypothesis scores)
  agents/judge.py               Lightweight validation node (Tier 2 and 3)
  agents/llm_client.py          Provider-agnostic LLM wrapper (Groq / Anthropic)
  agents/prompts/               27 hypothesis prompts + arbiter, judge, pattern interpreter
  config/hypothesis_library.yaml  27 hypothesis types, 7 domains, cost-of-error weights
  config/model_config.py        Model routing: Groq llama-3.3-70b (bulk), Anthropic (arbiter)
  config/domain_rules.yaml      Keyword → domain classification rules
  config/tier_rules.yaml        Tier 1/2/3 assignment thresholds
  nodes/                        Deterministic pipeline nodes (classifier, assessor, context assembler)
  memory/pattern_state.py       Redis-backed DBSCAN cluster detection

data/
  cost_matrix.yaml              Asymmetric cost model (FM:Project = 1:10) with stress-test ratios
  hypothesis_library/           Symlink → src/config/hypothesis_library.yaml (YAML spec)
  rag_documents/                6 operational reference documents (AMC scope, SOPs, audit, safety)
  sample/synthetic_300.jsonl    300 synthetic complaints matching production schema and distributions

eval/
  run_paper_eval.py             Primary eval: ARIA + GPT-4o on PostgreSQL complaint corpus
  isolation_ablation.py         Ablation: isolated vs pooled vs single-prompt evidence
  cost_sensitivity.py           Cost reduction robustness across 2×–50× weight ratios
  cfpb_second_domain.py         Generalizability: CFPB consumer complaint dataset
  nyc311_second_domain.py       Generalizability: NYC 311 service request dataset
  results/                      Stored aggregate results (JSON); raw CSVs excluded (NDA)

scripts/
  run_eval.py                   Reviewer quick-start: runs ARIA on a JSONL file, no DB needed
  run_paper_additional_experiments.py  Secondary experiments (cost sensitivity, ablation, second domain)
  anonymize_data.py             Data anonymization pipeline (for NDA-gated corpus sharing)
```

---

## Quick start

```bash
git clone https://github.com/freezeindigo/Resolv
cd Resolv
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # fill in GROQ_API_KEY (required) and OPENAI_API_KEY (optional)

# Run ARIA on the synthetic sample — no database needed
python scripts/run_eval.py \
  --dataset data/sample/synthetic_300.jsonl \
  --config eval/config.yaml
```

This produces:
- `eval/results/synthetic_eval_summary.json` — ARIA vs GPT-4o accuracy and cost metrics
- `eval/results/ablation_table.md` — markdown table suitable for a reviewer to inspect

If `OPENAI_API_KEY` is not set, the script runs ARIA-only and skips the GPT-4o baseline comparison.

Expected directional result: ARIA weighted cost < GPT-4o weighted cost (ARIA reduces Project misroutes at some expense in FM false alarms, consistent with the 1:10 asymmetric objective). Absolute numbers will differ from the paper because n=300 and there is no building-history context (no PostgreSQL corpus).

---

## What's NOT public and why

The following are not in this repository and are available to reviewers and PC chairs under NDA on request:

| Asset | Why not public |
|---|---|
| Production complaint corpus (17,098 complaints) | Proprietary operational data from a major Indian residential developer. Contains resident complaint text and metadata. |
| CRM operative labels (FM / Project ground truth) | Derived from internal CRM issue_type field; embedded in the corpus. |
| Building-history retrieval store (PostgreSQL) | Contains flat-level complaint history and resolution outcomes. Required for context-aware Tier 2 and Tier 3 routing; eval with this context is NDA-only. |
| Structural and audit document store | Building-specific operational documents beyond the sanitized samples in `data/rag_documents/`. |

**To request NDA access** (for reviewers verifying full result reproducibility): contact **k.kolanupaka@[institution TBD]** or through the EasyChair review system.

---

## Reproducing each table and figure

| Paper item | How to reproduce |
|---|---|
| **Table 1** — category-level FM/Project split | `python scripts/run_eval.py --dataset data/sample/synthetic_300.jsonl --config eval/config.yaml` (distributions approximate; exact counts need full corpus) |
| **Table 2** — main results (accuracy, cost reduction, CI) | Full corpus: `python eval/run_paper_eval.py --sample 9259 --run --paper-mode --bootstrap 1000 --save-json eval/results/full_ambiguous_eval_9259.json` (NDA). Stored result: `eval/results/full_ambiguous_eval_9259.json` |
| **Table 3** — isolation ablation | `python eval/isolation_ablation.py` (needs 300-row eval CSV; stored: `eval/results/isolation_ablation_summary.json`) |
| **Figure — cost sensitivity** | `python eval/cost_sensitivity.py` (uses stored 300-row CSVs; stored result: `eval/results/cost_sensitivity_results.json`) |
| **Figure — label efficiency** | `python eval/label_efficiency.py && python eval/plot_label_efficiency.py` |
| **Figure — two traps** | `python eval/plot_two_traps.py` |
| **TAT validation** | `python eval/resolution_time_validation.py` (NDA — needs full corpus) |
| **Generalizability (CFPB)** | `python eval/cfpb_second_domain.py` (public dataset, no NDA) |
| **Generalizability (NYC311)** | `python eval/nyc311_second_domain.py` (public dataset, no NDA) |

---

## Citation

```bibtex
@inproceedings{kolanupaka2026aria,
  title     = {Cost-Sensitive Decision Making with {LLM}s under Asymmetric Risk:
               A Multi-Agent Approach to Complaint Routing},
  author    = {Kolanupaka, Kartheek},
  booktitle = {Proceedings of the 35th ACM International Conference on
               Information and Knowledge Management (CIKM)},
  series    = {CIKM '26},
  year      = {2026},
  note      = {Applied Research Track. [DOI TBD on acceptance]}
}
```

---

## License

Code: [MIT](LICENSE) | Documentation and hypothesis YAML library: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
