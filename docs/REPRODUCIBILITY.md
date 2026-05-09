# ARIA Reproducibility Guide

This document specifies exact dependencies, model configurations, hyperparameters, random seeds, and expected outputs for reproducing results from the CIKM '26 paper.

---

## Environment

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Tested on 3.11.9 |
| `anthropic` | ≥0.18.0 | Arbiter (claude-haiku-4-5-20251001 in paper eval) |
| `groq` | ≥0.11.0 | Tier 2, hypothesis agents, judge, pattern interpreter |
| `openai` | ≥1.0.0 | GPT-4o baseline only |
| `langgraph` | ≥0.2.0 | Pipeline orchestration |
| `fastapi` | ≥0.109.0 | API server (not needed for eval) |
| `pydantic` | ≥2.0.0 | |
| `pyyaml` | ≥6.0.0 | Config loading |
| `numpy` | ≥1.26.0 | Bootstrap CI |
| `scikit-learn` | ≥1.4.0 | DBSCAN clustering in pattern state |
| `redis` | ≥5.0.0 | Pattern state store (not needed for eval with `--no-context`) |
| `psycopg2-binary` | ≥2.9.9 | PostgreSQL (not needed for synthetic eval) |
| `asyncpg` | ≥0.29.0 | Async DB queries (not needed for synthetic eval) |
| `python-dotenv` | ≥1.0.0 | `.env` loading |

Install:
```bash
pip install -e ".[dev]"
# or: pip install -r requirements.txt
```

---

## LLM models used in paper evaluation

The headline numbers (22.7% cost reduction, n=9,259) were generated using `--paper-mode`, which activates the following model configuration:

| Pipeline role | Provider | Model ID |
|---|---|---|
| Tier 2 reasoning | Groq | `llama-3.3-70b-versatile` |
| Hypothesis agents (all) | Groq | `llama-3.3-70b-versatile` |
| Pattern interpreter | Groq | `llama-3.3-70b-versatile` |
| Arbiter | Anthropic | `claude-haiku-4-5-20251001` |
| Judge | Groq | `llama-3.1-8b-instant` |
| GPT-4o baseline | OpenAI | `gpt-4o` (temperature=0) |

The production `src/config/model_config.py` uses `claude-sonnet-4-20250514` for the arbiter; `--paper-mode` overrides this to the above configuration at runtime without modifying the config file.

---

## Hyperparameters

| Parameter | Value | Where set |
|---|---|---|
| Asymmetric cost ratio (FM:Project) | 1:10 | `data/cost_matrix.yaml`, `eval/run_paper_eval.py` |
| Bootstrap resamples | 1,000 | `--bootstrap 1000` |
| Random seed (bootstrap) | 42 | `eval/run_paper_eval.py:bootstrap_cost_ci()` |
| Random seed (stratified sample) | 42 | `eval/run_paper_eval.py:stratified_sample()` |
| ARIA concurrency (async LLM calls) | 3 | `--concurrency 3` |
| Tier 2 max tokens | 800 | `src/config/model_config.py` |
| Hypothesis agent max tokens | 1,024 | `src/config/model_config.py` |
| Arbiter max tokens | 1,500 | `src/config/model_config.py` |
| Judge max tokens | 600 | `src/config/model_config.py` |
| GPT-4o baseline temperature | 0 | `eval/run_paper_eval.py:run_gpt4o_baseline()` |
| Cost sensitivity ratios | [1,2,3,5,7,10,15,20,30,50] | `data/cost_matrix.yaml`, `eval/cost_sensitivity.py` |

---

## Reproducing paper numbers (full corpus, NDA)

```bash
# Step 1: Load production corpus into PostgreSQL (NDA-gated)
python scripts/load_complaints_xlsx.py --db resolv

# Step 2: Run ARIA + GPT-4o on the 9,259 ambiguous complaints
python eval/run_paper_eval.py \
  --db resolv \
  --sample 9259 \
  --run \
  --paper-mode \
  --bootstrap 1000 \
  --save-json eval/results/full_ambiguous_eval_9259.json \
  --yes

# Step 3: Run isolation ablation (n=300 subset)
python eval/isolation_ablation.py

# Step 4: Cost sensitivity sweep
python eval/cost_sensitivity.py

# Step 5: Generate figures
python eval/plot_label_efficiency.py
python eval/plot_two_traps.py
```

Stored results from the above runs are in `eval/results/`:
- `full_ambiguous_eval_9259.json` — headline numbers (Table 2)
- `isolation_ablation_summary.json` — ablation results (Table 3)
- `cost_sensitivity_results.json` — sensitivity curve
- `fig1_label_efficiency.{pdf,png}` — Figure 1
- `fig2_two_traps.{pdf,png}` — Figure 2

---

## Reproducing from public repo (synthetic sample, no NDA)

```bash
python scripts/run_eval.py \
  --dataset data/sample/synthetic_300.jsonl \
  --config eval/config.yaml
```

Expected outputs after this command:
- `eval/results/synthetic_eval_summary.json` — ARIA vs GPT-4o metrics on 300-row sample
- `eval/results/ablation_table.md` — markdown summary table

**What reviewers should see:**

1. ARIA weighted cost < GPT-4o weighted cost on the synthetic sample. The exact % will differ from 22.7% (smaller n, no building-history context, synthetic complaint distributions).
2. ARIA Project accuracy > GPT-4o Project accuracy. GPT-4o strongly prefers FM for ambiguous complaints (paper: 13.2% Project accuracy) because the text surface alone under-signals structural/DLP issues.
3. ARIA FM accuracy ≤ GPT-4o FM accuracy. The cost-aware design intentionally trades some FM accuracy for fewer high-cost Project misses.

If results are directionally reversed (GPT-4o costs less), check: (a) GROQ_API_KEY is valid, (b) the synthetic data label distribution (verify `synthetic_300.jsonl` has ~59.3% FM / 40.7% Project), (c) the complaint text quality (very simple complaints may not trigger Tier 3 hypothesis deliberation).

---

## Eval flags reference

| Flag | Effect |
|---|---|
| `--paper-mode` | Overrides model config to paper-eval settings (Groq Tier2/hyp/judge, Haiku arbiter) |
| `--no-context` | Disables all DB + RAG retrieval (flat history, adjacency, docs) |
| `--no-docs` | Keeps DB context, disables document retrieval only |
| `--bootstrap N` | Computes bootstrap 95% CI on cost reduction using N resamples |
| `--checkpoint PATH` | Saves ARIA progress CSV after each 500-row chunk; resumes from PATH if exists |
| `--gpt-checkpoint PATH` | Same for GPT-4o baseline |
| `--yes` | Skips interactive confirmation prompt |

---

## Note on the Judge node

The pipeline includes a `judge` node (`src/agents/judge.py`, model: Groq `llama-3.1-8b-instant`) that validates Tier 2 and Tier 3 routing decisions before the execute node. In the stored paper-eval results the judge rarely overrides the arbiter; its primary effect is catching malformed outputs. This node is not discussed in the paper but is present in all eval runs.
