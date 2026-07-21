# Multi-Step Agent Repair: Which Restart Strategy Wins??

When a multi-step AI agent fails, is it better to **redo just the one broken step**,
or **restart the whole trajectory**? And can the agent's own **uncertainty** tell us
which step to fix — even when the real error happened *upstream* of the uncertainty peak?

**Pipeline:** Task → ReAct agent → step-wise uncertainty → predicted failure point →
{Targeted / Full Restart / Random / Oracle repair} → evaluation & causal analysis.

## Experiment scope

| Axis | Coverage |
|---|---|
| **Models** | 8 models across 3 tiers — Small (7–9B), Medium (12–27B), Large (70–72B) |
| **Datasets** | HotpotQA, FEVER, 2WikiMultiHopQA, MuSiQue |
| **Strategies** | 33 repair strategies (3 baselines + 5 metrics × 6 rules) |
| **Inference** | vLLM + AWQ 4-bit quantization on A100-80GB |

### Models

| Tier | Models |
|---|---|
| Small | Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B |
| Medium | Qwen2.5-14B, Mistral-Nemo-12B, Gemma-2-27B |
| Large | Llama-3.3-70B, Qwen2.5-72B |
| Judge | Qwen2.5-72B-Instruct-AWQ |

### Datasets

| Dataset | Dev size | Task type | Stratify by |
|---|---|---|---|
| HotpotQA | 7,405 | Multi-hop QA | level, type |
| FEVER | 19,998 | Fact verification | answer label |
| 2WikiMultiHopQA | 12,576 | Multi-hop QA | type |
| MuSiQue | 2,417 | Multi-hop QA | hop count |

## Research questions

- **RQ1** — Oracle-targeted repair vs Full Restart: *is targeting worth it?*
- **RQ2** — Uncertainty-targeted vs Oracle: *what does imperfect localization cost?*
- **RQ3** — Uncertainty-targeted vs Random: *is uncertainty better than luck?*

## How to run

| Environment | Instructions |
|---|---|
| **Local / H100** | See **[RUN_LOCAL.md](RUN_LOCAL.md)** |
| **Google Colab A100** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kishormorol/agent-repair/blob/main/run_colab.ipynb) |
| **Multi-model matrix** | `python scripts/run_experiment.py --config config/config_experiment.yaml` |

### Per-dataset Colab notebooks (run in parallel)

| Dataset | Colab |
|---|---|
| **HotpotQA** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kishormorol/agent-repair/blob/main/run_colab_hotpotqa.ipynb) |
| **FEVER** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kishormorol/agent-repair/blob/main/run_colab_fever.ipynb) |
| **2WikiMultiHopQA** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kishormorol/agent-repair/blob/main/run_colab_2wikimultihopqa.ipynb) |
| **MuSiQue** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kishormorol/agent-repair/blob/main/run_colab_musique.ipynb) |

## The pipeline

| Stage | Script | What it does | ~Time (A100, 500 Qs) |
|---|---|---|---|
| 0 | `run_setup.py` | Download dataset, create folders | 1 min |
| 1 | `run_generate.py` | Run the ReAct agent; keep the **failed** trajectories | 1–2 h |
| 2 | `run_uncertainty.py` | Score every step with **5 uncertainty metrics** | 1–2 h |
| 3 | `run_annotate.py` | 72B judge marks the **true broken step** (+ `label_human.py` validates) | 1–2 h |
| 4 | `run_localize.py` | Does uncertainty point at the true step? (5 metrics × 6 rules) | < 1 min |
| 5 | `run_repair.py` | **33 repair strategies** on every failure; batched + deduplicated | 3–4 h |
| 6 | `run_eval.py` | Tables, significance tests, causal analysis, figures | < 1 min |

Every stage is **resumable** — if the process dies, re-run the same command and it picks up where it left off.

## Repair strategies (33 total)

**3 baselines:** Full Restart, Random Step, Oracle (fix the true broken step).

**30 uncertainty-guided** = 5 metrics × 6 localization rules:

| | argmax | top-k | earliest>thr | cascade-up | cascade-grad | cascade-wt |
|---|---|---|---|---|---|---|
| Token entropy | | | | | | |
| Perplexity | | | | | | |
| Max-token-prob | | | | | | |
| Self-consistency | | | | | | |
| Verbalized conf. | | | | | | |

**Cascade-aware rules** (new): In our 500-question HotpotQA pilot, ~70% of localization
misses occurred because the real error was *upstream* of the uncertainty peak — errors
propagate forward through reasoning steps. The three cascade rules address this:

- **cascade-upstream** — from the uncertainty peak, look back *k* steps
- **cascade-gradient** — pick the step with the largest uncertainty *increase*
- **cascade-weighted** — composite of normalized uncertainty and position bias toward earlier steps

## Key design choices

- **Distractor setting** → retrieval runs offline over each question's local
  paragraphs (reproducible, no live Wikipedia), and gold supporting facts make
  error annotation objective.
- **Fair comparison** → identical nudge (retry hint + T=0.7) for *every* strategy,
  matched token budget, 3 seeds, paired McNemar tests with Holm correction.
- **Validated oracle** → 72B judge labels, checked against 50 human labels (κ).
- **Dedup** → strategies picking the same step share one repair (same step + same
  seed = same result), cutting GPU work by ~70–90%.
- **Batching** → up to 64 ReAct rollouts in flight at once; full dev runs overnight.
- **Multi-dataset** → dataset registry pattern — each dataset provides its own
  environment, download function, and scorer.

## Layout

```
config/
  config_colab.yaml        Colab A100 profile (500 questions)
  config_local.yaml        Local H100 profile (full dev sets)
  config_experiment.yaml   Multi-model × multi-dataset matrix
  models.yaml              8-model catalog (small / medium / large)
  datasets.yaml            4-dataset catalog

scripts/
  run_setup.py … run_eval.py   the 7 pipeline stages
  run_experiment.py            master runner for the full model × dataset matrix
  run_paper_tables.py          aggregate cross-experiment results into paper tables

src/
  env/                     dataset environments (HotpotQA, FEVER, 2WikiMQA, MuSiQue)
  agent/                   ReAct loop (resumable) + batched runner
  llm/                     vLLM client with per-token logprobs
  uncertainty/             5 uncertainty metrics
  annotate/                programmatic + 72B-judge error labels + human validation
  localize/                6 localization rules (incl. 3 cascade-aware)
  repair/                  33 repair strategies
  eval/, analysis/         statistics, ensemble, causal failure-mode analysis

outputs/                   created on first run: trajectories, tables, figures
run_colab.ipynb            one-click Colab notebook
```

## Reading the results

- `outputs/tables/summary.txt` — answers RQ1–RQ3 in plain English
- `outputs/tables/main_results_readable.csv` — percentages per strategy
- `outputs/tables/main_results.csv` — raw fractions (e.g. `0.352` = 35.2%)
- `outputs/tables/rq_tests.json` — statistical tests with p-values
- `outputs/tables/failure_modes.json` — why localization misses
- `outputs/figures/` — headline bars, metric×rule heatmap, Pareto cost-vs-success, failure modes
