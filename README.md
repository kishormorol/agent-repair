# Multi-Step Agent Repair: Which Restart Strategy Wins?

When a multi-step AI agent fails, is it better to **redo just the one broken step**,
or **restart the whole trajectory**? And can the agent's own **uncertainty** tell us
which step to fix?

**Pipeline:** Task → ReAct agent → step-wise uncertainty → predicted failure point →
{Targeted / Full Restart / Random / Oracle repair} → evaluation & causal analysis.
**Setting:** HotpotQA (distractor, full dev = 7,405 questions) with Qwen2.5-7B-Instruct.

## Research questions
- **RQ1** — Oracle-targeted repair vs Full Restart: *is targeting worth it?*
- **RQ2** — Uncertainty-targeted vs Oracle: *what does imperfect localization cost?*
- **RQ3** — Uncertainty-targeted vs Random: *is uncertainty better than luck?*

## ▶ How to run
See **[RUN_LOCAL.md](RUN_LOCAL.md)** — install, smoke-test, then six commands.

## The experiment
| Stage | Script | What it does |
|---|---|---|
| 0 | `run_setup.py` | Download HotpotQA, create folders |
| 1 | `run_generate.py` | Run the ReAct agent; keep the **failed** trajectories |
| 2 | `run_uncertainty.py` | Score every step with **5 uncertainty metrics** |
| 3 | `run_annotate.py` | 72B judge marks the **true broken step** (+ `label_human.py` validates it) |
| 4 | `run_localize.py` | Does uncertainty point at the true step? (5 metrics × 3 rules) |
| 5 | `run_repair.py` | **18 repair strategies** on every failure; batched + deduplicated |
| 6 | `run_eval.py` | Tables, significance tests, causal analysis, figures |

**18 strategies** = 3 baselines (Full Restart, Random, Oracle) + 5 uncertainty
metrics × 3 localization rules (argmax / top-k / earliest-above-threshold).

**Uncertainty metrics:** token entropy, perplexity, max-token-probability,
self-consistency, verbalized confidence.

## Key design choices
- **Distractor setting** → retrieval runs offline over each question's 10 local
  paragraphs (reproducible, no live Wikipedia), and the gold supporting facts make
  error annotation objective.
- **Fair comparison** → identical nudge (retry hint + T=0.7) for *every* strategy,
  matched token budget, 3 seeds, paired McNemar tests with Holm correction.
- **Validated oracle** → 72B judge labels, checked against 50 human labels (κ).
- **Dedup** → strategies picking the same step share one repair (same step + same
  seed ⇒ same result), cutting GPU work by ~90%.
- **Batching** → 64 ReAct rollouts in flight at once; full dev runs overnight.

## Layout
```
config/config_local.yaml   all experiment settings (models, sizes, batch sizes, seeds)
scripts/                   the 8 CLI stages (run these)
src/env/                   HotpotQA tools (search/lookup/finish) + EM/F1 scorer
src/agent/                 ReAct loop (resumable) + batched runner
src/llm/                   vLLM client with per-token logprobs
src/uncertainty/           the 5 metrics
src/annotate/              programmatic + 72B-judge error labels + human validation
src/localize/              argmax / top-k / earliest-above-threshold rules
src/repair/                the 18 repair strategies
src/eval/, src/analysis/   statistics, ensemble, causal "when does it work" model
outputs/                   created on first run: trajectories, tables, figures
```

## Reading the results
`outputs/tables/summary.txt` answers RQ1–RQ3 in plain English.
`main_results_readable.csv` has percentages; `main_results.csv` has raw fractions
(e.g. `0.352` = **35.2%**).
