# Running and Reproducing Agent Repair

Run commands from the repository root. Separate CPU analysis of archived
results from GPU generation of new observations. Neither a successful build
nor a passing CPU test suite independently reproduces the paper's results.

The current cloud allocation is **$119 using AWS credits**. Use the reduced defaults in
[run_iclr2027.ipynb](run_iclr2027.ipynb) and the
[AWS setup guide](CLOUD_GPU_SETUP.md#current-allocation-119-on-aws).
The direct CLI pilot YAML below still describes the earlier expanded
diagnostic experiment; do not run it unchanged as the low-budget profile.
An offline worksheet is available with
`python scripts/estimate_budget.py --total-usd 119 --reserve-usd 20 --hourly-usd 8.944 --spent-usd 0`.
This uses the September 8 London Linux p5.4xlarge base quote; recheck it at launch.
It does not enforce a provider billing limit.

## CPU Tests and Working Draft

Python 3.10 or newer is required by the source syntax. Use the CPU dependency
list instead of installing the GPU serving stack on a laptop:

```bash
python -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r requirements-analysis.txt
python -m pytest -q tests
python scripts/build_iclr_draft.py
```

The paper build checks the audited main study, diagnosis follow-up,
completed six-cell replication and runtime component before generating their assets. Historical
aggregates and explicitly indexed notebook tables retain separate provenance.
Generated LaTeX tables live in `paper/generated/tables/`; PDF/PNG figures
live in `paper/generated/figures/`. Numeric macros and input-hash manifests
remain in `paper/generated/`. Completed replication tables replace the
earlier manually maintained status table. Recovered source records are required
to rebuild audited results; passing CPU tests alone is not their reproduction.

Reproduce all six extension cells from the downloaded raw archives with:

```bash
python scripts/finalize_extension_study.py --base output/aws-experiment/2026-09-15-extension-completion
```

`scripts/build_extension_paper.py` requires that successful reproduction,
checks its input hashes and complete coverage, and runs in the default paper
build. It rejects changed or incomplete evidence. The completed runtime
component has its own validated builder, `scripts/build_runtime_paper.py`.

With `latexmk`, a TeX distribution, and required font packages installed:

```bash
make -C paper iclr-draft
```

The PDF is written to `output/pdf/agent-repair-iclr2027-draft.pdf`.
The official style needs packages including `eso-pic` and `fancyhdr`, and
Times, Helvetica, and Courier fonts. Use your distribution's package manager
if the build reports a missing package. Do not replace the official style
or change margins to fit the paper.

## Inspect the Intended Matrix

This command does not load models or download datasets. It creates the
configured directories and an experiment log:

```bash
python scripts/run_experiment.py \
  --config config/config_experiment.yaml \
  --model qwen2.5-32b --dataset hotpotqa --dry-run
```

Inspect other QA dataset names with `--dataset musique` or
`--dataset 2wikimultihopqa`. Omitting filters lists the catalog, including
unvalidated FEVER and planned model configurations. A catalog entry or a
successful dry run is not a completed experiment.

The 32B catalog entry now resolves; empty filter matches raise an error.
A non-dry matrix run requires `--run-id` and writes isolated absolute paths
for raw/processed data, model caches, checkpoints, and outputs. An existing
resolved configuration cannot be changed in place.

## Before Running Models

The GPU stack in [requirements.txt](requirements.txt) uses version ranges,
including `vllm>=0.19.0`; it is not a pinned reproduction environment.
The previous instructions for vLLM 0.6.3 and a fixed overnight runtime were
stale. Validate a compatible serving environment on the actual GPU, record
its exact package versions, and measure a pilot before estimating run time.
The completed AWS studies preserve their actual GPU preflights and pinned
runtime records in the linked reports. A new environment still needs its
own smoke test; those records do not validate every version in the ranges.

The [study protocol](docs/iclr2027_study_protocol.md#implementation-gates-before-gpu-runs)
contains implementation requirements for a confirmatory ICLR run. Do not
start the unfiltered full matrix as that experiment. In particular:

- Generation and repair manifests reject changed inputs/configuration/code
  and incompatible legacy outputs. Stage-2 scoring and judge checkpoints
  still require fresh paths when their code or settings change. Preserve
  old outputs; do not remove manifests to force an incompatible resume.
- `--limit` truncates a stage's input list. It is not a train/test split,
  and limiting initial questions is different from limiting failed repairs.
- Profile dataset values now override catalog defaults. Confirmatory
  `dataset.split: test` requires `id_manifest` and `exclude_ids_manifest`:
  JSON lists of unique IDs (or objects with an `ids` list). Their overlap is
  rejected. The excluded list must contain every explored question; the
  software cannot establish that it is complete.
- The repair CLI executes all configured multipliers. Set
  `repair.step_budget_mode: new` and `repair.match_restart_hint: true` for
  the primary controlled comparison; defaults retain historical semantics.
- Exact model/tokenizer revisions and serving packages still need pinning.
  Model-name/cache checks are not revision pinning. Do not use the direct
  stage `--model` flags to switch experiments; put the model in a new profile.
- Do not silently fall back to a different agent, judge, or quantization.
  Record the model actually loaded, its revision, and the saved cache.
- FEVER needs real evidence and task-specific prompting before any result
  from it is usable. Inspect notebook cleanup cells before executing them.

## Pipeline Entry Points

Each script accepts `--config` and `--limit`; inspect its help without
loading a model:

```bash
python scripts/run_generate.py --help
python scripts/run_repair.py --help
python scripts/run_eval.py --help
```

Run stages only after an isolated, validated run configuration is available:

| Order | Script | Requires |
| --- | --- | --- |
| 0 | `scripts/run_setup.py` | Dataset access and correct evidence loader |
| 1 | `scripts/run_generate.py` | Frozen IDs, agent, tools, and initial decoding |
| 2 | `scripts/run_uncertainty.py` | Failed trajectories and stored log probabilities |
| 3 | `scripts/run_annotate.py` | Reference answers, evidence, and declared judge |
| 4 | `scripts/run_localize.py` | Uncertainty records and reference annotations |
| 5 | `scripts/run_repair.py` | Failed pool, origins, matched prompts and allowances |
| Analysis | `scripts/run_paired_analysis.py` | Complete raw repair rows and adjacent manifests |

A pilot should include several distinct token budgets and trajectories with
retained prefixes. Check actual generated tokens, restored tool state,
answer scoring, model identity, and seed coverage before scaling up.
Synthetic unit tests do not test model quality, memory use, or GPU batching.

## Controlled Development Pilot

For a portable Jupyter workflow, use [run_iclr2027.ipynb](run_iclr2027.ipynb)
with its matching [code bundle](output/notebooks/agent-repair-iclr2027-code.zip).
The [cloud guide](CLOUD_GPU_SETUP.md) explains GPU choices and launch steps.
The notebook adds snapshot pinning, environment/configuration locks, a
repair-count guard, paired analysis and raw-evidence export; it defaults
to plan-only mode and does not rent or provision a GPU.

The new [pilot profile](config/config_iclr_pilot.yaml) requests 30 initial
HotpotQA questions, three repair seeds, every eligible origin, and 0.5x/1x
generated-token caps. This is development-only, not a confirmatory sample
size or a selected uncertainty method. At eight eligible steps it requests
at most `48 * N_failed` repair executions across the two caps, plus initial
generation. It uses only stored-token uncertainty, with no judge calls.
No commands below have been run on a real GPU during this review.

On the intended GPU, after validating and recording its environment:

```bash
python scripts/run_setup.py --config config/config_iclr_pilot.yaml
python scripts/run_generate.py --config config/config_iclr_pilot.yaml
python scripts/run_uncertainty.py --config config/config_iclr_pilot.yaml
python scripts/run_repair.py --config config/config_iclr_pilot.yaml --dry-run
python scripts/run_repair.py --config config/config_iclr_pilot.yaml
```

The repair dry run needs the real generated pool, failures, saved model
identity, trajectories, and uncertainty files. It checks planning without
loading a model or writing repair results; it does not synthesize missing
data. Steps 3/4 (judge/localization) are unnecessary for these primary
conditions. Do not use legacy `run_eval.py` as their confirmatory analysis;
it assumes the older broad grid and reference annotations.

After complete results exist, a development-only analysis is:

```bash
python scripts/run_paired_analysis.py \
  --results runs/iclr2027/hotpotqa/qwen32b/pilot-v1/outputs/repairs/results.jsonl \
  --strategy unc__perplexity__argmax \
  --baselines full_restart random_step --seeds 0 1 2 --multiplier 1 \
  --output runs/iclr2027/hotpotqa/qwen32b/pilot-v1/outputs/tables/paired_pilot.json
```

The JSON contains paired mean effects, intervals, explicit sign-flip
assumptions, Holm-adjusted macro contrasts, and per-dataset EM/F1
sensitivities. Effects are fractions; multiply by 100 for percentage points.
The one-dataset pilot output is not a QA3 confirmatory result. For that
study, supply one complete file per QA dataset from the frozen policy and
same model, with separately audited disjoint cohorts. Policy choice and
planned sample size must be frozen before viewing test outcomes.

Unique raw executions live in `outputs/repairs/executions/` beneath the run;
`origins.jsonl` covers the sweep and `results.jsonl` maps policies to shared
executions. Sum physical costs by unique execution ID, not copied strategy
rows. Acquisition fields with `null` are unknown costs, not free calls.
Record model/tokenizer snapshot hashes separately. Inspect missing logprob
rates and origin-zero fallback rates before interpreting any signal.

## Human Validation

`scripts/label_human.py` collects one human labeling record per trajectory
and writes `_human_labels.json` under the configured annotations directory.
It writes `judge_human_agreement.json` under the tables directory when
labels are available. Inspect its help with:

```bash
python scripts/label_human.py --help
```

Run interactive labeling only against the correct frozen pool. The current
CLI does not implement blinded multi-annotator assignment or adjudication;
those are specified in the study protocol and remain to be prepared.
Configuring a sample size does not mean the labels have been collected.

## Required Result Package

Keep the resolved configuration, Git revision, environment export, hardware
record, split IDs and hashes, evidence snapshot, model revisions, initial
trajectories, uncertainty acquisitions, reference and human annotations,
unique repair execution records, checkpoints, and analysis outputs together.
The [readiness audit](docs/iclr2027_readiness.md#files-needed-from-google-drive)
lists the raw artifacts missing from the current repository snapshot.
