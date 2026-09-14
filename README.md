# Uncertainty-Guided Trajectory Repair

**Targeted Regeneration versus Full Restart**

When a tool-using language model agent fails, how much of its previous
trajectory should recovery retain? This repository studies suffix
regeneration, full restart, and uncertainty-based selection of repair origins
in ReAct-style agents with question-specific offline evidence.

> **ICLR 2027 preparation, September 14, 2026:** the six-condition AWS
> HotpotQA study and the separate diagnosis/replay follow-up are complete and
> audited. Neither found a clear uncertainty-repair advantage. The manuscript
> reports both completed studies alongside historical
> exploratory results, with new execution, cost and scoring diagnostics and a
> verified statistical supplement. The follow-up report adds a same-model
> diagnosis/replay comparator and three allowances. A separately frozen
> two-model, three-dataset replication and runtime extension is executing
> within its $35 session ceiling. Its outcomes and audits are pending;
> human validation remains incomplete.
> This is not a completed or accepted submission. FEVER
> results are excluded.

## Paper and Submission Status

| Artifact | Purpose |
| --- | --- |
| [Start here](START_HERE.md) | AWS allocation, completed runs, backups and resource status |
| [Completed AWS main study](docs/aws_main_2026-09-12.md) | Audited 250-question, six-condition HotpotQA results, paired comparisons and cost estimate |
| [Current expert review and fixes](docs/iclr_expert_review_2026-09-13.md) | September 13 assessment, six gaps, implemented diagnostics and remaining experimental design |
| [Completed diagnosis/replay follow-up](docs/aws_diagnosis_2026-09-14.md) | Audited 250-question comparison across three allowances: 2,079 repairs, 3,186 trial rows, no clear uncertainty advantage; EC2 stopped |
| [Active replication and runtime extension](docs/aws_extension_2026-09-14.md) | Frozen two-model, three-dataset experiment; 600 main model/question evaluations, seven policies and a separate latency study |
| [Statistical supplement](output/reviewer-2026-09-12/agent-repair-statistical-supplement.zip) | Checksummed records and standalone reproduction of four primary and eight secondary contrasts; full GPU replay is separate |
| [ICLR manuscript](paper/iclr2027.tex) | Anonymous working draft: main text ends on page eight, followed by references and technical appendices |
| [Title and abstract](paper_title_abstract.md) | Current evidence-limited abstract, aligned with the manuscript |
| [Experiments to complete](docs/experiments_to_run.md) | Full-paper restoration map, prioritized experiments, and deadline workback |
| [Section-by-section review](docs/paper_section_review.md) | September 9 revisions, six additional related papers, and unresolved evidence checks |
| [Readiness audit](docs/iclr2027_readiness.md) | Identified validity problems, evidence paths, and deadlines |
| [Study protocol](docs/iclr2027_study_protocol.md) | Contribution positioning, controlled comparisons, and completion criteria |
| [Controlled experiment notebook](run_iclr2027.ipynb) | Defaults to the $119 AWS allocation: four conditions, one token cap, matched controls and a hashed local code bundle |
| [Cloud GPU guide](CLOUD_GPU_SETUP.md) | AWS eligibility, single-GPU setup, billing precautions and measured-throughput planning |
| [Historical result notes](results/cross_dataset/README.md) | Which archived summaries can and cannot be compared |

Build the PDF with `make -C paper iclr-draft`. It is written to
`output/pdf/agent-repair-iclr2027-draft.pdf`. The original `paper/main.tex`
and `paper/main.pdf` are historical AAAI drafts with superseded claims;
they are not the ICLR submission.

The manuscript includes the six-condition HotpotQA results, a figure of the
four paired confidence intervals, and the completed diagnosis follow-up's
nine policy/allowance results, six paired contrasts and measured policy costs.
Both studies retain their execution audits and historical-ID limitations.
Historical signal definitions, recovery costs, hint comparisons
and localization diagnostics are preserved in the appendix. The abstract and
conclusion report no clear uncertainty-repair advantage; missing comparisons
and unverified mechanisms remain explicitly identified.
Review-stage diagnostics show that treatment differs from restart on 43 of
113 failed questions and uses 29.5% more prompt tokens with similar output
tokens. These exploratory findings qualify the result and its cost claims;
they do not change the primary estimand or establish what a larger allowance
would achieve. The statistical supplement reproduces locally without cloud
access once its Python dependencies are installed.
The [completed main follow-up](docs/aws_main_2026-09-12.md) evaluated 250 HotpotQA
questions, six conditions, three seeds, one token allowance and one model.
All five batches, 1,058 unique repairs and 2,034 condition/seed rows were
downloaded and independently audited. Uncertainty repair succeeded on 9.73%
of failed-question seed trials versus 11.21% for restart; the paired difference
was −1.47 percentage points (95% bootstrap interval −4.13 to +0.88).
All four primary Holm-adjusted p-values were 1.000. This does not establish
equivalence. Estimated infrastructure usage is $9.16 before credits,
tax and transfer, within the $25 session allowance and original $119 allocation.
The other two QA datasets are being evaluated in the separate extension.

The [September 14 diagnosis/replay follow-up](docs/aws_diagnosis_2026-09-14.md)
completed a separate frozen cohort of 250 questions under 0.5×, 1× and 2×
recovery allowances. Its 2,079 unique repairs and 3,186 trial rows passed
independent replay and scoring audits. At 1×, uncertainty repair succeeded
on 9.89% of failed-question seed trials, versus 10.45% for restart and
10.17% for same-model diagnosis/replay. All six primary Holm-adjusted
p-values are 1.000. The report includes diagnosis costs, verified downloads
and the final AWS stopped-state observation. This follow-up has its own
protocol and comparison family.

The long historical diagnostics are preserved separately. The
[September 10 AWS pilot](docs/aws_run_2026-09-10.md) completed ten initial
questions and 25 unique repairs; it showed no uncertainty-repair advantage.
The [September 11 development run](docs/aws_development_2026-09-11.md)
completed 50 additional questions, 186 unique repairs and 288 condition/seed
rows after scoring, replay and shutdown fixes. Its paired comparisons remain
inconclusive. Those artifacts are also downloaded and audited.
After main-study retrieval, AWS independently confirmed EC2 **stopped**, with
no public IP, on September 12 at 19:02:37 UTC. EBS is retained at approximately
$0.62/day. The main study's historical exclusions
are reconstructed, with provenance and remaining verification limits in its report.
The previous $200 cash ceiling is not added to the AWS allocation. Do not fund
Runpod for the current plan. Cloud quota, rate and shutdown checks remain required.

The official ICLR 2027 deadlines are **September 18, 2026 for abstracts**
and **September 25, 2026 for full papers and supplements**, both 11:59 p.m.
Anywhere on Earth. Submission main text is limited to nine pages.
Check the [author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
for anonymity, author registration, reciprocal reviewing, and AI disclosure.
No submission has been made by the preparation tooling.

## Research Question and Contribution

**Does a prespecified uncertainty signal select a more useful repair origin
than simple alternatives once information access, recovery allowance, and
selection bias are controlled?**

The candidate contribution is a controlled empirical comparison of
low-overhead uncertainty signals for choosing repair origins. Backtracking,
uncertainty-triggered resets, and evaluating interventions by their outcomes
are not claimed as new ideas. Related work includes
[DoVer](https://www.microsoft.com/en-us/research/publication/dover-intervention-driven-auto-debugging-for-llm-multi-agent-systems/),
[ERGO](https://aclanthology.org/2025.uncertainlp-main.23/), and
[REFLECT](https://arxiv.org/abs/2606.09071). The September 9 review also adds
[Doctor-RAG](https://arxiv.org/abs/2604.00865v2), the closest task-level prior,
and [SymTrace](https://arxiv.org/abs/2608.25920v2), which directly examines
repair versus resampling. The [section review](docs/paper_section_review.md)
records all six additions; the protocol identifies the comparisons needed.

The planned tests ask:

1. Does held-out uncertainty-guided repair improve on restart and matched
   random-origin controls, rather than just a test-selected grid maximum?
2. Does moving upstream help because of uncertainty information, or can
   a fixed early origin and a position-matched random control explain it?
3. What changes when uncertainty acquisition and prompt processing are
   included in recovery costs?

These are research questions, not established conclusions. Judge-label
agreement is a separate diagnostic outcome, not proof of causal localization
or a guarantee of successful repair.

## Available Evidence

The archived QA summaries describe `Qwen/Qwen2.5-32B-Instruct-AWQ` as the
agent and `Qwen/Qwen2.5-72B-Instruct-AWQ` as the reference judge. Each dataset
started with 500 questions. Repairs used three generation seeds. The table
contains **historical descriptive point estimates**, not newly reproduced
results or statistically confirmed differences.

| Mean repair success (%) | HotpotQA | MuSiQue | 2WikiMHQA |
| --- | ---: | ---: | ---: |
| Failed questions / 500 | 226 | 425 | 203 |
| Judge-targeted | 10.9 | 4.9 | 10.0 |
| Judge-targeted + backtrack 2 | 12.4 | 5.6 | 22.0 |
| Full restart | 14.2 | 5.9 | 21.2 |
| Retrospective best uncertainty variant | 12.8 | 5.6 | 25.5 |

Source: [archived mean-seed aggregates](results/cross_dataset/cross_dataset_traj_stats.csv).
There are 854 distinct failed QA questions, not hundreds of thousands of
independent examples. Strategy rows may reuse the same generated repair.

Important qualifications:

- **Best uncertainty is retrospective.** The configuration was chosen on
  the reported questions. The HotpotQA and MuSiQue maxima also use
  judge-informed hints, so this row is not an uncertainty-only policy.
- **Historical budget equality is unverified.** The earlier runner could
  overshoot a token cap. Local fixes do not change archived outcomes.
- **Prompt and step allowances differ.** Targeted repairs receive a retry
  hint; restart does not. Retained steps count toward the eight-step total,
  leaving targeted repairs fewer new steps. These are additional confounds.
- **FEVER is invalid in the inspected loader.** It substitutes the claim
  for Wikipedia evidence and uses a generic QA prompt. Its historical
  results must not support the submission's empirical claims.
- **No trajectory-length routing rule is established.** The archived
  `avg_tool_calls` measures restart repairs, not original failed traces.
  It cannot justify a short-versus-long policy or a causal context claim.
- **Candidate union is not an ensemble policy.** Counting success whenever
  any strategy succeeds uses reference answers and does not define an
  implementable answer selector.
- **Raw trials and human labels are absent from this snapshot.** Confidence
  intervals, significance, budget compliance, and judge reliability cannot
  be independently verified from these aggregates alone.

The eight-model catalog is planned execution support, not evidence that an
eight-model study has been completed. Existing non-significant or negative
findings must be retained when the corrected comparison is run.

## Framework

![Trajectory repair pipeline](assets/arch.png)

The diagram describes the experimental design, not completed validation.
Dashed reference-annotation paths use gold answers and supporting evidence.
Informed hints therefore belong to a privileged diagnostic condition.

```text
Question + offline evidence
  -> ReAct trajectory with stored token log probabilities
  -> benchmark correctness gate
  -> failed trajectory
  -> uncertainty scores and repair-origin selection
  -> prefix replay + suffix regeneration, or full restart
  -> recovery success, localization agreement, and measured costs
```

The correctness gate uses benchmark answers to select failed attempts.
This is recovery conditional on a known failure, not online failure
detection for a deployed agent.

### Signals and Origins

| Signal family | Implemented measurement |
| --- | --- |
| Token entropy | Entropy after renormalizing the returned top-K probabilities, not full-vocabulary entropy |
| Sampled-token probability | `1 - p(sampled token)`; the legacy key is `max_token_prob` |
| Perplexity | Exponentiated mean sampled-token surprisal, with numerical clipping |
| Action disagreement | Disagreement among sampled action-and-argument pairs |
| Verbalized uncertainty | One minus normalized model-reported confidence |

Six rules select an origin: argmax, earliest among the top-k scores,
earliest above a percentile, upstream lookback, largest increase, and
position-weighted uncertainty. Backtracking subtracts an offset from that
origin, clamping at zero. Code uses zero-based indices and retains
`steps[:origin]` before regeneration. The diagram uses one-based notation.

The archived grid describes 30 metric-rule combinations, two backtrack
offsets, and two hint types, plus baselines: 126 nominal configurations.
Current configuration files need not reproduce that grid; for example,
the experiment profile lists four offsets. Always record the resolved grid.

### Budgets and Evaluation

The current CLI repair runner uses a nominal token cap equal to the original
trajectory's generated tokens (`m = 1`). Sequential and batched generation
now limit each call to the remaining allowance. The configured multiplier
list is not yet an implemented sweep.

Recovery-token equality does not mean equal total compute. Count repeated
prefix processing, uncertainty calls, selection, annotation when used by
the policy, and unique generated candidates separately. Report the step
allowance and prompt convention alongside the token cap.

The archived success gate is `EM == 1 or token F1 >= 0.5`; this is a
study-specific threshold. Report strict EM and benchmark-appropriate F1 as
well. The primary descriptive fix rate averages seed outcomes within each
question, then averages questions. Question-level bootstrap intervals are
implemented. Existing McNemar tests use majority-over-seeds outcomes and
are a secondary repeatability analysis, not a test of the mean-seed rate.
The confirmatory paired-difference analysis is implemented in
`scripts/run_paired_analysis.py`; it requires complete raw trials and frozen
manifests, which are absent for the archived results.

## Quick Start

Run commands from the repository root with Python 3.10 or newer. The CPU
path does not install vLLM, download models, or require a GPU:

```bash
python -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r requirements-analysis.txt
python -m pytest -q tests
python scripts/build_iclr_draft.py
```

The builder regenerates draft tables, figures, and a source-hash manifest
from archived CSVs and explicitly indexed saved notebook tables. It checks
baseline agreement across these sources and discards obsolete confidence
intervals. It does not generate new observations or valid
confidence intervals. With `latexmk`, a TeX distribution, and the template's
font packages installed:

```bash
make -C paper iclr-draft
```

For GPU execution, read [RUN_LOCAL.md](RUN_LOCAL.md) first. A CPU-only
inspection of the intended matrix is:

```bash
python scripts/run_experiment.py \
  --config config/config_experiment.yaml \
  --model qwen2.5-32b --dataset hotpotqa --dry-run
```

The dry run creates configured directories and a log, but does not run
models. The 32B catalog entry is now present; this inspection does not
reproduce the archived study. Do not launch the unfiltered matrix as the
ICLR confirmatory study. The new
[controlled pilot](config/config_iclr_pilot.yaml) activates matched prompts,
equal new-step allowances, an origin sweep, and isolated paths. See
[pilot commands](RUN_LOCAL.md#controlled-development-pilot) and the
[reviewer report](docs/reviewer_report.md) for fixed defects and remaining
experimental blockers. The six-condition HotpotQA GPU run and artifact audit
are complete. Broader replication, historical provenance verification and
full-cost comparisons remain pending.

### Pipeline Map

| Stage | Entry point | Output |
| --- | --- | --- |
| Setup | `scripts/run_setup.py` | Downloaded dataset |
| Generate | `scripts/run_generate.py` | Pool, failures, trajectories, model choice |
| Score | `scripts/run_uncertainty.py` | Step-level uncertainty |
| Annotate | `scripts/run_annotate.py` | Judge reference labels |
| Localize | `scripts/run_localize.py` | Predicted origins and label agreement |
| Repair | `scripts/run_repair.py` | Deduplicated rollouts and repair rows |
| Analyze | `scripts/run_eval.py` | Summaries, secondary tests, figures |
| Controlled paired analysis | `scripts/run_paired_analysis.py` | Manifest-checked question-level paired contrasts |
| Human validation | `scripts/label_human.py` | Human labels and judge agreement |

These scripts contain checkpoints, but a changed code revision, model,
dataset, prompt, or budget requires a fresh run directory. Existing cached
results must not silently become results for a different experiment.

The [HotpotQA](run_colab_hotpotqa.ipynb), [MuSiQue](run_colab_musique.ipynb),
[2Wiki](run_colab_2wikimultihopqa.ipynb), and
[cross-dataset analysis](run_cross_dataset_analysis.ipynb) notebooks are
historical entry points. Inspect setup and cleanup cells before execution;
some delete Drive outputs. The [FEVER notebook](run_colab_fever.ipynb) is
diagnostic only until its dataset and prompting are repaired.

## Reproducibility and Release

The [readiness audit](docs/iclr2027_readiness.md#files-needed-from-google-drive)
lists the required raw Drive folders. A result release needs question IDs
and split manifests, evidence snapshots, exact model and tokenizer revisions,
resolved configurations, package and hardware versions, raw trajectories,
unique repair execution IDs, token accounting, and completed annotations.

The public repository is **not an anonymous supplement**: its URLs, historical
paper, notebook metadata, and author material can identify the author.
Prepare and inspect a separate anonymized archive without Git history;
do not link this author-identifying repository from the blind submission.

## Citation and License

No accepted ICLR paper or archival paper identifier is asserted here.
For current code use, cite the repository URL and the exact Git revision
used. Add the final paper citation only when its title, author list, and
publication record are established.

The historical README stated Apache-2.0, but this snapshot has no `LICENSE`
file. The repository owner needs to confirm and include the intended
license before a reproducible code release; this audit does not grant one.
