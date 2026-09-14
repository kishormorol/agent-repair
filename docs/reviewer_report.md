# Reviewer Assessment and Resolution Record

> Historical September 8 assessment. The subsequent controlled AWS study
> is complete. Use the [September 13 expert review and revision record](iclr_expert_review_2026-09-13.md)
> for current findings, implemented fixes and remaining gaps.

Reviewed September 8, 2026. This is an author-requested mock review, not an
official ICLR review. It covers the expanded manuscript, the historical
`paper/main.tex`, README, saved result summaries, and the execution and
analysis code. The historical manuscript and PDF have not been overwritten.

## Recommendation

**I would not recommend acceptance in the present evidence state.** The
central question is useful: does uncertainty identify a better repair
origin than simply restarting or choosing an earlier step? However, the
available results do not isolate that effect, and no raw trials are locally
available to establish its uncertainty or reproduce it. Additional prose
or a larger uncontrolled configuration grid will not resolve this.

This assessment follows the emphasis on a concrete question, supported
claims, and useful new knowledge in the
[ICLR 2027 reviewer guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines).
A controlled negative or conditional finding could be valuable; a positive
uncertainty advantage must not be assumed in advance. No experiment package
guarantees acceptance.

## Decision-Relevant Findings

### 1. Critical: The Primary Effect Is Not Identified

The archived uncertainty maxima are selected on the evaluation questions;
the HotpotQA and MuSiQue winners also receive gold-informed hints. Restart
has a different prompt, and later origins receive fewer new steps under
the historical total-step limit. Consequently, a success difference does
not isolate the contribution of uncertainty. See the manuscript's
[information access and budget definitions](../paper/iclr2027.tex) and
[archived evidence audit](iclr2027_readiness.md#critical-findings).

**Resolved locally:** matched generic restart hints, explicit equal-new-step
mode, fixed-early and random/backtracked controls, configurable budget
multipliers, and an all-origin execution sweep are implemented and CPU
tested. The manuscript labels the archived estimates exploratory.

**Still required:** recover all previously explored IDs, choose one policy
on development questions, freeze fresh disjoint test IDs and sample size,
and run the controlled comparison. Position-matched random selection fitted
on development data remains to be implemented. An all-origin sweep alone
does not solve test-set selection or establish causal attribution.

### 2. Critical: Reproducibility and Statistical Evidence Are Missing

Summary CSVs and rounded notebook displays cannot recover paired trials,
budget overruns, missing questions, judge fallback rates, or confidence
intervals. Historical baseline majority-success and uncertainty mean-success
rates were also mixed. The 854 archived failures are distinct questions;
three repair seeds do not make 2,562 independent questions.

**Resolved locally:** comparable mean-success summaries are used in the
draft. New analysis averages seeds within each question, bootstraps paired
question differences within datasets, weights datasets equally for the
macro contrast, and applies Holm correction to the declared contrast
family. Sign-flip tests explicitly state their exchangeability assumption.
Missing/duplicate trials, incomplete seed sets, and even a question omitted
by both methods are rejected against the frozen manifest. Legacy McNemar
outputs are labeled secondary majority-repeatability results.

**Still required:** recover the raw Drive artifacts listed in the
[readiness audit](iclr2027_readiness.md#files-needed-from-google-drive), audit
them without overwriting them, then analyze complete corrected runs with
[run_paired_analysis.py](../scripts/run_paired_analysis.py). Do not rebuild
historical intervals from rounded means. Prespecification remains a research
process requirement; a hash cannot prove when a policy was selected.

### 3. High: Novelty and Full-Cost Efficiency Are Unestablished

Backtracking and recovery-oriented intervention overlap with existing
diagnosis/replay methods. Random and restart alone are not a close-method
comparison. A generated-token cap excludes repeated prompt processing,
uncertainty acquisitions, diagnosis, and answer selection. Oracle candidate
coverage is not a deployable ensemble, and nominal strategy rows can reuse
the same physical generation.

**Resolved locally:** related-work claims are narrowed; unique executions,
prompt hashes, prompt/generated-token counts, model requests, and available
acquisition costs are recorded separately. Unknown costs remain unknown,
not zero. Stored-logprob-only scoring no longer loads a model for disabled
sampling metrics. The draft removes unsupported ensemble speedup claims.

**Still required:** one documented gold-free diagnosis/replay comparison
after reading its complete method, repeated restart with a frozen gold-free
answer selector, and acquisition-inclusive cost curves. Judge-call costs,
hardware/runtime behavior, prompt-cache effects, and replay overhead still
need auditing. Prompt counts are not FLOPs. A second model family is the
next priority for generality; without it, retain model-specific claims.

### 4. High: Judge and Mechanism Claims Exceed Their Evidence

No completed human labels are locally available. A search/lookup action
that disagrees with a judge is not necessarily correct exploration.
Restart-repair tool counts are not original trajectory lengths. A judge's
earliest error annotation is not a proven optimal repair origin.

**Resolved locally:** the exploration flag is renamed descriptively;
unsupported 70%/20% prevalence estimates, a four-step routing rule, causal
context conclusions, and assertions of completed human validation are
removed or explicitly qualified in the revised manuscript.

**Still required:** 150 failed trajectories sampled independently of repair
success, 50 per dataset, preferably two blinded annotators (300 initial
judgments), preserved disagreements and adjudication. Report exact/within-one
step and error-class agreement, uncertain cases, fallback counts, and
trace-based examples. The existing single-label CLI does not provide the
complete blinded multi-annotator workflow. A causal context claim needs a
separate controlled intervention; otherwise keep it a hypothesis.

### 5. High: The Runtime Could Silently Change the Experiment

Nested generated configurations resolved paths incorrectly, processed pools
and model caches were shared across runs, and the requested 32B model was
missing from the matrix catalog. Missing sampled-token log probabilities
were replaced with invented finite values; empty entropy distributions
appeared perfectly certain. These problems can invalidate a rerun even
when a notebook completes without an exception.

**Resolved locally:** isolated absolute paths, immutable matrix run IDs and
configurations, a 32B catalog entry, errors for empty filters and incompatible
model caches, explicit-only judge fallback, generation/repair input and
source manifests, held-out ID/exclusion checks, and resumable raw execution
caches. Missing or invalid log probabilities remain missing; nonfinite
scores cannot select an origin. If no score is usable, the legacy origin-zero
fallback remains and its frequency must be reported. Entropy is over the
returned alternatives, not the full vocabulary.

**Still required:** a real vLLM pilot on the intended hardware. Pin exact
model/tokenizer snapshots and package versions; name/dtype checks do not
pin revisions. Stage-2 scoring and judge stages still use legacy checkpoints,
so changed settings or code require a fresh complete run. CPU fixtures do
not validate GPU memory use, token limits on actual generations, truncation,
batching, evidence quality, or scoring correctness.

## Concrete Next Experiments

| Order | Run | Completion criterion |
| --- | --- | --- |
| 0 | Development GPU pilot: 30 initial HotpotQA questions, recorded 32B agent, all eligible origins, three seeds, 0.5x/1x caps | Every execution within its token/new-step allowance; identical collapsed-restart prompt hashes; correct prefix replay and scoring; complete trials; measured throughput and missing-logprob rate. This is not an efficacy result. |
| 1 | Frozen QA3 comparison: one development-selected uncertainty policy versus matched restart, matched random and fixed-early | Disjoint IDs, prespecified sample size/precision, full raw outcomes, paired macro and per-dataset effects, strict EM/F1 sensitivities, uncertainty and negative results reported. |
| 2 | Same-policy offset 0/2 plus development-fitted position-matched control | Quantify origin position, how often backtracking becomes restart, and whether any gain remains beyond early placement. Reuse compatible origin executions. |
| 3 | Close diagnosis/replay adaptation and repeated restart with declared total allowance | Gold-free prompts and final-answer selector; explicit adaptation details; complete acquisition/selection costs and unique execution counts. |
| 4 | Human study, alongside GPU work | 150 traces, ideally two independent labels each; source records, ambiguity handling and agreement analysis. |
| 5 | Core comparison with a second model family | Separately generated initial failures and model-specific results; no mixing of model cohorts. |

The pilot profile is [config_iclr_pilot.yaml](../config/config_iclr_pilot.yaml);
commands and outputs are in [RUN_LOCAL.md](../RUN_LOCAL.md#controlled-development-pilot).
Its uncertainty strategy is a smoke-test candidate, not a selected winner.
Do not launch the broad historical matrix or rebuild FEVER before the core
validity issues are addressed. Full details are in
[experiments_to_run.md](experiments_to_run.md).

## Verification and Delivery Status

CPU tests cover sequential/batched allowances, paired inference, complete
cohorts, all-origin coverage, shared executions, interrupted-run recovery,
configuration/model incompatibility, prompt accounting, and missing
uncertainty measurements. The synthetic repair workflow performs 16 unique
executions and produces 32 policy rows, then resumes without more model
calls; these are test fixtures, not research results.

Run `python -m pytest -q tests` and `make -C paper iclr-draft` from the
repository root. Updated verification on September 8: **112 CPU tests passed**;
the budget-aligned draft builds with nine main-text pages and 16 pages total, six tables
and four figures. All rendered pages were inspected, citations resolve,
and PDF author metadata is empty. The 32B/HotpotQA matrix dry run selects
exactly one experiment without model loading. `git diff --check` passes.
No new GPU observations, human labels, or submission were
produced by this review. The historical point estimates are unchanged.

The [official call](https://iclr.cc/Conferences/2027/CallForPapers) lists
September 18, 2026 for abstracts and September 25 for papers, both 11:59
p.m. AoE, checked September 8. The immediate dependencies are the raw Drive
outputs, a compatible GPU and fixed environment, and human annotators.
The final paper must follow the completed evidence, including an inconclusive
or negative result if that is what the controlled study finds.
