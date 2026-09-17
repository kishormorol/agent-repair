# Full-Paper Completion Plan

This checklist concerns the manuscript and scientific work only, not
conference submission. The updated paper is `paper/iclr2027.tex`; its PDF
is `output/pdf/agent-repair-iclr2027-draft.pdf`. The original full manuscript,
`paper/main.tex`, is unchanged and should be the structural backbone of the
revision. The expanded ICLR working draft now restores the substantive
sections, with main text, statements, references and technical appendices.
The [September 9 section review](paper_section_review.md) records the earlier
section revisions, literature and evidence checks. The authorized main,
diagnosis, replication and runtime experiments are complete and independently
audited; human validation and broader optional extensions remain separate.
The working manuscript now includes the
six-condition results, paired intervals, audit counts, cost estimate and
historical-ID limitations, with a synchronized abstract and conclusion.
The longer historical exploratory analyses are preserved in the appendix.
The [September 13 expert review](iclr_expert_review_2026-09-13.md) is the
current assessment and next-experiment design. It adds completed execution
overlap, policy-cost, termination, scoring and example analyses, plus a
verified [statistical supplement](../output/reviewer-2026-09-12/agent-repair-statistical-supplement.zip).
The older restoration map below remains a record of intended scope; it is
not a request to repeat the completed HotpotQA comparison or pilot.

## Current Execution

The [September 14 extension](aws_extension_2026-09-14.md) completed its
separately frozen two-model, three-dataset comparison on September 15 at
18:00 UTC. All six cells, 600 main evaluations, 4,323 unique repairs and
8,016 trial rows pass independent audits and local/remote reproduction
with zero mismatches. All 24 primary Holm-adjusted p-values are 1.000.
The final recovery verified both archives and all inherited records;
AWS independently confirmed stopped with its GPU configuration restored.
The resource-only amendment kept the $50 extension cap inside the original
$119 allocation and $20 reserve. The reconciled extension estimate is
$43.90 and combined retrieval is bounded by $1.22, subject to the report's
shutdown assumptions and tax/transfer exclusions.

The frozen design retains 100 main and 20 development questions per dataset,
seven policies, three repair seeds and one 24-comparison primary family.
Full replication reporting now includes all six audits and matching local/remote exports.
The completed 25-question runtime component is separately reportable under
its prespecified two-comparison ten-second family and is now in the paper,
alongside the completed diagnosis follow-up. The abstract matches the manuscript.

## Active Budget Constraint

The [completed September 12 main study](aws_main_2026-09-12.md) evaluated the
position-matched random and no-backtracking controls for 250 HotpotQA
questions in five batches. Its $25 gross allowance fits inside the existing
core envelope. All five batches, 1,058 unique repairs and 2,034 condition/seed
rows are downloaded and audited. Uncertainty repair succeeded on 9.73% of
failed-question seed trials versus 11.21% for restart; no primary comparison
showed a clear advantage. Estimated infrastructure usage is $9.16 before
credits, tax and transfer. Historical exclusions are reconstructed, with
the original full-file comparison still unavailable. Multi-dataset replication
and the same-model diagnosis/replay adaptation are complete; reproducing
the full trained prior method remains outside the executed scope. Human validation
remains necessary if making judge-localization or mechanism claims; it is
not needed to calculate the main deterministic answer metric.

The current allocation is **$119 using AWS credits**, not $119 plus the
earlier $200 cash fallback. Follow the
[reduced execution plan](../CLOUD_GPU_SETUP.md#current-allocation-119-on-aws):
one 80GB-class GPU, one model, three QA datasets, four matched core conditions,
three seeds, and one token allowance. The portable notebook now disables
the full-origin sweep and extra offset diagnostics by default. Reserve
$20 for infrastructure and contingency; the other $99 includes setup,
development, initial generation, repair and retries. These are allocations,
not measured costs or an assurance of paper acceptance.

The broader experiments below remain the scientific wish list and validity
criteria, not authorization to exceed that ceiling. Prioritize the core
comparison and essential position/nearest-method checks over a new dataset,
second model family or large judge. Use pilot throughput and development
variance to freeze an affordable test cohort. If that cohort cannot resolve
small effects, report the resulting uncertainty; do not treat an
underpowered null as equivalence or replace missing controls with claims.
Any omitted diagnostic must be removed from claimed contributions or
identified explicitly as a limitation. Human annotation has not been funded
or performed by this plan.

## Goal and Research Decision

Complete a full manuscript with reproducible results, defensible claims,
restored substantive analyses, and an anonymous supplement. Completion means
every reported empirical claim can be traced to records and analysis, not
merely that the PDF compiles or reaches a page count.

Recommended central question: **Does uncertainty help choose a useful
repair origin beyond simply choosing an earlier step, at a declared
recovery allowance?** Treat agreement with a judge's error location as a
secondary diagnostic. Neither backtracking alone nor a large configuration
grid establishes a new contribution.

Keep the three QA datasets and the archived 32B configuration as the core.
Prioritize a second model family over an eight-model grid, additional budget
settings, or FEVER reconstruction. A one-model study is possible but supports
narrower claims. The nearest-method comparison below is important for
positioning; random and restart alone do not resolve overlap with existing
diagnosis-and-replay work.

## Restore the Original Paper

Revise `paper/iclr2027.tex`, not the untouched historical `paper/main.tex`.
Keep `paper_title_abstract.md` synchronized and extend
`scripts/build_iclr_draft.py` only when the required inputs are available.

| Original content | Completion action |
| --- | --- |
| Motivation, problem, uncertainty measures, localization rules, repair procedure | Preserve and sharpen. Distinguish a suspected error step from a useful regeneration origin, and define the actual available information. |
| Main repair results and backtracking | Restore full tables and plots using a frozen held-out comparison. Move the exploratory configuration grid to the appendix. |
| Localization | Keep, with judge agreement and human agreement clearly separated from repair utility. Do not call the judge an oracle upper bound. |
| Cost and Pareto analysis | Restore using unique executions, prompt and generated tokens, acquisition costs, and a gold-free selection policy. Oracle candidate coverage is a diagnostic, not a deployable ensemble. |
| Informed versus generic nudges | Retain as a paired diagnostic at identical origins and allowances. Gold-informed hints must stay outside the primary comparison. |
| Trajectory-length and failure-mode analysis | Recompute from original failed trajectories, not restart continuations. Validate whether exploratory actions are correct; the existing search/lookup flag does not establish correctness. |
| Discussion, limitations, conclusion | Restore depth while separating observations, hypotheses, and deployment recommendations. Remove the unsupported four-step routing threshold and unverified 70%/20% failure-mode proportions. |
| Prompts, full configurations, additional plots, annotation instructions, examples | Preserve in a technical appendix rather than deleting useful detail to fit the main text. |

Saved notebook outputs contain some recoverable baseline costs and paired
nudge summaries. They can help reconstruct historical descriptive tables,
but are not replacements for raw logs, valid paired intervals, or corrected
held-out experiments. Do not mix different saved execution snapshots.

The historical restoration uses hashed CSVs and specified notebook cells.
The completed main study adds audited result, interval, execution and cost
tables and a paired-interval figure; review-stage additions supply overlap,
policy-cost, EM/F1 and full-cohort tables plus three reproducible illustrations.
Remaining entries in the map concern evidence completion
and replacing exploratory estimates, not recovering deleted prose.

## Essential Experiments

Before spending on a confirmatory cohort, pass the replay-observation and
dataset-source/scoring checks in the section review. The ten-question GPU
pilot is development-only. The closest task-matched prior is now
[Doctor-RAG](https://arxiv.org/abs/2604.00865v2); a clearly labeled same-model
diagnosis/replay adaptation may be cheaper than reproducing its trained
system, but is not a substitute to be mislabeled as that reproduction.
[SymTrace](https://arxiv.org/abs/2608.25920v2) also makes the
repair-versus-resampling framing insufficient as a stand-alone novelty claim.

Keep the existing four-condition pilot unchanged. The separate main study
added and completed the position-distribution and no-backtracking controls.
The [September 14 follow-up](aws_diagnosis_2026-09-14.md) completed the
same-model diagnosis/replay adaptation and 0.5×, 1× and 2× allowances on a
separate 250-question cohort. All five batches, 3,186 trial rows and paired
comparisons are independently audited; uncertainty showed no clear advantage
in the six primary contrasts. The extension completed all six replication
cells and a 25-question measured-runtime comparison. Active deadline
cancellation, full end-to-end cost and a full trained-method reproduction
remain untested.
Any further additions require implementation, an affordable allocation and
a new frozen condition manifest before test outcomes. Report the completed
six-condition comparison and the separate diagnosis follow-up with their limitations.

| Priority | Experiment | Concrete design | Evidence it should produce |
| --- | --- | --- | --- |
| 1 | Cost-aware recovery comparison | Completed: gold-free diagnosis/replay, three allowances, measured policy costs and the prespecified ten-second runtime comparison. End-to-end latency and active cancellation remain outside scope. | Diagnosis and runtime results, costs and paired intervals are in the reports and manuscript. Adaptations are labeled explicitly. |
| 2 | Generality of the controlled comparison | Completed: Qwen32B and Mistral12B on HotpotQA, MuSiQue and 2Wiki, with 100 main and 20 development questions per dataset and model. | All six audited cells, model/dataset-specific repair rates, and the fixed 24-comparison family. |
| 3 | Backtracking and position controls | Complete in the main study and all six replication cells, including no-backtracking and position-matched random. | Policy tables and paired intervals are in the manuscript; broader new claims need new frozen cohorts. |
| 4 | Human validation, if retaining localization claims | Sample failures independently of repair success. The original three-dataset target is 50 traces per dataset, preferably two blinded annotators with disagreements preserved. | Exact/within-one step and error-class agreement, uncertainty and ambiguity counts. No annotations have been completed. |

Judge-targeted conditions are privileged diagnostics, not a proven upper
bound. Primary uncertainty conditions must not use judge-informed hints.
If completed human labels or compatible raw runs already exist elsewhere,
audit and reuse them where valid instead of repeating that work.

## Shared Repair-Origin Sweep

Use one explicit origin-outcome matrix to support the main comparison,
backtracking, localization-versus-utility analysis, and origin-cost plots.
For each frozen failed trajectory with `T` eligible steps, run repair from
every origin `k = 0, ..., T - 1`, with the same generic hint, three fixed
generation seeds, and identical new-generation allowances. Origin zero is
the matched restart. Freeze policies on development data before opening
held-out outcomes, then look up each policy's prespecified selected origin.
Charge that policy for its selected execution and acquisition overhead,
not for the whole evaluation sweep.

With the current eight-step limit, this is at most `24 * N_failed` unique
repair executions per model, hint, and budget setting. It excludes initial
trajectory generation, judge calls, extra uncertainty sampling, additional
hints, and diagnosis-based baselines. The old runner already deduplicates
strategies selecting the same origin and hint: this proposal improves
coverage and accounting, and is not a demonstrated GPU speedup.

Under the active $119 allocation, use the four core conditions on the full
planned test set. A small prespecified diagnostic sweep is optional only
if measured cost leaves room for essential controls first. The previous
preference for a complete sweep is deferred. Choose scope using pilot cost
and development variance, not test success.

Three noisy outcomes per origin do not identify a true optimal origin.
Do not select the best test origin and call it a deployable method. Any
best-origin diagnostic needs either an explicitly optimistic label or
independent origin-selection and evaluation seeds on a diagnostic subset.
An origin sweep also does not by itself establish causal error attribution.

## Controls to Fix First

- Recover all previously explored question IDs. Use those questions for
  development and fresh, disjoint IDs for confirmation. Freeze selection
  before examining new test outcomes.
- Give all primary methods the same generic retry prompt, including restart,
  the same generated-token cap, and the same number of **new** steps.
  Implemented via `match_restart_hint: true` and `step_budget_mode: new`,
  with CPU regressions. Existing historical outputs remain confounded.
- Isolate processed pools, saved model choices, and checkpoints by dataset,
  model, split, and immutable run ID. Do not resume an old experiment under
  new settings.
- Run a small real-GPU pilot to verify budget enforcement, prefix replay,
  model identity, and scoring. CPU regression tests are not this experiment.
  Compare stored and replayed tool observations and page/lookup state. The
  completed AWS development and main audits verified observations and replay
  state. Repeat these checks for changed runtimes or environments. Preserve
  failed checks as infrastructure defects.
- Plan held-out sample size using development variance and a stated precision
  target. The historical 854 failures are not the size of the future test
  set, and three seeds do not triple the number of independent questions.

The fallback four-condition comparison costs at most `12 * N_failed`
executions before other conditions, with exact duplicates shared. GPU hours
cannot be estimated reliably without a pilot on the intended hardware and
models. Estimate question count from paired-difference variance and the
desired interval width; do not substitute a round sample count for this.

These controls are now implemented in the repair planner/runner, both
agent loops, and configuration/cohort handling. CPU tests cover origin
coverage, identical collapsed-restart execution, new-step allowances,
physical execution reuse, split isolation, and interrupted-run recovery.
The [current expert review](iclr_expert_review_2026-09-13.md) records the remaining gaps;
[RUN_LOCAL.md](../RUN_LOCAL.md#controlled-development-pilot) provides a
bounded development pilot. Position-matched random is implemented and
evaluated. The same-model diagnosis/replay adaptation is also complete;
the full trained-method reproduction and human-labeling workflow remain
outside scope. Human labels are conditional on retaining localization claims.

## Required Analyses, Not New Model Runs

Average repair seeds within each question; compute paired strategy deltas
and bootstrap questions, not individual seed rows. Report strict exact
match and token F1 alongside the historical `EM == 1 or F1 >= 0.5` gate.
Audit missing trials, duplicate executions, budget overruns, and model
identity. Analyze localization agreement separately from repair success.
Raw records are needed; summary CSVs cannot supply these checks. Use
`scripts/run_paired_analysis.py` for the controlled study: it verifies
question/seed coverage against the frozen manifests and does not perform
test-outcome-based policy selection. First compute EM/F1 on the same frozen
failure cohort. An EM-only failure gate defines a larger eligible population
and needs outcomes for its additional initial failures; it cannot be created
by changing the success column on already selected rows.

Required outputs are a paired main-results table, backtracking/position
controls, localization-versus-repair analysis, cost curves, human agreement,
and trace-based failure examples. The last two can proceed while GPU runs
are running. Do not infer correct exploration from action type alone, or a
repair-selection rule from a model trained only to predict judge agreement.
The paired main table, backtracking/position controls, recorded policy costs,
EM/F1 sensitivities and explicitly outcome-selected examples are now supplied
for the main HotpotQA study. Full cost curves and localization/human evidence
remain absent. The examples do not establish failure-type prevalence.

## Stronger Evidence and Optional Extensions

- **Completed generality check:** the frozen core comparison now includes
  two model families on three QA datasets. Claims remain specific to those
  evaluated models and the offline environment.
- **Completed allowance check:** the separate Qwen HotpotQA diagnosis cohort
  evaluates 0.5x, 1x and 2x recovery-token caps. This does not establish
  allowance robustness for every replication cell or equal total compute.
- **Only if retaining a causal context claim:** intervene on retained
  context while controlling evidence, prompt, and recovery allowance.
  Restart winning alone does not establish context contamination.
- **Optional FEVER extension:** rebuild real Wikipedia evidence and the
  task-specific prompt, then rerun the full dependent pipeline. Otherwise
  leave FEVER out of the empirical claims.

The [full protocol](iclr2027_study_protocol.md) specifies selection,
statistics, related-work comparisons, and implementation requirements.
These experiments may support positive, negative, or inconclusive findings;
the final claims must follow the observations.

## Deadline Workback

The abstract deadline is September 18, 2026, and the paper deadline is
September 25, 2026, both 11:59 p.m. AoE. Main text is limited to nine pages;
references and appendices can preserve additional detail. Sources:
[official call](https://iclr.cc/Conferences/2027/CallForPapers) and
[author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).
Submission itself remains the author's responsibility; this plan does not
authorize uploading, registering, or submitting anything.

| Target dates | Deliverable and gate |
| --- | --- |
| September 8-9 | Full structure and CPU-tested core controls are restored. Recover and audit raw records; complete the real-GPU pilot. Freeze scope only after measured throughput and development variance are available. |
| September 9-14 | Execute the frozen core study; begin blinded human labeling and draft the full methods/appendix concurrently. Preserve all failures and infrastructure retries. |
| September 14-18 | Complete backtracking/position and cost/nearest-method comparisons; prioritize second-family replication if the core is sound and resources permit. Prepare a genuine abstract based only on established work. |
| September 18-22 | Finish prespecified runs, paired statistics, all main figures, failure examples, and the complete results/discussion revision. Do not extend runs to chase significance. |
| September 23-24 | Reproduce tables from raw records; audit claims/citations, anonymity, AI-use disclosure, supplement, and every rendered PDF page. Keep the final day as buffer. |

This is a target schedule, not a runtime guarantee. The main HotpotQA raw
records, controlled experiment and GPU audits are complete. Missing historical
raw records still prevent corrected historical intervals; they do not block
reporting the independently audited main study. New inference depends on an
affordable frozen design and GPU availability. Human annotators are needed
only for the proposed localization/mechanism extension.

The close-comparator adaptation and additional allowances are complete.
Independent replication and measured runtime await retrieval and final
audits under the separate extension protocol. The
[current review](iclr_expert_review_2026-09-13.md#concrete-next-experimental-design)
specifies the design boundaries. Statistical packaging is complete; full
GPU replay packaging and final author review remain separate work.
