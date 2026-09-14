# ICLR 2027 Study Protocol

Updated September 12, 2026. **The six-condition HotpotQA main study is
completed; the broader protocol below remains partly proposed and was not
publicly preregistered.** Its resolved HotpotQA choices were frozen before
new test outcomes. The [readiness audit](iclr2027_readiness.md) records earlier
evidence and blocking defects. Publication cannot be guaranteed by a
checklist; the results must contribute useful, well-supported knowledge.

## Contribution to Establish

The candidate contribution is a controlled account of when inexpensive
uncertainty measurements improve the choice of a regeneration origin in
a single tool-using QA agent. The current tables do not establish this.
Neither a large strategy grid nor the use of backtracking is sufficient
novelty on its own.

The completed [AWS main study](aws_main_2026-09-12.md) contains six
conditions on 250 HotpotQA questions, three seeds, one token allowance, and
a $25 allowance within the $119 total ceiling. The position and
no-backtracking controls were evaluated across all five batches. All 1,058
unique repairs and 2,034 condition/seed rows passed independent audit.
The four primary comparisons found no clear uncertainty-repair advantage;
this is not evidence of equivalence. Other expanded controls below remain
future work. See the
[section review](paper_section_review.md) for the September 9 manuscript audit.

### Closest Related Work

| Work | Relevant overlap | Required distinction or comparison |
| --- | --- | --- |
| [Doctor-RAG, v2 June 12, 2026](https://arxiv.org/abs/2604.00865v2) | Localized repair on the same three QA benchmarks. | Highest-priority task-matched comparator; control evidence access, diagnosis overhead and failure gate. |
| [SymTrace, v2 August 29, 2026](https://arxiv.org/abs/2608.25920v2) | Repair versus resampling using prefix reconstruction. | Audit replay fidelity separately from repair outcomes; do not claim deterministic suffix replay. |
| [AgentRewind, August 14, 2026](https://arxiv.org/abs/2608.14380v1) | Context and workspace rollback with rewind memory. | Our local search/lookup replay does not restore arbitrary external effects. |
| [AgenticRAG-FP, August 20, 2026](https://arxiv.org/abs/2608.20627v1) | Attribution against injected faults after downstream re-execution. | Natural failed traces do not supply known causal error locations. |
| [AgentRx, v2 August 31, 2026](https://arxiv.org/abs/2602.02475v2) | Constraint-supported diagnosis; the revision reports acceptance to Findings of EMNLP 2026. | A reference localizer is not a recovery policy. Use current metadata, not the older 115-trace description. |
| [ReAgent, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.202/) | Reversible multi-agent QA reasoning. | Rollback and multi-hop QA are established prior work, not new contributions here. |
| [DoVer, ICLR 2026](https://www.microsoft.com/en-us/research/publication/dover-intervention-driven-auto-debugging-for-llm-multi-agent-systems/) | Tests debugging hypotheses through interventions and evaluates task recovery rather than attribution alone. | Do not claim the attribution-versus-recovery distinction is new. Evaluate inexpensive origin selection against a diagnosis-based alternative with explicit information and cost accounting. |
| [ERGO, UncertaiNLP 2025](https://aclanthology.org/2025.uncertainlp-main.23/) | Uses changes in entropy to trigger conversational prompt consolidation. | Distinguish resetting multi-turn input from suffix regeneration in a fixed tool environment. Do not claim the first uncertainty-triggered reset. |
| [REFLECT, June 2026 preprint](https://arxiv.org/abs/2606.09071) | Uses diagnosis-specific replay around candidate error steps to refine attribution. | Backtracking around a suspected error is already explored. Compare repair-origin quality without privileged hints; label any adapted baseline as an adaptation. |
| [Who&When, ICML 2025](https://proceedings.mlr.press/v267/zhang25cq.html) | Evaluates failure attribution to agents and steps. | Reference-step agreement is a secondary outcome; report whether it predicts successful repair in this single-agent setting. |
| [Causal Agent Replay, June 2026 preprint](https://arxiv.org/abs/2606.08275) | Studies step interventions and outcome-distribution changes for attribution. | An origin sweep is an evaluation design, not a claim to have invented intervention-based attribution. Full-method review is still required before choosing a comparison. |
| [CausalFlow, May 2026 preprint](https://arxiv.org/abs/2605.25338) | Connects counterfactual step attribution with targeted repair. | Position cheap uncertainty selection against existing repair-oriented attribution, rather than claiming the first connection between localization and repair. Full-method review remains pending. |

This is a targeted related-work check, not an exhaustive novelty search.
Read the complete method and evaluation of a chosen baseline before
implementing it. Do not compare percentages from different benchmarks as
head-to-head results. REFLECT is cited as a preprint, not an accepted
conference publication.

## Scope and Data Freeze

1. Retain HotpotQA, MuSiQue, and 2WikiMultiHopQA with their offline evidence
   environments. Exclude FEVER unless both evidence and prompting are
   repaired and the entire dependent pipeline rerun.
2. Recover the IDs of every previously explored question. Treat these as
   development data; do not relabel them as an untouched test split.
3. Create a deterministic, disjoint held-out ID list from remaining eligible
   records. Preserve dataset strata, deduplicate IDs, and hash the manifests
   and evidence before generating test trajectories. If old IDs cannot be
   recovered, do not claim disjointness.
4. Freeze the agent and tokenizer revisions, quantization, prompts, tools,
   decoding, step limits, and failure gate. Use the same initial failed
   trajectory for every strategy on a question. A second model needs its
   own initial failures and must be reported separately.
5. Use development data to estimate paired-difference variability and plan
   test sample size for a stated precision target. Record the maximum
   number of questions and compute allowance in advance. Do not stop or
   expand the run based on whether a desired result becomes significant.

Begin with the archived 32B agent configuration only if the available GPU
supports it. Do not silently substitute a smaller model, tokenizer, judge,
or quantization. Add a second model family after the core experiment works;
without that replication, keep the conclusion explicitly model-specific.

## Minimal Controlled Comparison

Primary conditions use no gold answer, gold evidence labels, judge error
type, or test outcome to select an origin or final answer. The benchmark
gate may identify that the original attempt failed, consistently for all
conditions. This remains an offline known-failure experiment.

| Condition | Origin | Role |
| --- | --- | --- |
| Restart, matched prompt | Zero | Primary baseline |
| Uniform random, matched backtracking | Uniform eligible step, then the selected method's offset | Primary localization control |
| Fixed early origin | `min(1, T - 1)` with zero-based indexing | Tests whether retaining almost no prefix suffices |
| Selected uncertainty method | One development-selected signal, rule, and offset | Primary treatment |
| Same method without backtracking | Identical signal and rule, offset zero | Separates score information from upstream movement |
| Position-matched random | Sample normalized origins from a distribution fitted on development data only | Tests an early-origin preference without question-specific uncertainty |
| Judge-targeted reference | Judge annotation, generic prompt | Privileged diagnostic, not an upper bound |
| Judge-targeted plus backtrack two | Same judge and prompt | Diagnostic origin sensitivity |

Select one uncertainty configuration globally using development macro-mean
repair success across the three datasets, with deterministic tie breaking.
A bounded candidate grid is three stored-token signals (entropy,
perplexity, sampled-token complement), three rules (argmax, top-three
earliest, position-weighted), and offsets zero/two. Do not use informed
variants for this selection. Log every candidate tried. Additional choices
after inspection belong to an explicitly exploratory analysis.
A single policy can instead be specified without an outcome-based search;
record the rationale before evaluation. The notebook's perplexity/argmax/
backtrack-two setting is a pilot candidate, not a demonstrated winner, and
the broader candidate grid is not automatically included in the $119 plan.

All primary conditions receive the exact same generic retry message and
sampling temperature, including restart. Use three prespecified generation
seeds per question. Pair outcomes by question and seed, while recognizing
that matching seed numbers does not make different prompts identical draws.
Backtracking that reaches zero must have the same effective prompt and
allowance as the corresponding restart; verify this with a prompt hash.

Hold the **new-generation step allowance** constant across origins for the
primary comparison, as well as the generated-token cap. The revised runner
supports `repair.step_budget_mode: new`, tested in both execution loops;
activate it together with `repair.match_restart_hint: true`. Keep the historical
total-step convention as a separately labeled sensitivity analysis.

The first six rows were evaluated in the completed HotpotQA study; the notebook
defaults retain four conditions until a frozen position profile is supplied.
The implemented position control samples the empirical normalized effective
origins of failed development trajectories and maps them to each target
length with nearest-position, half-up rounding. This supplies a deterministic
rule for unseen lengths without fitting test uncertainty profiles.
Report actual origin distributions and frequency of collapse to restart.

### Execution Layout: Core First, Sweep Deferred

The active six-condition HotpotQA core requires at most `18 * N_failed` repair
executions; identical effective executions are shared. This excludes initial
generation, diagnosis, setup and development. It is not a wall-clock bound.
The position control and closest-method comparator take priority over an
all-origin sweep or another dataset. Revise the condition list and precision
plan before test outcomes if pilot costs permit those additions.

The broader diagnostic design is a complete matrix over eligible origins
`0, ..., T - 1` and the three fixed generation seeds, under one shared
generic hint and allowance. Each frozen policy selects its origin without
looking at held-out repair outcomes; identical effective executions are
reused, not counted as independent trials. Random controls use prespecified
origin draws, and position-matched controls use development-only fits.
Policy cost includes only its required execution and selection/acquisition
work; the full matrix is evaluation overhead.

At an eight-step limit this costs at most 24 repair executions per failed
question for one model, hint, and budget. The existing runner already
deduplicates origin/hint selections, so this is not a measured speedup over
the historical strategy grid. If the pilot shows the complete matrix will
not fit the available compute, freeze a four-condition core on the full
test set and an all-origin diagnostic on a prespecified stratified subset.
Resolve and record this decision before inspecting test outcomes.

The matrix supports origin sensitivity, not a claim that the empirical
maximum over three seeds is the true best repair origin. Keep any such
maximum explicitly optimistic, or use independent seed groups for selecting
and evaluating an origin on a diagnostic subset. Do not turn test-origin
selection into a claimed deployable baseline or infer causal attribution
from the sweep alone.

## Costs and Closest-Method Comparison

Start with an audited cap `B(q) = original_generated_tokens(q)` and require
every outcome to satisfy `recovery_gen_tokens <= B(q)`. Changing this rule
requires a new protocol version, run directory, and complete rerun.

Record prompt tokens, generated tokens, uncertainty acquisitions, selection
calls, unique repair executions, tool calls, and elapsed inference time.
Reused results are one execution, not several independent observations.
Count evaluation-only judging separately from calls used by a recovery
policy. Do not call generated-token matching equal FLOPs or equal total cost.

For sampling-based uncertainty or diagnosis-and-replay methods, add a
repeated-restart baseline at the same declared incremental allowance.
Freeze a gold-free final-answer selector (for example normalized-answer
plurality with deterministic ties) and charge selection costs. Report
oracle candidate coverage separately. A small adapted diagnosis/replay
comparison can test whether the cheap signal offers value; document the
adaptation rather than claiming a full Doctor-RAG, DoVer or REFLECT reproduction.

For a same-model diagnosis/replay adaptation, first evaluate its origin with
the same generic hint and prefix-only recovery information as uncertainty.
A separate full-policy arm can use diagnosis-specific guidance, with access
and costs declared. Never use reference answers or supporting-fact labels in
either selector. Doctor-RAG's published rates are not a comparison against
our system. Its corpus, failure gate and operators differ. A faithful
reproduction additionally requires validating the released implementation,
checkpoints and training setup; this review has not completed that work.

Do not make causal context-contamination claims from these comparisons.
They change retained information and feasible continuation jointly. A
mechanism claim needs a separate intervention that manipulates prefix
content while controlling prompt, evidence access, and recovery allowance.

## Estimands and Analysis

For question `q`, strategy `a`, and generation seed `r`, let `Y(q,a,r)`
indicate repair success under the declared gate. Compute the seed mean
within each question, then average questions equally. Report both the
conditional repair rate and the original failure count. The primary gate
remains `EM == 1 or F1 >= 0.5` for comparability; strict EM and token F1 are
required sensitivity outcomes, not substitutes chosen after seeing results.

These sensitivity metrics first reuse the same frozen failure cohort.
Changing to an EM-only failure gate changes eligibility and requires outcomes
for the extra initial failures. Do not silently change the denominator.
For an ideal reference-gated, single-repair policy leaving successful initial
answers untouched, population success is `A0 + (1 - A0) * repair_rate` under
one common correctness criterion. It does not measure an online detector's
misses, false alarms, regressions or cost.

The two primary contrasts are selected uncertainty minus matched restart,
and selected uncertainty minus matched random. For each contrast:

1. Require identical question and seed coverage. Do not drop failed jobs or
   missing cells from only one strategy. Resolve infrastructure failures
   under a documented retry rule; retain genuine budget and format failures.
2. Compute paired differences of question-level seed means, then bootstrap
   entire questions with their paired outcomes and all seeds intact.
3. Use 10,000 bootstrap resamples with a fixed analysis seed and report
   95% intervals for effect sizes in percentage points. Report each dataset
   and an equal-dataset macro-average; do not pool all seed rows as trials.
4. For confirmatory testing, use question-level paired sign-flip permutation
   tests for the two macro-average contrasts, state the exchangeability
   assumption, and apply Holm correction to that prespecified family.
   Label dataset-specific intervals and other comparisons exploratory.
5. Preserve negative and inconclusive outcomes. A non-significant difference
   is not evidence of equivalence. A practical equivalence or noninferiority
   claim requires a margin fixed before the test results are opened.

`scripts/run_paired_analysis.py` now implements manifest-checked paired
question means, within-dataset bootstrap resampling, equal-dataset macro
contrasts, sign-flip tests, Holm correction, and EM/F1 sensitivities. It
does not choose a winning strategy. CPU tests cover the inference logic,
not actual study results. `summarize_strategies` supplies per-strategy
intervals; the labeled legacy McNemar output tests majority-seed
repeatability and cannot replace the primary analysis.

## Human Validation

Sample 50 failed trajectories per QA dataset independently of repair
success, for 150 trajectories total. Prefer two annotators labeling every
sample independently, blinded to judge labels, method identity, and repair
outcomes. This means 300 initial judgments, not 150 independent experiments.
Record uncertain or unjudgeable cases, original labels, and adjudication
separately. Report exact/within-one step agreement, error-class agreement,
sample counts, uncertainty, and the annotation instructions.

The existing single-annotator CLI is only a starting point: it does not
manage blinded multi-annotator records or adjudication. Human reviewers need
the question, answer rubric, and evidence to assess correctness; a forced
step number alone does not establish a uniquely causal error. Do not create
human labels with an LLM or claim completed validation without the records.

## Implementation Gates Before GPU Runs

Checked entries mean implemented and CPU tested, not validated on a GPU or
used to produce new empirical findings. See the
[reviewer report](reviewer_report.md) and [pilot commands](../RUN_LOCAL.md#controlled-development-pilot).

- [ ] Recover and hash raw artifacts; verify coverage, model identity,
  deduplication, invalid evidence, and historical budget overruns.
- [x] Isolate processed pools, model-choice caches, checkpoints, and outputs
  by dataset/model/run ID; the resolved dataset configuration freezes split
  information. Generation/repair input and source manifests reject changed
  runs; the matrix also refuses changes to its resolved configuration.
- [x] Enforce held-out ID/exclusion manifests and preserve frozen ID order.
  Explicit profiles now override catalog sample-size defaults. Completeness
  of the historical exclusion list still requires external verification.
- [x] Implement explicit strategy selection, honored baseline lists, uniform
  random with configurable backtracking, and a fixed-early control.
- [ ] Implement the separate development-fitted position-matched random control.
- [x] Equalize retry prompts and new-step limits; test prefix allowances and
  equality when a backtracked origin collapses to restart.
- [ ] Reject mismatches between stored and replayed observations; validate
  page, lookup cursor and retrieved-title state on frozen records.
- [ ] Pin dataset sources and conversions, preserve source hashes, and audit
  paragraph splitting, answer aliases and custom versus official scoring.
- [ ] Verify remaining-token limits with a real vLLM smoke test, including
  heterogeneous batched budgets. CPU stubs are not GPU validation.
- [x] Add raw execution IDs, prompt hashes/counts, generated-token and request
  counts, available uncertainty-acquisition costs, and paired analysis.
- [ ] Complete full-cost auditing, including judge costs, replay overhead,
  model/tokenizer revision pins, serving versions, and runtime validation.
- [x] Test eligible-origin coverage, policy-to-execution mapping, and restart
  equivalence before using a shared origin sweep. Diagnose cache/resume
  compatibility using the effective prompt and configuration, not the
  strategy name alone.
- [x] Reject invented missing log probabilities and nonfinite origin scores;
  stored-token-only uncertainty does not load a model for disabled metrics.
- [ ] Pilot evidence handling, answer scoring, and human labeling before
  launching the bounded main run. Preserve all pilot artifacts separately.

The portable notebook uses the four-condition AWS defaults. The separate
`config/config_iclr_pilot.yaml` is a broader development profile, not the
funded notebook configuration. Stage-2 and judge checkpoints retain legacy
resume behavior: a changed code revision or configuration requires fresh
complete run paths, not selective reuse of unverified outputs.

## Submission Gates

ICLR evaluates the question, motivation and literature, evidence supporting
claims, and significance of the contribution; a new leaderboard best is
not required. These are the [official reviewer criteria](https://iclr.cc/Conferences/2027/ReviewerGuidelines),
not a promise that a particular experiment package will be accepted.

- [ ] By September 17: freeze a genuine title and abstract, confirm every
  author and OpenReview profile, and check reviewing eligibility and quotas.
- [ ] By September 18, 11:59 p.m. AoE: submit the abstract. Finalize author
  membership by this deadline; do not use a placeholder abstract.
- [ ] By September 23: every main claim has a raw-data source, an analysis
  command, and an honest scope statement. Remove unsupported claims instead
  of filling missing evidence with estimates or intended outcomes.
- [ ] By September 24: inspect every PDF page, references, metadata, and a
  separate anonymous code archive. Resolve the absent repository license
  with the owner; do not package author-identifying Git history or notebooks.
- [ ] By September 25, 11:59 p.m. AoE: upload the paper and supplement. Main
  text must fit nine pages; references and permitted statements are excluded.
  Finalize the required AI-use statement in both paper and submission form.
- [ ] Remove internal completion notes only when their gates are resolved.
  If the evidence cannot support a meaningful completed study, do not label
  the audit draft submission-ready simply because the deadline has arrived.

Deadline and submission requirements: [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).
AI disclosure: [ICLR 2027 author AI policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).
