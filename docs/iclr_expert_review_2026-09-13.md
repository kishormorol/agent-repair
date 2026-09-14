# Expert review and revision record

Reviewed September 13, 2026. This is an author-requested mock ICLR review,
not an official conference review. It assesses the current
[manuscript](../paper/iclr2027.tex), the completed
[AWS main study](aws_main_2026-09-12.md), raw archived evidence and analysis
code. It supersedes the scientific status in the
[September 8 review](reviewer_report.md); that earlier assessment predates
the controlled experiment.

**Recommendation: lean reject on the current evidence; confidence: moderate.**
The controlled comparison now supports its narrow conclusion, and the
earlier concerns about missing primary trials, unmatched hints and untested
GPU execution have been addressed for this study. The remaining concern is
the amount of new, transferable knowledge: one prespecified perplexity
selector is inconclusive against simple controls in one restricted setting,
without a close diagnosis/replay comparator or allowance replication.
The negative result itself is not grounds for rejection. The
[ICLR 2027 reviewer guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines)
ask whether claims are supported and the work provides useful new knowledge;
state-of-the-art performance is not required.

## What the paper establishes

A pinned Qwen2.5-32B-Instruct-AWQ agent attempted 250 HotpotQA questions.
The 113 initial failures received six conditions and three repair seeds
under identical generic retry hints, generated-token caps and new-step
allowances. The treatment chooses the maximum stored-token perplexity step,
backtracks by two and regenerates the suffix. Seeds are averaged within
questions; questions, not the 2,034 condition/seed rows, are resampled.

| Treatment minus control | Success difference, percentage points | Individual 95% bootstrap interval |
| --- | ---: | ---: |
| Full restart | −1.47 | [−4.13, +0.88] |
| Uniform random with backtracking | −0.88 | [−4.13, +2.06] |
| Development-fitted position-matched random | 0.00 | [−2.95, +2.95] |
| Same uncertainty rule without backtracking | +1.18 | [−2.95, +5.31] |

All four primary Holm-adjusted sign-flip p-values are 1.000. Treatment
success is 9.73%, versus 11.21% for restart. This supports “no clear advantage
in this setting,” not equivalence, a general failure of uncertainty, or
selection of the numerically best alternative. These primary estimates
were not changed during this review. Source:
[pooled analysis](../output/aws-experiment/2026-09-11-main/pooled-analysis-local.json).

## Strengths

- The question is concrete and the controls address different explanations:
  uncertainty, upstream placement, the development origin distribution and
  backtracking. The fixed-early sixth condition is correctly descriptive.
- The completed study preserves frozen IDs, conditions, seeds, model and
  source identities. All five batches and 1,058 unique repairs are available
  locally, with independent replay, reference-scoring and allowance audits.
- The primary analysis respects the paired question structure and declared
  four-comparison family. Shared executions are identified explicitly.
- The revised manuscript acknowledges close prior work and reports the
  unfavorable result. It does not infer causal localization from terminal
  answer success or transfer historical audit guarantees to missing records.

## Major gaps and work completed

### 1. The tested intervention differs from restart less often than the nominal study size suggests

Treatment and restart share 210 of 339 executions, covering all three seeds
on 70 failed questions. Only 43 questions have a nonzero treatment origin.
Within those 43, the success difference is −3.88 points, with interval
[−10.85, +2.33]. The full effect decomposes as
`(43 / 113) × (−3.88) ≈ −1.47` points. Across all 113 questions, treatment
has a higher three-seed mean on five, a lower mean on eight and ties on 100.

**Implemented:** an execution-overlap table, this mixture decomposition,
the explicitly exploratory nonzero-origin analysis, and wins/losses/ties
for every primary comparator. The primary sample remains 113; dropping
zero differences would change the estimand. Position-matched random differs
on 93 questions, so collapse into restart alone does not explain all four
inconclusive comparisons.

**Remaining:** enough independent failures to characterize small effects,
particularly when a prefix is retained. A prospective approximation using
the observed question-difference variance gives 179, 257, 248 and 476 failed
questions for a two-percentage-point individual 95% interval half-width,
respectively. These are planning sensitivities, not power estimates,
simultaneous coverage, guaranteed precision or permission for optional
stopping. Freeze a meaningful effect and feasible precision target before
new outcomes.

### 2. Equal output allowances do not establish equal computation or an efficiency advantage

Treatment uses 269.9 output and 4,083.3 prompt tokens per attempted repair,
versus restart's 271.6 and 3,153.3. That is **29.5% more recorded prompt
tokens**, with a paired increase of 930.0 tokens, interval [579.2, 1321.9].
The output difference is −1.7 tokens, interval [−8.9, +5.7]. Prefix retention
does not demonstrate lower prompt processing. Cached prefill, latency and
FLOPs are unmeasured.

Original-failure allowances range from 77 to 548 tokens, median 344.
Treatment records 103 token-budget stops and 65 step-limit stops: 168/339
repairs emit no final answer. Separately, 152/339 use their entire token
cap; cap equality and the recorded stopping reason are not interchangeable.
All 637 failed-trace steps have usable perplexity, with zero empty-profile
fallbacks. Missing signal records therefore do not explain the outcome.

**Implemented:** per-policy prompt/output/request/tool accounting, paired
cost intervals, termination and cap-utilization diagnostics, plus explicit
separation of costs charged to each policy from deduplicated study usage.
All six conditions' values are in Appendix F and the
[diagnostic report](../output/reviewer-2026-09-12/diagnostics.json).

**Remaining:** new executions at other allowances and direct measurements
of policy runtime, acquisition/diagnosis cost and cache behavior. A stopped
trace cannot reveal whether more tokens would improve the answer. The
observed extra prompt tokens also do not prove a wall-clock slowdown.

### 3. Novelty and comparison with the closest repair methods remain unresolved

The defensible contribution is a controlled empirical study of a cheap
selector. Prefix reuse, rollback and outcome-based evaluation already
exist. Doctor-RAG is a close QA comparator, but uses a trained diagnoser,
an EM-only failure gate and repair operators that can retain documents
retrieved across the failed trace. These information and training differences
must be handled explicitly in a comparison.
[Doctor-RAG method](https://arxiv.org/html/2604.00865v2).

SymTrace directly studies repair versus resampling and the limits of
inferring corrected failure mechanisms from terminal success.
[SymTrace paper](https://arxiv.org/html/2608.25920v2).
ERGO uses entropy changes for conversation resets, a different intervention
from the present post-failure perplexity selector.
[ERGO publication](https://aclanthology.org/2025.uncertainlp-main.23/).

**Implemented:** a narrower title and contribution statement, an explicit
statement that only perplexity was tested in the controlled study, and a
main argument centered on the audited comparison. Confounded historical
three-dataset results remain in a clearly marked appendix. The rationale
for stored-token uncertainty is its lack of additional selection-model
requests, not proof that it is the best uncertainty estimator.

**Remaining:** a verified close-method reproduction if feasible, or a
clearly named same-model diagnosis/replay adaptation. Published results
from different cohorts are not valid rows in this study's comparison.
An adaptation must not be represented as a full Doctor-RAG reproduction.

### 4. The failure gate and restricted environment limit interpretation

The primary gate is answer EM equal to one or token F1 at least 0.5.
Forty initial answers pass that gate but fail strict EM and were never
repaired. The tools search question-specific supplied context, not the web
or all Wikipedia: 248 questions contain ten paragraphs, one contains two
and one six. Search uses title matching; lookup returns matching sentences
from the current paragraph. These facts matter for recovery difficulty.

**Implemented:** the actual context-size distribution and tool behavior,
all eight EM/F1 sensitivity intervals, and overall outcomes under the
implemented reference gate. Treatment reaches 59.20% gate success and
42.00% EM over all 250 questions, versus 59.87% and 42.67% for restart.
These preserve initially accepted answers and average single-repair
outcomes across seeds. They are neither an EM-only rerun nor a deployed
failure detector, majority vote or oracle best-of-three result.

| Treatment minus control | EM difference, pp [95% interval] | F1 difference × 100 [95% interval] |
| --- | ---: | ---: |
| Restart | −1.47 [−3.83, +0.59] | −1.67 [−4.28, +0.64] |
| Random with backtracking | +0.29 [−2.06, +2.65] | −1.27 [−4.14, +1.45] |
| Position-matched random | +1.47 [−0.59, +3.83] | +0.11 [−2.48, +2.62] |
| Uncertainty without backtracking | +2.95 [+0.29, +5.60] | +1.80 [−1.35, +5.07] |

The positive EM interval against no backtracking is reported explicitly.
It is one of eight unadjusted secondary comparisons and does not replace
the primary family. The descriptive treatment-versus-fixed-early primary
difference is −1.77 points [−6.19, +2.65]; it does not establish a winner.

**Remaining:** an EM-gated experiment needs outcomes for the additional
eligible initial failures. A second independently generated model cohort
and another dataset/environment are needed for broader claims. Human labels
are relevant to error-location and mechanism claims; they are not required
to compute the deterministic primary answer metric.

### 5. Reproducibility is substantially improved, but has a defined boundary

**Implemented:** a diagnostic loader verifies 398 input-to-archive
comparisons against all five checksummed archives, checks complete trial
coverage and shared execution consistency, and rescores original and repair
answers. It reproduces the frozen primary estimates before producing new
analyses. The post-results
[diagnostic plan](reviewer_diagnostics_plan_2026-09-12.md) is retained and
hashed; no new analysis is mislabeled prespecified.

A [statistical supplement](../output/reviewer-2026-09-12/agent-repair-statistical-supplement.zip)
now contains 250 initial records, 2,034 trial rows, the frozen manifest,
reference results, analysis implementation and file checksums. Extracted
outside the repository package structure, `python reproduce.py` reproduces
four primary contrasts including Holm correction, eight EM/F1 contrasts,
six policies' costs and population rates, and unique execution totals to
absolute tolerance `1e-12`. The script makes no network calls. Its test also
confirms that a changed input fails checksum verification.

**Remaining:** this package reproduces statistics from measured CSVs, not
GPU generations or the full raw-trajectory audit. The full runtime/raw-data
release still needs packaging and author review. Direct local author/account
identifiers were checked in the whitelisted statistical package; this is not
a guarantee of conference anonymity. Historical exclusion IDs were
reconstructed, and the original full historical pool file was not compared.
No claim of disjointness from every undocumented prior run is warranted.

### 6. Mechanism and failure examples need evidence beyond successful answers

**Implemented:** three reproducibly selected illustrations, one each where
treatment has a higher mean, restart has a higher mean, and both fail.
The first sorted question ID in each outcome stratum is used, and all three
seeds are displayed. The examples are labeled outcome-selected, not
representative or causal. None of the 18 displayed answers passes strict EM;
partial-answer gate successes are visible rather than hidden.

**Remaining:** mechanism claims require trace-based hypotheses and suitable
interventions. Claims about judge localization need independent annotation
and agreement analysis. If retaining that extension, sample trajectories
independently of repair outcome, blind annotators to policy and outcome,
preserve disagreement, and report ambiguous cases. No human labels or
failure-type prevalence estimates were fabricated for this revision.

## Concrete next experimental design

The completed primary experiment stays frozen. Its failures are now
outcome-exposed; any added condition on them is a follow-up exploratory
comparison. New confirmatory claims need a newly frozen disjoint cohort.

| Order | Work | Design and completion criterion |
| --- | --- | --- |
| 1 | Close diagnosis/replay control | Prototype on declared development traces only. Use the same agent model, no gold answer/support labels, a frozen diagnosis prompt and parser, and a recorded origin/diagnosis. Separate origin-only replay with the common hint from any richer diagnosed-hint variant. Charge diagnosis requests and tokens. Document whether discarded evidence is available. Test malformed-output handling and prefix fidelity before freezing the policy. |
| 2 | Allowance sensitivity | Freeze 0.5×, 1× and 2× original generated-token caps, the same eight-new-step limit, the treatment and matched restart. Add the close control after development. Record both exhausted limits; a token-cap sweep alone does not remove the step limit. Reuse an execution only when all effective inputs, cap and seed agree. |
| 3 | Confirmatory evaluation | Exclude all documented historical, pilot, development and current 250 IDs. Fix initial-question count and primary contrasts before new outcomes, using an explicit precision target and measured affordability. Do not stop when significance is reached or replace the treatment with the observed best variant. Report conditional and actual gated full-cohort results separately. |
| 4 | Generality | Generate a fresh initial cohort with a pinned model from another family; failures must be selected from that model's own attempts. Replicate a frozen small core on another QA dataset if the measured budget permits. Report model/dataset-specific effects before any declared macro average. |
| 5 | Optional mechanism extension | Only if retaining localization/mechanism contributions, collect blinded independent labels and prespecify context interventions. Otherwise keep those claims outside the paper's established contribution. |

All new model work must fit the existing AWS allocation and use a recorded
cost/stop plan. The historical $9.16 main-study estimate is a cutoff snapshot,
not a current credit balance or a price quote for these additions. This
review did not launch additional inference or incur new GPU-run charges.

## Reviewer questions and current answers

1. **Does uncertainty still help when it chooses a distinct origin?**
   The new 43-question subgroup is inconclusive, with a wide interval. It
   neither rescues a positive claim nor invalidates the original estimand.
2. **Are the methods matched for total inference cost?**
   No. The output and new-step allowances match; prompt usage differs and
   runtime/cache measurements are absent. The manuscript now states this.
3. **Is the result robust to official answer metrics?**
   EM/F1 comparisons on the same failure cohort are supplied, including
   the positive exploratory no-backtracking EM interval. The experiment
   does not evaluate an EM-only repair gate.
4. **What is new beyond existing repair/replay work?**
   A bounded, audited comparison of a prespecified stored-token selector
   against positional controls. A stronger methodological or general claim
   remains unsupported without the proposed empirical extensions.

## Deliverables and verification

- [Revised manuscript source](../paper/iclr2027.tex) and synchronized
  [title/abstract](../paper_title_abstract.md).
- [Diagnostic analysis](../scripts/analyze_study_diagnostics.py),
  [analysis implementation](../src/analysis/review_diagnostics.py),
  [results and provenance](../output/reviewer-2026-09-12/diagnostics.json),
  and five generated diagnostic table/case fragments.
- [Statistical supplement builder](../scripts/build_statistical_supplement.py)
  and the verified ZIP linked above.
- Tests cover incomplete/duplicate trials, inconsistent shared executions,
  scoring and allowance violations, archive tampering, subgroup weighting,
  policy/study accounting, the actual reference gate, paper-source
  consistency and isolated statistical reproduction.

The final test, PDF-build and rendered-page inspection record is
[agent-repair-iclr2027-validation.json](../output/pdf/agent-repair-iclr2027-validation.json).
All 59 targeted tests passed; the 43 paper-asset/documentation tests passed
again after the final layout changes. The PDF has 24 pages, with main text
ending on page eight, 15 tables and five figures. All rendered pages were
inspected; the build has no overfull boxes or unresolved references/citations.
The review-stage additions improve interpretation and reproducibility;
they do not fill missing experimental cells or guarantee acceptance.
