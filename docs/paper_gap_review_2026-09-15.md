# Paper review: remaining scientific and release gaps

Reviewed September 15, 2026, after the completed replication was integrated.
This is an author-requested assessment, not an official conference review.
It supersedes the open-gap assessment in the September 13 review where the
diagnosis, allowance, replication and runtime studies were still missing.

## Assessment

**The narrow empirical conclusion is defensible, but the paper needs a
stronger account of what the inconclusive comparisons teach us.** The most
important newly quantified issue is imperfect positional matching in the
replication. Statistical precision, scope and access to the new evidence are
the other immediate priorities. A negative result can be useful; a positive
result or a new GPU run is not a prerequisite for improving this paper.

The [current manuscript](../paper/iclr2027.tex) has nine pages through the
conclusion and 33 pages including statements, references and appendices.
The reviewed PDF SHA-256 is
`0932755e0d385f6e51dc6e9b8d019f12787070ce25a652ac81fb8c92f989df2f`.
The manuscript and frozen analyses were not changed during this review.

I inspected the paper, relevant analysis and execution code, development
profiles, all 600 replication initial records, and the 8,016-row extension
export. I independently reconstructed all 24 replication effect estimates
and their reported Holm adjustments. I also computed exact sign-flip
diagnostics from the paired question differences. These checks found no
numerical contradiction with the reported primary conclusions. This review
does not repeat the complete raw-trajectory audits or GPU generations.

Reproducible review evidence:

- [Review check script](../output/reviewer-2026-09-15-paper-gaps/review_checks.py)
- [Computed diagnostics and source hashes](../output/reviewer-2026-09-15-paper-gaps/replication-diagnostics.json)

Run from the repository root:

```bash
python output/reviewer-2026-09-15-paper-gaps/review_checks.py
```

These are post-results review diagnostics, not additional prespecified tests.

## What is already resolved

- Two model families and three datasets now have completed controlled results.
- A same-model diagnosis/replay adaptation and three allowance settings have
  been evaluated on a separate cohort.
- A 25-question measured-runtime component has complete outcomes.
- Trial coverage, reference scoring, replay and recorded allowances have
  audited evidence; replication local/remote exports agree.
- The runtime questions are now correctly identified as a replication subset.
- The manuscript distinguishes individual confidence intervals from corrected
  tests, and does not claim equivalence or a general computational advantage.

The earlier requests to add a second model, another dataset, a diagnosis arm
and allowance tests should therefore be treated as completed.

## G1 — High priority: the replication position control does not closely match the treatment

**September 16 status:** current-paper diagnostics and interpretation are
addressed in [the gap 1 resolution](position_balance_2026-09-16.md).
The experimental imbalance remains; the original review evidence below is
preserved. Gaps 2–6 remain queued.

**Evidence.** The [replication methods](../paper/iclr2027.tex#L1515) specify
20 development questions per cell, but the distribution is fitted only to
their failures. Two Qwen cells have just six development failures. Five of
those six choose origin zero in each cell. The realized test distributions
then differ substantially from the uncertainty treatment:

| Cell | Development failures used | Main failures | Uncertainty origin zero | Position control origin zero |
| --- | ---: | ---: | ---: | ---: |
| Qwen / HotpotQA | 6 | 43 | 62.79% | 82.17% |
| Qwen / MuSiQue | 16 | 72 | 33.33% | 29.63% |
| Qwen / 2Wiki | 6 | 28 | 60.71% | 86.90% |
| Mistral / HotpotQA | 13 | 78 | 67.95% | 75.21% |
| Mistral / MuSiQue | 19 | 83 | 54.22% | 48.59% |
| Mistral / 2Wiki | 11 | 67 | 59.70% | 53.23% |

Percentages count policy attempts over the three seeds. For example, Qwen
2Wiki has 73/84 origin-zero control rows versus 51/84 treatment rows: a
26.19-percentage-point difference. HotpotQA differs by 19.38 points.

**Why it matters.** The comparison remains a valid evaluation of the two
frozen policies. However, it does not fully isolate uncertainty information
from positional preferences. The main study's close origin-zero balance
cannot be assumed to carry over to replication. This is an interpretation
limitation, not evidence of a runner bug or an invalid frozen result.

**Work using current records.** Report development failure counts, complete
normalized-origin distributions, retained-prefix lengths and execution
overlap in all six cells. Describe the comparator as development-fitted and
show the observed imbalance alongside its results. A zero-origin check alone
does not establish balance elsewhere in the distribution.

**If strengthening the experiment.** Use a larger development failure sample
and prespecified position/length balance diagnostics before a fresh test
cohort. Do not retune the control using the completed test outcomes and call
that repaired comparison confirmatory.

**Done when:** readers can assess positional balance in every replication
cell and the contribution statement reflects its actual strength.

## G2 — High priority: the paper needs effect bounds and replication-level diagnostics

**Evidence.** The [replication results](../paper/iclr2027.tex#L433) emphasize
24 adjusted p-values of 1.000. There are 28–83 failed questions per cell,
and only 3–23 questions per primary contrast have different three-seed mean
success between the two policies. Those counts are not replacement sample
sizes: all paired questions, including ties, belong in the estimand.

Uncertainty shares the restart execution on 33.33%–67.95% of replication
attempts. The question counts with any different execution from restart are
16, 48 and 11 for Qwen HotpotQA, MuSiQue and 2Wiki, and 25, 38 and 27 for
the corresponding Mistral cells. The paper currently gives this detailed
decomposition for the original main study, but not replication.

The most striking example is Qwen 2Wiki. Its +5.95-point estimate and
individual 95% interval [+1.19, +11.90] come from four questions with higher
treatment mean, none with lower mean and 24 ties. The exact two-sided
sign-flip p-value is 0.125; the frozen 10,000-draw Monte Carlo value is
0.131. Both give an adjusted value of 1.000. This is not a numerical error;
it makes the sparse paired evidence easier to understand than a generic
statement that interval and test constructions differ.

Separately, 621/1,113 uncertainty trial rows in replication stop at the
token or step limit. This does not establish that additional generation
would help; it identifies an important descriptive bottleneck to report.

**Why it matters.** A nonsignificant test does not measure how much benefit
has been ruled out. The original HotpotQA interval has an upper bound of
only +0.88 points, while the Qwen 2Wiki interval permits a considerably
larger gain. Treating all cells as equally informative loses useful nuance.

**Work using current records.** Extend the execution-overlap, wins/losses/ties,
termination and cost analyses to every cell. Present the effect intervals as
practical bounds under the stated assumptions. Explain the Qwen 2Wiki
example explicitly; any exact test added now should be labeled a sensitivity
check, with the frozen primary outputs retained. Avoid a new pooled
confirmatory claim; any exploratory pooled analysis must account for shared
question IDs across models.

**If stronger precision is needed.** Choose a meaningful effect size and a
prospective precision or power target before new outcomes. A two-point
target is a possible design choice, not an already prespecified equivalence
margin. Increase independent questions if the desired claim requires it,
rather than adding seeds and treating them as independent questions.

**Done when:** the reader can distinguish narrow bounds, broad uncertainty,
mechanical overlap and unsuccessful completion without equating p=1 with
equivalence.

## G3 — High priority: sharpen the contribution around the policy actually tested

**Evidence.** The [title](../paper/iclr2027.tex#L10) asks about stored-token
uncertainty, while the [controlled method](../paper/iclr2027.tex#L194) tests
perplexity/argmax with two-step backtracking and its no-backtracking ablation.
Other uncertainty signals appear only in confounded historical results.
The diagnosis comparator is explicitly an untrained, origin-only adaptation.
Its valid JSON and zero replication fallbacks establish parser coverage,
not diagnosis quality.

**Why it matters.** The empirical question is narrower than uncertainty-based
repair as a method class. Audit completeness is valuable but does not alone
explain the new scientific insight. Prefix reuse and controlled replay are
already present in closely related work. Doctor-RAG combines trained
diagnosis with error-specific operators; SymTrace couples execution control
with symptom-guided interventions. Their reported rates cannot be inserted
as comparable baselines here. Sources:
[Doctor-RAG](https://arxiv.org/html/2604.00865v2),
[SymTrace](https://arxiv.org/html/2608.25920v2).

**Work using current records.** State the central finding as a controlled
test of this fixed perplexity policy. A title such as “A Controlled Test of
Perplexity-Based Trajectory Repair” would match the evidence more directly.
Use the replication diagnostics to explain what was learned about origin
selection, positional matching and measured resources. Keep claims about
the whole uncertainty family, causal error localization and trained repair
systems outside the established contribution.

**Optional empirical extension.** If the desired contribution remains about
uncertainty broadly, compare a small, development-frozen set of distinct
signals on fresh evaluation data. If it is about competing repair methods,
first assess a verified stronger diagnosis-based comparator. Doctor-RAG has
a [public implementation](https://github.com/Fdioa/dr_rag_open), but its
distillation and retrieval requirements need a feasibility assessment; a
ready-to-use checkpoint and affordable reproduction were not established
in this review. Recomputed scores alone do not supply missing repair
outcomes for newly selected origins.

**Done when:** the title, contribution bullets, comparator names and
conclusion all answer the same bounded question. New inference is optional
if the paper adopts that scope.

## G4 — High priority for release: the supplement omits the studies now central to the paper

**Evidence.** The [reproducibility statement](../paper/iclr2027.tex#L606)
correctly says that the standalone supplement covers only the original
250-question HotpotQA study and its 2,034 trial rows. I inspected the ZIP:
it contains those records, four primary contrasts and eight secondary
comparisons. It does not package the diagnosis, six-cell replication or
runtime evidence. Those records are locally retained and checked, which is
different from giving a reviewer a complete independent reproduction path.

**Work using current records.** Build a combined anonymous supplement with
all frozen protocols, initial and repair exports, analyses, source/model
identities, checksums, and clear commands. Separate statistical reproduction,
raw replay/scoring audit and GPU generation reruns. Preserve dataset and
third-party notices, and review identifying metadata before release.

**Done when:** an isolated extraction reproduces the 4-, 6-, 24- and 2-test
primary families, their trial counts and paper tables without depending on
private workspace paths or cloud credentials. The documented raw-audit and
generation environments should have explicit, separately tested scope.

## G5 — Medium priority: outcome reporting is less complete for replication

**Evidence.** The [replication scoring section](../paper/iclr2027.tex#L1506)
uses the declared EM-or-F1-at-least-0.5 gate. Across the 600 initial
model/question evaluations, 229 pass that gate but only 177 pass strict EM.
Thus **52 initially accepted non-EM answers are never repaired**. This is
8.67% of initial evaluations, not an annotation-error estimate.

The paper reports per-policy replication EM/F1 means, but the paired EM/F1
sensitivity and reference-gated full-cohort analyses are developed mainly
for the original HotpotQA study. Strict metrics on the current failure
cohort do not establish performance under an EM-only repair gate.

**Work using current records.** Add per-cell exploratory EM/F1 contrasts and
full-cohort outcomes that preserve the actually accepted original answers.
Report how many answers pass only the F1 threshold, including repaired
answers. Keep these distinct from a newly executed EM-gated experiment.
Use “accepted under the study gate” where “correct” could imply strict EM.

**If an EM-gated claim is needed.** Additional eligible failures need repair
outcomes; existing conditional results cannot substitute for those runs.
Any confirmatory extension needs a separately frozen design.

**Done when:** readers can see the dependence of the conclusion on the
success definition without confusing rescoring with a new failure cohort.

## G6 — Medium priority, conditional on efficiency claims: runtime scope remains narrow

**Evidence.** The [runtime experiment](../paper/iclr2027.tex#L447) uses 25
Qwen HotpotQA failures on one execution setup, three seeds, disabled prefix
caching, and completed attempts scored retrospectively against deadlines.
Diagnosis acquisition is measured once per question. Its ten-second
uncertainty-minus-restart interval is [-5.33, +2.67] points. These limitations
are already acknowledged; the experiment is not missing or invalid.

**Why it matters.** The observed result answers a completion-time question
for this setup. It cannot establish savings under deployed caching,
concurrency, repeated acquisition or a matched total resource allowance.

**Work using current records.** Keep the claim descriptive and show the
available latency distribution and completion/termination counts. Explain
what is included in time measurement and what the deadline does.

**Optional empirical extension.** A broader efficiency claim needs a
prespecified runtime design with an adequate question sample, repeated
acquisition timing, realistic cache/concurrency settings and a suitable
total-resource comparator. Active cancellation is needed only for a claim
about resource savings from stopping late attempts. Scope and affordability
should be settled before starting new inference.

**Done when:** either the paper keeps its current measured-deadline scope,
or stronger efficiency language has directly corresponding evidence.

## Editorial and submission items

- The nine-page main text fits the current limit; the optional reproducibility
  and required AI-use statements are excluded from that limit. Appendices
  are allowed after references, but reviewers need not read them. Put the
  most decision-relevant replication diagnostics in the main text. These
  rules were checked against the
  [official author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).
- The [AI-use statement](../paper/iclr2027.tex#L616) still tells authors to
  complete verification/disclosure. Replace this working-copy instruction
  with an accurate final statement after author review.
- The long historical appendix is transparently labeled, but it can distract
  from the completed controlled evidence. Compress or move historical
  bookkeeping while retaining traceability and essential method definitions.
- Human error-step annotation is needed for localization or mechanism
  claims, not for the current deterministic answer-scoring comparison.
  Likewise, live retrieval and additional model scales are scope extensions,
  not automatic prerequisites for this bounded paper.
- Reconstructed historical exclusions remain a disclosed provenance limit.
  Resolving it requires the original pool or other evidence, not new wording.

## Recommended work order

| Order | Work | New inference? | Completion criterion |
| --- | --- | --- | --- |
| 1 | Replication position balance, overlap, discordant questions and termination diagnostics (G1/G2) | No | Six-cell tables/figure generated from checked records; explicitly exploratory; frozen estimates unchanged. |
| 2 | Paired metric sensitivities and gated full-cohort outcomes (G5) | No | Per-cell EM/F1/gate results, original-gate exceptions and clear cohort definitions. |
| 3 | Sharpen title, contribution and discussion using those results (G3) | No | One concrete empirical question with claims matching tested policies and precision. |
| 4 | Complete the anonymous reproducibility package (G4) | No for statistics/raw audits | Isolated reproduction covers every completed study and stated verification layer. |
| 5 | Decide whether a stronger scientific claim is worth another study | Only if selected | A frozen design tied to one remaining claim, a precision target and an affordable allocation. |

**Recommendation:** finish steps 1–4 before choosing another experiment.
The next experiment, if any, should address a defined remaining uncertainty
rather than aim to produce a significant result. ICLR's
[reviewer guidance](https://iclr.cc/Conferences/2027/ReviewerGuidelines)
emphasizes supported claims and useful new knowledge; state-of-the-art
performance is not required.
