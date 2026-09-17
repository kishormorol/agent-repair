# Applicability review: three proposed experimental figures

## Decision

All three proposed figures are supported by completed, independently audited
records. They require new analysis/plotting, **not new GPU experiments**.
This is a review of the supplied figure plan; the figures have not yet been
added to the manuscript. The separately requested new position-matching
experiment addresses a different question and must retain its own protocol,
cohort and results.

The check reran `load_extension()` and `load_followup()`, verified the
incremental cost identities, and computed candidate overlap rates and
point-estimate frontiers directly from trial exports. Evidence is saved in
`output/figure-plan-review-20260916/applicability-check.json`.

## Figure-by-figure assessment

| Proposed figure | Available evidence | Assessment |
| --- | --- | --- |
| Effective origins and execution overlap | Six replication cells; 371 failed model/question pairs; 7,791 token-policy rows; all seven policies, three seeds each | Applicable. Table 20 supplies uncertainty-versus-development-control summaries; derive the other requested policies/overlaps from the complete audited trial CSV. Exclude the 225 runtime rows. |
| Success versus measured resource use | Same six cells; success, incremental prompt/generated tokens and requests for every policy | Applicable as a descriptive tradeoff figure. Compute the frontier within each cell, using standalone costs including acquisition. |
| Recovery-budget sensitivity | Separate Qwen/HotpotQA follow-up: 250 initial questions, 118 failures, 3 policies × 3 allowances × 3 seeds = 3,186 rows | Applicable. All nine points are available. It is within-cohort allowance sensitivity, not cross-model or cross-dataset generalization. |

## Required statistical and interpretive details

1. **Use the complete trial exports, not rounded table entries.** For the
   existing studies, average seeds within questions, then weight questions
   equally. Keep models and datasets separate. The shared source questions
   across models are not six independent samples of a common population.
2. **Figure 5: execution equality is an ID check.** Join on cell, question and
   seed; compare execution IDs and verify matching origin/prompt/allowance.
   The denominator is `3*N_f`. Origin zero means the same recovery execution
   as restart under this frozen token protocol, while policy acquisition
   charges can still differ. Retain the full normalized-origin curves:
   zero/nonzero bars alone cannot establish positional balance.
3. **Figure 6: use paired differences and ratio of means.** Compute each
   policy's question-level success minus the same question's restart success,
   then bootstrap questions for vertical pointwise intervals. Do not subtract
   independently estimated interval endpoints. The token coordinate is the
   ratio of the two mean incremental token counts, not the mean of individual
   question ratios. Report vertical differences in percentage points.
4. **New descriptive intervals do not change the primary family.** Most
   policy-versus-restart contrasts in Figure 6 are not the prespecified
   uncertainty-versus-control contrasts. Label additional intervals exploratory
   and pointwise. Preserve the original Holm families and omit significance
   stars. Fixed early remains descriptive, including when it lies on a frontier.
5. **Token counts are a chosen resource proxy.** Prompt-plus-generated tokens
   gives equal numerical weight to two different processing stages. It is not
   FLOPs, latency, dollars, or a universal efficiency ordering. Show the raw
   components in the existing tables; a prompt-token or request companion
   can reveal sensitivity to the proxy. Include full diagnosis acquisition
   once per standalone attempt, not once per deduplicated study execution.
6. **Pareto uncertainty remains unresolved by a line.** Use “observed
   point-estimate Pareto frontier.” Both measured cost and success vary across
   questions. Vertical intervals alone do not test statistical dominance,
   establish a joint frontier, or justify selecting a winning policy.
7. **Figure 7: distinguish cap from consumption.** The x-axis in (a) is a
   recovery-generation allowance, not a total token-processing budget.
   Panel (b) includes prompt processing and acquisition. Resample the same
   question indices across all nine conditions for uncertainty summaries.
   Lines connect allowance settings and do not imply monotonic per-question
   success or prove the absence of policy-by-budget interactions.

## Verified descriptive checks

Origin-zero percentages for the three proposed Figure 5 policies:

| Cell | Uncertainty + BT2 | Dev.-fitted random | Uncertainty, no BT |
| --- | ---: | ---: | ---: |
| Qwen / HotpotQA | 62.79 | 82.17 | 16.28 |
| Qwen / MuSiQue | 33.33 | 29.63 | 6.94 |
| Qwen / 2Wiki | 60.71 | 86.90 | 21.43 |
| Mistral / HotpotQA | 67.95 | 75.21 | 15.38 |
| Mistral / MuSiQue | 54.22 | 48.59 | 18.07 |
| Mistral / 2Wiki | 59.70 | 53.23 | 32.84 |

All seven named policies are available in every replication cell, including
Random + BT2 and Fixed early. Under the proposed sum-of-tokens proxy, the
observed point-estimate frontier varies across cells; for example, it contains
restart alone for Qwen/MuSiQue, while Qwen/2Wiki includes restart, both
uncertainty variants and descriptive Fixed early. This supports separate
cell panels and does not establish statistically superior policies.

## Layout and naming corrections

- The current main text already occupies nine pages. The cross-model/dataset
  forest plot is currently in the appendix, not the main replication section.
  Adding both large figures to the main text and moving the forest plot there
  requires replacing or condensing existing material, not simple insertion.
- Prefer Figure 5 in the main text if it replaces the older main-study-only
  overlap table. Figure 6 is a main-text candidate only after a page-budget
  revision; otherwise place the complete six-cell grid in the appendix and
  reference its finding. Figure 7 belongs with the allowance follow-up appendix.
- Treat the proposed numbers as placeholders. Existing historical and new
  appendix figures already use these numbers. Use LaTeX labels and automatic
  numbering instead of hard-coded “Figure 5/6/7” references.
- Use **Dev.-fitted random** consistently. “Position-matched” is a legacy
  identifier/design intent, not an achieved-balance claim for these runs.
- Use one policy color/marker mapping in policy plots. Figure 5's stacked-bar
  segments instead encode origin state; use a consistent state encoding and
  explicitly label policy rows so the two visual meanings are not confused.
- FEVER remains excluded, and judge/reference-selected origins must not be
  presented as a ground-truth oracle. Do not mix historical rows into these
  three controlled figures.

## Relation to the new experiment

These figures make existing interventions and resource use easier to inspect.
They do not remove the imbalance already measured in the completed replication.
The user separately authorized a fresh gap-1 experiment on September 16.
Its outcomes must be shown as new evidence, rather than merged into the six
completed replication cells or described as part of their frozen protocol.
