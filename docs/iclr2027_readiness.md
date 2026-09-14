# ICLR 2027 Completion Checklist

> This is the September 7–9 historical readiness audit. The controlled
> HotpotQA study has since completed and passed its independent audits.
> The [September 13 expert review](iclr_expert_review_2026-09-13.md) records
> current evidence, new diagnostics, the statistical supplement and the
> remaining scientific gaps. Statements below describe the earlier state.

Initial audit against repository commit `756f80d`; resolutions updated
September 9, 2026. See the [reviewer report](reviewer_report.md) for the
decision-relevant findings and their current status.

**Status: working draft prepared; empirical validation is not complete.**
The original `paper/main.tex` and `paper/main.pdf` are the historical AAAI
draft. Use `paper/iclr2027.tex` for the new anonymous ICLR working draft.
The new draft uses the three QA datasets provisionally and omits FEVER
from its empirical claims. This is not a statement that the remaining
results have been independently reproduced.

The [study protocol](iclr2027_study_protocol.md) now specifies contribution
positioning, a held-out comparison, matched prompts and new-step allowances,
paired analysis, and implementation gates. The root README, local run guide,
and standalone title/abstract have been rewritten to reflect this scope.
The [section-by-section review](paper_section_review.md) records the latest
manuscript edits, verified prior work and remaining replay/data checks.

## Deadlines and Format

| Requirement | Current official information |
| --- | --- |
| Abstract | September 18, 2026, 11:59 p.m. AoE |
| Full paper and supplement | September 25, 2026, 11:59 p.m. AoE |
| Main text | At most 9 pages at submission |
| Authors | Finalize membership by the abstract deadline |
| Submission | Anonymous manuscript and supplement; mandatory AI use statement |

Sources checked September 7: [official dates](https://iclr.cc/Conferences/2027/Dates),
[author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), and
[AI disclosure policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).
The official [2027 style archive](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip)
supplies the included, unmodified `.sty` and `.bst` files.

## Critical Findings

| Priority | Finding | Evidence and required action |
| --- | --- | --- |
| P0 | Raw evidence is absent | Only summary CSVs and saved notebook outputs are tracked. Recover trial logs, trajectories, configurations, and human labels before claiming reproduction or statistical significance. |
| P0 | FEVER evidence is invalid in the inspected loader | `src/env/fever_env.py`, `_fever_jsonl_to_records`, creates each evidence paragraph with `context.append([title, [claim]])`. These are claims, not Wikipedia evidence sentences. Exclude the current results, or rebuild evidence and regenerate every dependent stage. |
| P0 | Recovery could exceed its budget | Both execution loops requested a full step's tokens even when less budget remained. Fixed locally and reproduced with tests. Audit historical `recovery_gen_tokens > budget`, then rerun affected comparisons with the corrected cap in a fresh output directory. |
| P0 | Human validation is claimed but unavailable | No `_human_labels.json` or `judge_human_agreement.json` is tracked. The original abstract says the judge was validated. The new draft treats labels as judge references and states that validation is pending. |
| P0 | Baseline and uncertainty rates used different definitions | The cross-dataset notebook used majority success for baselines but mean seed success for `best_unc`. For example, HotpotQA restart is 10.6% in the mixed CSV versus 14.2% in mean-seed aggregates. The notebook now uses mean seed success for both and its stale outputs are cleared. Do not cite the historical `cross_dataset_main_results.csv` as a comparable table. |
| P1 | Confidence intervals ignored question clusters | `summarize_strategies` bootstrapped individual seed rows. Fixed to average seeds per question and bootstrap questions; duplicates and mixed datasets/models/budgets are rejected. Old intervals must be recomputed from raw data. |
| P1 | Best uncertainty is selected on the evaluation questions | The notebook maximizes success over the grid on the reported questions. Freeze selection on a separate development split or use a selection-aware analysis. |
| P1 | Some uncertainty results use privileged hints | The archived HotpotQA and MuSiQue maxima use `__informed` variants. Those hints derive from gold-informed judge annotations. Primary claims must use generic hints and matched baselines; report informed conditions separately. |
| P1 | The trajectory-length explanation uses the wrong variable | `cross_dataset_traj_stats.csv:avg_tool_calls` is computed from full-restart repairs, not original failed trajectories. Remove the four-step routing recommendation. Use original lengths and within-dataset comparisons for any new moderator analysis. |
| P1 | Ensemble success assumes access to correct answers | `ensemble_rows` takes the maximum of success flags. This measures retrospective candidate coverage, not a deployed answer selector. Costs also sum strategy rows despite generation reuse. |
| P1 | Claimed budget sweep was not executed by the historical script | The revised runner executes all configured multipliers, with CPU regression coverage. Archived results still describe one nominal cap unless actual separate runs are recovered. |
| P1 | Total inference cost is not covered by recovery-token matching | Self-consistency, confidence elicitation, input tokens, and annotation are additional costs. Evaluate a repeated-restart comparator at the same total allowance. |
| P1 | Historical origin comparisons change both prompt and new-step allowance | Explicit matched-hint and equal-new-step controls, fixed-early/random controls, and an origin sweep are implemented and CPU tested. The new pilot activates them; archived results require a rerun. |
| P1 | Matrix caches and nested paths could mix experiments | Fixed isolated absolute paths, immutable run configurations, generation/repair source/input manifests, model-cache rejection, and test-ID/exclusion checks. Legacy scoring/judge checkpoints still require fresh run paths after changes. |
| P1 | The documented model and scope did not match the catalog | Added the configured 32B entry; its filtered dry run now selects one experiment. Unknown models and empty matrix filters raise errors. Real model identity and snapshot revisions still need runtime validation. |
| P1 | Missing uncertainty measurements could become false certainty | Missing sampled-token logprobs were imputed, and empty entropy distributions returned zero. Revised metrics preserve missingness and localization skips nonfinite scores; historical profiles must be audited/recomputed. |
| P1 | Novelty needs comparison to closer work | Doctor-RAG directly overlaps on QA repair; SymTrace addresses repair versus resampling. Added six missing papers and narrowed the claim. Position-distribution and closest-method comparisons still require implementation and evidence. |
| P1 | Prefix replay fidelity is not checked | The batch runner re-executes tools but does not compare their observations with the stored prefix. Add mismatch rejection and audit page/cursor/title state before confirmatory runs. CPU budget tests do not establish replay fidelity. |
| P1 | Dataset and scoring provenance needs audit | Loaders permit mirror fallbacks; MuSiQue splits paragraphs at periods and drops answer aliases. Freeze sources and conversions, inspect records, and distinguish the custom gate from official task evaluation. |
| P2 | Release licensing was asserted without a license file | The old README said Apache-2.0, but no `LICENSE` file is present. Owner confirmation is required before adding one; the audit does not select or grant a license. |

FEVER additionally uses the generic multi-hop QA system prompt, which does
not specify its three output labels. Any FEVER rerun needs both real
evidence and a task-specific prompt, with scoring checked in the actual
execution path. Correcting only the evidence loader is insufficient.

## What Can Be Recovered Now

`results/cross_dataset/cross_dataset_traj_stats.csv` consistently uses mean
seed success for its reported strategies. It supplies the draft's
judge/restart/grid-maximum table rows and backtracking changes.
`cross_dataset_localization.csv` supplies descriptive argmax agreement
with the judge. Selected saved notebook displays supply random-origin
rates, recovery costs, and fixed-origin hint comparisons. Their baseline
rates are checked against the CSVs at displayed precision; their earlier
confidence intervals are not reused.

| Mean repair success (%) | HotpotQA | MuSiQue | 2WikiMHQA |
| --- | ---: | ---: | ---: |
| Judge-targeted | 10.9 | 4.9 | 10.0 |
| Judge-targeted + backtrack 2 | 12.4 | 5.6 | 22.0 |
| Full restart | 14.2 | 5.9 | 21.2 |
| Retrospective best uncertainty variant | 12.8 | 5.6 | 25.5 |

The last row includes privileged informed hints on HotpotQA and MuSiQue
and is selected on the evaluation questions. These are historical point
estimates, not new measurements or verified matched-budget results.

The archived source notebook at `756f80d`, cell 27 (zero-based), records
the best configuration names:

- HotpotQA: `unc__max_token_prob_max__topk__informed`.
- MuSiQue: `unc__perplexity__cascade_weighted__bt2__informed`.
- 2WikiMHQA: `unc__perplexity__topk__bt2`.

The historical paper instead reports 13.2% for the HotpotQA best strategy;
the available final cross-dataset summary is 12.8%. The new draft uses the
reproducible aggregate value without inventing an explanation for that
remaining snapshot discrepancy.

## Files Needed From Google Drive

The saved notebooks point to these directories under `MyDrive`:

- `agent-repair-hotpotqa/`
- `agent-repair-musique/`
- `agent-repair-2wikimultihopqa/`
- `agent-repair-fever/` for diagnosis only

Recover these paths within each directory:

```text
data/processed/pool.json
data/processed/failed_ids.json
data/processed/agent_model.json
outputs/trajectories/
outputs/uncertainty/
outputs/annotations/
outputs/annotations/_human_labels.json        # if actually completed
outputs/repairs/results.jsonl
outputs/tables/
outputs/tables/judge_human_agreement.json     # if actually completed
outputs/logs/
```

Also recover the configuration and code revision actually used for each
run. Keep original artifacts unchanged and calculate hashes for imported
files. Do not execute notebook cleanup cells that delete Drive output
directories. The current local machine has CPU analysis tools and LaTeX;
no remote GPU allocation or experiment rerun was started in this audit.

## Shortest Defensible Completion Path

1. **September 7-9: recover evidence and freeze scope.** Audit the three QA
   datasets, trial coverage, model identity, budgets, and copied execution
   rows. Freeze the QA-only scope unless a valid FEVER rebuild and rerun can
   be completed. Confirm whether human labels exist.
2. **September 9-13: rerun the core comparison.** Use corrected token caps,
   generic hints, the three existing QA datasets, and a small prespecified
   set: full restart, random, a fixed early origin, judge reference, and
   a selected uncertainty method with/without backtracking. Treat existing
   explored questions as development data for selecting the method; use
   fresh held-out IDs for the reported confirmatory comparison. Keep each
   run in a new output directory so old checkpoints cannot suppress it.
3. **September 10-16: validate and analyze.** Sample 50 trajectories per
   retained dataset (150 total). The study protocol recommends two blinded
   annotators and preserved adjudication; `scripts/label_human.py` currently
   supports only a single annotation record per trajectory.
   Compute question-level paired deltas and confidence intervals, strict
   EM sensitivity, and acquisition-inclusive costs. Add a second model
   family only after the core comparison is valid.
4. **By September 17: freeze title, abstract, and author list.** Use claims
   supported by the verified results, then complete the abstract submission
   by the official September 18 deadline. No OpenReview submission has
   been made by this agent.
5. **September 18-23: complete the manuscript.** Replace historical
   aggregates with verified outputs, report judge agreement, document
   configuration selection, and finalize related work and the AI
   disclosure. Remove the working-draft completion appendix only once
   the underlying work is complete.
6. **September 24: final artifact check.** Build the PDF, check the main
   text page count, inspect all pages, confirm anonymity including metadata,
   and prepare reproducible anonymous supplementary code for September 25.

This schedule is a priority order, not a verified GPU-time estimate. The
eight-model expansion in `CLOUD_GPU_SETUP.md` is not required to address
the current validity problems and should not delay the core rerun.

## Local Changes and Verification

- Regression tests reproduced budget overshoot and incorrect seed-row
  bootstrap intervals before the fixes.
- Sequential and batched generation now cap requests by remaining budget.
  Per-prompt limits use the supported vLLM sampling-parameter list API:
  [vLLM 0.19.0 documentation](https://docs.vllm.ai/en/v0.19.0/api/vllm/entrypoints/llm/).
- Summary confidence intervals now resample questions. The existing
  McNemar tests are labeled secondary majority repeatability. A new
  manifest-checked paired-mean CLI implements the primary macro contrasts,
  question bootstrap, sign-flip inference and Holm adjustment, with
  per-dataset EM/F1 sensitivities. It requires actual raw trial records.
- The ICLR draft removes unsupported causal and human-validation claims,
  identifies privileged comparisons, uses the official anonymous template,
  and includes an explicitly unfinished AI-use disclosure.
- Corrected bibliography metadata for Huang et al., Lightman et al., Tian
  et al., and Reflexion; added the closely related Who&When paper.
- Added DoVer, ERGO, and REFLECT to the related-work discussion, with the
  last explicitly identified as a preprint. The study does not claim that
  backtracking or recovery-based evaluation was invented here.
- Added documentation regression checks for the README's aggregate values,
  question counts, local links, abstract synchronization, and catalog filters.
- Added a controlled origin planner and raw execution cache with prompt
  hashes, matched restart hints, equal-new-step mode, full multiplier
  iteration, policy-specific acquisition costs, and interruption recovery.
- Added explicit cohort ID/exclusion checks, isolated matrix paths, frozen
  generation/repair manifests, exact configured model-name checks, and a
  bounded development pilot. Exact snapshot revision pinning is still needed.
- Missing token probabilities are no longer invented; missing entropy is
  not treated as certainty. Failure flags describe action type rather than
  claiming exploration is correct. No archived rates were changed by these
  implementation fixes.

Run the CPU checks and build from the repository root:

```bash
python -m pytest -q tests
make -C paper iclr-draft
```

Output: `output/pdf/agent-repair-iclr2027-draft.pdf`.
Asset provenance: `paper/generated/iclr2027_provenance.json`.
The expanded manuscript restores cost, fixed-origin hint, localization,
failure-mode, and implementation/annotation sections. The asset builder
parses selected saved notebook tables, rejects inconsistent baseline
snapshots, and records their cell indices and hashes. It omits obsolete
intervals rather than reconstructing them from rounded means.

Local verification: 112 CPU tests passed on September 8, including the
budget notebook and staged upload package. The draft compiles with resolved
citations and empty author metadata; main text ends on page 9, with 16
pages total, six tables, and four figures. The active four-condition plan
has a US$119 resource ceiling; its GPU results remain pending. Every rendered page was visually
inspected. The documented
filtered matrix dry run selects one catalog entry without loading a model.
The tests exercise the analysis and generation interfaces with synthetic
fixtures. They do not validate actual model performance or replace a GPU
smoke test and experimental rerun.
