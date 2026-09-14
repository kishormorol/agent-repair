# Section-by-Section Paper Review

Reviewed September 9, 2026 against the original long manuscript, current
ICLR draft, archived aggregate sources, repair planner, agent loops, tool
environments, loaders, scoring and paper tests. This is an author-facing
revision record, not independent peer review or evidence of completed runs.

## Reviewer Assessment

**Not yet a defensible completed ICLR study.** The biggest risks are missing
paired raw evidence, confounded historical comparisons, and close prior work.
Writing can fix inaccurate claims and clarify the experiment, but cannot
replace the controlled results. The central claim to test is:

> Does a frozen stored-token signal choose a better repair origin than
> positional controls, without a separate diagnosis model, under declared
> information access and recovery allowances?

A positive, negative or inconclusive answer is possible. The four-condition
core can compare those specific policies, but cannot by itself isolate all
position effects or establish an advantage over diagnosis-based repair.

## Section Decisions

The original [main.tex](../paper/main.tex) and `paper/main.pdf` are preserved.
Edits are in [iclr2027.tex](../paper/iclr2027.tex), its bibliography, and the
synchronized [abstract](../paper_title_abstract.md). Archived measurements
have not been changed or replaced with intended outcomes.

| Section | Review and revision | Evidence still needed |
| --- | --- | --- |
| Title and abstract | Keep the literal recovery comparison. Clarify the stored-token, no-additional-diagnoser question; retain the exact distinction between historical observations and pending follow-up. | Frozen-policy effect sizes and valid intervals before rewriting as a completed study. |
| 1. Introduction | Remove novelty implications for prefix reuse or attribution-versus-recovery; cite directly overlapping work. Present decomposition, descriptive evidence and the pending protocol separately. | Show value beyond simple origins and a close diagnosis-based comparator. |
| 2. Related work | Add six missing papers; distinguish QA repair, replay infrastructure, failure attribution and uncertainty-triggered resets. Use current versions and publication status. | Verify the chosen comparator's released code/checkpoints before implementation. No cross-paper rate comparison. |
| 3. Problem and evaluation targets | Retain the custom failure gate, zero-based origins, clamping, unobserved best origin and question-weighted seed mean. These definitions already fit the inspected implementation. | Runtime confirmation of the gate and report failure denominators; do not interpret known-failure recovery as online detection. |
| 4. Historical signal catalog | Retain five signals, sampled-token complement naming, truncated entropy and extra-call costs. Distinguish the broad historical catalog from the one-policy follow-up. | Missing-logprob/fallback frequencies and acquisition costs from raw records. |
| 4. Localization rules | Retain tie breaking, scored-position versus trajectory-index offsets, missing-score handling and compounded upstream preferences. | Effective-origin histograms and a development-fitted position control. The notebook's offset-matched random control is not this control. |
| 4. Information access | Add the boundary: the selector sees failed-trace uncertainty; the recovery model receives question, retained prefix and generic hint, with the same searchable corpus. No suffix observations are directly inserted. | Audit rendered prompts, not only strategy labels. Validate any diagnosis adaptation separately. |
| 4. Budgets | Retain historical overshoot and total-step confounds; explicitly separate equal allowances from equal realized tokens, prompt cost, time or FLOPs. | Corrected GPU execution records, full costs and an audited billing log. |
| 5. Evidence and active study | Keep 854 archived failures separate from the future test cohort. Preserve QA3, one model, four conditions, three seeds and one cap. | Recover old IDs; freeze disjoint test IDs, policy, revisions and affordable sample size after the pilot. |
| 6. Archived results | Preserve source-derived tables/figures and descriptive labels. Remove editorial correction prose from the nudge discussion; compare hints only at fixed origins. | Per-question logs for intervals, seed coverage and missing-trial audits. No new result table has been invented. |
| 7. Analysis and discussion | Retain hypotheses rather than causal claims, remove the unsupported length router and distinguish gold-selected candidate coverage from an answer selector. | Within-dataset original-length data, targeted context interventions only if retaining mechanism claims, and validated failure examples. |
| 8. Limitations | Add replay and external-state boundaries. Keep positional confounding, one-model scope, custom scoring, conditional populations and precision limits explicit. | These limitations cannot be declared resolved by prose or extra citations. |
| 9. Conclusion | Require position and closest-method controls before asserting uncertainty's additional value. Do not promise a positive result. | Conclusions must follow the new paired results, including null or mixed findings. |
| Reproducibility statement | Keep provenance, source hashes, CPU checks and missing raw artifacts distinct from empirical reproduction. | Exact GPU/software/dataset manifest, complete logs and a clean rebuild by another person. |
| AI use statement | Retain the disclosure of assisted writing, coding and literature lookup. | Authors must verify all assisted content and complete the project-wide disclosure. |
| Appendix A: execution/annotation | Add the replay-observation equality audit and seed reproducibility boundary. Keep exact hints, annotation fallbacks and localizer conventions. | Observation/state replay checks and genuine human validation if judge-accuracy claims remain central. |
| Appendix B: archived diagnostics | Preserve costs, hint deltas, localization, coverage and seed estimands from available sources. | Recover hidden grid rows only from records; never infer them from incomplete notebook displays. |
| Appendix C: completion protocol | Add dataset-source/alias audit and same-cohort EM/F1 sensitivity; separate online utility from ideal reference gating. Preserve the $119 ceiling. | Source and scoring validation before confirmation; actual throughput before choosing test size. |
| Appendix D: prior-work/control comparison | Add an explicit overlap matrix, nearest-method adaptation rules, position-distribution design and a scope-revision gate. | Extra controls are unimplemented and not silently included in the four-condition pilot. |

## New Literature Checked

Focused primary-source update through September 9, 2026, not a systematic
or exhaustive search. Venue acceptance below is reported only where the
source states it; a recent preprint is not treated as peer-reviewed evidence.

| Paper | Version inspected | Consequence for this paper |
| --- | --- | --- |
| [Doctor-RAG](https://arxiv.org/abs/2604.00865v2) | June 12, 2026, v2; revised title and nine-author list | Closest task-level overlap. Test an affordable, clearly identified diagnosis comparator before expanding datasets. |
| [Repair or Resample? / SymTrace](https://arxiv.org/abs/2608.25920v2) | August 29, 2026, v2 | Prefix fidelity and suffix randomness need separate checks. Generic repair-versus-resampling framing alone is insufficient novelty. |
| [AgentRewind](https://arxiv.org/abs/2608.14380v1) | August 14, 2026, v1 | Limit claims to the environment actually restored; our offline tool state is not a general sandbox rollback system. |
| [When Failures Propagate / AgenticRAG-FP](https://arxiv.org/abs/2608.20627v1) | August 20, 2026, v1 | Useful intervention-grounded attribution design, not evidence that our judge labels identify causal origins. |
| [AgentRx](https://arxiv.org/abs/2602.02475v2) | August 31, 2026, v2; reports acceptance to Findings of EMNLP 2026 | Cite the current 170-trajectory revision, not the older 115-trajectory account. We do not use its reported localization gains as repair results. |
| [ReAgent](https://aclanthology.org/2025.emnlp-main.202/) | EMNLP 2025 proceedings, pp. 4067-4089 | Important earlier QA rollback precedent; bibliography follows the Anthology's author record. |

The [DoVer final ICLR 2026 paper](https://openreview.net/pdf?id=mrEK16Jy6h)
was also checked; its displayed author names replace abbreviated older
metadata. Existing ERGO, Who&When, REFLECT, Causal Agent Replay and CausalFlow
citations remain. Full reproduction of those systems was not attempted.

## Remaining Checks in Code and Data

1. **Replay fidelity:** [the batch runner](../src/agent/batch_runner.py)
   replays tools but ignores returned observations, while the prompt retains
   stored observations. Add mismatch rejection and verify page/cursor/title
   state against a fresh environment on frozen records. This is not yet
   implemented; changed source data could otherwise yield inconsistent state.
2. **Dataset provenance:** [MuSiQue](../src/env/musique_env.py) and
   [2Wiki](../src/env/wikimultihop_env.py) allow mirror fallbacks. Pin source
   revisions, hash both source and converted records, and inspect conversions.
   MuSiQue's period-based splitting and dropped aliases require an explicit
   scoring/data audit, not a claim of official benchmark reproduction.
3. **Sensitivity denominators:** EM/F1 on the original fixed failure cohort
   can be computed from saved predictions. An EM-only failure population adds
   initially threshold-accepted answers and requires additional repair runs.
4. **Closest-method information:** origin-only comparisons must retain the
   same generic hint and recovery evidence; full diagnosis policies require
   separate labeling and complete cost accounting.
5. **Novelty beyond position:** implement a development-fitted origin
   distribution conditional on length, freeze its fallback, and record actual
   test distributions. Matched backtrack offsets alone do not meet this test.

These are scoped completion requirements, not claims that new controls or
GPU results were produced during the writing pass. The
[experiment plan](experiments_to_run.md) defines the order and budget gates.

## Immediate Actions

Recover the original IDs and raw logs while awaiting sufficient AWS quota.
London's last verified quota is 8 P-instance vCPUs; the machine needs 16.
The [AWS appeal](aws_quota_appeal.md) was sent with user approval on September 9
at 13:48 EDT; reassessment is pending. Run only the
bounded development pilot after access, rate and stop checks are resolved.
Profile the essential extra controls before freezing a confirmatory cohort;
do not spend the budget on another dataset first.

## Verification Record

- `python -m pytest -q`: 115 passed, with two Jupyter deprecation warnings.
- After final edits, the paper, documentation and notebook subset passed
  all 37 tests, including plan-only notebook execution without GPU/network.
- `make -C paper iclr-draft`: successful. Final PDF has 17 pages: nine main,
  two for statements/references and six for appendices; seven tables and
  four figures. Every final page was checked visually, with unchanged
  main-page renders also compared byte-for-byte against the inspected set.
- No undefined citations/references or overfull boxes in the final LaTeX
  log. Original `paper/main.tex` and `paper/main.pdf` have no Git diff.
- AWS upload folder rebuilt with the matching notebook/code bundle; no
  GPU provisioned or empirical results generated.

These checks establish document and local-tooling consistency, not research
validity, GPU performance or likely acceptance.
