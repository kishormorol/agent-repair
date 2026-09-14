# Diagnosis/replay follow-up: implementation and launch package

**Execution update:** the [AWS follow-up is complete and independently audited](aws_diagnosis_2026-09-14.md).
The corrected run retains the same frozen questions and study settings.
The preparation snapshot below records the original package state before launch.

September 13, 2026. The local implementation and launch package are complete.
**GPU validation and the new experiment have not run.** The completed
[September 12 main study](aws_main_2026-09-12.md) retains its original cohort,
comparisons and results. This follow-up has a separate protocol and run IDs.

The package is at
[`output/aws-experiment/2026-09-13-diagnosis/prepared/`](../output/aws-experiment/2026-09-13-diagnosis/prepared/).
Its [README](../output/aws-experiment/2026-09-13-diagnosis/prepared/README.md)
contains the exact execution and retrieval instructions. The
[validation record](../output/aws-experiment/2026-09-13-diagnosis/validation.json)
records local checks and artifact identities.

The [upload ZIP](../output/aws-experiment/2026-09-13-diagnosis/agent-repair-diagnosis-followup.zip)
contains the complete prepared package (17.3 MB). All 33 prepared input files
were hash-verified after extracting the ZIP outside the repository, and its
standalone plan-only command passed. The final suite passed **255 tests**
in 25.69 seconds, with three dependency deprecation warnings; `git diff
--check` also passed. The
[test log](../output/aws-experiment/2026-09-13-diagnosis/validation-tests.log)
and archive SHA256 are preserved alongside the validation record.

## Frozen design

| Item | Specification |
| --- | --- |
| Initial questions | 250 HotpotQA questions sampled before follow-up outcomes, in five batches of 50 |
| Exclusions | 799 documented historical, pilot, development and main-study IDs; 6,606 eligible IDs remain before sampling |
| Development check | Five previously failed development questions, using imported initial traces; excluded from the new cohort |
| Model | Qwen2.5-32B-Instruct-AWQ, revision `5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c` |
| Policies | Full restart; maximum-perplexity origin with two-step backtracking; same-model diagnosis/replay |
| Recovery allowances | 0.5×, 1× and 2× original generated tokens; eight new steps; three seeds |
| Primary comparisons | Uncertainty versus restart and diagnosis/replay at each allowance; six two-sided tests in one Holm family |
| Unit of analysis | Question mean across three repair seeds; paired question bootstrap |
| Secondary outcomes | Twelve unadjusted EM/F1 contrasts, policy costs, termination counts and reference-gated full-cohort outcomes |
| Completion | All 250 initial questions and every applicable policy/seed/allowance row; an interrupted cohort stays incomplete |

This is an untrained diagnosis/replay adaptation, not a Doctor-RAG
reproduction. The diagnosis model sees the question and failed trace, with
reference labels and stored uncertainty excluded. Recovery receives the
retained prefix and common retry hint. Diagnosis prose and discarded suffix
evidence are not added to its prompt. Malformed diagnosis output falls back
to restart, with the failure and acquisition costs recorded.

The original full historical Drive pool remains unavailable for byte-level
comparison. The new cohort is disjoint from the documented reconstruction;
no claim about undocumented prior runs is added. The resource-bounded sample
does not guarantee a particular effect precision or statistical power.

## Completed implementation

- The runner caches diagnoses and physical repairs, resumes saved work, and
  rejects changes to the frozen inputs. It records all 27 policy/seed/allowance
  rows per failed question, while deduplicating shared physical executions.
- The analyzer independently replays tool observations and retained prefixes,
  rescores answers with the frozen official answer scorer, validates origins
  and allowances, and checks complete trial coverage.
- The completed accounting charges diagnosis cost to every diagnosis-policy
  attempt and counts each physical diagnosis once in study totals. Request
  and tool counts are checked against replayed steps. A recorded final answer
  must agree with an actual finish action.
- Pooled analysis now accepts the distinct run IDs of the five declared
  batches while validating every row's batch assignment. Previously it would
  reject the completed multi-batch study before producing comparisons.
- The AWS controller verifies package identity and fresh live inputs, performs
  GPU/development checks before the main cohort, enforces deadlines, archives
  partial batches and schedules a guest stop on exit.
- The package includes a deliberately unfilled live-observation template and
  standalone instructions. Default pytest discovery now targets `tests/`,
  avoiding duplicate modules inside historical PDF backup directories.

Tests exercise measured accounting, altered raw records, missing questions,
duplicate rows, inconsistent shared executions and diagnosis costs,
multi-batch pooling, package tampering, repeatable cohort selection, cached
resume, launch limits and failure-time archival/shutdown. Synthetic tests
validate the workflow; they are not model-quality results.

## Rebuild and verify

From the repository root:

```bash
python -m pytest -q
python scripts/prepare_diagnosis_followup.py --output output/aws-experiment/2026-09-13-diagnosis/prepared
python scripts/run_aws_diagnosis_study.py --session output/aws-experiment/2026-09-13-diagnosis/prepared --plan-only
```

Preparation is repeatable only while the bundled source and frozen inputs
match. After changing code or the design, use a new output directory,
`--run-tag` and `--remote-session`; preserve the existing package.

The package's planned session ceiling is $25 gross within the original $119
allocation, with $20 reserved and at most four hours from EC2 start. Its
recorded hourly ceiling is a guard, not a current quote. Launch still needs
the intended cloud scope and fresh resource, rate, credit, cumulative usage
and external-stop observations. No cloud resource was started by this local
completion work, and no new empirical results were added to the manuscript.
