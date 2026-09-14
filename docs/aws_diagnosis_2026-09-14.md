# Completed AWS diagnosis/replay follow-up

All five frozen main batches and the pooled analysis completed on **September 14, 2026 at 08:01:04 UTC**. The 250-question study produced **2,079 unique repairs, 118 diagnoses and 3,186 policy/seed/allowance rows**. All six archives, including development, are downloaded and independently audited.

**Uncertainty-guided repair showed no clear advantage over full restart or same-model diagnosis/replay at any of the three allowances.** All six Holm-adjusted primary p-values are 1.000. This does not establish equivalence.

EC2 was independently verified **stopped**, with no public IP, at **2026-09-14T12:04:20.441008+00:00**. The conservative combined infrastructure estimate is **$18.29**, including failed attempts and retrievals, within the original **$25** ceiling. It includes a $1.80 storage/IPv4/overhead allowance and excludes tax and transfer; it is not an invoice. The encrypted EBS volume and prior study records remain retained, with ongoing storage charges.

## Pooled results

Initial generation succeeded on **132/250 questions (52.8%)**, with **89 exact matches (35.6%)** and mean answer F1 **0.4980**. The 118 initial failures each received three policies, three seeds and three generated-token allowances: **354 seed trials per policy/allowance**. Success means official HotpotQA exact match or answer F1 at least 0.5.

| Allowance | Policy | Successful trials | Success | Exact match | Mean F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.5× | Full restart | 10/354 | 2.82% | 2.54% | 0.0303 |
| 0.5× | Uncertainty, backtrack two | 10/354 | 2.82% | 2.54% | 0.0286 |
| 0.5× | Same-model diagnosis/replay | 15/354 | 4.24% | 3.67% | 0.0649 |
| 1× | Full restart | 37/354 | 10.45% | 7.63% | 0.1206 |
| 1× | Uncertainty, backtrack two | 35/354 | 9.89% | 7.06% | 0.1122 |
| 1× | Same-model diagnosis/replay | 36/354 | 10.17% | 7.91% | 0.1306 |
| 2× | Full restart | 44/354 | 12.43% | 9.32% | 0.1654 |
| 2× | Uncertainty, backtrack two | 39/354 | 11.02% | 8.19% | 0.1500 |
| 2× | Same-model diagnosis/replay | 39/354 | 11.02% | 9.32% | 0.1515 |

Each contrast below is **uncertainty minus control**, in percentage points. The paired unit is a question's mean across three seeds. Intervals use 10,000 paired question bootstrap resamples; they are individual, unadjusted 95% intervals. Six two-sided sign-flip tests, assuming within-question exchangeability, share one Holm family.

| Allowance | Control | Difference (pp) | 95% interval (pp) | Raw p | Holm p |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.5× | Full restart | +0.00 | [-2.26, +2.26] | 1.000 | 1.000 |
| 0.5× | Same-model diagnosis/replay | -1.41 | [-4.52, +1.41] | 0.471 | 1.000 |
| 1× | Full restart | -0.56 | [-3.67, +2.54] | 0.876 | 1.000 |
| 1× | Same-model diagnosis/replay | -0.28 | [-4.24, +3.39] | 1.000 | 1.000 |
| 2× | Full restart | -1.41 | [-4.80, +1.41] | 0.492 | 1.000 |
| 2× | Same-model diagnosis/replay | +0.00 | [-3.67, +3.39] | 1.000 | 1.000 |

The [full analysis](../output/aws-experiment/2026-09-14-diagnosis-v2/pooled-analysis-local.json) retains all twelve exploratory EM/F1 contrasts, termination counts and reference-gated full-cohort outcomes. Cross-allowance rate changes are descriptive. Reference labels identify the initial failures, so the full-cohort calculation is not an autonomous deployment policy.

## Measured recovery and diagnosis costs

The allowances match generated tokens available for recovery, with eight new steps. Diagnosis adds its own measured cost: **40.31 generated tokens, 1,234.25 prompt tokens and one request per policy attempt** on average. Every standalone diagnosis-policy attempt pays for the full diagnosis; shared physical diagnoses are counted once in study totals.

| Allowance | Policy | Mean incremental generated tokens | Mean incremental prompt tokens | Mean model requests |
| --- | --- | ---: | ---: | ---: |
| 0.5× | Full restart | 151.3 | 1463.2 | 3.49 |
| 0.5× | Uncertainty, backtrack two | 151.6 | 2146.9 | 3.41 |
| 0.5× | Same-model diagnosis/replay | 188.5 | 3289.1 | 4.19 |
| 1× | Full restart | 274.3 | 3311.8 | 5.38 |
| 1× | Uncertainty, backtrack two | 274.6 | 4504.8 | 5.36 |
| 1× | Same-model diagnosis/replay | 296.1 | 5371.0 | 5.84 |
| 2× | Full restart | 295.5 | 3522.6 | 5.56 |
| 2× | Uncertainty, backtrack two | 295.1 | 4710.7 | 5.53 |
| 2× | Same-model diagnosis/replay | 311.3 | 5555.6 | 5.97 |

Initial generation used 58,010 generated tokens. Unique repairs used **496,780 generated tokens, 7,763,726 prompt tokens, 9,731 model requests and 8,259 retrieval calls**. The 118 physical diagnoses used 4,756 generated tokens, 145,641 prompt tokens and 118 requests. No diagnosis required the malformed-output fallback. A `finish` action counts as a model request but not a retrieval call. Token counts do not establish equal runtime or FLOPs.

## Audit and reproduction

| Batch | Initial questions | Initial failures | Unique repairs | Trial rows | Audit |
| --- | ---: | ---: | ---: | ---: | --- |
| [1](../output/aws-experiment/2026-09-14-diagnosis-v2/audits/main/batch01.json) | 50 | 22 | 378 | 594 | Passed |
| [2](../output/aws-experiment/2026-09-14-diagnosis-v2/audits/main/batch02.json) | 50 | 24 | 441 | 648 | Passed |
| [3](../output/aws-experiment/2026-09-14-diagnosis-v2/audits/main/batch03.json) | 50 | 23 | 414 | 621 | Passed |
| [4](../output/aws-experiment/2026-09-14-diagnosis-v2/audits/main/batch04.json) | 50 | 27 | 459 | 729 | Passed |
| [5](../output/aws-experiment/2026-09-14-diagnosis-v2/audits/main/batch05.json) | 50 | 22 | 387 | 594 | Passed |

The independent main audit replayed and rescored **2,329 trajectories**, checked **13,504 observations**, **2,637 retained prefix steps** and **388 input hashes**, and found **zero mismatches**. It also verified complete trials, source identity, original questions, planned origins, prompts, generated-token allowances, eight-new-step caps, final answers and policy accounting.

The [reproduction check](../output/aws-experiment/2026-09-14-diagnosis-v2/analysis-reproduction-check.json) compared 455 numeric values and all 3,186 rows across 43 trial columns. Local and AWS results agree within `1e-12`; the largest numeric difference is `1.1e-16`. All [six archives](../output/aws-experiment/2026-09-14-diagnosis-v2/archive-verification.json) match their live AWS SHA-256, status record and checksum file. All 33 prepared input files also match the frozen package.

## Audit correction and preserved protocol

The first development attempt completed its 135 trial rows but failed the audit before any main question executed. The audit incorrectly summed the environment's `is_tool_call` flag, which is also true for `finish`; the runner records retrieval calls as `search` and `lookup`. This explained all 45 mismatches in the 104 downloaded development trajectories. Two audit expressions were corrected, with a regression test that failed before the fix. The full local suite passed **256 tests**; all **49 GPU-environment regression tests** passed, followed by the real development audit.

The [amendment verification](../output/aws-experiment/2026-09-14-diagnosis-v2/amendment-verification.json) confirms that only the auditor and its test changed in the source bundle. The same 250 IDs and order, 799 exclusions, five development IDs, Qwen2.5-32B-Instruct-AWQ revision `5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c`, diagnosis prompt, seeds, allowances and comparisons were retained. Distinct run IDs preserve both source versions and all prior responses. This is an untrained same-model diagnosis/replay adaptation, not a Doctor-RAG reproduction. Diagnosis receives the failed trace without reference labels or uncertainty; recovery receives the retained prefix and common retry hint, without diagnosis prose or discarded suffix evidence.

Source identity: `515069e6d005f3aad48060c02ae7219828bde3c8502dc73d28a4ade0613d709b`. Main protocol identity: `9dc15ffe49874be12a2b82b124e28e209daf7a3476337fb3dfe92cb6edfbe81b`. The [September 12 six-condition study](aws_main_2026-09-12.md) remains a separate experiment and comparison family.

## Timing, retrieval and resource state

The corrected EC2 session started at **06:07:37 UTC**, its controller started at **06:09:01**, and the pooled analysis finished at **08:01:04**. The guest journal records shutdown at **08:04:11**. The local monitor downloaded the development archive, then experienced long connection gaps; this did not interrupt remote inference. A final retrieval started at **11:52:47**, downloaded the entire session and verified all archives, and requested stop at **11:58:16**. Fresh independent AWS stops bounded both execution and retrieval.

The [cost worksheet](../output/aws-experiment/2026-09-14-diagnosis-v2/cost-estimate.json) records all six billing intervals, the verified $5.84531/hour rate and the final [AWS stopped-state observation](../output/aws-experiment/2026-09-14-diagnosis-v2/final-state.json). [Completion and shutdown evidence](../output/aws-experiment/2026-09-14-diagnosis-v2/remote-completion-evidence.json), [raw session files](../output/aws-experiment/2026-09-14-diagnosis-v2/retrieved/), [trial data](../output/aws-experiment/2026-09-14-diagnosis-v2/pooled-analysis-local.trials.csv) and [extracted runs](../output/aws-experiment/2026-09-14-diagnosis-v2/results/) are saved locally.

## Interpretation limits

The evidence concerns one model, one HotpotQA cohort, offline evidence and the declared eight-step recovery policy. It supplies the diagnosis comparator and additional allowances missing from the earlier controlled study, while other models, new datasets, human validation and a matched total-runtime comparison remain untested. The original full historical Drive pool remains unavailable for byte-level comparison; cohort disjointness is relative to the documented reconstruction. Nonsignificant contrasts do not establish that the policies are equivalent.
