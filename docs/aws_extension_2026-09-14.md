# AWS replication and runtime extension

The separately frozen extension started on **September 14, 2026 at
17:18:37 UTC**. At the **17:42 UTC** check it was executing Qwen32B repairs
on HotpotQA. The AWS console showed the existing `g7e.2xlarge` instance
running with all three health checks passing. Completed-model archives and
the final independent audit are still pending; no extension success rates
are claimed here.

## Frozen scope

The [protocol](../output/aws-experiment/2026-09-14-extension/prepared/protocol.json)
fixes **100 main and 20 disjoint development questions per dataset**, shared
across two models: **600 main model/question evaluations and 120 development
evaluations**. The model-specific initial failures determine each repair
cohort, so conditional repair rates do not directly rank model reliability.

| Model | Immutable revision | Precision |
| --- | --- | --- |
| Qwen/Qwen2.5-32B-Instruct-AWQ | `5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c` | AWQ, automatic dtype |
| mistralai/Mistral-Nemo-Instruct-2407 | `04d8a90549d23fc6bd7f642064003592df51e9b3` | bfloat16 |

| Dataset | Main questions | Development questions | Documented historical exclusions |
| --- | ---: | ---: | ---: |
| HotpotQA | 100 | 20 | 1,049 |
| MuSiQue | 100 | 20 | 500 |
| 2WikiMultiHopQA | 100 | 20 | 500 |

All seven policies use three seeds, one original-generated-token recovery
allowance, eight new steps and the same generic retry hint: full restart,
offset-matched random, fixed early, uncertainty with two-step backtracking,
development-fitted position-matched random, uncertainty without backtracking,
and same-model diagnosis/replay. The diagnosis policy pays its full measured
acquisition cost. Shared origin/prompt/seed executions are deduplicated only
for the token study.

Four uncertainty-versus-control comparisons in each of six model/dataset
cells share a **24-comparison Holm family**. Individual question-bootstrap
intervals use 10,000 resamples. The bounded sample size does not promise
power to resolve small effects or support equivalence claims.

The runtime experiment uses the first 25 frozen Qwen32B HotpotQA failures
(all if fewer), three policies and three seeds. Each policy's recovery executes
separately in shuffled order with a warmed model and prefix caching disabled.
The measured time includes origin acquisition and prefix replay. Recovery
has no total generated-token cap, but retains eight new steps and the
512-token per-request limit. Success by **10 seconds** is primary, with two
comparisons in a separate Holm family; 5 and 20 seconds are exploratory.
Full attempts are measured, then evaluated against deadlines. This does not
measure savings from actively cancelling work at a deadline. Diagnosis
acquisition is measured once per question and its full measured cost is
charged to every diagnosis-policy attempt; the study does not repeatedly
measure diagnosis latency for each seed.

## Input validation

The [data audit](../output/aws-experiment/2026-09-14-extension/prepared/data-audit.json)
records **174,191 official-scorer checks with zero mismatches**. MuSiQue
retains full source paragraphs, repeated titles in source order, and answer
aliases. 2Wiki scoring retains source aliases and demonyms. Support labels
are available only to evaluation. Original historical Drive files remain
unavailable for a full byte comparison; exclusions retain that limitation.

Before launch, **273 local tests passed**. The continuation check found and
fixed a later mismatch between the paper and standalone abstracts; the
full 273-test suite then passed. The subsequent table-integrity and final
reproduction checks have their own regression tests. These local reporting
changes do not change the frozen execution package. All **46 GPU preflight
tests passed** before the extension began inference.

Protocol SHA-256:
`be6f3170150ca7541cf5a4cb3d8742dddb7582d1ec995293959a7b76b2fc305c`.
Uploaded ZIP SHA-256:
`e7dc416afc69ec9e29b022fdb11c08300f156a7d63c4d22607f3617ddadb49ac`.

## Spending and stop controls

The [launch checks](../output/aws-experiment/2026-09-14-extension/live-checks.json)
record the existing **$35 session ceiling inside the original $119 AWS
allocation**, with a $20 reserve and a conservative $45 prior-usage bound.
The checked rate is $5.84531/hour. The maximum 330-minute window plus $2.50
overhead gives a $34.65 infrastructure bound, before tax and transfer.

The independent AWS stop schedule targets **22:46:36 UTC**. The recorded
session deadline is **22:48:36 UTC**. The controller also renews a guest
stop timer, stops inference before the deadline to preserve results, and
requests shutdown eight minutes after completion or failure for retrieval.
The encrypted 200 GiB EBS volume is retained. This run reuses the existing
instance; it does not add another machine or a cash fallback.

## Completion and reproduction

The [retrieval monitor](../output/aws-experiment/2026-09-14-extension/monitor_extension.py)
downloads completed-model archives, checks their SHA-256 and preserves
conflicting local records. Its latest observation is
[latest-status.json](../output/aws-experiment/2026-09-14-extension/latest-status.json).
Completion requires all frozen cells, policy/seed rows, official rescoring,
tool/prefix replay, verified local archives and an independently observed
EC2 stopped state. A partial cohort remains incomplete.

After the retrieval monitor reports complete, run the finalizer. It checks
all archive hashes, invokes the downloaded frozen auditor, verifies every
cell audit, and compares the full local/remote JSON and trial exports:

```bash
python scripts/finalize_extension_study.py
```

The finalizer refuses active or failed runs. Its successful output is
`analysis-reproduction-check.json`, with hashes, audited counts and the
comparison tolerance. Verify EC2 stopped independently before closing the
run. The existing manuscript currently
reports the completed six-condition HotpotQA study and diagnosis follow-up.
