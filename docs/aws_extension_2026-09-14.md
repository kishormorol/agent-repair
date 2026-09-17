# AWS replication and runtime extension

## Completed and independently reproduced

The extension finished on **September 15, 2026 at 18:00:00 UTC**, before
the authorized deadline. Both model archives are downloaded and
[checksum-verified](../output/aws-experiment/2026-09-15-extension-completion/archive-verification.json).
The [final reproduction](../output/aws-experiment/2026-09-15-extension-completion/analysis-reproduction-check.json)
passes all **six cell audits**, covering **600 main evaluations, 120
development evaluations, 4,323 unique repairs and 8,016 trial rows**, with
**zero mismatches**. Local and remote trial CSVs are byte-identical across
all **29 columns**. All result summaries reproduce; platform-specific
roundoff diagnostics are separately validated within the frozen tolerance.

| Model | Dataset | Initial failures / 100 | Unique repairs | Trial rows | Audit mismatches |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen32B | HotpotQA | 43 | 662 | 1,128 | 0 |
| Qwen32B | MuSiQue | 72 | 880 | 1,512 | 0 |
| Qwen32B | 2Wiki | 28 | 308 | 588 | 0 |
| Mistral12B | HotpotQA | 78 | 804 | 1,638 | 0 |
| Mistral12B | MuSiQue | 83 | 940 | 1,743 | 0 |
| Mistral12B | 2Wiki | 67 | 729 | 1,407 | 0 |
| Total | | 371 | 4,323 | 8,016 | 0 |

The 371 initial failures contribute 7,791 token-policy rows (seven policies,
three seeds); Qwen HotpotQA also contributes the separate 225 runtime
attempts. Conditional repair rates and restart contrasts are:

| Model | Dataset | Restart | Uncertainty | Diagnosis/replay | Uncertainty minus restart (pp) | Individual 95% interval (pp) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen32B | HotpotQA | 13.95% | 11.63% | 10.85% | -2.33 | [-7.75, +1.55] |
| Qwen32B | MuSiQue | 7.87% | 4.63% | 7.41% | -3.24 | [-7.41, +0.93] |
| Qwen32B | 2Wiki | 15.48% | 21.43% | 16.67% | +5.95 | [+1.19, +11.90] |
| Mistral12B | HotpotQA | 11.54% | 12.39% | 9.40% | +0.85 | [-2.56, +4.27] |
| Mistral12B | MuSiQue | 7.23% | 6.02% | 7.63% | -1.20 | [-3.21, +0.80] |
| Mistral12B | 2Wiki | 16.92% | 15.92% | 14.43% | -1.00 | [-6.97, +4.48] |

**All 24 primary Holm-adjusted p-values are 1.000.** The family compares
uncertainty with restart, position-matched random, no-backtracking
uncertainty and diagnosis/replay in every cell. Qwen 2Wiki's individual
percentile interval excludes zero, while its unadjusted sign-flip p-value
is 0.131 and adjusted p-value is 1.000. These are different prespecified
constructions; the interval is not an inversion of that test and does not
have simultaneous family-wise coverage. The results provide no clear
primary-test advantage and do not establish equivalence. Model-specific
failure cohorts prevent conditional repair rates from ranking overall
model reliability.

The [final preservation check](../output/aws-experiment/2026-09-15-extension-completion/final-preservation-check.json)
verifies all **8,536 inherited records** and all **8,662 records present at
the last restart** remain byte-identical. The final CPU retrieval completed
at **21:41:14 UTC**; AWS independently confirmed **stopped**, no public IP,
and the original GPU configuration restored at **21:41:43 UTC** in the
[final state record](../output/aws-experiment/2026-09-15-extension-completion/final-state.json).
The [transport cleanup](../output/aws-experiment/2026-09-15-extension-completion/transport-cleanup.json)
verified deletion of the temporary endpoint and restoration of the original
SSH rule at **21:43:50 UTC**. The temporary stop schedules are removed.
The [cost reconciliation](../output/aws-experiment/2026-09-15-extension-completion/cost-reconciliation-20260915.json)
estimates **$43.90** for the combined extension and bounds combined retrievals
by **$1.22**, within the authorized $50 and $2 limits. The inherited
first-session shutdown assumption, tax/transfer exclusions and retained
storage limits remain explicit below. All **314 local tests** and five
transport checks passed. The paper now reports the completed replication,
all policy results and costs, all primary contrasts, and the runtime study.

## September 15 completion continuation

The approved completion resumed at **16:55:31 UTC**, with its controller
running from **16:56:20 UTC** and the existing **18:12:37 UTC** deadline.
All **8,662 saved JSON records** were preserved, including the **8,536**
records inherited from the prior runs. The frozen scientific package is
unchanged. The latest execution state is saved in the
[completion status](../output/aws-experiment/2026-09-15-extension-completion/latest-status.json).

The temporary managed SSH endpoint restored access. A subsequent failure
was traced to the short bootstrap shutdown remaining scheduled after the
longer experiment timer was installed. The previous boot journal confirms
that it blocked new SSH logins at **16:43:57 UTC**, triggering the original
monitor's stop request. The corrected timer setup cancels that short
shutdown only after verifying its replacement; it preserves the bootstrap
stop if verification fails. The regression failed before the fix, then all
**314 local tests** passed. Five operational checks also cover authenticated
tunnel routing and resumable archive downloads. The
[transport amendment](../output/aws-experiment/2026-09-15-extension-completion/transport-amendment.json)
preserves the findings, checks, timestamps and unchanged resource deadline.
The continuation executed only missing jobs and completed all six-cell
audits and the final reproduction reported above.

## Original extension and first continuation

The separately frozen extension started on **September 14, 2026 at
17:18:37 UTC**. All Qwen32B executions completed, but two audit defects
prevented progression to Mistral. Authenticated retrieval recovered the
original records. After reproducing and fixing both defects, all three
Qwen dataset audits passed: **300 main evaluations, 1,850 unique repairs
and 3,228 trial rows, with zero mismatches**.

Mistral resumed on **September 15 at 05:22:59 UTC** and stopped at
**07:16:24 UTC** after saving a partial archive. The recovered HotpotQA cell
was complete and locally audited: **804 unique repairs, 1,638 trial rows and
zero mismatches**. At that earlier stop, MuSiQue had 100 saved originals and
83 failures: 62 had all repairs, one had five missing executions, and 20
had not entered repair. Mistral 2Wiki had not started. Four of six replication
cells were locally audited before the final completion continuation.

The completed **25-question runtime component** is now in the paper, with
225 policy/seed attempts. At the primary ten-second threshold, uncertainty
succeeds on 12/75 attempts (16.0%), restart on 13/75 (17.3%), and diagnosis
on 9/75 (12.0%); both primary Holm-adjusted p-values are 1.000.
The earlier partial replication was retained until the full six-cell comparison completed.

The user authorized the prepared **$50 combined extension cap** on September
15, retaining the original $119 total and $20 reserve. The
[completion resource amendment](../output/aws-experiment/2026-09-15-extension-completion/resource-amendment.json)
changes resource limits only; all scientific protocol bytes remain unchanged.
Its 150-minute maximum window has an independent AWS stop at **18:10:37 UTC**
and a hard deadline at **18:12:37 UTC**. Initial SSH access failed because the
operator's network address changed; the failed startup was stopped and is
included within that same window. The retry verifies and updates the single
allowed SSH address before launch. No cohort, policy, seed or outcome-based
stopping rule changes.

## Recovery and audit amendments

The [continuation record](../output/aws-experiment/2026-09-14-extension/retrieval-check-2026-09-14.json)
preserves the first blocked retrieval check at **September 15, 00:02:54 UTC**.
Access was restored, and a separate $2 retrieval window recovered a
112,023,169-byte Qwen archive and the original status/log metadata. Their
[SHA-256 checks](../output/aws-experiment/2026-09-14-extension/archive-verification.json)
passed. The original status records an audit failure around September 14,
20:43 UTC; Mistral had not started. The guest journal records shutdown at
20:51:13 UTC. AWS independently confirmed the retrieval instance stopped,
with no public IP, on September 15 at 05:08:20 UTC before the continuation.

The [source amendment](../output/aws-experiment/2026-09-14-extension/audit-amendment-v2-20260915.json)
and its patch preserve the original and corrected source hashes. Only the
envelope loader, uncertainty comparison and their regression tests changed:

1. JSON converts integer token IDs to string keys. Lexicographic sorting of
   keys such as `"10"` and `"2"` changed the hash of decoded token records.
   Restoring integer keys in the exact token schema reproduces the original
   saved hash. Raw records, probability values and checksums are untouched.
2. Recomputing stored uncertainty produced 3,542 floating-point differences,
   with a maximum absolute difference of **1.78 × 10⁻¹⁵**. The auditor now
   permits **10⁻¹² absolute roundoff** while recomputing all position profiles
   and policy origins. Those decisions must still match exactly. Tests reject
   altered probabilities, larger differences and even tolerated roundoff
   that changes an origin.

Both failures were reproduced before correction. All **292 local tests
passed** afterward. The [independent Qwen audit](../output/aws-experiment/2026-09-14-extension/qwen-analysis-audited.json)
verifies the frozen inputs, all tool observations, prefixes, reference scores,
policy/seed coverage and allowances. It correctly reports `complete: false`
because that historical audit contains only Qwen; the completed pooled
analysis above now reports `complete: true`.

| Qwen dataset | Main questions | Initial failures | Development questions | Unique repairs | Trial rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | 100 | 43 | 20 | 662 | 1,128 |
| MuSiQue | 100 | 72 | 20 | 880 | 1,512 |
| 2WikiMultiHopQA | 100 | 28 | 20 | 308 | 588 |
| Total | 300 | 143 | 60 | 1,850 | 3,228 |

HotpotQA includes 225 additional timed attempts on a 25-failure subset of
the same replication cohort. The continuation verified all **4,739 original Qwen result files**
against a per-file manifest before copying them. Its fresh Qwen archive
contains the amended audit source and identical raw results.

The [Qwen reproduction check](../output/aws-experiment/2026-09-15-extension-resume/qwen-reproduction-check.json)
compares every result field and all **3,228 rows × 29 columns**; the trial
CSV files are byte-identical. Linux and macOS report different counts of
tiny uncertainty roundoff, so the finalizer validates those two diagnostic
fields separately on each platform and preserves them in its report.
All other analysis fields and trial columns must match within 10⁻¹².
After adding finalization and reporting regressions, the complete local
suite passed **306 tests**; the continuation's GPU preflight passed **51 tests**.

Mistral's startup emitted a Transformers tokenizer-regex warning. A
[direct check](../output/aws-experiment/2026-09-15-extension-resume/tokenizer-check.json)
found that the complete tokenizer backend is byte-identical with and without
the suggested fix for the pinned snapshot. No tokenizer setting changed.

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

The independent AWS stop schedule targeted **September 14, 22:46:36 UTC**.
The recorded session deadline was **September 14, 22:48:36 UTC**. The
controller also renews a guest stop timer, stops inference before the
deadline to preserve results, and
requests shutdown eight minutes after completion or failure for retrieval.
The encrypted 200 GiB EBS volume is retained. This run reuses the existing
instance; it does not add another machine or a cash fallback.

The [cost reconciliation](../output/aws-experiment/2026-09-14-extension/cost-reconciliation-20260915.json)
combines CloudTrail, the guest shutdown journal and AWS state observations.
It bounds the original execution at **$20.81 compute**, including a 60-second
shutdown grace. The separate retrieval used at most **412 billed seconds**,
or **$1.02 including its $0.35 overhead allowance**, inside its $2 ceiling.
For a normal stop, instance usage is not billed in the stopping state;
retained EBS remains billable ([AWS lifecycle documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html)).

The [September 15 launch checks](../output/aws-experiment/2026-09-15-extension-resume/live-checks.json)
reverified the $5.84531/hour rate and bounded the continuation at 115 minutes.
Original compute plus that full window and the original $2.50 overhead
allowance was bounded by **$34.52** before launch. This remains inside the $35 extension ceiling and
the overall $119 allocation with its $20 reserve. The continuation
stop targeted **07:15:52 UTC**, two minutes before the session deadline;
the renewed guest timer provides a separate stop path. These are conservative
infrastructure estimates, not invoices; tax and transfer are excluded.

The [reconciled continuation estimate](../output/aws-experiment/2026-09-15-extension-resume/cost-reconciliation-20260915.json)
is **$34.36** for the original extension and first continuation, plus a
**$1.09 combined retrieval bound**. The original estimate assumes a
60-second transition after guest shutdown; it is not an invoice or an
independently verified first-session transition bound. The GPU retrieval
restart failed for insufficient capacity. Recovery instead used a temporary
CPU configuration from 10:46 UTC, verified both archives, and independently
confirmed stopped with the original GPU type restored at **10:47:33 UTC**.
Its Mistral archive SHA-256 is
`6e56c146aad7e0d3894cd3b54c2eabe5db8eb709da93cfe5030a9e7cbbbd5003`.

The authorized completion preflight used the verified **$5.84531/hour** quote
and **$60.98** credits from the AWS Free Tier API. It deducts $20 of other
credits and $14 for pending usage/storage before relying on $26.98. The
full new window including $0.25 overhead gives a combined extension
estimate of **$49.23**, an overall estimate of **$96.23**, and at least
**$22.77** unallocated. Failed startup time remains inside that window.

The final reconciliation includes all four successful GPU starts, including
failed staging. Start-to-AWS-stopping intervals total at most **5,718
seconds**, giving **$9.2843** completion compute plus the $0.25 overhead
allowance. Adding the inherited $34.3586 estimate gives **$43.8929**, reported
conservatively as **$43.90**. The guest journal records shutdown at
18:08:08 UTC; the independent AWS stop found the instance already stopping
at 18:11:12 UTC, before the 18:12:37 deadline. This reconciliation retains
the earlier original-session transition assumption and is not an invoice.
The entire approved-window estimate remains $49.23.

Final retrieval reused the stopped instance as `m7i.large`, at a live
$0.11655/hour quote. Its 16-minute ceiling, $0.10 overhead and all earlier
retrieval bounds total **$1.2176**, rounded upward to **$1.22**. It ran no
inference, verified both archives, stopped the instance, and restored
`g7e.2xlarge`. The overall estimate using the prior $45 bound is **$90.12**,
leaving at least **$28.88** of the original allocation under these assumptions.
The encrypted EBS remains retained and billable; later storage, tax and
transfer are excluded from these experiment estimates.

## Completion and reproduction

The automatic monitor saved Qwen before local monitoring ended. The bounded
CPU recovery retrieved the final Mistral archive, logs and pooled exports,
verified checksums and byte counts, refused conflicting records, and
independently verified shutdown. All prior records passed the final
per-file preservation check. The complete six-cell comparison now has every
required record, independent audit and matching local/remote analysis.

The successful reproduction and asset-generation commands are:

```bash
python scripts/finalize_extension_study.py --base output/aws-experiment/2026-09-15-extension-completion
python scripts/build_extension_paper.py --base output/aws-experiment/2026-09-15-extension-completion
```

The finalizer refuses active or failed runs. It checks archive hashes,
invokes the downloaded amended auditor, validates all six cell audits, and
compares every local/remote result and trial column. Platform-specific
roundoff counts remain separately validated diagnostics. Its successful
output is `analysis-reproduction-check.json`.

The current paper reports the completed main study, diagnosis follow-up,
full replication and runtime component; its extension appendix records the
frozen design, every policy's results and costs, all paired contrasts and
recovery history. Tables and figures are organized under
`paper/generated/tables/` and `paper/generated/figures/`. The complete
replication builder rejects incomplete or changed evidence and is now included
in `make -C paper iclr-draft`. The runtime builder validates its separately
prespecified component. The title and standalone abstract match the manuscript.
