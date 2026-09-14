# AWS diagnosis/replay follow-up execution

**September 14 update:** this attempt stopped at the development audit,
before any main-study question executed. Its archive is downloaded and
hash-verified. The audit incorrectly counted `finish` as a retrieval call;
the runner counts only `search` and `lookup`. A regression test reproduced
the failure, and all 256 tests pass with the correction. The
[corrected execution](aws_diagnosis_2026-09-14.md) preserves the same questions,
model, prompts, seeds, allowances and comparisons in a new source package.
The original $25 ceiling covers all attempts and retrievals together.

## Startup and recovery

The retained `g7e.2xlarge` instance started September 13 at 18:09:33 UTC.
The environment probe passed, but three regression fixtures required the
declared `nbformat` dependency, which was absent from the GPU environment.
The controller exited before any development or main question executions.
An EC2 stop request was recorded at 18:18:39 UTC, and AWS later independently
confirmed the instance stopped. The first compute interval was 9 minutes
6 seconds, approximately $0.887 at the verified rate.

The instance restarted at 23:38:16 UTC to retrieve the error and recover.
Installing `nbformat==5.10.4` resolved the failure; all 49 selected regression
tests passed in the GPU environment. The original startup directory is
preserved as
`/workspace/aws119-session/diagnosis-20260913-v1-startup-failed-20260913T181726Z`.
Recovery asserts that no study run directories exist before preserving the
old startup and restoring the exact same prepared package. The code, model,
questions, diagnosis prompt, conditions, seeds and analysis family remain
unchanged. New execution-window and recovery records identify the retry.

The recovered controller started at **23:44:35 UTC**. Its GPU probe,
dependency check and 49 regression tests passed; the development run then
started loading the pinned model.

## Budget and independent stop

AWS Pricing returned **$5.84531/hour** for Linux On-Demand `g7e.2xlarge` in
London. The Billing console showed $99.20 total estimated credits remaining,
including $79.20 with EC2 explicitly listed as an applicable service and an
April 11, 2027 expiry. These estimates can lag actual usage. Recovery uses
$77.70 as a conservative eligible balance after the first-attempt allowance.

The recovery deadline is **September 14, 03:18:16 UTC**, at most 220 minutes
from the second EC2 start, including retrieval and setup. The cost worksheet
allows $1.50 for the first attempt and $1.40 for recovery overhead:
`220/60 × 5.84531 + 1.40 + 1.50 = $24.3328033`. This is a conservative
infrastructure planning bound, not an invoice or a current credit offset.

An AWS EventBridge Scheduler stop is enabled for **03:16:16 UTC** under
`aws119-diagnosis-retrieve-20260913`. Its role can only stop this instance,
is restricted to this account's scheduler group, and has an expiring
permission. The controller also refreshes a separate guest stop for each
batch and preserves partial archives on failure. The scheduler configuration
uses AWS's documented [universal target](https://docs.aws.amazon.com/scheduler/latest/UserGuide/managing-targets-universal.html)
and [scoped execution-role trust](https://docs.aws.amazon.com/scheduler/latest/UserGuide/cross-service-confused-deputy-prevention.html).

## Evidence and remaining work

Local records are under
[`output/aws-experiment/2026-09-13-diagnosis/`](../output/aws-experiment/2026-09-13-diagnosis/):

- `aws-launch.json` and `aws-live-observation.json`: first launch and independent-stop observations.
- `controller.log` and `gpu-preflight.log`: preserved failed-startup logs.
- `aws-recovery-budget.json` and `live-checks-recovery.json`: shortened window and conservative balance inputs.
- `recover_preflight.py`: preservation and retry procedure; it rejects any previously started study question runs.
- `validation.json`: original local package verification, with 255 passing CPU tests.

The failed development attempt produced 99 unique repairs, five diagnoses
and 135 policy/seed/allowance rows. The guest journal records its shutdown
at September 13, 23:55:49 UTC. A retrieval attempt at September 14, 03:07:33
ended with the independent scheduled stop at 03:16:17. The final retrieval
ran from 05:53:45 to the stop request at 05:59:04; AWS independently confirmed
stopped state at 06:04:52. The
[verified archive](../output/aws-experiment/2026-09-13-diagnosis/archive-verification-20260914.json),
[shutdown evidence](../output/aws-experiment/2026-09-13-diagnosis/shutdown-journal-20260914.txt)
and [failure diagnosis](../output/aws-experiment/2026-09-13-diagnosis/audit-failure-diagnosis-20260914.json)
are retained. This attempt is incomplete and supplies no main-study results.
