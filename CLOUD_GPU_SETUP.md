# GPU Execution for the Controlled ICLR Study

Updated September 12, 2026. **This is model inference and evaluation.**
The ten-question AWS pilot and subsequent 50-question development run
completed on a 96GB RTX PRO 6000 GPU. The development run produced 186 unique
repairs and 288 condition/seed rows; its paired comparisons are inconclusive.
See the [development report](docs/aws_development_2026-09-11.md) for audited
results, timing, costs and stopped resource state. The
[250-question main study](docs/aws_main_2026-09-12.md) completed all five
batches with six conditions, 1,058 unique repairs and 2,034 condition/seed
rows. All archives and independent audits passed. It found no clear
uncertainty-repair advantage; estimated infrastructure usage is about $9
before credits, tax and transfer. AWS independently confirmed the instance
stopped, with no public IP, on September 12 at 19:02:37 UTC. Retained EBS
costs approximately $0.62/day.

## Current Allocation: $119 on AWS

The user has selected **$119 of AWS credits** for the active plan. This
supersedes the earlier $200 Runpod allocation: do not fund Runpod or add
the $200 cash fallback to this amount. Credits are not a hard spending cap.
The notebook cannot query billing, guarantee coverage or stop AWS charges.
The allocation is not a promise that a complete ICLR study fits within it.

| Allocation | Maximum planned spend |
| --- | ---: |
| Setup and ten-question pilot | $10 |
| Fifty-question development experiment | $5 |
| Frozen core experiment, including initial generation | $74 |
| Infrastructure retries or essential follow-up controls | $10 |
| Storage and uncommitted contingency | $20 |
| Total gross usage allocation | $119 |

Use the **live EC2 Linux On-Demand rate**, not a Capacity Blocks price.
`GPU_HOURLY_USD = None` deliberately requires entering the current quote
before GPU execution. The London EC2 launch selector displayed **$8.944/hour**
for `p5.4xlarge`, Linux On-Demand, on September 8. This base rate excludes
EBS, public IPv4, transfers and taxes, and is not a capacity reservation.
At that rate, $99 buys at most **11.07 billed compute hours**, including
installation, downloads, idle time and failed attempts, before prior usage.
No throughput or study completion is implied. Reduce compute if storage or
other account usage consumes more than the reserve. Taxes are not covered
by promotional credits and may require cash; they are not authorized by
this credit-only plan.

The notebook defaults use one 80GB-class GPU, one pinned Qwen32B model,
three repair seeds, one 1x token allowance, and four conditions:
full restart, random origin with the selected policy's backtracking offset,
fixed early origin, and one prespecified uncertainty policy. The notebook
sets `ORIGIN_SWEEP = False` and `INCLUDE_DIAGNOSTICS = False`.
Equal new-step/token allowances and generic hints are unchanged.

The authorized main-study freeze supplies `POSITION_PROFILE` and adds
position-matched random and uncertainty without backtracking. It completed
250 HotpotQA questions in five batches under a $25 gross allowance, with
four primary contrasts and Holm correction across that family. This
six-condition profile requires at most `18 * N_failed` unique repairs;
the actual number is lower when effective executions coincide. All 180
CPU tests and the final plan-only notebook passed. The other QA datasets
remain pending.

This needs at most `12 * N_failed` unique repairs rather than the previous
two-budget sweep's `48 * N_failed`. The reduction in the job-count upper
bound is 75%; neither actual runtime nor cost is guaranteed to fall by 75%.
The first pilot is ten initial questions with a 120-new-repair guard.
Increase the guard only after measuring throughput and freezing an affordable
cohort. It is not a billing control or a prescription for test sample size.

For scale only, 100 initial questions per dataset means at most 300 failed
questions and 3,600 repairs. At a hypothetical 60 seconds per repair this
alone needs 60 GPU hours; do not assume such a cohort fits the credits.
This is a scaling example, not measured throughput or a sample-size
recommendation. Use development-only timing
and paired-difference variance to choose the test cohort before seeing
test outcomes. A small cohort may only support wide intervals and
inconclusive findings. Do not stop early because significance looks favorable.

Defer extra datasets, second-model replication, the 72B judge, full-origin
curves, and additional budget sweeps. Do not describe the four-condition
comparison as isolating every position effect or reproducing a close method:
the diagnosis/replay comparison remains a gap. Position-matched random
was implemented and evaluated in the completed six-condition HotpotQA study.
Use the follow-up allocation on further controls if feasible; otherwise narrow
the manuscript's claims and report the omissions. Human validation requires
actual annotators; this allocation does not purchase their labor.

### Credit and Quota Checks

Before the September 12 main run, the signed-in console displayed
$108.66 estimated total credits remaining, including $88.66 explicitly
verified as EC2-eligible. Expiry April 11, 2027 was rechecked. These figures
precede the main run's estimated usage. The user authorized using $119.
This observation is not a live ledger: estimated balances update daily,
other workloads may consume them, and some allocation history was marked
incomplete. Recheck the Bills and Credits pages before every session.
See [AWS credit details](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/useconsolidatedbilling-credits.html).

The September 10 live check confirmed **16 On-Demand P vCPUs** in London.
The [appeal](docs/aws_quota_appeal.md) records the earlier partial approval.
H100 capacity was unavailable across London at launch, so the pilot used
one `g7e.2xlarge` under the existing 8-vCPU G quota. Its live Linux On-Demand
quote was **$5.84531/hour**. Sufficient quota and physical capacity remain
separate requirements; recheck both before another session.
If a Free plan blocks GPU access, a Paid-plan upgrade is a separate user
decision with overage liability; never upgrade silently.

### Prevent Accidental Overspending

- Check EC2, EBS, IPv4 and any data-transfer charges in the chosen region.
  Avoid paid Marketplace images, Reserved Instances, Savings Plans and
  Capacity Blocks: standard promotional terms exclude upfront fees, and
  Capacity Blocks bill upfront. Do not treat credits as a prepaid account
  that automatically stops at zero. See [credit terms](https://aws.amazon.com/awscredits/)
  and [Capacity Blocks billing](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-pricing-billing.html).
- Arrange an independent stop for each session and check it works, including
  time billed before Jupyter opens. Set EC2 instance-initiated shutdown
  behavior to **Stop**, not Terminate. Keep a console stop available.
  Never expose Jupyter publicly; use a localhost listener and SSH tunnel.
- Use AWS Budgets alerts on gross usage, **excluding credit offsets**. Alerts
  can lag by hours and are not a hard cap. See [Budgets caveats](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html).
- Update `SPENT_SO_FAR_USD` with gross usage since allocating $119, including
  other workloads consuming the credit pool. Do not enter the net $0 invoice.
  The notebook's $10 `SESSION_ALLOWANCE_USD` is a worksheet allocation, not
  an implemented timeout, account ledger or provider spending limit.
- Use EBS-backed storage and keep verified local backups. EC2 instance-store
  NVMe data is lost on stop; EBS remains billable while stopped. Root EBS is
  normally deleted on termination. See [EC2 stop/start behavior](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html).

Calculate the remaining allocation locally without a GPU or network:

```bash
python scripts/estimate_budget.py --total-usd 119 --reserve-usd 20 --hourly-usd 8.944 --spent-usd 0 --session-usd 10
```

Recheck the rate before use. The notebook rejects invalid or unaffordable session allocations and
requires billing-rate/provider-control confirmations before installations
or GPU calls. Neither check can verify your account or enforce $119 in
provider billing. No funds or cloud resources are managed by these files.

## Recommended Machine

The AWS candidate is **one `p5.4xlarge` in London (`eu-west-2`)**: one H100
80GB, 16 vCPUs and 256 GiB RAM. AWS documents On-Demand purchasing there;
the actual quote, quota and immediate capacity must still be confirmed.
See [P5 specifications](https://aws.amazon.com/ec2/instance-types/p5/)
and the [single-GPU availability announcement](https://aws.amazon.com/about-aws/whats-new/2025/08/amazon-p5-single-gpu-instances-now-available/).
Do not substitute `p5.48xlarge` or a P4 node: they have eight GPUs, and this
notebook only uses one. A smaller-memory G instance fails the current preflight.

The earlier cash-provider comparison is retained for reference only; none
of these vendors can consume AWS credits:

| Provider / GPU | Publicly displayed hourly GPU price | Role |
| --- | ---: | --- |
| Runpod A100 SXM 80GB | $1.59 | Previous cash fallback; not the active AWS plan |
| Runpod H100 SXM 80GB | $3.49 | Comparison price only, not the current budget recommendation |
| Lambda single-GPU H100 SXM 80GB | $4.29 | Comparison price only |

Prices checked September 8, 2026 on the official
[Runpod pricing page](https://www.runpod.io/pricing) and
[Lambda pricing page](https://lambda.ai/pricing). These are not reservations
or live availability checks; location, cloud tier, storage, transfers and
taxes can change the quote. Confirm the console price before renting.

The notebook estimates repair time only after observing actual pilot
execution throughput. It uses the configured conditions when projecting,
so a reduced-profile estimate no longer silently assumes a full sweep.

Colab can be useful for a pilot if a suitable GPU is available, but do not
depend on it as the only deadline-critical resource: GPU types, availability
and runtime limits are variable. See the
[official Colab FAQ](https://research.google.com/colaboratory/faq.html).
The notebook's conservative preflight expects an 80GB-class GPU; it will
not silently switch models on a T4 or a smaller-memory device.

## Files to Upload

The [short launch checklist](START_HERE.md) gives the exact first-run steps.
The builder also stages the matching notebook, code bundle and checklist
together in `output/aws-upload/`, with a file-hash manifest. Upload only
the notebook and code bundle into the same authenticated Jupyter folder.

- [run_iclr2027.ipynb](run_iclr2027.ipynb): portable notebook, defaulting to plan-only mode.
- [agent-repair-iclr2027-code.zip](output/notebooks/agent-repair-iclr2027-code.zip): matching, hashed snapshot of the current corrected code, configs, tests and requirements.

**Use both files together.** The code fixes are local and have not been
pushed to GitHub. A fresh clone of the remote repository does not contain
this complete revision. The bundle includes no `.git`, `.env`, model
weights, raw results or author manuscript. It is an execution package,
not an anonymity-reviewed supplementary artifact.

Rebuild the pair after any code/config/test change:

```bash
python scripts/build_iclr_notebook.py
```

Do not rebuild or modify the source of an active run. The notebook verifies
the matching code digest and refuses to overwrite changed extracted code.
The old Colab notebooks contain destructive cleanup cells; they are not
the entry points for the corrected study.

## Start the Pilot

1. After quota approval, credit checks and launch confirmation, rent one
   `p5.4xlarge` with a Linux x86-64 environment, Python 3.10-3.13 and a
   compatible NVIDIA driver. The [AWS Base GPU Ubuntu 24.04 AMI](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-24-04.html)
   is the candidate; select an official AWS image without third-party fees.
   Avoid spot interruption for the first deadline-critical pilot.
2. Attach persistent storage. Allocate 200 GB initially and monitor it;
   the notebook requests at least 100 GiB free before the first download.
   Larger studies and backups can need more. Keep credentials out of notebooks.
3. Open the provider's authenticated Jupyter interface and upload both files
   into the same folder. Open `run_iclr2027.ipynb`.
4. Run with `EXECUTE_GPU = False` first to verify the package and budget
   worksheet (it deliberately omits hours until the quote is entered). For
   actual execution, confirm the live hourly quote, eligible credit balance
   and independent stop, then set `AWS_CREDITS_CONFIRMED = True`,
   `BILLING_RATE_CONFIRMED = True`, `PROVIDER_BUDGET_CONTROLS_CONFIRMED = True`,
   `PERSISTENT_STORAGE_CONFIRMED = True` and `EXECUTE_GPU = True`.
5. Keep the initial settings: `PHASE = "pilot"`, `DATASET = "hotpotqa"`,
   ten initial questions, batch size 2, three seeds, four core conditions,
   no origin sweep or offset diagnostics, and one 1x token cap. The selected
   strategy is only a pilot candidate. Preserve IDs from every pilot dataset.
6. Run the numbered cells in order. The notebook runs CPU regressions,
   pins/downloads a model snapshot, generates initial trajectories, computes
   stored-token uncertainty, prints the exact repair plan, enforces a job-count
   guard, runs repair, and performs manifest-checked paired analysis.
7. Audit actual token/new-step allowances, evidence/scoring, missing logprob
   rates, prefix replay and trial coverage. Export the raw result archive
   and checksum. Do not call a successful notebook execution human validation
   or completion of the scientific study.

The notebook uses a separate venv with `vllm==0.19.0`, checks dependencies,
and records resolved package versions. It does not assume the old Docker
image or CUDA recipe still works. The matching
[vLLM installation documentation](https://docs.vllm.ai/en/v0.19.0/getting_started/installation/gpu/)
recommends a fresh environment because PyTorch/CUDA binary combinations
matter. Inspect full errors instead of suppressing failed installs.

## Get Results Faster Without Changing the Question

- **Skip unnecessary model calls:** the primary stored-token signals are
  computed from original log probabilities; self-consistency, confidence
  elicitation and the 72B judge are not run in this core notebook.
- **Share effective executions:** several policies selecting the same origin
  and hint use the same raw rollout. The four-condition, one-budget profile
  needs at most `12 * N_failed` repair jobs across three seeds.
- **Tune batching on development-only pilots:** try 2, 4 and 8 with new run
  IDs. Choose using measured throughput, memory and correctness. Freeze
  batching before test execution; do not quietly change it after an OOM.
- **Use one GPU sequentially for this budget:** the notebook can run
  each dataset with isolated paths and a reused, pinned weight cache.
  Change `DATASET`, keep the same frozen study settings, and never share
  dataset/run write paths. Do not rent three idle copies of the environment.
- **Avoid an unnecessary distributed setup:** the notebook uses one GPU per
  process, not tensor parallelism. Renting a multi-GPU node alone does not
  make it use every GPU. Multiple GPU instances are outside this budget profile.

Keep mount paths consistent across machines, such as
`/workspace/agent-repair-iclr2027`. Result model identities include the local
pinned snapshot path, so arbitrary per-machine paths must not be mixed in
the paired analysis. Use the same model revision and package versions.
Do not share a writable venv across concurrent instances installing packages.
The simplest setup is a separate persistent volume for each worker with
the same mount path, rather than concurrent installation into one shared volume.

## From Pilot to Confirmatory Results

The notebook supports `pilot`, `development` and `test` phases. Pilot and
development observations are exploratory. Before test execution:

- Recover all historical/pilot/development question IDs and use fresh
  disjoint IDs for confirmation. Preserve source records and manifests.
- Select one uncertainty policy from the bounded development grid, with
  declared tie breaking. Plan sample size from paired-effect variability
  and a precision target, not from whether a p-value looks favorable.
- Complete the optional policy-freeze cell. It hashes the actual QA3 ID
  lists, stores the selected strategy, resolved model revision, code digest,
  batch size, seeds, allowances, diagnostic-condition choice, sample counts
  and selection rationale. Leave `STUDY_ORIGIN_SWEEP` and
  `STUDY_INCLUDE_DIAGNOSTICS` false for the budget profile. Missing diagnostic
  fields in older manifests mean the old expanded profile, not four conditions.
- Set `MODEL_REVISION`, `TEST_IDS`, `EXPLORED_IDS` and `FROZEN_POLICY` in
  each test notebook. Incomplete/overlapping cohorts and policy mismatches
  are rejected. These checks do not establish that the excluded-ID list
  is complete or that the policy was genuinely selected before evaluation.
- Use a new run ID for changed code, environment, configuration or model.
  Resuming the unchanged run reuses persisted unique executions.

After all three complete files are available, run
`scripts/run_paired_analysis.py --results` with one `results.jsonl` per
dataset and the frozen strategy. Its two primary baselines must be
`full_restart` and the random strategy with the selected policy's offset,
for example `random_step__bt2`. The notebook automatically makes this
choice for its per-dataset analysis. Raw results live beneath
`runs/qwen32b/<phase>/<run-id>/<dataset>/outputs/repairs/`.

The code supports paired question-level intervals and mean-success, EM and
F1 analyses. `scripts/fit_position_control.py` fits the development-only
position distribution, and `scripts/analyze_study_batches.py` verifies and
pools every frozen main-study batch before testing the four primary contrasts.
The notebook does **not** complete the close-method comparison,
gold-free repeated-restart selector or
blinded human-labeling workflow. See the
[experiment priorities](docs/experiments_to_run.md).

## Persistence and Billing

Runpod's container disk is temporary. Its volume disk survives stops but
is lost when the pod is deleted; a network volume exists independently of
the pod. Both commonly appear at `/workspace`, so inspect the actual
storage type. Source: [Runpod storage documentation](https://docs.runpod.io/pods/storage/types).

The final notebook cell archives raw evidence and writes a SHA256 checksum.
Download and verify that archive before terminating compute, and retain the
matching source bundle. A persistent volume is not a backup. Stopping or
deleting compute can still leave billable storage; review the provider's
console. Neither a notebook cell finishing nor closing Jupyter stops billing.

## Verification

The notebook is schema-validated, its code cells and embedded subprocess
scripts are syntax-checked, and the full plan-only path executes in a local
Jupyter kernel without model downloads or GPU calls. CPU tests exercise
cloud configuration isolation, immutable resumes, model snapshot pinning,
test-cohort/policy checks, failure propagation and repair-count accounting.
Real GPU compatibility, throughput, and research outcomes remain unmeasured.
