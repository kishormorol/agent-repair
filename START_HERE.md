# The $119 AWS Agent Repair Study

## Current Extension and Completed Studies

The [replication and runtime extension](docs/aws_extension_2026-09-14.md)
is complete. Its final continuation finished on **September 15 at 18:00 UTC**:
two models, three datasets, **600 main evaluations, 120 development
evaluations, 4,323 unique repairs and 8,016 trial rows**. All six cells pass
independent replay, scoring and allowance audits with zero mismatches;
local and remote exports reproduce across all 29 trial columns and result
summaries. All **24 primary Holm-adjusted p-values are 1.000**.

Both completed model archives are downloaded and checksum-verified.
The final retrieval used CPU only; AWS independently confirmed **stopped**,
with no public IP, and the original GPU configuration restored. The original
scientific protocol, prior partial archives and all 8,536 inherited result
files are preserved. The continuation passed **51 GPU preflight tests**;
the local suite passed **314 tests**, plus five transport checks.

The user-authorized **$50 extension cap** remains within the original
**$119 allocation and $20 reserve**. The reconciled extension estimate is
**$43.90** (the full-window estimate was $49.23), and all combined retrievals
are bounded by **$1.22**, before tax and transfer and subject to the report's
inherited shutdown assumption. The extension report records the start/stop evidence, resource
amendments, earlier audit fixes and final reproduction.

The manuscript includes the complete replication, diagnosis follow-up and
runtime component: 25 questions, 225 attempts, and no clear advantage at
the primary ten-second deadline. Its title, abstract, discussion and
limitations reflect the completed multi-dataset study; tables and figures
have separate directories. The standalone abstract is synchronized.

The [diagnosis/replay follow-up is complete](docs/aws_diagnosis_2026-09-14.md).
All 250 questions, 2,079 unique repairs and 3,186 trial rows are downloaded
and independently audited, with zero mismatches. Uncertainty repair showed
no clear advantage over restart or diagnosis/replay at 0.5×, 1× or 2× token
allowances; all six primary Holm-adjusted p-values are 1.000. The audit
correction passed 256 local tests and 49 GPU tests. AWS independently
confirmed EC2 **stopped**, with no public IP, on September 14 at
**12:04:20 UTC**. The conservative combined infrastructure estimate is
**$18.29**, including failed attempts and retrievals, within the original
$25 ceiling; tax and transfer are excluded. The encrypted EBS is retained.

The [September 13 expert review](docs/iclr_expert_review_2026-09-13.md)
adds execution-overlap, cost, termination and scoring diagnostics, a revised
manuscript and a verified statistical supplement. This review uses downloaded
records; it did not launch another GPU run.

The [250-question main study](docs/aws_main_2026-09-12.md) completed all five
batches on September 12 at 05:25:03 UTC. Its six conditions produced 1,058
unique repair executions and 2,034 condition/seed rows; all archives and
independent audits passed. Uncertainty repair succeeded on 9.73% of failed-question
seed trials versus 11.21% for restart, with no clear advantage in the four
prespecified comparisons. Estimated infrastructure usage is about $9 before
credits, tax and transfer, within the authorized $25 session allowance.

AWS independently confirmed the instance **stopped**, with no public IP,
on September 12 at **19:02:37 UTC**. All execution, retrieval, auditing and
shutdown verification are complete. The retained encrypted EBS volume
costs approximately $0.62/day. The report contains the precise observations,
cost assumptions, local archives and audit records.

The 50-question AWS development run completed on September 11, 2026 using
the pinned Qwen32B model on a 96GB RTX PRO 6000 GPU. All 186 unique repair
executions and 288 strategy/seed rows were downloaded and audited. Uncertainty
repair recovered 8.33% of seed trials, versus 5.56% for restart and 8.33% for
random; paired comparisons are inconclusive. Read the
[development report](docs/aws_development_2026-09-11.md) for results, costs,
fixes and main-study planning. The [ten-question pilot](docs/aws_run_2026-09-10.md)
is preserved separately, with its subsequent F1 correction documented.
Before the main run, all 180 CPU tests and the final plan-only notebook
passed. The documented historical pool was reconstructed, producing 549
combined exclusions; full comparison to the original Drive JSON remains
unavailable. The new cohort is disjoint from that reconstructed history.
This is inference and evaluation.

Use the $119 AWS allocation. Do not fund Runpod or add the $200 cash fallback
to this plan. The available credits are not a provider spending cap.

## 1. Resolve AWS GPU Access

Open [AWS Billing](https://console.aws.amazon.com/costmanagement/) and
select Credits. Before the September 12 main run, the check showed
$108.66 estimated total credits remaining, including $88.66 explicitly
verified as EC2-eligible, with expiry April 11, 2027. These balances precede
the main study. Recheck balance,
current unbilled usage and eligible products before renting. Do not paste
passwords, credit codes, keys or payment details into chat or notebooks.

In [Service Quotas](https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas/),
select **Europe (London), eu-west-2**, then **Running On-Demand P instances**.
The September 10 check confirmed **16 vCPUs applied**. H100 launch attempts
still failed with insufficient capacity in all three London zones and with
automatic placement. The pilot therefore used `g7e.2xlarge` under the existing
8-vCPU G quota, at a verified $5.84531/hour. The model and pilot conditions
were retained. Check the run report before resuming the existing instance;
do not create a duplicate instance or reuse an expired stop deadline.

If the account is restricted to the Free plan, review any proposed Paid-plan
upgrade separately. Upgrading permits charges beyond credits; it is not
automatically authorized by these instructions.

## 2. Review One Machine and Its Total Rate

- One **EC2 p5.4xlarge**, Linux On-Demand, default/shared tenancy in London:
  one H100 80GB, 16 vCPUs and 256 GiB RAM. Not `p5.48xlarge` or an eight-GPU P4.
- An official AWS **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu
  24.04)**, x86-64, with no paid Marketplace software subscription. Record the
  exact AMI ID and test `nvidia-smi`; do not assume a named image is validated.
- **200 GiB gp3 EBS root storage**, encrypted, default IOPS/throughput.
  Keep `/workspace` on EBS, not `/opt/dlami/nvme` instance storage. At least
  100 GiB free is needed before the first model download. A volume is not a backup.
- Instance-initiated shutdown behavior **Stop**, not Terminate. Normal EBS
  deletion-on-termination makes premature termination destructive.
- SSH restricted to your IP, not `0.0.0.0/0`. No public Jupyter port.
  Avoid NAT gateways, load balancers, extra instances and idle reservations.
- Inspect the live Linux **On-Demand** rate and additional EBS/IPv4 costs
  before approving launch. A Capacity Blocks price is not an On-Demand quote.

The London launch selector showed **$8.944/hour** for Linux `p5.4xlarge` on
September 8, excluding storage, IPv4, transfer and tax. The $99 compute
allocation therefore allows at most **11.07 billed hours** at that base rate,
not 11.07 hours of useful inference. Recheck at launch; quota is still required.

The $119 allocation is **$10 pilot, $5 development, $74 core study, $10
follow-up/retries and $20 reserve**. The development allowance came from the
original $79 core-study envelope. Count installation, downloads, idle time and failures.
Taxes are not covered by credits and may be cash charges. Do not assume a
Paid AWS account stops billing when credits run out. AWS Budgets alerts
are delayed, not a hard limit; monitor usage excluding credit offsets.

## 3. Arrange a Stop, Then Open Jupyter Securely

After launch approval and successful SSH connection, verify the configured
shutdown behavior in EC2. Set an independent stop before installing models,
subtracting time already billed from the session allowance. One conservative
first checkpoint is **60 billed minutes**, provided the live quote fits $10.
This is a billing checkpoint, not a prediction that setup will finish.
At the checked base rate, one hour consumes $8.944 plus noncompute charges.

On the Ubuntu instance, `sudo shutdown -h +MINUTES` can schedule a guest OS
shutdown after the chosen remaining whole minutes; replace `MINUTES` with
the value calculated from the live rate and launch time. Confirm the timer
with `sudo shutdown --show`. Keep EC2 console access and a separate reminder
to verify state becomes **stopped**. A guest timer is independent of Jupyter
but can fail or be canceled; re-arm it after every restart. Do not use `halt`.
The pilot verified an independent systemd stop timer on the real instance.
The development run replaced the old persistent timer using
[`scripts/configure_aws_stop.py`](scripts/configure_aws_stop.py): a
nonpersistent calendar deadline plus an elapsed-time fallback. Install a
fresh deadline within the next 60 minutes and verify the timer after each
restart. The completed run also verified an external EC2 stop during retrieval.

For an official Ubuntu image with username `ubuntu`, the following commands
prepare Jupyter on the instance, after the stop has been arranged:

```bash
sudo install -d -o ubuntu -g ubuntu /workspace
findmnt -T /workspace
df -h /workspace
nvidia-smi
python3 -m venv /workspace/jupyter-env
/workspace/jupyter-env/bin/python -m pip install jupyterlab ipykernel
/workspace/jupyter-env/bin/jupyter lab --ip=127.0.0.1 --port=8888 --no-browser --notebook-dir=/workspace
```

Use the launch's actual username, key and public DNS; keep the private key
local. In a separate local terminal, an SSH tunnel has this form:

```bash
ssh -i /path/to/your-key.pem -L 8888:127.0.0.1:8888 ubuntu@INSTANCE_PUBLIC_DNS
```

Open `http://127.0.0.1:8888` locally and enter Jupyter's token yourself.
Do not disable token authentication or open port 8888 to the internet.
The notebook runner, GPU environment and model inference were validated in
the completed pilot; a public Jupyter service was not needed for that run.

## 4. Upload the Matching Pair and Check It

Use **`output/aws-upload/`**, not the previous Runpod upload folder. Upload
these two files into the same authenticated Jupyter folder:

1. `run_iclr2027.ipynb`
2. `agent-repair-iclr2027-code.zip`

Do not upload the manuscript, `.env`, credentials or old result directories.
The corrected code is local and has not been pushed to GitHub; the notebook
checks the bundled code digest. `launch_manifest.json` records local hashes.

Leave `EXECUTE_GPU = False` and run all cells first. This checks the matching
package without installations, model downloads or inference. Until you
enter the live `GPU_HOURLY_USD`, the worksheet intentionally estimates no
hours. A plan-only notebook running on EC2 still consumes instance billing.

## 5. Completed Runs and Preserved Cohorts

| Run | Initial questions | Unique repairs | Estimated gross usage |
| --- | ---: | ---: | ---: |
| September 10 pilot, including setup and recovery | 10 | 25 | About $7.50 at its report cutoff |
| September 11 development, including recovery | 50 | 186 | About $3.30 at its report cutoff |
| September 12 main study, including retrieval | 250 | 1,058 | About $9 before credits, tax and transfer |

The development run's [frozen parameters](output/aws-experiment/2026-09-11-development/parameters.json)
and [verified archive](output/aws-experiment/2026-09-11-development/attempt-20260911T133800Z.tar.gz)
are saved locally. It used four conditions, three seeds, a 1x generated-token
allowance and eight new steps. All 160 local tests, 106 AWS preflight tests,
and the complete artifact audit passed.

The [main-study planning worksheet](output/aws-experiment/2026-09-11-development/main-study-planning.json)
is preserved as pre-run planning evidence. The resolved six-condition
[main-study freeze](output/aws-experiment/2026-09-11-main/study-freeze.json)
and all five [verified archives](output/aws-experiment/2026-09-11-main/archive-verification.json)
are saved locally. Historical exclusions combine the reconstructed 500-question
July pool with both AWS development runs, yielding 549 distinct IDs.
The full original Drive JSON comparison remains a provenance limitation.
Preserve the completed outputs and frozen source. Any future cohort needs
a new run ID, a fresh allocation/rate check and an independent stop deadline.

## 6. Preserve Backups and Manage Future Work

Export the timestamped `.tar.gz` and SHA256 from
`/workspace/agent-repair-iclr2027/exports/`. Download and verify the backup
before termination; export partial files before a scheduled stop when
possible. Stop the instance in EC2 and verify it stopped. EBS remains billable.
Do not delete data until backups are verified, then remove unneeded resources
with explicit approval to avoid ongoing storage charges.

Provide the archive, GPU model, actual billed time and gross cost for paper
analysis. It contains raw executions, question IDs, config, model pin, timing
and paired analysis. Pilot results are exploratory, not held-out results.

Use development throughput and variance to freeze an affordable disjoint
test cohort before outcomes. Keep HotpotQA, MuSiQue and 2WikiMultiHopQA as
the intended scope, but do not promise all three fit $119. If the pilot
shows they do not, revise scope transparently before test execution. An
interrupted cohort is not a completed experiment; keep negative results.

See [cloud details and official sources](CLOUD_GPU_SETUP.md) and
[experiments to complete](docs/experiments_to_run.md).
