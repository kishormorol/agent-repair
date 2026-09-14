# AWS P Quota: Status and Sent Appeal

September 10 update: the live London On-Demand P quota is now 16 vCPUs.
The [AWS pilot report](aws_run_2026-09-10.md) records the completed run.
The correspondence below preserves the earlier partial-approval history.

Checked September 9, 2026 in the signed-in AWS console. London
(`eu-west-2`), Running On-Demand P instances, now has **8 vCPUs applied**
and zero displayed utilization. The request was for 16. AWS's latest case
message, September 8 at 23:54 ET, partially approves 8 and invites a
detailed-use-case reassessment. One `p5.4xlarge` needs 16, so this does not
unblock the planned machine. No instance was launched during this check.

**Sent with the user's approval on September 9, 2026 at 13:48:35 EDT.**
The reply below is visible in the correspondence for existing AWS support
case `178892065500209`. After submission the case status is **Customer action
completed**; AWS reassessment is pending. No duplicate quota request was
created. This does not authorize launching compute, upgrading the account
plan, or exceeding $119, and is not approval of the requested 16 vCPUs.

## Sent Reply

Thank you for approving 8 vCPUs. Please reassess the existing request for a
total quota of 16 vCPUs for Running On-Demand P instances in eu-west-2.
Our intended instance is one p5.4xlarge, which requires 16 vCPUs, so the
partial approval does not permit a single instance of this type.

The use case is a short, single-instance research inference and evaluation
pilot for a paper on language-model agent recovery. We plan to run the
Qwen2.5-32B-Instruct-AWQ model using vLLM and evaluate recovery on public
multi-hop question-answering datasets. This is not model pretraining or a
production service. The planned notebook targets one H100 80GB GPU; it does
not request a multi-GPU cluster or concurrent P instances.

We have allocated a maximum of $119 in AWS credits, including a $20 reserve
for noncompute costs and contingency. The first run will be supervised and
limited to a one-hour billing checkpoint, subject to a fresh rate and credit
check. Before inference, we will configure and verify an instance shutdown
timer, monitor billing, and stop the instance after the session. We understand
credits and billing alerts do not constitute a hard spending cap.

Please increase the total regional quota to 16 vCPUs so that this one-instance
pilot is possible. We understand quota approval does not guarantee capacity
and will separately verify cost and capacity before launching. Thank you.

## After a Decision

Verify the applied quota on the London Service Quotas page, not just an email
or request status. Then follow [START_HERE](../START_HERE.md) for the fresh
credit, rate, storage and stop checks. No GPU experiment has run yet.
