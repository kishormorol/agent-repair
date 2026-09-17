# Second-Account GPU Quota Request

The $100 credit sits on a second AWS account. AWS does not transfer credits
between accounts, so that balance can only be spent where it already is, and a
new account's default quota for GPU instance families is **0 vCPUs**. The
quota, not the money, is the constraint.

The first account's history sets expectations: the September 9 appeal in
[`aws_quota_appeal.md`](aws_quota_appeal.md) took from September 8 to
September 10 and returned a *partial* approval of 8 of the 16 vCPUs requested.
A new account has no billing history to support the case, so assume slower.

## What to request

| Field | Value |
| --- | --- |
| Region | `eu-west-2` (London) |
| Service | EC2 |
| Quota | **Running On-Demand G and VT instances** |
| New value | **8 vCPUs** |
| Target instance | one `g7e.2xlarge`, NVIDIA RTX PRO 6000 Blackwell 96GB |

Eight is the whole ask: `g7e.2xlarge` is 8 vCPUs and the design needs exactly
one instance. Do **not** request P instances. The September 10 pilot found no
`p5.4xlarge` capacity in any London zone and fell back to `g7e.2xlarge`; asking
for a family you will not use weakens the case and slows review.

## Verified account state (September 17, 2026)

Read from the configured AWS CLI profiles on the author's machine. No
mutation was performed.

| Profile | Account | On-Demand G and VT, `eu-west-2` |
| --- | --- | --- |
| `agent-repair-aws119` | 692430448570 | 8.0 - the account that ran every experiment |
| `alexa-hackathon` | 304118843563 | **0.0** - the second account holding the credit |
| `alexa-ai` | - | cannot assume its role; not usable |

`alexa-hackathon` is the only usable second account and its GPU quota is zero,
which matches the expectation for an account that has never run one. The quota
`L-DB2E81BA` is adjustable and regional, and the change history is empty, so a
request would not duplicate an existing case.

## The command to file it

The Service Quotas API has no justification field, so a request filed this way
carries the numbers but not the use case below. For a zero-to-eight GPU
increase on an account with no billing history, expect AWS to ask for the use
case in the support case the request opens. Have the text ready.

```bash
aws service-quotas request-service-quota-increase \
  --service-code ec2 --quota-code L-DB2E81BA --desired-value 8 \
  --region eu-west-2 --profile alexa-hackathon
```

Filing through the console instead attaches the use case up front, which is
the better path if the deadline allows it.

## Use-case text

> This is a short, single-instance research inference run for a paper on
> language-model agent recovery, submitted to ICLR 2027. We run
> Qwen2.5-32B-Instruct-AWQ under vLLM and evaluate trajectory repair on public
> multi-hop question-answering datasets (HotpotQA, 2WikiMultihopQA). This is
> not pretraining and not a production service.
>
> We need one g7e.2xlarge, which requires 8 vCPUs of On-Demand G quota. We do
> not request a multi-GPU cluster or concurrent instances. A prior phase of
> this work ran on the same instance type in eu-west-2c under an equivalent
> 8-vCPU G quota.
>
> A maximum of $100 in credits is allocated. Sessions are supervised and
> bounded by a billing checkpoint; a shutdown timer is configured and verified
> before inference, and the instance is stopped at the end of each session. We
> understand that credits and billing alerts are not a hard spending cap, and
> that a quota approval does not guarantee capacity.

## Before spending anything, check the expiry

AWS credits carry an expiry date. Confirm it in **Billing and Cost Management
-> Credits** on that account. The plan below spends them during the ICLR
rebuttal period, which is months after submission. Credits that expire first
would have to be spent earlier or not at all, and that changes the plan.

## Nothing is authorized here

This file is a request template and a cost note. It does not authorize
launching an instance, redeeming a credit, or spending against any allocation.
