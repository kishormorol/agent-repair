# AWS development experiment — September 11, 2026

**Completed and audited.** The notebook ran from **13:21:54 to 13:38:00 UTC**
(9:21:54–9:38:00 AM EDT), exited with code 0, and saved its archive before
automatically shutting down. The downloaded archive passed its SHA256 check.
EC2 was confirmed **stopped, without a public IP, at 19:16:34 UTC**.

This is development evidence. It does not establish an uncertainty-repair
advantage, and the held-out study remains unexecuted.

## Results

Fifty HotpotQA development questions were selected before outcomes, excluding
all ten questions from the earlier AWS pilot. Of the 50 initial attempts,
**26 succeeded** under the configured **exact match OR answer token F1 ≥ 0.5**
gate. **18 were exact matches**; initial mean F1 was **0.5067**.
All 24 failed questions received four conditions and seeds 0, 1 and 2.

| Repair condition | Successful seed trials | Success rate | Exact-match rate | Mean F1 |
| --- | ---: | ---: | ---: | ---: |
| Full restart | 4 / 72 | 5.56% | 2.78% | 0.0654 |
| Random origin, backtrack 2 | 6 / 72 | 8.33% | 5.56% | 0.0931 |
| Fixed early origin | 7 / 72 | 9.72% | 6.94% | 0.1353 |
| Perplexity argmax, backtrack 2 | 6 / 72 | 8.33% | 4.17% | 0.0904 |

There are **288 complete condition/seed rows and 186 unique repair executions**.
Identical requests share an execution; the rows are not independent questions.
The paired analysis averages seeds within each question and resamples whole
questions, retaining all conditions and seeds together.

| Uncertainty contrast | Success difference | 95% question-bootstrap interval | Holm-adjusted p |
| --- | ---: | ---: | ---: |
| Versus full restart | +2.78 percentage points | −2.78 to +9.72 points | 1.00 |
| Versus random origin | 0.00 percentage points | −5.56 to +5.56 points | 1.00 |

These are exploratory comparisons over 24 failed questions. The intervals
include both gains and losses; non-significance does not demonstrate
equivalence. Fixed early's highest observed rate is not grounds for declaring
a winner or silently changing the prespecified uncertainty policy.

Uncertainty selected origin zero for **14 of 24 questions** (58.33% of its
seed rows). Random with backtrack two selected zero in 43 of 72 rows (59.72%).
The frequent overlap with restart makes position controls relevant to any
stronger claim about information supplied by uncertainty.

## Corrections and verification

Before running, the following fixes passed **160 local tests** and the rebuilt
plan-only notebook. The real GPU environment passed **106 preflight tests**.

- HotpotQA scoring now follows the categorical-answer and empty-overlap rules
  in the [official evaluator](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py).
  Both sequential and batched execution use the environment's scorer.
- Prefix replay checks step order, actions, observations, tool-call flags and
  retrieved titles before requesting new inference. Regression tests cover
  stale evidence and restoration of lookup state in both loops.
- An incomplete action at the generated-token cap is labeled `budget` while
  retaining `invalid_action_step`. Its unsuccessful outcome is unchanged.
- The guest stop uses a nonpersistent calendar deadline plus an elapsed-time
  fallback. Expired deadlines are rejected when installing a new timer.
- Development IDs and their known exclusions are copied into the frozen run
  configuration. The source bundle and transferred raw dataset hashes matched.

The local artifact audit checked every expected question/condition/seed row,
all **51 recorded input hashes**, the repair manifest, and source consistency.
It replayed all **236 trajectories**, verified **1,521 observations** and
**222 retained prefix steps**, and recomputed all answer scores using the
three scoring functions from the downloaded official evaluator.
There were **zero mismatches and zero token or new-step budget violations**.

The 186 unique repairs ended as follows: 72 with an answer, 75 at the token
budget and 39 at the step limit. Of the budget stops, 66 retained an incomplete
action at the exact cap. All remain in the results. Initial generation used
11,718 tokens; unique repairs used 56,082 generated and 836,852 prompt tokens.

The [earlier pilot report](aws_run_2026-09-10.md) now includes a derived scoring
correction: its initial mean F1 is 0.4810, with no binary success changes.
Its original archives and recorded outputs were preserved unchanged.

## Configuration and timing

| Item | Recorded value |
| --- | --- |
| Run ID | `aws119-development50-20260911-v1` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 96GB |
| Instance | `i-03b33e00c47b11be6`, `g7e.2xlarge`, eu-west-2c |
| Model | `Qwen/Qwen2.5-32B-Instruct-AWQ` |
| Model/tokenizer revision | `5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c` |
| Source snapshot SHA256 | `078308104c78745b9e387add9ac1bd84c9070f15baee27b4e50b81675d168e23` |
| Code ZIP SHA256 | `fdd203304326499372254c9c3380f0986ee08f1195d8ed79b19ac4a53160db8d` |
| Cohort selection | Uniform without replacement from sorted eligible IDs; seed 20260911 |
| Repair limits | Original trajectory's generated-token count; eight new steps; batch size 2 |
| Initial-generation stage | 319.28 seconds, including engine startup |
| Repair stage | 622.55 seconds, including engine startup |
| Notebook wall time | 16 minutes 6 seconds |

This was the same GPU/model configuration as the pilot, not an H100 run.
No judge, additional dataset, fine-tuning or origin sweep ran.

## Spending and shutdown

The session's **$5 cap** was allocated from the original $79 main-study
envelope, leaving **$74 for the core study**, with the $20 reserve and $119
total ceiling unchanged. The pre-run AWS check showed **$112.14 estimated
credits remaining and $7.86 estimated used**. EC2 eligibility and the live
Linux On-Demand rate of **$5.84531/hour** were verified. These estimated credit
figures precede this development run and are not a current invoice balance.

| Session | Conservative compute allowance used | Estimated compute |
| --- | ---: | ---: |
| Expired-timer boot, 13:14:28–13:14:47 UTC | One-minute billing minimum | $0.097 |
| Development and archive grace, 13:20:18–13:41:11 UTC | 20m 53s | $2.034 |
| Archive retrieval, 15:10:12–15:20:14 UTC | 10m 2s | $0.977 |

Estimated gross session usage is **about $3.30 before credits** through
19:17:52 UTC: $3.108 compute, $0.156 EBS since development startup and a
conservative $0.030 public-IPv4 bound. Tax, transfer and subsequent storage
are excluded. This is an estimate, not an AWS invoice. Stopped hours are not
counted as compute; EBS storage during those hours is included.

The retained old timer fired during the first boot. After a clean guest
shutdown, force-stop completed the EC2 transition. An immediate restart hit
a temporary vCPU-quota rejection; a later retry succeeded. The corrected
timer was then installed with a 13:59:27 UTC deadline. The successful wrapper
saved the archive at 13:38 and scheduled shutdown three minutes later; the
journal confirms powerdown at **13:41:11 UTC**.

A network-address change interrupted monitoring. The instance remained
stopped while disconnected. It was restarted only to retrieve the archive,
with SSH restricted to the new operator address and a fresh ten-minute guest
and external stop. The CloudShell log confirms a successful EC2 stop request
at **15:20:12 UTC**, and the console confirms clean powerdown. The original
45-minute CloudShell watchdog had no recorded output; its execution is not
asserted. Final EC2 state was verified separately.

The encrypted 200 GiB gp3 volume is retained and costs approximately
**$0.62/day** ($18.56 per 30 days). No volume was deleted. Each future session
still needs a freshly installed and verified deadline.

## Main-study planning

The saved [planning worksheet](../output/aws-experiment/2026-09-11-development/main-study-planning.json)
retains the four conditions and the prespecified uncertainty policy. A
proposed fixed cohort of **250 initial HotpotQA questions**, in five disjoint
50-question batches, would produce about 120 failed questions if the observed
48% failure rate persisted. Using the development paired standard deviations,
normal approximations give roughly ±3.0 and ±2.5 percentage-point precision
for uncertainty versus restart and random respectively. These estimates are
uncertain because only 24 failed development questions informed them.

Scaling the measured stages gives about 79 minutes and $7.65 compute before
additional setup, retrieval and noncompute costs. The worksheet proposes a
$5 cap for each of five batches ($25 maximum). This is a planning envelope;
no main-study run has been launched or additional spend incurred.

The [known exclusion union](../output/aws-experiment/2026-09-11-development/known-explored-ids-after-development.json)
contains all **60 AWS questions**. The older Google Drive run IDs have not been
recovered; committed and current local notebook/CSV artifacts did not recover
them. Therefore a final disjoint held-out ID freeze is still pending. Do not
claim historical disjointness from the 60-ID union alone. Position-matched
random and no-backtrack controls remain gaps for a stronger mechanistic
claim, and throughput on the other datasets has not been calibrated.

## Saved evidence

- [Verified archive](../output/aws-experiment/2026-09-11-development/attempt-20260911T133800Z.tar.gz), SHA256 `e4eff35d51b8fc9ce6a1421a1ef9fee6d57868853a1e1467ac172abf898f5590`.
- [Executed notebook](../output/aws-experiment/2026-09-11-development/results/session/executed.ipynb) and [frozen parameters](../output/aws-experiment/2026-09-11-development/parameters.json).
- [Audited results and paired analysis](../output/aws-experiment/2026-09-11-development/development-summary.json) and [audit script](../output/aws-experiment/2026-09-11-development/audit_development.py).
- [Cohort manifest](../output/aws-experiment/2026-09-11-development/development-manifest.json), [cost estimate](../output/aws-experiment/2026-09-11-development/cost-estimate.json), [shutdown journal](../output/aws-experiment/2026-09-11-development/development-shutdown.txt) and [final AWS state](../output/aws-experiment/2026-09-11-development/final-state.json).
