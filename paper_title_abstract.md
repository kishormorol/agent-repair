# ICLR 2027 Title and Abstract

**Status: all six replication cells and the runtime experiment are audited and reproduced; not submitted.**
The manuscript reports the six-condition study, separate diagnosis/replay
follow-up at three allowances, two-model replication on three QA datasets,
and a runtime experiment on 25 failures from the replication cohort. Older
summaries remain exploratory evidence. Conclusions concern the evaluated models, offline environment,
failure gate and recovery policies.

## Title

A Length-Matched Control Erases the Apparent Benefit of Perplexity-Guided Trajectory Repair

## Abstract
Recovery from a failed agent trajectory can restart the task or preserve a prefix and
regenerate its suffix. We test whether stored-token perplexity helps select a useful
recovery origin beyond positional controls in ReAct-style QA agents. A prespecified
250-question HotpotQA study with Qwen2.5-32B-Instruct-AWQ compares six policies under
matched retry hints, generated-token caps and new-step allowances. Two-step backtracking
from the highest-perplexity step achieves 9.73% repair success versus 11.21% for
restart: -1.47 percentage points (question-level 95% bootstrap interval, -4.13 to
+0.88). None of four primary comparisons shows a clear advantage. A separate
250-question diagnosis/replay follow-up finds none across three token allowances and six
primary comparisons. A frozen replication evaluates Qwen32B and Mistral-Nemo on
HotpotQA, MuSiQue and 2WikiMultiHopQA: 600 main model/question evaluations, seven
policies and three repair seeds. All 24 primary comparisons remain inconclusive. A
further frozen study pairs failed questions by identical original length and swaps each
pair's chosen origins, holding the two arms' origin distributions exactly equal by
construction rather than by fitting; that comparison is also inconclusive. Within that
extension, 25 failed Qwen HotpotQA questions receive 225 additional timed attempts;
success by ten seconds is 16.0% for uncertainty, 17.3% for restart and 12.0% for
diagnosis, with neither primary contrast significant. Each experiment uses its own
prespecified Holm family. All new experiments pass independent replay, scoring and
allowance audits. Exploratory main-study diagnostics find 29.5% more prompt tokens than
restart despite similar generated-token use. Historical exclusions rely on reconstructed
pools. These results limit claims for this uncertainty policy in the evaluated offline
setting; they establish neither equivalence nor an advantage under equal total compute.

## Before Submitting

This abstract matches the current [ICLR manuscript](paper/iclr2027.tex)
and its [main study report](docs/aws_main_2026-09-12.md) and
[diagnosis follow-up report](docs/aws_diagnosis_2026-09-14.md). The six-condition
experiment and the separate three-policy, three-allowance follow-up each
cover 250 HotpotQA questions with one pinned Qwen32B model. Both use three
repair seeds. Neither found a clear uncertainty-repair advantage;
a nonsignificant result is not evidence of equivalence. Historical exclusions
are reconstructed and the full original pool comparison remains unavailable.

The [completed frozen extension](docs/aws_extension_2026-09-14.md) has
600 main evaluations, 4,323 unique repairs and 8,016 independently reproduced
trial rows across all six cells. None of the 24 primary comparisons shows a
clear advantage. These totals include 225 additional timed attempts on 25
failed replication questions; the runtime component adds no new initial
evaluations. It finds no clear advantage in the two primary ten-second comparisons.
Human reference validation is
needed for localization and mechanism claims. Follow
the [remaining scientific work](docs/experiments_to_run.md) when developing
stronger claims. The [September 13 expert review](docs/iclr_expert_review_2026-09-13.md)
records the implemented execution, cost and scoring diagnostics. A local
[statistical supplement](output/reviewer-2026-09-12/agent-repair-statistical-supplement.zip)
reproduces the main HotpotQA study's primary and secondary statistics; it
does not reproduce GPU generations. The paper now records each study's
verification and packaging status. A combined anonymous supplement, full
GPU replay environment and final author review remain separate work.
