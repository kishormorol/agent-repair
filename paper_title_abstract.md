# ICLR 2027 Title and Abstract

**Status: working draft with two audited HotpotQA studies; not submitted.**
The manuscript reports the completed six-condition study and separate
diagnosis/replay follow-up at three allowances. Older three-dataset summaries
remain exploratory evidence. Its conclusions concern the evaluated model,
environment, failure gate and recovery policies.

## Title

Does Stored-Token Uncertainty Improve Trajectory Repair? A Controlled HotpotQA Study

## Abstract

When a language model agent fails, recovery can preserve a trajectory prefix
and regenerate its suffix, or restart from the original task. We test whether
stored-token uncertainty helps select a recovery origin beyond simple
positional choices in a ReAct-style agent. A controlled experiment with
Qwen2.5-32B-Instruct-AWQ freezes 250 HotpotQA questions before
inference. Its 113 initial failures receive six repair conditions
and three seeds under matched retry hints, generated-token caps and new-step
allowances. The prespecified perplexity/argmax policy with two-step
backtracking achieves 9.73% repair success, versus
11.21% for full restart: a difference of -1.47
percentage points, with a question-level 95% bootstrap interval from
-4.13 to +0.88. Its success rate equals the
position-matched random control; none of the four prespecified comparisons
shows a clear advantage after Holm correction. Exploratory trace diagnostics
show that the treatment differs from restart on only 43
failed questions and uses 29.5% more prompt tokens, with
similar generated-token use. All 1,058 unique repair executions
pass independent replay, scoring and allowance audits. Historical exclusion
IDs are reconstructed, with the original full pool comparison unavailable.
A separate 250-question cohort compares restart, uncertainty, and same-model
diagnosis/replay at three allowances, with 2,079 additional unique repairs.
Uncertainty again shows no clear advantage in six Holm-adjusted comparisons.
These results do not establish equivalence or an advantage under equal total compute.

## Before Submitting

This abstract matches the current [ICLR manuscript](paper/iclr2027.tex)
and its [main study report](docs/aws_main_2026-09-12.md) and
[diagnosis follow-up report](docs/aws_diagnosis_2026-09-14.md). The six-condition
experiment and the separate three-policy, three-allowance follow-up each
cover 250 HotpotQA questions with one pinned Qwen32B model. Both use three
repair seeds. Neither found a clear uncertainty-repair advantage;
a nonsignificant result is not evidence of equivalence. Historical exclusions
are reconstructed and the full original pool comparison remains unavailable.

Replication on the other two QA datasets, a second model and a measured
runtime comparison are being evaluated in a separate frozen extension.
Its pending outcomes are not evidence for the current abstract. Human reference validation is
needed for localization and mechanism claims. Follow
the [remaining scientific work](docs/experiments_to_run.md) when developing
stronger claims. The [September 13 expert review](docs/iclr_expert_review_2026-09-13.md)
records the implemented execution, cost and scoring diagnostics. A local
[statistical supplement](output/reviewer-2026-09-12/agent-repair-statistical-supplement.zip)
reproduces the primary and secondary statistics; it does not reproduce GPU
generations. Full runtime/raw-data packaging and final author review remain
separate work.
