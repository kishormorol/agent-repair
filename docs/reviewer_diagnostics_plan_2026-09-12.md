# Review-stage diagnostic analysis plan

Recorded September 12, 2026 (America/New_York), after the main results were
known. These are exploratory analyses requested during manuscript review,
not new prespecified primary tests. The original six conditions, cohort,
four-comparison Holm family, and primary estimates remain fixed.

Use all five audited batches from study
`e4ea08c8291b0f797d05b15110d762634ec48d57ae53e05aaf4373c805986308`.
Load complete question/condition/seed coverage through the existing study
validator. Check diagnostic inputs against the archived audit hashes and
check shared executions for identical outcomes and recorded costs.

1. **Distinct interventions.** For each primary control, count identical
   execution IDs, different origins, questions with any different execution,
   and question-level positive/negative/zero outcome differences. Decompose
   the full effect into identical and different-execution contributions.
   For treatment versus restart, additionally report the effect restricted
   to questions with nonzero treatment origins. This subgroup is defined by
   the original trajectory and fixed policy, but was requested after the
   outcomes and remains exploratory. Do not count seed rows as questions.
2. **Allowance and termination.** Report initial-failure length and token
   distributions, each condition's budget/step/finished termination counts,
   token-cap utilization, and whether a final answer was emitted. Stopping
   at a cap is observed; improvement under a larger cap is unobserved.
3. **Policy accounting.** Report mean recovery generated tokens, prompt
   tokens, model requests and tool calls per attempted repair, averaging
   seeds within questions. Charge a shared execution to each separately
   evaluated policy that uses it, while deduplicating it for total study
   usage. Include paired question intervals for differences in these costs.
   Keep prompt and output counts separate; do not label their sum
   FLOPs or infer wall-clock speedups from batched logs.
4. **Evaluation sensitivity.** Surface the already computed strict-EM and
   continuous-F1 paired intervals on the same 113 initial failures. Include
   treatment versus fixed early as an explicitly descriptive contrast, outside
   the original four-comparison family. Also
   reconstruct outcomes over all 250 questions under the actual
   reference-gated policy: retain initially accepted answers and repair only
   the 113 rejected ones. These are not a deployed failure detector or a
   rerun under an EM-only failure gate. Report initially accepted answers
   that failed strict EM.
5. **Signal and environment checks.** Count invalid/missing perplexity
   profiles, fallback origins and original trajectory lengths. State the
   actual offline distractor collection and title-matching search behavior.
6. **Illustrative cases.** Select at most one question from each treatment
   better/restart better/equal-and-unsuccessful stratum, ordered by question
   ID, and include all three seeds. These outcome-selected illustrations
   do not estimate failure-type prevalence or validate causal diagnoses.
7. **Prospective precision.** Use the observed paired-question variance to
   show approximate independent-question requirements for 2 and 5 percentage
   point 95% interval half-widths. This is a sensitivity calculation for a
   future design, not observed power, equivalence testing, or a stopping rule
   for the completed study. Freeze any future allowance and comparison
   before generating its new outcomes.

Save source hashes and the analysis-code hash with the outputs. Test
complete and incomplete paired coverage, shared-execution consistency,
policy-versus-study accounting, subgroup weighting, and the reference gate.
No new model inference is part of this diagnostic analysis.
