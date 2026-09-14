# Historical Summary Status

These files are retained as historical artifacts, not regenerated results.
See `docs/iclr2027_readiness.md` for the September 7, 2026 audit.

- `cross_dataset_main_results.csv` mixes majority-over-seeds baseline
  rates with mean uncertainty rates. Do not use it for a direct comparison.
  The corrected source notebook must be rerun after recovering raw logs.
- `cross_dataset_traj_stats.csv` uses mean rates consistently, but its
  `avg_tool_calls` column describes **full-restart repairs**, not original
  failed-trajectory lengths. Its `best_unc` is selected on the evaluated
  questions and includes informed hints in the search space.
- `cross_dataset_localization.csv` reports agreement with a model judge;
  human-validation results are absent from this repository snapshot.
- `cross_dataset_backtrack_gains.csv` contains descriptive historical rates.

The expanded manuscript additionally uses the final saved cascade-results
displays in the three dataset notebooks. The builder records exact cell
indices and hashes in `paper/generated/iclr2027_provenance.json` and extracts
visible rows into `paper/generated/iclr2027_archived_summaries.json`.
Those displays recover random-origin rates, baseline recovery costs, and
fixed-origin nudge comparisons. They do not expose the full strategy grid
or raw outcomes, and their historical confidence intervals are discarded.

FEVER evidence construction is invalid in the inspected loader. Historical
generation budgets may be exceeded. None of these summaries is a substitute
for raw-trial validation, corrected-budget reruns, or new confidence intervals.
