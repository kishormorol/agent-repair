"""Evaluation & statistics (Stage 6)."""
from .metrics import (
    bootstrap_ci, summarize_strategies, per_qid_success, mcnemar,
    holm_correction, rq_comparisons, pareto_points, ensemble_rows,
)

__all__ = [
    "bootstrap_ci", "summarize_strategies", "per_qid_success", "mcnemar",
    "holm_correction", "rq_comparisons", "pareto_points", "ensemble_rows",
]
