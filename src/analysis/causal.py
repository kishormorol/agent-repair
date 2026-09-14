"""Descriptive reference-label agreement, not causal attribution or repair utility.

Exploratory action type does not establish correctness. The fitted models
predict annotation agreement and cannot validate a repair-selection policy.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def add_failure_mode_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Given a per-trajectory feature frame with columns
    [oracle_step, pred_argmax, pred_step_action, argmax_top1], add flags.
    """
    df = df.copy()
    df["error_upstream_of_peak"] = (df["oracle_step"] < df["pred_argmax"]).astype(int)
    df["error_downstream_of_peak"] = (df["oracle_step"] > df["pred_argmax"]).astype(int)
    explore = df.get("pred_step_action", pd.Series("", index=df.index)).isin(["search", "lookup"])
    df["exploratory_action_at_missed_peak"] = ((df["argmax_top1"] == 0) & explore).astype(int)
    return df


def summarize_failure_modes(df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate rates of localization success and the two failure modes."""
    miss = df[df["argmax_top1"] == 0]
    n_miss = max(1, len(miss))
    return {
        "n": int(len(df)),
        "localization_top1": round(float(df["argmax_top1"].mean()), 4),
        "error_upstream_of_peak_rate": round(float(df["error_upstream_of_peak"].mean()), 4),
        "among_misses_upstream": round(float(miss["error_upstream_of_peak"].sum() / n_miss), 4),
        "n_misses": len(miss),
        "among_misses_explore": round(float(miss["exploratory_action_at_missed_peak"].sum() / n_miss), 4),
        "exploration_interpretation": "Action type only; correctness is not validated.",
    }


def fit_localization_model(df: pd.DataFrame, feature_cols: List[str],
                           label_col: str = "argmax_top1",
                           seed: int = 0) -> Dict[str, Any]:
    """Fit logistic regression + shallow decision tree predicting reference
    agreement. Returns cross-validated accuracy, LR coefficients, and tree
    feature importances (the interpretable 'rule')."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    data = df.dropna(subset=feature_cols + [label_col])
    X = data[feature_cols].astype(float).values
    y = data[label_col].astype(int).values
    out: Dict[str, Any] = {"n": int(len(data)), "features": feature_cols}
    if len(np.unique(y)) < 2 or len(data) < 20:
        out["note"] = "insufficient/degenerate data for modeling"
        return out

    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    out["logreg_cv_acc"] = round(float(cross_val_score(logreg, X, y, cv=5).mean()), 4)
    logreg.fit(X, y)
    coefs = logreg.named_steps["logisticregression"].coef_[0]
    out["logreg_coef"] = {f: round(float(c), 4) for f, c in zip(feature_cols, coefs)}

    tree = DecisionTreeClassifier(max_depth=3, random_state=seed)
    out["tree_cv_acc"] = round(float(cross_val_score(tree, X, y, cv=5).mean()), 4)
    tree.fit(X, y)
    out["tree_importances"] = {f: round(float(i), 4)
                               for f, i in zip(feature_cols, tree.feature_importances_)}
    out["tree_rules"] = export_text(tree, feature_names=list(feature_cols))
    return out
