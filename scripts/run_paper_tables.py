"""Generate publication-ready tables and figures from multi-model x multi-dataset results.

Usage:
    python scripts/run_paper_tables.py --config config/config_experiment.yaml

Produces LaTeX tables and matplotlib figures suitable for an AAAI submission:
  - Table 1: Main results (repair success by strategy, aggregated across datasets)
  - Table 2: Model size effect (Small vs Medium vs Large)
  - Table 3: Per-dataset breakdown
  - Table 4: Localization accuracy by metric
  - Figure 1: Headline bar chart (baselines vs best uncertainty vs ensemble)
  - Figure 2: Metric x Rule heatmap
  - Figure 3: Model size scaling curve
  - Figure 4: Per-dataset Pareto (cost vs success)
  - Figure 5: Failure mode analysis
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.utils import load_config, get_logger, read_jsonl, load_json


# =========================================================================== #
# Data collection
# =========================================================================== #
def collect_all_results(base: str, datasets: list, models_cat: dict,
                        tiers: list) -> pd.DataFrame:
    """Collect repair results from all (dataset, model) experiment directories."""
    rows = []
    for ds in datasets:
        for tier in tiers:
            if tier not in models_cat:
                continue
            for model_key in models_cat[tier]:
                results_path = os.path.join(
                    base, "outputs", ds, model_key, "repairs", "results.jsonl")
                if not os.path.exists(results_path):
                    continue
                for rec in read_jsonl(results_path):
                    rec["dataset"] = ds
                    rec["model_key"] = model_key
                    rec["tier"] = tier
                    rec["family"] = models_cat[tier][model_key].get("family", "unknown")
                    rows.append(rec)
    return pd.DataFrame(rows)


def collect_localization(base: str, datasets: list, models_cat: dict,
                         tiers: list) -> pd.DataFrame:
    """Collect localization summary CSVs."""
    frames = []
    for ds in datasets:
        for tier in tiers:
            if tier not in models_cat:
                continue
            for model_key in models_cat[tier]:
                loc_path = os.path.join(
                    base, "outputs", ds, model_key, "tables",
                    "localization_summary.csv")
                if not os.path.exists(loc_path):
                    continue
                df = pd.read_csv(loc_path)
                df["dataset"] = ds
                df["model_key"] = model_key
                df["tier"] = tier
                frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def collect_generation_stats(base: str, datasets: list, models_cat: dict,
                             tiers: list) -> pd.DataFrame:
    """Collect initial generation success rates."""
    rows = []
    for ds in datasets:
        for tier in tiers:
            if tier not in models_cat:
                continue
            for model_key in models_cat[tier]:
                traj_dir = os.path.join(base, "outputs", ds, model_key, "trajectories")
                if not os.path.exists(traj_dir):
                    continue
                n_total = 0
                n_success = 0
                for f in os.listdir(traj_dir):
                    if not f.endswith(".json"):
                        continue
                    try:
                        t = load_json(os.path.join(traj_dir, f))
                        n_total += 1
                        if t.get("success", False):
                            n_success += 1
                    except Exception:
                        pass
                if n_total > 0:
                    rows.append({
                        "dataset": ds, "model_key": model_key, "tier": tier,
                        "family": models_cat[tier][model_key].get("family", "unknown"),
                        "n_total": n_total, "n_success": n_success,
                        "success_rate": n_success / n_total,
                        "n_failed": n_total - n_success,
                    })
    return pd.DataFrame(rows)


# =========================================================================== #
# LaTeX table helpers
# =========================================================================== #
def df_to_latex(df: pd.DataFrame, caption: str, label: str,
                fmt: dict = None) -> str:
    """Convert DataFrame to a LaTeX table string."""
    fmt = fmt or {}
    latex = df.to_latex(index=False, escape=False, caption=caption, label=label,
                        column_format="l" + "c" * (len(df.columns) - 1),
                        formatters=fmt)
    return latex


# =========================================================================== #
# Paper tables
# =========================================================================== #
def table_1_main_results(df: pd.DataFrame) -> pd.DataFrame:
    """Table 1: Repair success rate by strategy, averaged across models & datasets."""
    if df.empty:
        return pd.DataFrame()
    # Filter to multiplier=1.0 if present
    if "multiplier" in df.columns:
        df = df[df.multiplier == 1.0]

    # Majority vote across seeds per (qid, strategy, model, dataset)
    agg = (df.groupby(["dataset", "model_key", "qid", "strategy"])["success"]
           .mean().reset_index())
    agg["success_binary"] = (agg["success"] > 0.5).astype(int)

    summary = (agg.groupby("strategy")
               .agg(fixed_pct=("success_binary", "mean"),
                    n=("success_binary", "count"))
               .reset_index())
    summary["fixed_pct"] = (100 * summary["fixed_pct"]).round(1)
    return summary.sort_values("fixed_pct", ascending=False)


def table_2_model_size(df: pd.DataFrame, gen_df: pd.DataFrame) -> pd.DataFrame:
    """Table 2: Effect of model size tier on initial success and repair."""
    if df.empty:
        return pd.DataFrame()
    if "multiplier" in df.columns:
        df = df[df.multiplier == 1.0]

    # Initial success by tier
    init = gen_df.groupby("tier").agg(
        initial_success=("success_rate", "mean")).reset_index()

    # Best repair success by tier
    best_repair = (df.groupby(["tier", "strategy"])["success"]
                   .mean().reset_index()
                   .groupby("tier")["success"].max().reset_index()
                   .rename(columns={"success": "best_repair"}))

    result = init.merge(best_repair, on="tier", how="outer")
    result["initial_success"] = (100 * result["initial_success"]).round(1)
    result["best_repair"] = (100 * result["best_repair"]).round(1)

    tier_order = {"small": 0, "medium": 1, "large": 2}
    result["_order"] = result["tier"].map(tier_order)
    return result.sort_values("_order").drop(columns=["_order"])


def table_3_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Table 3: Per-dataset breakdown of key strategies."""
    if df.empty:
        return pd.DataFrame()
    if "multiplier" in df.columns:
        df = df[df.multiplier == 1.0]

    key_strategies = ["full_restart", "oracle_targeted", "random_step"]
    # Also find best uncertainty strategy per dataset
    unc = df[df.strategy.str.startswith("unc__")]

    rows = []
    for ds in df.dataset.unique():
        ds_df = df[df.dataset == ds]
        row = {"dataset": ds}
        for s in key_strategies:
            s_df = ds_df[ds_df.strategy == s]
            if not s_df.empty:
                row[s] = f"{100 * s_df['success'].mean():.1f}"
        # Best unc for this dataset
        ds_unc = unc[unc.dataset == ds]
        if not ds_unc.empty:
            best = ds_unc.groupby("strategy")["success"].mean().idxmax()
            best_val = ds_unc.groupby("strategy")["success"].mean().max()
            row["best_unc"] = f"{100 * best_val:.1f}"
            row["best_unc_name"] = best.replace("unc__", "").replace("__", "/")
        rows.append(row)

    return pd.DataFrame(rows)


def table_4_localization(loc_df: pd.DataFrame) -> pd.DataFrame:
    """Table 4: Localization accuracy by uncertainty metric."""
    if loc_df.empty:
        return pd.DataFrame()
    # Average across models and datasets
    summary = (loc_df.groupby("metric")
               .agg(top1=("argmax_top1", "mean"),
                    within1=("argmax_within1", "mean"),
                    mrr=("mrr", "mean"))
               .reset_index())
    summary["top1"] = (100 * summary["top1"]).round(1)
    summary["within1"] = (100 * summary["within1"]).round(1)
    summary["mrr"] = summary["mrr"].round(3)
    return summary.sort_values("top1", ascending=False)


# =========================================================================== #
# Paper figures
# =========================================================================== #
def fig_1_headline(df: pd.DataFrame, path: str):
    """Bar chart: baselines vs best uncertainty vs ensemble."""
    if df.empty:
        return
    if "multiplier" in df.columns:
        df = df[df.multiplier == 1.0]

    strat_success = df.groupby("strategy")["success"].mean()
    unc_strats = [s for s in strat_success.index if s.startswith("unc__")]
    best_unc = strat_success[unc_strats].idxmax() if unc_strats else None

    order = ["random_step", "full_restart"]
    if best_unc:
        order.append(best_unc)
    order.append("oracle_targeted")

    labels = {"random_step": "Random", "full_restart": "Full Restart",
              "oracle_targeted": "Oracle"}
    colors = {"random_step": "#999999", "full_restart": "#0072B2",
              "oracle_targeted": "#009E73"}

    order = [s for s in order if s in strat_success.index]
    vals = [strat_success[s] for s in order]
    lbls = [labels.get(s, s.replace("unc__", "").replace("__", "/")) for s in order]
    clrs = [colors.get(s, "#E69F00") for s in order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(order))
    ax.bar(x, vals, color=clrs, edgecolor="white", linewidth=1.5)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.01, f"{100*v:.1f}%", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, rotation=15, ha="right")
    ax.set_ylabel("Repair success rate")
    ax.set_title("Repair strategies (averaged across all models and datasets)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_3_model_scaling(gen_df: pd.DataFrame, repair_df: pd.DataFrame, path: str):
    """Line plot: success rate vs model size tier."""
    if gen_df.empty:
        return

    tier_order = {"small": 0, "medium": 1, "large": 2}
    tier_labels = {"small": "Small\n(7-9B)", "medium": "Medium\n(12-27B)",
                   "large": "Large\n(70-72B)"}

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Initial success by tier
    init = gen_df.groupby("tier")["success_rate"].mean()
    tiers = sorted(init.index, key=lambda t: tier_order.get(t, 99))
    x = [tier_order[t] for t in tiers]
    ax.plot(x, [init[t] for t in tiers], "o-", color="#0072B2",
            label="Initial success", linewidth=2, markersize=8)

    # Best repair success by tier (if available)
    if not repair_df.empty:
        if "multiplier" in repair_df.columns:
            repair_df = repair_df[repair_df.multiplier == 1.0]
        best = (repair_df.groupby(["tier", "strategy"])["success"]
                .mean().reset_index()
                .groupby("tier")["success"].max())
        ax.plot(x, [best.get(t, 0) for t in tiers], "s--", color="#E69F00",
                label="Best repair", linewidth=2, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([tier_labels.get(t, t) for t in tiers])
    ax.set_ylabel("Success rate")
    ax.set_title("Effect of model size on agent performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_4_per_dataset_pareto(df: pd.DataFrame, path: str):
    """Per-dataset Pareto plot: cost vs success."""
    if df.empty:
        return
    if "multiplier" in df.columns:
        df = df[df.multiplier == 1.0]

    datasets = sorted(df.dataset.unique())
    n_ds = len(datasets)
    if n_ds == 0:
        return

    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4.5), squeeze=False)
    colors = {"random_step": "#999999", "full_restart": "#0072B2",
              "oracle_targeted": "#009E73"}

    for idx, ds in enumerate(datasets):
        ax = axes[0][idx]
        ds_df = df[df.dataset == ds]
        strat_agg = ds_df.groupby("strategy").agg(
            success=("success", "mean"),
            cost=("recovery_gen_tokens", "mean")).reset_index()

        # Uncertainty strategies as grey dots
        unc = strat_agg[strat_agg.strategy.str.startswith("unc__")]
        ax.scatter(unc.cost, unc.success, s=26, color="#b0b8c0", zorder=1)

        # Baselines as colored dots
        for s, c in colors.items():
            row = strat_agg[strat_agg.strategy == s]
            if not row.empty:
                ax.scatter(row.cost.values[0], row.success.values[0],
                           s=90, color=c, zorder=3, edgecolor="white")
                label = {"random_step": "Rand", "full_restart": "Restart",
                          "oracle_targeted": "Oracle"}[s]
                ax.annotate(label, (row.cost.values[0], row.success.values[0]),
                            textcoords="offset points", xytext=(7, 5),
                            fontsize=8, color=c)

        ax.set_xlabel("Recovery tokens")
        ax.set_ylabel("Success rate")
        ax.set_title(ds)

    fig.suptitle("Cost vs success across datasets", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# =========================================================================== #
# Main
# =========================================================================== #
def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate paper tables and figures")
    p.add_argument("--config", default="config/config_experiment.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    log = get_logger("paper", cfg.path("logs"))

    exp = cfg.raw.get("experiment", {})
    models_file = exp.get("models_file", "config/models.yaml")
    datasets_file = exp.get("datasets_file", "config/datasets.yaml")

    with open(models_file) as f:
        models_cat = yaml.safe_load(f)
    with open(datasets_file) as f:
        datasets_cat = yaml.safe_load(f)

    tiers = exp.get("tiers", ["small", "medium", "large"])
    datasets = exp.get("datasets", list(datasets_cat.keys()))

    # Output directory for paper artifacts
    paper_dir = os.path.join(cfg.base, "outputs", "paper")
    os.makedirs(os.path.join(paper_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(paper_dir, "figures"), exist_ok=True)

    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "font.size": 11})

    # Collect data
    log.info("Collecting results across all experiments...")
    repair_df = collect_all_results(cfg.base, datasets, models_cat, tiers)
    loc_df = collect_localization(cfg.base, datasets, models_cat, tiers)
    gen_df = collect_generation_stats(cfg.base, datasets, models_cat, tiers)

    log.info(f"Repair results: {len(repair_df)} rows")
    log.info(f"Localization results: {len(loc_df)} rows")
    log.info(f"Generation stats: {len(gen_df)} models x datasets")

    if repair_df.empty:
        log.warning("No repair results found. Run experiments first.")
        return

    # Generate tables
    log.info("Generating tables...")

    t1 = table_1_main_results(repair_df)
    t1.to_csv(os.path.join(paper_dir, "tables", "table1_main_results.csv"), index=False)
    log.info(f"Table 1: {len(t1)} strategies")

    t2 = table_2_model_size(repair_df, gen_df)
    t2.to_csv(os.path.join(paper_dir, "tables", "table2_model_size.csv"), index=False)
    log.info(f"Table 2: {len(t2)} tiers")

    t3 = table_3_per_dataset(repair_df)
    t3.to_csv(os.path.join(paper_dir, "tables", "table3_per_dataset.csv"), index=False)
    log.info(f"Table 3: {len(t3)} datasets")

    if not loc_df.empty:
        t4 = table_4_localization(loc_df)
        t4.to_csv(os.path.join(paper_dir, "tables", "table4_localization.csv"), index=False)
        log.info(f"Table 4: {len(t4)} metrics")

    # Generate figures
    log.info("Generating figures...")

    fig_1_headline(repair_df,
                   os.path.join(paper_dir, "figures", "fig1_headline.png"))

    fig_3_model_scaling(gen_df, repair_df,
                        os.path.join(paper_dir, "figures", "fig3_model_scaling.png"))

    fig_4_per_dataset_pareto(repair_df,
                             os.path.join(paper_dir, "figures", "fig4_pareto.png"))

    # Save generation stats
    gen_df.to_csv(os.path.join(paper_dir, "tables", "generation_stats.csv"), index=False)

    # LaTeX snippets
    latex_dir = os.path.join(paper_dir, "latex")
    os.makedirs(latex_dir, exist_ok=True)

    if not t1.empty:
        with open(os.path.join(latex_dir, "table1.tex"), "w") as f:
            f.write(df_to_latex(t1, "Main repair results across all models and datasets.",
                                "tab:main"))

    # Summary
    print("\n" + "=" * 70)
    print("PAPER ARTIFACTS GENERATED")
    print("=" * 70)
    print(f"Tables:  {os.path.join(paper_dir, 'tables')}")
    print(f"Figures: {os.path.join(paper_dir, 'figures')}")
    print(f"LaTeX:   {latex_dir}")
    print()

    # Print quick summary
    if not gen_df.empty:
        print("Initial success rates by model:")
        for _, row in gen_df.iterrows():
            print(f"  {row['model_key']:20s} x {row['dataset']:15s}: "
                  f"{100*row['success_rate']:.1f}% ({row['n_success']}/{row['n_total']})")


if __name__ == "__main__":
    main()
