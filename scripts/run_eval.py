"""Stage 6 — evaluation, statistics, causal analysis, figures (CPU).

    python scripts/run_eval.py --config config/config_local.yaml
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

from _common import parse_args, boot  # type: ignore

from src.eval import (summarize_strategies, rq_comparisons, ensemble_rows)
from src.localize import get_step_scores, rule_argmax
from src.repair import parse_strategy
from src.analysis import (add_failure_mode_flags, summarize_failure_modes,
                          fit_localization_model)
from src.utils import load_item, load_json, save_json, read_jsonl

BASE_LABELS = {"random_step": "Random", "full_restart": "Full Restart",
               "oracle_targeted": "Oracle",
               "uncertainty_ensemble_any": "Ensemble (any)"}
BASE_COLORS = {"random_step": "#999999", "full_restart": "#0072B2",
               "oracle_targeted": "#009E73", "uncertainty_ensemble_any": "#E69F00"}
METRIC_LABELS = {"token_entropy_max": "Entropy", "perplexity": "Perplexity",
                 "max_token_prob_max": "MaxProb", "self_consistency": "SelfConsist",
                 "verbalized_confidence": "Verbalized"}
RULE_LABELS = {"argmax": "argmax", "topk": "top-k",
               "earliest_above_threshold": "earliest>thr",
               "cascade_upstream": "cascade-up",
               "cascade_gradient": "cascade-grad",
               "cascade_weighted": "cascade-wt"}


def main() -> None:
    args = parse_args("Evaluation, statistics, figures")
    cfg, log = boot("stage6", args)
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "font.size": 11})
    FIG = cfg.path("figures")
    TAB = cfg.path("tables")

    # ---- load + ensemble --------------------------------------------------- #
    df = pd.DataFrame(list(read_jsonl(os.path.join(cfg.path("repairs"), "results.jsonl"))))
    if df.empty:
        log.error("No repair results found. Run stage 5 first.")
        return
    if "multiplier" not in df.columns:
        df["multiplier"] = 1.0
    if "backtrack" not in df.columns:
        df["backtrack"] = df["strategy"].apply(lambda s: parse_strategy(s)["backtrack"])
    df = df[df.multiplier == 1.0].copy()
    df = df.drop_duplicates(subset=["qid", "strategy", "seed"])   # safety
    unc_strats = [s for s in df.strategy.unique() if s.startswith("unc__")]
    # Ensemble uses bt0 uncertainty strategies only
    bt0_unc = [s for s in unc_strats if parse_strategy(s)["backtrack"] == 0]
    df = pd.concat([df, ensemble_rows(df, bt0_unc, "uncertainty_ensemble_any")],
                   ignore_index=True)
    log.info(f"rows: {len(df)} | uncertainty strategies: {len(unc_strats)} "
             f"(bt0: {len(bt0_unc)}) | + ensemble")

    # ---- main table -------------------------------------------------------- #
    summary = summarize_strategies(df, iters=cfg.raw["evaluation"]["bootstrap_iters"],
                                   ci=cfg.raw["evaluation"]["ci"])
    summary.to_csv(os.path.join(TAB, "main_results.csv"), index=False)   # raw fractions
    si = summary.set_index("strategy")

    # human-readable version: fractions -> percentages
    readable = pd.DataFrame({
        "strategy": summary["strategy"],
        "fixed_%": (100 * summary["success"]).round(1),
        "95%_CI": [f"{100*lo:.1f}-{100*hi:.1f}%"
                   for lo, hi in zip(summary["success_lo"], summary["success_hi"])],
        "avg_tokens": summary["recovery_gen_tokens"].round(0).astype(int),
        "avg_tool_calls": summary["recovery_tool_calls"].round(2),
        "hit_true_step_%": (100 * summary["targeted_oracle_match"]).round(1),
    })
    readable.to_csv(os.path.join(TAB, "main_results_readable.csv"), index=False)
    print("\n" + "=" * 78)
    print("MAIN RESULTS  (fixed_% = share of FAILED trajectories that the repair fixed)")
    print("=" * 78)
    print(readable.to_string(index=False))

    # ---- metric x rule heatmap (bt0 only) --------------------------------- #
    u = df[df.strategy.isin(bt0_strats)] if bt0_strats else df[df.strategy.isin(unc_strats)]
    piv = u.groupby(["metric", "rule"])["success"].mean().unstack("rule")
    mo = [m for m in cfg.raw["repair"]["uncertainty_metrics"] if m in piv.index]
    ro = [r for r in cfg.raw["repair"]["uncertainty_rules"] if r in piv.columns]
    piv = piv.loc[mo, ro]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    im = ax.imshow(piv.values, cmap="YlGn", vmin=0,
                   vmax=max(0.01, float(np.nanmax(piv.values))), aspect="auto")
    ax.set_xticks(range(len(ro))); ax.set_xticklabels([RULE_LABELS.get(r, r) for r in ro])
    ax.set_yticks(range(len(mo))); ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in mo])
    for i in range(len(mo)):
        for j in range(len(ro)):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title("Repair success: uncertainty metric x rule")
    fig.colorbar(im, ax=ax, label="success rate")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "metric_rule_heatmap.png")); plt.close(fig)

    # ---- headline bars ------------------------------------------------------ #
    # best uncertainty among bt0 (no backtrack) strategies only
    bt0_strats = [s for s in unc_strats
                  if parse_strategy(s)["backtrack"] == 0]
    best_unc = (u[u.strategy.isin(bt0_strats)]
                .groupby("strategy")["success"].mean().idxmax()
                if bt0_strats else u.groupby("strategy")["success"].mean().idxmax())

    def blab(s):
        if s in BASE_LABELS:
            return BASE_LABELS[s]
        p = parse_strategy(s)
        if p["metric"] and p["rule"]:
            label = f"{METRIC_LABELS.get(p['metric'], p['metric'])}/{RULE_LABELS.get(p['rule'], p['rule'])}"
            if p["backtrack"] > 0:
                label += f" bt{p['backtrack']}"
            return label
        if p["backtrack"] > 0:
            return f"{p['base']} bt{p['backtrack']}"
        return s

    order = [s for s in ["random_step", "full_restart", best_unc,
                         "uncertainty_ensemble_any", "oracle_targeted"] if s in si.index]
    hs = si.loc[order]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(order))
    err = [hs["success"] - hs["success_lo"], hs["success_hi"] - hs["success"]]
    ax.bar(x, hs["success"], color=[BASE_COLORS.get(s, "#E69F00") for s in order],
           yerr=err, capsize=4, edgecolor="white", linewidth=1.5)
    for xi, v in zip(x, hs["success"]):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([blab(s) for s in order], rotation=15, ha="right")
    ax.set_ylabel("Repair success rate")
    ax.set_title("Baselines vs best single metric vs ensemble")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "headline_success.png")); plt.close(fig)

    # ---- Pareto ------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for s in unc_strats:
        if s in si.index:
            ax.scatter(si.loc[s, "recovery_gen_tokens"], si.loc[s, "success"],
                       s=26, color="#b0b8c0", zorder=1)
    for s in [x for x in BASE_COLORS if x in si.index]:
        ax.scatter(si.loc[s, "recovery_gen_tokens"], si.loc[s, "success"], s=90,
                   color=BASE_COLORS[s], zorder=3, edgecolor="white")
        ax.annotate(BASE_LABELS[s], (si.loc[s, "recovery_gen_tokens"], si.loc[s, "success"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=9,
                    color=BASE_COLORS[s])
    ax.set_xlabel("Recovery cost (generated tokens)")
    ax.set_ylabel("Repair success rate")
    ax.set_title("Cost vs success (grey = single-metric strategies)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "pareto_cost_success.png")); plt.close(fig)

    # ---- backtrack ablation -------------------------------------------------- #
    # Parse backtrack offset from strategy names
    if "backtrack" not in df.columns:
        df["backtrack"] = df["strategy"].apply(lambda s: parse_strategy(s)["backtrack"])

    has_bt = df["backtrack"].max() > 0
    if has_bt:
        # Oracle backtrack curve
        oracle_bt = df[df.strategy.str.startswith("oracle_targeted")]
        oracle_by_bt = oracle_bt.groupby("backtrack")["success"].mean()

        # Average across all uncertainty strategies per backtrack offset
        unc_all = df[df.strategy.str.startswith("unc__")]
        unc_by_bt = unc_all.groupby("backtrack")["success"].mean()

        # Best uncertainty strategy per backtrack offset
        def best_per_bt(bt_val):
            sub = unc_all[unc_all.backtrack == bt_val]
            if sub.empty:
                return None
            return sub.groupby("strategy")["success"].mean().max()

        bt_vals = sorted(unc_all["backtrack"].unique())
        best_unc_by_bt = pd.Series({bt: best_per_bt(bt) for bt in bt_vals})

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        if len(oracle_by_bt) > 1:
            ax.plot(oracle_by_bt.index, oracle_by_bt.values, "o-",
                    color="#009E73", linewidth=2.5, markersize=8,
                    label="Oracle + backtrack", zorder=3)
        if len(unc_by_bt) > 1:
            ax.plot(unc_by_bt.index, unc_by_bt.values, "s--",
                    color="#E69F00", linewidth=2, markersize=7,
                    label="Avg uncertainty + backtrack", zorder=2)
        if len(best_unc_by_bt.dropna()) > 1:
            ax.plot(best_unc_by_bt.index, best_unc_by_bt.values, "D-.",
                    color="#D55E00", linewidth=2, markersize=7,
                    label="Best uncertainty + backtrack", zorder=2)
        # Full restart reference line
        if "full_restart" in si.index:
            ax.axhline(si.loc["full_restart", "success"], color="#0072B2",
                       linestyle=":", linewidth=1.5, label="Full Restart")
        ax.set_xlabel("Backtrack offset (additional steps upstream)")
        ax.set_ylabel("Repair success rate")
        ax.set_title("Effect of backtracking: fixing earlier steps helps")
        ax.set_xticks(bt_vals)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "backtrack_ablation.png"))
        plt.close(fig)

        # Save backtrack summary table
        bt_summary = []
        for bt in bt_vals:
            oracle_s = df[(df.strategy == f"oracle_targeted__bt{bt}" if bt > 0
                          else df.strategy == "oracle_targeted") &
                          (df.backtrack == bt)]
            # Simpler: just use the grouped values
            bt_summary.append({
                "backtrack": bt,
                "oracle_success": float(oracle_by_bt.get(bt, float("nan"))),
                "avg_unc_success": float(unc_by_bt.get(bt, float("nan"))),
                "best_unc_success": float(best_unc_by_bt.get(bt, float("nan"))),
            })
        bt_df = pd.DataFrame(bt_summary)
        bt_df.to_csv(os.path.join(TAB, "backtrack_ablation.csv"), index=False)
        save_json(bt_summary, os.path.join(TAB, "backtrack_ablation.json"))
        log.info(f"backtrack ablation:\n{bt_df.to_string(index=False)}")

    # ---- RQ tests ----------------------------------------------------------- #
    # Filter to bt0 strategies only for the standard RQ comparisons
    df_bt0 = df[df.backtrack == 0].copy() if has_bt else df.copy()
    rq = rq_comparisons(df_bt0)
    save_json(rq, os.path.join(TAB, "rq_tests.json"))
    n_sig = sum(1 for v in rq.values() if v.get("significant_0.05"))
    log.info(f"{len(rq)} comparisons, {n_sig} significant after Holm")

    # ---- causal analysis ---------------------------------------------------- #
    head = load_json(os.path.join(TAB, "localization_headline.json"))
    mk = head["best_metric"]
    failed_ids = load_json(os.path.join(cfg.path("data_processed"), "failed_ids.json"))
    feat = []
    for q in failed_ids:
        unc = load_item(cfg.path("uncertainty"), q)
        ann = load_item(cfg.path("annotations"), q)
        scores = get_step_scores(unc, mk)
        if not scores:
            continue
        pred = rule_argmax(scores)
        desc = sorted([v for _, v in scores], reverse=True)
        margin = (desc[0] - desc[1]) if len(desc) > 1 else 0.0
        n = len(scores)
        feat.append(dict(qid=q, n_steps=n, oracle_step=ann["broken_step"],
                         pred_argmax=pred, norm_pos=pred / (n - 1) if n > 1 else 0.0,
                         uncertainty_margin=margin,
                         pred_step_action=next((s["action"] for s in unc["steps"]
                                                if s["index"] == pred), ""),
                         error_type=ann["error_type"],
                         argmax_top1=int(pred == ann["broken_step"])))
    fdf = add_failure_mode_flags(pd.DataFrame(feat))
    fm = summarize_failure_modes(fdf)
    save_json(fm, os.path.join(TAB, "failure_modes.json"))
    model = fit_localization_model(fdf, ["n_steps", "norm_pos", "uncertainty_margin"])
    save_json({k: v for k, v in model.items() if k != "tree_rules"},
              os.path.join(TAB, "causal_model.json"))
    log.info(f"failure modes: {fm}")

    # ---- failure-mode figure ------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["Upstream of peak", "Uncertain-but-correct"],
                [fm["among_misses_upstream"], fm["among_misses_explore"]],
                color=["#0072B2", "#E69F00"], edgecolor="white")
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("Fraction of localization misses")
    axes[0].set_title("Why uncertainty mislocalizes")
    et = fdf.groupby("error_type")["argmax_top1"].mean().sort_values()
    axes[1].barh(et.index, et.values, color="#009E73", edgecolor="white")
    axes[1].set_xlabel("Top-1 localization accuracy")
    axes[1].set_title("Localization by error type")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "failure_modes.png")); plt.close(fig)

    # ---- final report ------------------------------------------------------- #
    def g(s):
        return float(si.loc[s, "success"]) if s in si.index else None

    report = {
        "n_failed": len(failed_ids),
        "RQ1_oracle_vs_restart": {
            "oracle": g("oracle_targeted"), "restart": g("full_restart"),
            "delta": g("oracle_targeted") - g("full_restart"),
            "p_holm": rq["RQ1_oracle_vs_restart"]["p_value_holm"],
            "oracle_cheaper": bool(si.loc["oracle_targeted", "recovery_gen_tokens"] <
                                   si.loc["full_restart", "recovery_gen_tokens"])},
        "best_single_strategy": {"name": best_unc, "success": g(best_unc)},
        "ensemble_any": {"success": g("uncertainty_ensemble_any"),
                         "gap_vs_oracle": g("oracle_targeted") - g("uncertainty_ensemble_any"),
                         "delta_vs_random": g("uncertainty_ensemble_any") - g("random_step")},
        "best_localizing_metric": head["best_metric"],
        "failure_modes": fm,
    }

    # Backtrack results
    if has_bt:
        bt_oracle = {int(bt): float(v) for bt, v in oracle_by_bt.items()}
        bt_best = {int(bt): float(v) for bt, v in best_unc_by_bt.items() if not pd.isna(v)}
        best_bt_oracle = max(bt_oracle, key=bt_oracle.get) if bt_oracle else 0
        best_bt_unc = max(bt_best, key=bt_best.get) if bt_best else 0
        report["backtrack_ablation"] = {
            "oracle_by_backtrack": bt_oracle,
            "best_unc_by_backtrack": bt_best,
            "best_oracle_backtrack": best_bt_oracle,
            "best_oracle_backtrack_success": bt_oracle.get(best_bt_oracle),
            "best_unc_backtrack": best_bt_unc,
            "best_unc_backtrack_success": bt_best.get(best_bt_unc),
        }
    save_json(report, os.path.join(TAB, "final_report.json"))

    # ---- plain-English summary --------------------------------------------- #
    def P(x):
        return "n/a" if x is None else f"{100*x:.1f}%"

    r1 = report["RQ1_oracle_vs_restart"]
    ens = report["ensemble_any"]
    lines = [
        "", "=" * 78,
        "PLAIN-ENGLISH SUMMARY", "=" * 78,
        f"Failed trajectories we tried to repair: {report['n_failed']}", "",
        "RQ1 — Is fixing ONE step better than restarting everything?",
        f"   Oracle (fix the true broken step) fixed {P(r1['oracle'])} of them.",
        f"   Full Restart fixed                    {P(r1['restart'])}.",
        f"   Difference: {P(r1['delta'])}  (p = {r1['p_holm']:.4f}"
        f"{' — significant' if r1['p_holm'] < 0.05 else ' — NOT significant'})",
        f"   Oracle was also cheaper in tokens: {r1['oracle_cheaper']}", "",
        "RQ2 — How much do we lose by using uncertainty instead of the true step?",
        f"   Best single uncertainty strategy: {report['best_single_strategy']['name']}",
        f"   It fixed {P(report['best_single_strategy']['success'])} "
        f"(vs Oracle's {P(r1['oracle'])}).", "",
        "RQ3 — Is uncertainty better than guessing?",
        f"   Ensemble (any of the bt0 strategies) fixed {P(ens['success'])}, "
        f"which is {P(ens['delta_vs_random'])} above Random.", "",
        f"Best metric at FINDING the broken step: {report['best_localizing_metric']}",
        f"Uncertainty pointed at the true step {P(fm['localization_top1'])} of the time.",
        f"   When it missed: {P(fm['among_misses_upstream'])} of misses were because the",
        f"   error happened BEFORE the uncertainty peak.",
    ]
    if has_bt:
        bta = report["backtrack_ablation"]
        lines += [
            "",
            "RQ4 — Does backtracking further upstream improve repair?",
            f"   Oracle bt0 (exact step):    {P(bt_oracle.get(0))}",
            f"   Oracle bt{bta['best_oracle_backtrack']} (best backtrack): "
            f"{P(bta['best_oracle_backtrack_success'])}",
            f"   Best unc bt0:               {P(bt_best.get(0, None))}",
            f"   Best unc bt{bta['best_unc_backtrack']}:               "
            f"{P(bta['best_unc_backtrack_success'])}",
            f"   Full Restart:               {P(g('full_restart'))}",
        ]
    lines += [
        "=" * 78,
        f"Tables : {TAB}",
        f"Figures: {FIG}",
    ]
    print("\n".join(lines))
    with open(os.path.join(TAB, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
