"""Build extension tables only from complete, independently reproduced results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scripts.build_followup_paper import require
from scripts.build_iclr_draft import write_table
from scripts.finalize_extension_study import compare_exports
from scripts.extension_position_balance import CONTROL, load_position_balance
from src.repair.controlled import fingerprint
from src.repair.diagnosis import TREATMENT, DIAGNOSIS_STRATEGY

BASE = ROOT / "output/aws-experiment/2026-09-15-extension-completion"
OUT = ROOT / "paper/generated"
MODELS = {"qwen32b": "Qwen32B", "mistral12b": "Mistral12B"}
DATASETS = {"hotpotqa": "HotpotQA", "musique": "MuSiQue", "2wikimultihopqa": "2Wiki"}
POLICIES = {"full_restart": "Restart", "random_step__bt2": "Random, backtrack 2",
            "fixed_early": "Fixed early", TREATMENT: "Uncertainty, backtrack 2",
            "position_matched_random": "Dev.-fitted random",
            "unc__perplexity__argmax": "Uncertainty, no backtracking",
            DIAGNOSIS_STRATEGY: "Diagnosis/replay"}
INPUTS = ["pooled-analysis-local.json", "pooled-analysis-local.trials.csv",
          "remote-pooled-analysis.json", "remote-pooled-analysis.trials.csv",
          "retrieved/package/protocol.json"]
MEANS = ["success", "em", "f1", "incremental_gen_tokens", "incremental_prompt_tokens",
         "incremental_model_requests", "incremental_wall_latency_s"]


def validate_coverage(analysis, protocol, trials):
    """Check the reporting denominator, policies, deadlines and measured means."""
    require(analysis.get("complete") is True, "Extension remains incomplete")
    cells = {(m, d) for m in protocol["models"] for d in protocol["cohorts"]}
    validate_cell_coverage(analysis, protocol, trials, cells)


def validate_cell_coverage(analysis, protocol, trials, cells):
    """Validate specified complete cells without declaring the extension complete."""
    declared = {(m, d) for m in protocol["models"] for d in protocol["cohorts"]}
    require(bool(cells) and cells <= declared, "Undeclared reporting cells")
    audits = {(a["model_key"], a["dataset"]): a for a in analysis["audits"]}
    require(len(audits) == len(analysis["audits"]) and set(audits) == cells
            and all(a["mismatches"] == 0 for a in audits.values()), "Incomplete or mismatched cell audits")
    keys = ["model_key", "dataset", "qid", "mode", "strategy", "seed"]
    require(not trials.duplicated(keys).any(), "Duplicate extension trials")
    expected, primary_keys, runtime_keys, summary_keys = set(), set(), set(), set()
    runtime_spec = protocol["runtime"]
    for (model, dataset), audit in audits.items():
        cohort = protocol["cohorts"][dataset]
        require(audit["main_questions"] == len(cohort["main_ids"])
                and audit["development_questions"] == len(cohort["development_ids"]), "Changed cohort size")
        cell = trials[(trials.model_key == model) & (trials.dataset == dataset)]
        failed = set(cell.loc[cell["mode"] == "token", "qid"])
        require(len(failed) == audit["main_failures"] and failed <= set(cohort["main_ids"]),
                "Changed failed-question coverage")
        runtime_ids = ([q for q in cohort["main_ids"] if q in failed][:runtime_spec["n_failures_max"]]
                       if (model, dataset) == (runtime_spec["model"], runtime_spec["dataset"]) else [])
        require(audit["runtime_questions"] == len(runtime_ids), "Changed runtime cohort")
        for mode, selected, policies in [("token", failed, protocol["strategies"]),
                                         ("runtime", runtime_ids, runtime_spec["strategies"])]:
            expected.update((model, dataset, q, mode, s, seed)
                            for q in selected for s in policies for seed in protocol["seeds"])
            if selected:
                summary_keys.update((model, dataset, mode, s, None) for s in policies)
        require(audit["trial_rows"] == len(cell)
                and audit["unique_repairs"] == len(cell.drop_duplicates(["mode", "execution_id"])),
                "Changed unique execution or trial counts")
        if failed:
            primary_keys.update((model, dataset, c) for c in protocol["primary_controls"])
        if runtime_ids:
            runtime_keys.update((model, dataset, seconds, c) for seconds in runtime_spec["deadlines_s"]
                                for c in ["full_restart", DIAGNOSIS_STRATEGY])
            summary_keys.update((model, dataset, "deadline", s, seconds)
                                for seconds in runtime_spec["deadlines_s"] for s in runtime_spec["strategies"])
    require(set(trials[keys].itertuples(index=False, name=None)) == expected,
            "Incomplete policy/seed or runtime coverage")
    family = analysis["primary_comparisons"]
    require(analysis["primary_family_size"] == len(declared)*len(protocol["primary_controls"])
            and len(family) == len(primary_keys)
            and {(r["model_key"], r["dataset"], r["strategy_b"]) for r in family} == primary_keys,
            "Changed primary comparison family")
    runtime = analysis["runtime_comparisons"]
    require(len(runtime) == len(runtime_keys)
            and {(r["model_key"], r["dataset"], r["deadline_s"], r["strategy_b"]) for r in runtime} == runtime_keys,
            "Changed runtime comparison family")
    for row in family + runtime:
        require(row["strategy_a"] == TREATMENT, "Changed comparison treatment")
        if "deadline_s" in row:
            require((row["p_value_holm"] is not None) == (row["deadline_s"] == runtime_spec["primary_deadline_s"]),
                    "Runtime primary/exploratory correction changed")
    summaries = analysis["summaries"]
    require(len(summaries) == len(summary_keys)
            and {(r["model_key"], r["dataset"], r["mode"], r["strategy"], r.get("deadline_s"))
                 for r in summaries} == summary_keys, "Changed policy summary coverage")
    for row in summaries:
        mode = "runtime" if row["mode"] == "deadline" else row["mode"]
        frame = trials[(trials.model_key == row["model_key"]) & (trials.dataset == row["dataset"])
                       & (trials["mode"] == mode) & (trials.strategy == row["strategy"])]
        success = frame.success
        if row["mode"] == "deadline":
            success = success & (frame.incremental_wall_latency_s <= row["deadline_s"])
        require(row["questions"] == frame.qid.nunique() and row["trials"] == len(frame)
                and row["successful_trials"] == int(success.sum()), "Changed policy denominator or successes")
        for key in (["success"] if row["mode"] == "deadline" else MEANS):
            value = success.mean() if key == "success" else frame[key].mean()
            require(math.isclose(value, row[key], rel_tol=0, abs_tol=1e-12),
                    f"Policy mean differs from measured trials: {row['strategy']}/{key}")


def load_extension(base=BASE):
    base = Path(base)
    reproduced = json.loads((base/"analysis-reproduction-check.json").read_text())
    require(reproduced.get("matched") is True and reproduced.get("mismatches") == 0,
            "Independent extension reproduction is incomplete")
    require(set(reproduced["sha256"]) == set(INPUTS), "Incomplete reproduction input hashes")
    for name in INPUTS:
        require(hashlib.sha256((base/name).read_bytes()).hexdigest() == reproduced["sha256"][name],
                f"Evidence changed after independent reproduction: {name}")
    analysis = json.loads((base/INPUTS[0]).read_text())
    frozen = json.loads((base/"retrieved/package/protocol.json").read_text())
    protocol = frozen["payload"]
    require(frozen["sha256"] == fingerprint(protocol) == analysis["protocol_sha256"]
            == reproduced["protocol_sha256"], "Extension protocol identity mismatch")
    comparison = compare_exports(base/INPUTS[0], base/"remote-pooled-analysis.json")
    require(comparison["trial_rows_compared"] == reproduced["trial_rows_compared"],
            "Extension reproduction row count changed")
    trials = pd.read_csv(base/"pooled-analysis-local.trials.csv", keep_default_na=False)
    validate_coverage(analysis, protocol, trials)
    require(reproduced["audited_cells"] == len(analysis["audits"])
            and reproduced["main_model_question_evaluations"] == sum(a["main_questions"] for a in analysis["audits"])
            and reproduced["unique_repairs"] == sum(a["unique_repairs"] for a in analysis["audits"]),
            "Extension reproduction audit counts changed")
    return analysis, protocol


def interval(row):
    return f"[{100*row['delta_lo']:+.2f}, {100*row['delta_hi']:+.2f}]"


def build_contrast_figure(out, analysis):
    """Show paired restart contrasts, including every audited replication cell."""
    rows = [r for r in analysis["primary_comparisons"] if r["strategy_b"] == "full_restart"]
    (Path(out)/"figures").mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(6.4, 3.0), layout="constrained")
        if rows:
            labels, positions = [], []
            for index, row in enumerate(rows):
                y = len(rows)-1-index
                color = "#21618c" if row["model_key"] == "qwen32b" else "#b96529"
                lo, hi, mean = [100*row[key] for key in ["delta_lo", "delta_hi", "delta"]]
                ax.hlines(y, lo, hi, color=color, linewidth=1.8)
                ax.vlines([lo, hi], y-.065, y+.065, color=color, linewidth=1.2)
                ax.scatter([mean], [y], color=color, s=28, zorder=3,
                           marker="o" if row["model_key"] == "qwen32b" else "s")
                labels.append(f"{MODELS[row['model_key']]} / {DATASETS[row['dataset']]} (n={row['n_questions']})")
                positions.append(y)
            ax.set(yticks=positions, yticklabels=labels, ylim=(-.45, len(rows)-.55))
            ax.tick_params(axis="y", labelsize=8, length=0)
        else:
            ax.text(.5, .5, "No conditional repair estimates: zero initial failures",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_yticks([])
        ax.axvline(0, color="#8c8c8c", linewidth=.8, linestyle="--", zorder=0)
        ax.grid(axis="x", color="#e4e7ea", linewidth=.6)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_xlabel("Uncertainty minus restart in repair success (percentage points)")
        for extension in ["pdf", "png"]:
            fig.savefig(Path(out)/"figures"/f"iclr2027_extension_contrasts.{extension}", dpi=300)
        plt.close(fig)


def build_assets(out, analysis, protocol):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    initial, overview = [], []
    for audit in analysis["audits"]:
        model, dataset = audit["model_key"], audit["dataset"]
        labels = [MODELS[model], DATASETS[dataset]]
        initial.append(labels + [str(audit["main_questions"]), str(audit["main_failures"]),
                       f"{100*audit['initial_success']:.1f}", f"{100*audit['initial_em']:.1f}", f"{audit['initial_f1']:.4f}"])
        summaries = {r["strategy"]: r for r in analysis["summaries"]
                     if (r["model_key"], r["dataset"], r["mode"]) == (model, dataset, "token")}
        contrast = next((r for r in analysis["primary_comparisons"]
                         if (r["model_key"], r["dataset"], r["strategy_b"]) == (model, dataset, "full_restart")), None)
        rates = [f"{100*summaries[s]['success']:.2f}" if s in summaries else "--"
                 for s in ["full_restart", TREATMENT, DIAGNOSIS_STRATEGY]]
        overview.append(labels + [str(audit["main_failures"])] + rates +
                        ([f"{100*contrast['delta']:+.2f}", interval(contrast), f"{contrast['p_value_holm']:.3f}"]
                         if contrast else ["--"]*3))
    write_table(out/"tables"/"iclr2027_extension_initial.tex", ["Model", "Dataset", "$N$", "Failures", "Success (\\%)", "EM (\\%)", "F1"], initial)
    write_table(out/"tables"/"iclr2027_extension_overview.tex", ["Model", "Dataset", "$N_f$", "Restart", "Unc.", "Diag.", "$\\Delta$ (pp)", "95\\% interval (pp)", "$p_H$"], overview)
    for model in protocol["models"]:
        results, costs, contrasts = [], [], []
        for dataset in protocol["cohorts"]:
            for strategy, label in POLICIES.items():
                row = next((r for r in analysis["summaries"] if
                            (r["model_key"], r["dataset"], r["mode"], r["strategy"]) ==
                            (model, dataset, "token", strategy)), None)
                if row is None:
                    continue  # A zero-failure cell has no conditional repair denominator.
                labels = [DATASETS[dataset], label]
                results.append(labels + [f"{row['successful_trials']}/{row['trials']}",
                               f"{100*row['success']:.2f}", f"{100*row['em']:.2f}", f"{row['f1']:.4f}"])
                costs.append(labels + [f"{row['incremental_gen_tokens']:.1f}", f"{row['incremental_prompt_tokens']:.1f}",
                             f"{row['incremental_model_requests']:.2f}"])
            for row in analysis["primary_comparisons"]:
                if (row["model_key"], row["dataset"]) == (model, dataset):
                    contrasts.append([DATASETS[dataset], POLICIES[row["strategy_b"]], f"{100*row['delta']:+.2f}",
                                      interval(row), f"{row['p_value']:.3f}", f"{row['p_value_holm']:.3f}"])
        write_table(out/"tables"/f"iclr2027_extension_{model}_results.tex", ["Dataset", "Policy", "Successes", "Success (\\%)", "EM (\\%)", "F1"], results)
        write_table(out/"tables"/f"iclr2027_extension_{model}_costs.tex", ["Dataset", "Policy", "Generated", "Prompt", "Requests"], costs)
        write_table(out/"tables"/f"iclr2027_extension_{model}_contrasts.tex", ["Dataset", "Control", "$\\Delta$ (pp)", "95\\% interval (pp)", "$p$", "$p_H$"], contrasts)
    runtime_rows, runtime_contrasts = [], []
    for strategy in protocol["runtime"]["strategies"]:
        raw = next((r for r in analysis["summaries"] if r["mode"] == "runtime" and r["strategy"] == strategy), None)
        if raw is None:
            continue
        rates = []
        for seconds in protocol["runtime"]["deadlines_s"]:
            row = next(r for r in analysis["summaries"] if r["mode"] == "deadline"
                       and r["strategy"] == strategy and r["deadline_s"] == seconds)
            rates.append(f"{row['successful_trials']}/{row['trials']} ({100*row['success']:.1f})")
        runtime_rows.append([POLICIES[strategy]] + rates + [f"{raw['incremental_wall_latency_s']:.2f}"])
    for row in analysis["runtime_comparisons"]:
        runtime_contrasts.append([f"{row['deadline_s']:g}", POLICIES[row["strategy_b"]], f"{100*row['delta']:+.2f}",
                                 interval(row), f"{row['p_value']:.3f}",
                                 "--" if row["p_value_holm"] is None else f"{row['p_value_holm']:.3f}"])
    write_table(out/"tables"/"iclr2027_extension_runtime.tex", ["Policy"] +
                [f"By {s:g} s (\\%)" for s in protocol["runtime"]["deadlines_s"]] + ["Mean time (s)"], runtime_rows)
    write_table(out/"tables"/"iclr2027_extension_runtime_contrasts.tex", ["Deadline (s)", "Control", "$\\Delta$ (pp)", "95\\% interval (pp)", "$p$", "$p_H$"], runtime_contrasts)
    build_contrast_figure(out, analysis)


def build_position_assets(out, diagnostics, trials):
    """Report all six realized distributions without claiming an exact match."""
    out = Path(out)
    rows = []
    for cell in diagnostics["cells"]:
        u, c = [cell["policies"][p] for p in [TREATMENT, CONTROL]]
        n = cell["paired_seed_rows"]
        values = ([f"{100*p['origin_zero_fraction']:.2f}" for p in [u, c]]
                  + [f"{p['mean_retained_steps']:.2f}" for p in [u, c]]
                  + [f"{cell['max_normalized_ecdf_gap']:.3f}", f"{cell['shared_execution_pairs']}/{n}"]
                  if n else ["--"]*6)
        rows.append([f"{MODELS[cell['model_key']]} / {DATASETS[cell['dataset']]}",
                     str(cell["development"]["n"]), str(cell["main_failures"])] + values)
    write_table(out/"tables"/"iclr2027_extension_position_balance.tex",
                ["Cell", "$N_d$", "$N_f$", "$P_0^U$", "$P_0^C$", "$\\bar{k}_U$", "$\\bar{k}_C$", "$D$", "Shared"], rows)
    # Store exact support counts and row-level joins alongside the PDF assets.
    (out/"iclr2027_extension_position_balance.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False)+"\n")
    columns = ["model_key", "dataset", "qid", "seed", "strategy", "origin", "n_steps", "normalized_origin",
               "execution_id", "budget", "prompt_sha256"]
    trials[columns].sort_values(columns[:5]).to_csv(out/"iclr2027_extension_position_rows.csv", index=False)
    (out/"figures").mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        cells = diagnostics["cells"]
        models = list(dict.fromkeys(c["model_key"] for c in cells))
        datasets = list(dict.fromkeys(c["dataset"] for c in cells))
        fig, axes = plt.subplots(len(models), len(datasets), figsize=(6.4, 2.25*len(models)),
                                 squeeze=False, sharex=True, sharey=True)
        for i, model in enumerate(models):
            for j, dataset in enumerate(datasets):
                ax = axes[i, j]
                cell = next(c for c in cells if (c["model_key"], c["dataset"]) == (model, dataset))
                for policy, label, color, style in [(TREATMENT, "Uncertainty", "#21618c", "-"),
                                                    (CONTROL, "Dev.-fitted random", "#b96529", "--"),
                                                    (None, "Development fit sample", "#606060", ":")]:
                    dist = (cell["policies"][policy] if policy else cell["development"])["normalized_origin"]
                    if not dist:
                        continue
                    x, y, cumulative = [0.], [0.], 0.
                    for row in dist:
                        cumulative += row["proportion"]
                        x.append(row["position"])
                        y.append(cumulative)
                    ax.step(x+[1.], y+[cumulative], where="post", color=color, linestyle=style,
                            linewidth=1.6 if policy else 1.1, label=label)
                if not cell["paired_seed_rows"]:
                    ax.text(.5, .4, "No initial failures", ha="center", transform=ax.transAxes)
                ax.set_title(f"{MODELS[model]} / {DATASETS[dataset]}\n"
                             f"{cell['development']['n']} dev. failures; {cell['main_failures']} main failures", fontsize=8)
                ax.set(xlim=(-.025, 1.025), ylim=(-.025, 1.025), xticks=[0, .5, 1], yticks=[0, .5, 1])
                ax.grid(color="#e4e7ea", linewidth=.5)
                if j == 0:
                    ax.set_ylabel("Cumulative fraction")
                if i == len(models)-1:
                    ax.set_xlabel("Normalized origin k / max(1, T−1)", fontsize=7)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, .075, 1, 1), h_pad=1.6, w_pad=1.)
        for extension in ["pdf", "png"]:
            fig.savefig(out/"figures"/f"iclr2027_extension_position_ecdf.{extension}", dpi=300)
        plt.close(fig)


def main(base=BASE, out=OUT):
    analysis, protocol = load_extension(base)
    positions, position_trials = load_position_balance(base, analysis, protocol)
    build_assets(out, analysis, protocol)
    build_position_assets(out, positions, position_trials)
    assets = [p for folder in ["tables", "figures"]
              for p in sorted((Path(out)/folder).glob("iclr2027_extension_*"))]
    assets.extend(Path(out)/name for name in ["iclr2027_extension_position_balance.json", "iclr2027_extension_position_rows.csv"])
    provenance = {"protocol_sha256": analysis["protocol_sha256"],
                  "checks": "Complete independent reproduction, frozen cohorts, policy/seed and deadline coverage, measured policy means; position diagnostics verify raw identities, dev-only profiles, and frozen origins/executions",
                  "source_sha256": {name: hashlib.sha256((Path(base)/name).read_bytes()).hexdigest()
                                    for name in INPUTS + ["analysis-reproduction-check.json"]},
                  "reporting_code_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                            for name in ["scripts/build_extension_paper.py", "scripts/extension_position_balance.py"]},
                  "asset_sha256": {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in assets}}
    (Path(out)/"iclr2027_extension_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print("Built reproduced extension tables and exploratory position-balance diagnostics for every cell.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    main(args.base, args.out)
