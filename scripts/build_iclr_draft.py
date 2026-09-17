"""Build paper assets from audited HotpotQA results and archived diagnostics.

The main-study intervals are read from the reproduced question-level analysis.
Historical CSVs and notebook displays remain descriptive; no tests or intervals
are inferred from those aggregates. This builder performs no model inference.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"HotpotQA": "Hotpot", "MuSiQue": "Musique", "2WikiMHQA": "Wiki"}
STRATEGIES = {
    "oracle": ("Judge-targeted", "Judge"),
    "oracle_bt2": ("Judge-targeted + backtrack 2", "Backtrack"),
    "full_restart": ("Full restart", "Restart"),
    "best_unc": ("Retrospective best uncertainty variant", "BestUnc"),
}
NOTEBOOKS = {
    "HotpotQA": ("hotpotqa", 34),
    "MuSiQue": ("musique", 35),
    "2WikiMHQA": ("2wikimultihopqa", 34),
}
BASELINES = {
    "random_step": "Random origin",
    "oracle_targeted": "Judge-targeted",
    "oracle_targeted__bt2": "Judge + backtrack 2",
    "full_restart": "Full restart",
}
MAIN_STUDY = ROOT / "output/aws-experiment/2026-09-11-main"
REVIEW_DIAGNOSTICS = ROOT / "output/reviewer-2026-09-12/diagnostics.json"
MAIN_STRATEGIES = {
    "full_restart": "Full restart",
    "random_step__bt2": "Random + backtrack 2",
    "fixed_early": "Fixed early (descriptive)",
    "unc__perplexity__argmax__bt2": "Uncertainty + backtrack 2",
    "position_matched_random": "Position-matched random",
    "unc__perplexity__argmax": "Uncertainty, no backtrack",
}
MAIN_INPUTS = (
    "study-freeze.json", "main-summary.json", "pooled-analysis-local.json",
    "analysis-reproduction-check.json", "position-profile.json",
    "historical-reconstruction.json", "archive-verification.json",
    "cost-estimate.json", "final-state.json",
    *(f"audits/batch{i:02}.json" for i in range(1, 6)),
)


def load_main_study(base: Path = MAIN_STUDY):
    """Reject incomplete or inconsistent evidence before rendering paper assets."""
    records = {name: json.loads((base / name).read_text()) for name in MAIN_INPUTS}
    freeze = records["study-freeze.json"]
    study = freeze["payload"]
    summary = records["main-summary.json"]
    analysis = records["pooled-analysis-local.json"]
    audits = [records[f"audits/batch{i:02}.json"] for i in range(1, 6)]

    def require(condition, message):
        if not condition:
            raise ValueError(f"Main-study evidence: {message}")

    def same_number(a, b):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=1e-12, rel_tol=0)

    digest = hashlib.sha256(json.dumps(study, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
    require(digest == freeze["sha256"] == summary["study_sha256"] == analysis["study_sha256"],
            "frozen study identity mismatch")
    require(summary["code_sha256"] == study["code_sha256"], "source identity mismatch")
    require(study["strategies"] == list(MAIN_STRATEGIES), "six-condition order changed")
    require(list(analysis["primary_contrasts"]) == study["primary_comparisons"],
            "primary comparison family changed")
    require(len(study["batches"]) == summary["batches_completed"] == analysis["n_complete_batches"] == 5,
            "all five frozen batches are required")
    ids = study["question_ids"]
    require(len(ids) == len(set(ids)) == study["n_initial_questions"] == 250,
            "initial cohort coverage mismatch")
    require(not set(ids).intersection(study["explored_ids"]), "declared exclusion overlap")
    require([qid for batch in study["batches"] for qid in batch["ids"]] == ids,
            "batch order or coverage mismatch")
    require(study["seeds"] == [0, 1, 2], "repair seeds changed")
    profile = records["position-profile.json"]
    profile_digest = hashlib.sha256(json.dumps(profile["payload"], sort_keys=True,
                                               separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(profile_digest == profile["sha256"] == study["position_profile_sha256"],
            "position profile identity mismatch")
    require(profile["payload"]["source_phase"] == "development" and
            not set(ids).intersection(profile["payload"]["source_question_ids"]),
            "position profile uses test questions")
    for audit, batch in zip(audits, study["batches"]):
        require(audit["run_id"] == batch["run_id"] and
                audit["study_sha256"] == digest and audit["code_sha256"] == study["code_sha256"],
                "batch audit identity mismatch")
        require(audit["initial_questions"] == len(batch["ids"]) == 50, "batch size mismatch")
        require(all(audit[key] == 0 for key in
                    ("mismatches", "token_budget_violations", "new_step_budget_violations")),
                "audit contains a mismatch or budget violation")
        require(set(audit["strategies"]) == set(MAIN_STRATEGIES), "audit strategy coverage mismatch")
    for key in ("initial_questions", "failed_questions", "initial_successes", "initial_exact_matches",
                "unique_repair_executions", "strategy_seed_rows", "observations_verified",
                "retained_prefix_steps_verified", "input_hashes_verified",
                "trajectories_replayed_and_reference_scored", "initial_generated_tokens",
                "unique_repair_generated_tokens", "unique_repair_prompt_tokens"):
        require(summary[key] == sum(audit[key] for audit in audits), f"audit total mismatch: {key}")
    require(analysis["n_initial_questions"] == summary["initial_questions"] and
            analysis["n_failed_questions"] == summary["failed_questions"] and
            analysis["n_strategy_seed_rows"] == summary["strategy_seed_rows"],
            "pooled analysis coverage mismatch")
    for strategy in MAIN_STRATEGIES:
        row = summary["strategies"][strategy]
        trials = sum(audit["strategies"][strategy]["seed_trials"] for audit in audits)
        successes = sum(audit["strategies"][strategy]["successful_seed_trials"] for audit in audits)
        require(trials == row["seed_trials"] == 3 * summary["failed_questions"] and
                successes == row["successful_seed_trials"], "strategy count mismatch")
        require(same_number(row["success_rate"], successes / trials), "strategy rate mismatch")
        for metric in ("exact_match_rate", "mean_f1"):
            expected = sum(audit["strategies"][strategy][metric] *
                           audit["strategies"][strategy]["seed_trials"] for audit in audits) / trials
            require(same_number(row[metric], expected), f"strategy {metric} mismatch")
        origins = {}
        for audit in audits:
            for origin, count in audit["strategies"][strategy]["target_step_counts"].items():
                origins[origin] = origins.get(origin, 0) + count
        require(origins == row["target_step_counts"] and sum(origins.values()) == trials,
                "origin histogram mismatch")
        require(same_number(row["restart_origin_fraction"], origins.get("0", 0) / trials),
                "origin-zero rate mismatch")
    require(records["analysis-reproduction-check.json"]["matched"], "local analysis was not reproduced")
    for control, contrast in analysis["primary_contrasts"].items():
        require(contrast["strategy_a"] == study["strategy"] and contrast["strategy_b"] == control and
                contrast["n_questions"] == summary["failed_questions"] and contrast["outcome"] == "success",
                "paired contrast identity mismatch")
        expected_delta = (summary["strategies"][study["strategy"]]["success_rate"] -
                          summary["strategies"][control]["success_rate"])
        require(same_number(contrast["delta"], expected_delta), "paired contrast delta mismatch")
        for key, value in contrast.items():
            recorded = summary["primary_analysis"]["primary_contrasts"][control][key]
            require(same_number(value, recorded) if isinstance(value, (int, float)) else value == recorded,
                    "local and recorded primary analysis disagree")
    cost = records["cost-estimate.json"]
    require(same_number(cost["estimated_total_usd"], sum(cost["components_usd"].values())),
            "cost component mismatch")
    return records


def main_study_macros(records):
    summary = records["main-summary.json"]
    treatment = summary["strategies"]["unc__perplexity__argmax__bt2"]
    restart = summary["strategies"]["full_restart"]
    contrast = records["pooled-analysis-local.json"]["primary_contrasts"]["full_restart"]
    values = {
        "MainInitial": f"{summary['initial_questions']:,}",
        "MainFailed": f"{summary['failed_questions']:,}",
        "MainUniqueRepairs": f"{summary['unique_repair_executions']:,}",
        "MainSeedRows": f"{summary['strategy_seed_rows']:,}",
        "MainTreatmentRate": f"{100 * treatment['success_rate']:.2f}",
        "MainRestartRate": f"{100 * restart['success_rate']:.2f}",
        "MainRestartDelta": f"{100 * contrast['delta']:.2f}",
        "MainRestartLow": f"{100 * contrast['delta_lo']:.2f}",
        "MainRestartHigh": f"{100 * contrast['delta_hi']:+.2f}",
        "MainEstimatedUSD": f"{records['cost-estimate.json']['estimated_total_usd']:.2f}",
        "MainObservations": f"{summary['observations_verified']:,}",
        "MainPrefixes": f"{summary['retained_prefix_steps_verified']:,}",
        "MainTrajectories": f"{summary['trajectories_replayed_and_reference_scored']:,}",
    }
    return [r"\newcommand{\%s}{%s}" % item for item in values.items()]


def load_review_diagnostics(records, path=REVIEW_DIAGNOSTICS):
    report = json.loads(path.read_text())
    summary = records["main-summary.json"]
    study = records["study-freeze.json"]

    def require(condition, message):
        if not condition:
            raise ValueError(f"Review diagnostics: {message}")

    def equal(a, b):
        return math.isclose(a, b, abs_tol=1e-12, rel_tol=0)

    require(report["analysis_status"] == "exploratory_review_stage", "exploratory status missing")
    require(report["study_sha256"] == study["sha256"] and
            report["frozen_code_sha256"] == study["payload"]["code_sha256"], "study identity mismatch")
    for key, target in [("n_initial_questions", "initial_questions"),
                        ("n_failed_questions", "failed_questions"), ("n_strategy_seed_rows", "strategy_seed_rows")]:
        require(report[key] == summary[target], "cohort coverage mismatch")
    for key in ["source_sha256", "analysis_code_sha256"]:
        for relative, expected in report[key].items():
            require(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected,
                    f"changed source or analysis code: {relative}")
    for name, expected in report["export_sha256"].items():
        require(hashlib.sha256((path.parent / name).read_bytes()).hexdigest() == expected, "changed trial export")
    plan = report["review_analysis_plan"]
    require(hashlib.sha256((ROOT / plan["path"]).read_bytes()).hexdigest() == plan["sha256"], "changed analysis plan")
    unique = report["unique_study_usage"]
    require(unique["executions"] == summary["unique_repair_executions"] and
            unique["recovery_gen_tokens"] == summary["unique_repair_generated_tokens"] and
            unique["recovery_prompt_tokens"] == summary["unique_repair_prompt_tokens"], "unique usage mismatch")
    require(list(report["policy_accounting"]) == list(MAIN_STRATEGIES), "strategy coverage mismatch")
    for name, row in report["policy_accounting"].items():
        require(row["n_seed_rows"] == 3 * summary["failed_questions"] and
                row["n_questions"] == summary["failed_questions"], "policy coverage mismatch")
        require(equal(row["success"]["mean"], summary["strategies"][name]["success_rate"]), "policy success mismatch")
        require(sum(row["termination_counts"].values()) == row["n_seed_rows"], "termination coverage mismatch")
        for field, value in row["mean_cost_per_attempt"].items():
            require(equal(value, row["policy_trial_totals"][field] / row["n_seed_rows"]), "policy cost mismatch")
        require(equal(row["reference_gated_full_cohort"]["success"]["mean"],
            (summary["initial_successes"] + summary["failed_questions"] * row["success"]["mean"])
            / summary["initial_questions"]), "reference-gated full-cohort outcome mismatch")
    require(list(report["execution_overlap"]) == study["payload"]["primary_comparisons"], "primary family changed")
    for name, row in report["execution_overlap"].items():
        require(row["same_execution_seed_rows"] + row["different_execution_seed_rows"] == 3 * summary["failed_questions"],
                "overlap coverage mismatch")
        require(sum(row[k] for k in ["question_mean_wins", "question_mean_losses", "question_mean_ties"])
                == summary["failed_questions"], "question difference coverage mismatch")
        require(equal(row["same_execution_contribution"] + row["different_execution_contribution"],
                      row["paired_success"]["delta"]), "effect decomposition mismatch")
        for field in ["delta", "delta_lo", "delta_hi", "mean_a", "mean_b"]:
            require(equal(row["paired_success"][field],
                    records["pooled-analysis-local.json"]["primary_contrasts"][name][field]), "primary estimate changed")
    active = report["nonzero_treatment_origin"]
    require(equal(active["question_fraction"], active["n_questions"] / summary["failed_questions"]), "subgroup weight mismatch")
    require(equal(active["question_fraction"] * active["paired_success_versus_restart"]["delta"],
            report["execution_overlap"]["full_restart"]["paired_success"]["delta"]), "subgroup decomposition mismatch")
    return report


def review_macros(report):
    treatment = report["policy_accounting"]["unc__perplexity__argmax__bt2"]
    restart = report["policy_accounting"]["full_restart"]
    subgroup = report["nonzero_treatment_origin"]
    effect = subgroup["paired_success_versus_restart"]
    values = {
        "ReviewActiveQuestions": str(subgroup["n_questions"]),
        "ReviewActiveDelta": f"{100 * effect['delta']:.2f}",
        "ReviewActiveLow": f"{100 * effect['delta_lo']:.2f}",
        "ReviewActiveHigh": f"{100 * effect['delta_hi']:+.2f}",
        "ReviewPromptIncrease": f"{100 * (treatment['mean_cost_per_attempt']['recovery_prompt_tokens'] / restart['mean_cost_per_attempt']['recovery_prompt_tokens'] - 1):.1f}",
        "ReviewBudgetStops": str(treatment["termination_counts"]["budget"]),
        "ReviewStepStops": str(treatment["termination_counts"]["max_steps"]),
        "ReviewBudgetStopRate": f"{100 * treatment['budget_stop_fraction']:.2f}",
        "ReviewWholeSuccess": f"{100 * treatment['reference_gated_full_cohort']['success']['mean']:.2f}",
        "ReviewWholeRestartSuccess": f"{100 * restart['reference_gated_full_cohort']['success']['mean']:.2f}",
    }
    return [r"\newcommand{\%s}{%s}" % item for item in values.items()]


def tex_text(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))


def build_review_assets(out, report):
    overlap_rows, sensitivity_rows = [], []
    for name, row in report["execution_overlap"].items():
        overlap_rows.append([MAIN_STRATEGIES[name], str(row["same_execution_seed_rows"]),
            str(row["questions_with_any_different_execution"]),
            " / ".join(str(row[k]) for k in ["question_mean_wins", "question_mean_losses", "question_mean_ties"])])
        cells = [MAIN_STRATEGIES[name]]
        for metric in ["em", "f1"]:
            contrast = row["secondary_outcomes"][metric]
            cells.append(f"{100 * contrast['delta']:+.2f} [{100 * contrast['delta_lo']:+.2f}, {100 * contrast['delta_hi']:+.2f}]")
        sensitivity_rows.append(cells)
    write_table(out / "tables" / "iclr2027_review_overlap.tex",
                ["Control", "Shared / 339", "Different q. / 113", "Wins / losses / ties"], overlap_rows)
    write_table(out / "tables" / "iclr2027_review_sensitivity.tex",
                ["Control", "EM delta [95\\% CI], pp", "100$\\Delta$F1 [95\\% CI]"], sensitivity_rows)
    cost_rows, population_rows = [], []
    for name, row in report["policy_accounting"].items():
        cost = row["mean_cost_per_attempt"]
        cost_rows.append([MAIN_STRATEGIES[name], f"{cost['recovery_gen_tokens']:.1f}",
            f"{cost['recovery_prompt_tokens']:.1f}", f"{cost['recovery_model_requests']:.2f}",
            f"{cost['recovery_tool_calls']:.2f}",
            " / ".join(str(row["termination_counts"][key]) for key in ["finished", "budget", "max_steps"])])
        overall = row["reference_gated_full_cohort"]
        population_rows.append([MAIN_STRATEGIES[name], f"{100 * overall['success']['mean']:.2f}",
            f"{100 * overall['em']['mean']:.2f}", f"{overall['f1']['mean']:.4f}"])
    write_table(out / "tables" / "iclr2027_review_costs.tex",
                ["Condition", "Output tokens", "Prompt tokens", "Requests", "Tool calls", "F / B / S"], cost_rows)
    write_table(out / "tables" / "iclr2027_review_population.tex",
                ["Recovery condition", "Gate success (\\%)", "EM (\\%)", "F1"], population_rows)
    case_text = []
    titles = {"treatment_higher": "Higher treatment seed mean", "restart_higher": "Higher restart seed mean",
              "both_unsuccessful": "Both policies unsuccessful"}
    for index, case in enumerate(report["illustrative_cases"], 1):
        case_text += [r"\par\medskip\noindent\begin{minipage}{\linewidth}",
            r"\paragraph{Case %d: %s.}" % (index, titles[case["selection_stratum"]]),
            r"\textit{%s} Reference answer: %s. The original trace has %d steps and a %d-token recovery allowance."
            % (tex_text(case["question"]), tex_text(case["gold_answer"]), case["original_steps"], case["token_allowance"]),
            r"\begin{center}\footnotesize", r"\begin{tabular}{@{}p{.20\linewidth}rp{.16\linewidth}p{.44\linewidth}@{}}",
            r"\toprule Condition & Origin & Gate; F1, seeds 0/1/2 & Answers, seeds 0/1/2 \\", r"\midrule"]
        for name, label in [("full_restart", "Restart"), ("unc__perplexity__argmax__bt2", "Uncertainty + bt2")]:
            rows = sorted([r for r in case["rows"] if r["strategy"] == name], key=lambda r: r["seed"])
            answers = "; ".join(tex_text(r["final_answer"]) if r["final_answer"] is not None else r"\textit{no answer}" for r in rows)
            case_text.append(" & ".join([label, str(rows[0]["target_step"]),
                "/".join(str(int(r["success"])) for r in rows) + "; " +
                "/".join(f"{r['f1']:.2f}" for r in rows), answers]) + r" \\")
        case_text += [r"\bottomrule\end{tabular}\end{center}",
            r"Question ID: \texttt{%s}." % case["qid"], r"\end{minipage}\par"]
    (out / "tables" / "iclr2027_review_cases.tex").write_text("\n".join(case_text) + "\n")


def build_main_assets(out, records):
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    summary = records["main-summary.json"]
    contrasts = records["pooled-analysis-local.json"]["primary_contrasts"]
    rows = []
    for name, label in MAIN_STRATEGIES.items():
        row = summary["strategies"][name]
        rows.append([label, f"{row['successful_seed_trials']}/{row['seed_trials']}",
                     f"{100 * row['success_rate']:.2f}", f"{100 * row['exact_match_rate']:.2f}",
                     f"{row['mean_f1']:.4f}", f"{100 * row['restart_origin_fraction']:.2f}"])
    write_table(out / "tables" / "iclr2027_main_results.tex",
                ["Condition", "Successes", "Success (\\%)", "EM (\\%)", "F1", "$k'=0$ (\\%)"], rows)
    rows = [[MAIN_STRATEGIES[name], f"{100 * row['delta']:+.2f}",
             f"[{100 * row['delta_lo']:+.2f}, {100 * row['delta_hi']:+.2f}]",
             f"{row['p_value']:.3f}", f"{row['p_value_holm']:.3f}"]
            for name, row in contrasts.items()]
    write_table(out / "tables" / "iclr2027_main_contrasts.tex",
                ["Control", "Delta (pp)", "95\\% interval (pp)", "Raw $p$", "Holm $p$"], rows)
    audit_rows = []
    for i in range(1, 6):
        audit = records[f"audits/batch{i:02}.json"]
        audit_rows.append([str(i), str(audit["initial_questions"]), str(audit["failed_questions"]),
                           str(audit["unique_repair_executions"]), str(audit["strategy_seed_rows"]), "0"])
    write_table(out / "tables" / "iclr2027_main_audit.tex",
                ["Batch", "Initial", "Failed", "Unique repairs", "Condition/seed rows", "Violations"], audit_rows)
    cost = records["cost-estimate.json"]
    cost_labels = {
        "main_compute_through_guest_powerdown_usd": "Main compute through guest powerdown",
        "retrieval_compute_through_observed_stopping_upper_estimate_usd": "Retrieval through observed stopping",
        "main_transition_contingency_60_seconds_usd": "One-minute transition contingency",
        "retained_ebs_from_main_start_through_report_usd": "Retained storage through report cutoff",
        "public_ipv4_full_elapsed_interval_upper_estimate_usd": "Conservative public IPv4 allowance",
    }
    write_table(out / "tables" / "iclr2027_main_cost.tex", ["Component", "Estimated USD"],
                [[label, f"{cost['components_usd'][key]:.3f}"] for key, label in cost_labels.items()] +
                [["Total before credits, tax and transfer", f"{cost['estimated_total_usd']:.2f}"]])
    fig, ax = plt.subplots(figsize=(7.2, 2.45), layout="constrained")
    for index, (name, row) in enumerate(contrasts.items()):
        delta, low, high = (100 * row[key] for key in ("delta", "delta_lo", "delta_hi"))
        ax.errorbar(delta, index, xerr=[[delta - low], [high - delta]],
                    fmt="o", color="#315d80", capsize=4, markersize=6, linewidth=1.8)
    ax.set_yticks(range(len(contrasts)), [MAIN_STRATEGIES[name] for name in contrasts])
    ax.invert_yaxis()
    ax.set_ylim(len(contrasts) - 0.6, -0.4)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlim(-5.5, 6)
    ax.set_xlabel("Treatment minus control in repair success (percentage points)")
    ax.grid(axis="x", alpha=0.2)
    for extension in ("pdf", "png"):
        fig.savefig(out / "figures" / f"iclr2027_main_contrasts.{extension}", dpi=300)
    plt.close(fig)


def read_notebook_summary(path: Path, cell_index: int) -> dict[str, dict[str, float]]:
    """Read only the visible rows of one frozen pandas HTML display.

    Pandas emits a well-formed table, parsed with the standard XML parser.
    Ellipsis rows and obsolete confidence intervals are deliberately excluded.
    """
    notebook = json.loads(path.read_text())
    cell = notebook["cells"][cell_index]
    displays = [output["data"]["text/html"] for output in cell.get("outputs", [])
                if "text/html" in output.get("data", {})]
    if len(displays) != 1:
        raise ValueError(f"Expected one archived table: {path}, cell {cell_index}")
    html = displays[0]
    html = "".join(html) if isinstance(html, list) else html
    start, end = html.find("<table"), html.find("</table>")
    if start < 0 or end < start:
        raise ValueError(f"Missing archived table: {path}")
    table = ElementTree.fromstring(html[start:end + len("</table>")])
    header = ["".join(item.itertext()).strip() for item in table.findall("./thead/tr/th")]
    fields = ("fixed_%", "avg_tokens", "avg_tool_calls")
    rows = {}
    for tr in table.findall("./tbody/tr"):
        values = ["".join(item.itertext()).strip() for item in tr]
        if len(values) != len(header):
            raise ValueError(f"Malformed archived row: {path}")
        record = dict(zip(header, values))
        strategy = record["strategy"]
        if strategy == "...":
            continue
        if strategy in rows:
            raise ValueError(f"Duplicate archived strategy: {strategy}")
        row = {key: float(record[key]) for key in fields}
        if not all(math.isfinite(value) and value >= 0 for value in row.values()):
            raise ValueError(f"Invalid archived numbers: {strategy}")
        if row["fixed_%"] > 100:
            raise ValueError(f"Invalid archived success: {strategy}")
        rows[strategy] = row
    required = set(BASELINES) | {
        "oracle_targeted__informed", "oracle_targeted__bt2__informed",
        "uncertainty_ensemble_any",
    }
    if not required <= rows.keys():
        raise ValueError(f"Incomplete archived baseline display: {path}")
    return rows


def load_notebook_summaries(aggregate_rows):
    summaries, sources = {}, {}
    for dataset, (directory, cell_index) in NOTEBOOKS.items():
        path = ROOT / f"results/{directory}/run_colab_{directory}.ipynb"
        summary = read_notebook_summary(path, cell_index)
        for strategy, key in (("oracle_targeted", "oracle"),
                              ("oracle_targeted__bt2", "oracle_bt2"),
                              ("full_restart", "full_restart")):
            expected = round(100 * float(aggregate_rows[dataset][key]), 1)
            if abs(summary[strategy]["fixed_%"] - expected) > 1e-6:
                raise ValueError(f"Notebook/CSV snapshot mismatch: {dataset}, {strategy}")
        summaries[dataset] = summary
        sources[str(path.relative_to(ROOT))] = {
            "cell_index_zero_based": cell_index,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scope": "Visible rounded rows only; not raw trials or a full strategy grid.",
        }
    return summaries, sources


def write_table(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = "l" + "r" * (len(header) - 1)
    lines = [r"\begin{tabular}{@{}" + columns + r"@{}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def build_restored_assets(out, summaries, aggregate_rows, localization_rows):
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    costs, nudges = [], []
    for strategy, label in BASELINES.items():
        values = []
        for dataset in DATASETS:
            row = summaries[dataset][strategy]
            values += [f"{row['avg_tokens']:.0f}", f"{row['avg_tool_calls']:.2f}"]
        costs.append([label, *values])
    write_table(out / "tables" / "iclr2027_costs.tex",
                ["Strategy", "HP tokens", "HP tools", "MS tokens", "MS tools",
                 "2W tokens", "2W tools"], costs)
    for dataset in DATASETS:
        for bt, strategy in ((0, "oracle_targeted"), (2, "oracle_targeted__bt2")):
            generic = summaries[dataset][strategy]["fixed_%"]
            informed = summaries[dataset][strategy + "__informed"]["fixed_%"]
            nudges.append([dataset, str(bt), f"{generic:.1f}", f"{informed:.1f}",
                           f"{informed - generic:+.1f}"])
    write_table(out / "tables" / "iclr2027_nudges.tex",
                ["Dataset", "Backtrack", "Generic (\\%)", "Informed (\\%)", "Delta (pp)"], nudges)

    metric_labels = ["Action disagreement", "Token entropy", "Sampled-token complement",
                     "Perplexity", "Verbalized uncertainty"]
    metric_keys = ["self_consistency", "token_entropy_max", "max_token_prob_max",
                   "perplexity", "verbalized_confidence"]
    by_metric = {row["metric"]: row for row in localization_rows}
    values = [[float(by_metric[key][dataset]) for dataset in DATASETS] for key in metric_keys]
    fig, ax = plt.subplots(figsize=(8.3, 2.4), layout="constrained")
    heatmap = ax.imshow(values, cmap="cividis", vmin=0, vmax=0.5, aspect="auto")
    ax.set_xticks(range(3), DATASETS)
    ax.set_yticks(range(5), metric_labels)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    color="white" if value < 0.25 else "black")
    fig.colorbar(heatmap, ax=ax, label="Exact judge agreement", shrink=0.9)
    for extension in ["pdf", "png"]:
        fig.savefig(out / "figures" / f"iclr2027_localization_heatmap.{extension}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True, layout="constrained")
    markers = ["o", "s", "^", "D"]
    colors = ["#777777", "#228477", "#d68d24", "#3c69a3"]
    short_labels = ["Random", "Judge", "Judge + bt2", "Restart"]
    for ax, dataset in zip(axes, DATASETS):
        for strategy, label, color, marker in zip(BASELINES, short_labels, colors, markers):
            row = summaries[dataset][strategy]
            ax.scatter(row["avg_tokens"], row["fixed_%"], label=label,
                       color=color, marker=marker, s=65)
        ax.set_title(dataset)
        ax.set_xlabel("Mean generated recovery tokens")
        ax.set_ylim(0, 27)
        ax.margins(x=0.18)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean repair success (%)")
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    for extension in ["pdf", "png"]:
        fig.savefig(out / "figures" / f"iclr2027_cost_success.{extension}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), layout="constrained")
    gains = [100 * (float(aggregate_rows[d]["oracle_bt2"]) - float(aggregate_rows[d]["oracle"]))
             for d in DATASETS]
    bars = axes[0].bar(range(3), gains, color="#228477", width=0.6)
    axes[0].bar_label(bars, fmt="%.1f", padding=3)
    axes[0].set_xticks(range(3), DATASETS)
    axes[0].set_ylim(0, 14)
    axes[0].set_ylabel("Backtrack-two gain (pp)")
    for offset, bt, color in ((-0.18, 0, "#3c69a3"), (0.18, 2, "#d68d24")):
        strategy = "oracle_targeted" + ("__bt2" if bt else "")
        deltas = [summaries[d][strategy + "__informed"]["fixed_%"] -
                  summaries[d][strategy]["fixed_%"] for d in DATASETS]
        bars = axes[1].bar([i + offset for i in range(3)], deltas, width=0.34,
                           color=color, label=f"Backtrack {bt}")
        axes[1].bar_label(bars, fmt="%+.1f", padding=3, fontsize=9)
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set_xticks(range(3), DATASETS)
    axes[1].set_ylim(-1.6, 3.1)
    axes[1].set_ylabel("Informed minus generic (pp)")
    axes[1].legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    for extension in ["pdf", "png"]:
        fig.savefig(out / "figures" / f"iclr2027_origin_hint_deltas.{extension}", dpi=300)
    plt.close(fig)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    source = ROOT / "results/cross_dataset/cross_dataset_traj_stats.csv"
    localization = ROOT / "results/cross_dataset/cross_dataset_localization.csv"
    rows = {row["dataset"]: row for row in read_rows(source)}
    summaries, notebook_sources = load_notebook_summaries(rows)
    localization_rows = read_rows(localization)
    main_records = load_main_study()
    review = load_review_diagnostics(main_records)
    out = ROOT / "paper/generated"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    macros = ["% Generated by scripts/build_iclr_draft.py; Main* macros are audited HotpotQA results."]
    for dataset, prefix in DATASETS.items():
        row = rows[dataset]
        macros.append(r"\newcommand{\%sFailed}{%s}" % (prefix, row["n_failed"]))
        for key, (_, suffix) in STRATEGIES.items():
            value = float(row[key])
            if not 0 <= value <= 1:
                raise ValueError(f"Invalid success rate for {dataset}: {key}")
            macros.append(r"\newcommand{\%s%s}{%.1f}" % (prefix, suffix, value * 100))
        gain = 100 * (float(row["oracle_bt2"]) - float(row["oracle"]))
        macros.append(r"\newcommand{\%sGain}{%.1f}" % (prefix, gain))
    macros.append(r"\newcommand{\TotalFailed}{%d}" % sum(
        int(rows[dataset]["n_failed"]) for dataset in DATASETS))
    macros.extend(main_study_macros(main_records))
    macros.extend(review_macros(review))
    (out / "iclr2027_numbers.tex").write_text("\n".join(macros) + "\n")

    table = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Strategy & HotpotQA & MuSiQue & 2WikiMHQA \\", r"\midrule"]
    table.append("Random origin & " + " & ".join(
        f"{summaries[d]['random_step']['fixed_%']:.1f}" for d in DATASETS) + r" \\")
    for key, (label, _) in STRATEGIES.items():
        if key == "best_unc":
            table.append(r"\midrule")
        values = [f"{100 * float(rows[dataset][key]):.1f}" for dataset in DATASETS]
        table.append(label + " & " + " & ".join(values) + r" \\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "iclr2027_results.tex").write_text("\n".join(table) + "\n")

    labels = {
        "self_consistency": "Action disagreement",
        "token_entropy_max": "Token entropy (max)",
        "max_token_prob_max": "Sampled-token complement (max)",
        "perplexity": "Perplexity",
        "verbalized_confidence": "Verbalized uncertainty",
    }
    table = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Metric & HotpotQA & MuSiQue & 2WikiMHQA \\", r"\midrule"]
    for row in localization_rows:
        values = [f"{float(row[dataset]):.3f}" for dataset in DATASETS]
        table.append(labels[row["metric"]] + " & " + " & ".join(values) + r" \\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "iclr2027_localization.tex").write_text("\n".join(table) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0), sharey=True, layout="constrained")
    colors = ["#228477", "#d68d24", "#3c69a3"]
    for ax, dataset in zip(axes, DATASETS):
        values = [100 * float(rows[dataset][key])
                  for key in ("oracle", "oracle_bt2", "full_restart")]
        bars = ax.bar(range(3), values, color=colors, width=0.65)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=3, fontsize=10)
        ax.set_xticks(range(3), ["Judge", "Judge\n+ bt2", "Restart"])
        ax.set_title(dataset)
        ax.set_ylim(0, 27)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean repair success (%)")
    for extension in ["pdf", "png"]:
        fig.savefig(out / "figures" / f"iclr2027_repair_rates.{extension}", dpi=300)
    plt.close(fig)
    build_restored_assets(out, summaries, rows, localization_rows)
    build_main_assets(out, main_records)
    build_review_assets(out, review)
    (out / "iclr2027_archived_summaries.json").write_text(json.dumps({
        "status": "rounded_archived_display_values_not_raw_trials",
        "datasets": summaries,
        "sources": notebook_sources,
    }, indent=2) + "\n")

    manifest = {
        "status": "audited_hotpotqa_main_with_archived_diagnostics_not_submission_ready",
        "controlled_main_study": {
            "study_sha256": main_records["study-freeze.json"]["sha256"],
            "code_sha256": main_records["main-summary.json"]["code_sha256"],
            "sources": {str((MAIN_STUDY / name).relative_to(ROOT)):
                        hashlib.sha256((MAIN_STUDY / name).read_bytes()).hexdigest()
                        for name in MAIN_INPUTS},
            "analysis": "Reproduced paired question analysis; seed means, 10000 bootstrap resamples, four Holm contrasts.",
            "historical_exclusions": "Reconstructed documented history; original full pool JSON not compared.",
            "cost": "Estimated infrastructure usage before credits, tax and transfer; not a policy-efficiency comparison.",
        },
        "review_stage_diagnostics": {
            "path": str(REVIEW_DIAGNOSTICS.relative_to(ROOT)),
            "sha256": hashlib.sha256(REVIEW_DIAGNOSTICS.read_bytes()).hexdigest(),
            "status": "Exploratory analyses added after the frozen primary results were known.",
            "primary_contrasts_unchanged": True,
        },
        "archived_commit": "756f80d",
        "datasets_in_draft": list(DATASETS),
        "excluded_dataset": {"FEVER": "Claim text was substituted for evidence in the loader."},
        "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (source, localization)},
        "notebook_sources": notebook_sources,
        "notebook_validation": "Judge, backtrack, and restart means agree with CSVs to displayed precision.",
        "costs": "Rounded recovery-only token/tool means; acquisition, prompt, and selector costs absent.",
        "uncertainty": "Historical intervals cannot be regenerated from aggregates; audited main-study intervals are separate.",
        "best_unc": "Retrospective maximum over all variants, including judge-informed hints.",
        "budget": "Historical runs may exceed requested caps; CPU fixes do not repair old outcomes.",
    }
    (out / "iclr2027_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Generated ICLR working-draft tables, restored analyses, figures, and provenance.")


if __name__ == "__main__":
    main()
