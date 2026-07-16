"""Human validation — hand-label N failed trajectories, then report judge agreement.

    python scripts/label_human.py --config config/config_local.yaml

Interactive. Progress is saved after every label, so you can quit (Ctrl-C) and
resume later. This is the ONLY manual step in the whole pipeline.
"""
from __future__ import annotations

import os
import random

from _common import parse_args, boot  # type: ignore

from src.annotate import render_for_human, compute_agreement
from src.utils import save_json, load_json, load_item


def main() -> None:
    args = parse_args("Hand-label failed trajectories & score judge agreement")
    cfg, log = boot("label_human", args)

    failed_ids = load_json(os.path.join(cfg.path("data_processed"), "failed_ids.json"))
    ann_dir = cfg.path("annotations")
    human_path = os.path.join(ann_dir, "_human_labels.json")
    human = load_json(human_path) if os.path.exists(human_path) else {}

    rng = random.Random(cfg.project.seed)
    n_val = args.limit or cfg.dataset.human_val_size
    val_ids = rng.sample(failed_ids, min(n_val, len(failed_ids)))

    etypes = cfg.raw["annotation"]["error_types"]
    menu = "  ".join(f"{i+1}={e}" for i, e in enumerate(etypes))

    todo = [q for q in val_ids if q not in human]
    print(f"\n{len(human)}/{len(val_ids)} already labeled. {len(todo)} to go.")
    print("Ctrl-C any time — progress is saved.\n")

    try:
        for n, q in enumerate(todo, 1):
            t = load_item(cfg.path("trajectories"), q)
            print("=" * 78)
            print(f"[{n}/{len(todo)}]")
            print(render_for_human(t))
            print(f"\nError types: {menu}")
            step = input(f"Earliest broken step (1-{len(t['steps'])}): ").strip()
            et = input("Error type number: ").strip()
            try:
                bs = max(0, min(len(t["steps"]) - 1, int(step) - 1))
                etype = etypes[int(et) - 1]
            except Exception:
                print("  -> skipped (bad input)")
                continue
            human[q] = {"broken_step": bs, "error_type": etype}
            save_json(human, human_path)
    except KeyboardInterrupt:
        print("\nStopped. Progress saved.")

    # ---- agreement --------------------------------------------------------- #
    ids = [q for q in val_ids if q in human]
    if not ids:
        print("No labels yet — nothing to score.")
        return
    h = [human[q] for q in ids]
    j = [load_item(ann_dir, q) for q in ids]
    agr = compute_agreement(h, j)
    save_json(agr, os.path.join(cfg.path("tables"), "judge_human_agreement.json"))
    print("\nJudge vs human agreement:", agr)
    if agr["error_type_kappa"] < 0.4 or agr["step_within1"] < 0.6:
        print("NOTE: low agreement. Step labels drive the experiment; error-type "
              "breakdowns should be treated as exploratory.")


if __name__ == "__main__":
    main()
