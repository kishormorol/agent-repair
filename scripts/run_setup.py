"""Stage 0 — download HotpotQA dev (distractor) and create output folders.

    python scripts/run_setup.py --config config/config_local.yaml
"""
from __future__ import annotations

import json
import os
import urllib.request

from _common import parse_args, boot  # type: ignore


def _try_direct(urls, dst, timeout=20) -> bool:
    for u in urls:
        try:
            print("Trying direct:", u)
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
                f.write(r.read())
            return True
        except Exception as e:
            print("  failed:", e)
    return False


def _from_huggingface(dst: str) -> int:
    """Fallback: pull from HF and reshape into the original HotpotQA JSON layout."""
    from datasets import load_dataset
    ds = None
    for repo in ["hotpotqa/hotpot_qa", "hotpot_qa"]:
        try:
            ds = load_dataset(repo, "distractor", split="validation",
                              trust_remote_code=True)
            break
        except Exception as e:
            print(f"  {repo} failed:", e)
    assert ds is not None, "Could not load HotpotQA from Hugging Face."
    recs = []
    for ex in ds:
        ctx = [[t, s] for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])]
        sf = [[t, i] for t, i in zip(ex["supporting_facts"]["title"],
                                     ex["supporting_facts"]["sent_id"])]
        recs.append({"_id": ex["id"], "question": ex["question"], "answer": ex["answer"],
                     "type": ex["type"], "level": ex["level"],
                     "supporting_facts": sf, "context": ctx})
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(recs, f)
    return len(recs)


def main() -> None:
    args = parse_args("Download HotpotQA + create folders")
    cfg, log = boot("setup", args)

    raw_dir = cfg.path("data_raw")
    os.makedirs(raw_dir, exist_ok=True)
    dst = os.path.join(raw_dir, cfg.dataset.raw_filename)

    if os.path.exists(dst):
        log.info(f"Dataset already present: {dst}")
    else:
        ok = _try_direct([cfg.dataset.url,
                          cfg.dataset.url.replace("http://", "https://")], dst)
        if not ok:
            log.info("Direct download unavailable -> Hugging Face fallback")
            n = _from_huggingface(dst)
            log.info(f"Converted {n} questions from Hugging Face.")

    size_mb = os.path.getsize(dst) / 1e6
    with open(dst, "r", encoding="utf-8") as f:
        n = len(json.load(f))
    log.info(f"Dataset ready: {dst} ({size_mb:.1f} MB, {n} questions)")
    log.info(f"Output dirs created under: {cfg.base}")


if __name__ == "__main__":
    main()
