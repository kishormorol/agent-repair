"""I/O helpers: atomic JSON/JSONL persistence + resumable checkpointing.

Design goal: a Colab disconnect at ANY moment must never corrupt output and
must let the next run resume where it stopped. We achieve this by:
  * writing to a temp file then os.replace() (atomic on the same filesystem),
  * storing per-item results as one JSON file per id (idempotent, resumable),
  * a lightweight `Checkpoint` that records completed ids in a JSONL log.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


# --------------------------------------------------------------------------- #
# Atomic single-file JSON
# --------------------------------------------------------------------------- #
def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Atomically write `obj` as JSON to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
        os.replace(tmp, path)          # atomic swap
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# JSONL (append-friendly logs)
# --------------------------------------------------------------------------- #
def append_jsonl(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    if not Path(path).exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# Per-item store: one JSON file per id, under a directory. Idempotent.
# --------------------------------------------------------------------------- #
def item_path(dir_: str | Path, item_id: str) -> Path:
    return Path(dir_) / f"{item_id}.json"


def save_item(dir_: str | Path, item_id: str, obj: Any) -> None:
    save_json(obj, item_path(dir_, item_id))


def load_item(dir_: str | Path, item_id: str) -> Any:
    return load_json(item_path(dir_, item_id))


def list_item_ids(dir_: str | Path) -> List[str]:
    d = Path(dir_)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def load_all_items(dir_: str | Path) -> List[Any]:
    return [load_json(p) for p in sorted(Path(dir_).glob("*.json"))]


# --------------------------------------------------------------------------- #
# Checkpoint: track completed ids so a stage can skip + resume.
# --------------------------------------------------------------------------- #
class Checkpoint:
    """Records completed item ids in a JSONL file for resumable processing.

    Usage:
        ckpt = Checkpoint(cfg.path("logs") + "/stage1.jsonl")
        for q in questions:
            if ckpt.is_done(q["id"]):
                continue
            ...do work...
            ckpt.mark_done(q["id"])
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._done: set[str] = set()
        for rec in read_jsonl(self.path):
            self._done.add(rec["id"])

    def is_done(self, item_id: str) -> bool:
        return item_id in self._done

    def mark_done(self, item_id: str, meta: Dict[str, Any] | None = None) -> None:
        if item_id in self._done:
            return
        self._done.add(item_id)
        rec = {"id": item_id}
        if meta:
            rec.update(meta)
        append_jsonl(rec, self.path)

    def __len__(self) -> int:
        return len(self._done)

    def done_ids(self) -> List[str]:
        return sorted(self._done)


def iter_remaining(items: Iterable[Dict[str, Any]], ckpt: Checkpoint,
                   id_key: str = "id") -> Iterator[Dict[str, Any]]:
    """Yield only items not yet marked done in the checkpoint."""
    for it in items:
        if not ckpt.is_done(str(it[id_key])):
            yield it
