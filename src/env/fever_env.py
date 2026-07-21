"""FEVER (Fact Extraction and VERification) environment.

Task: given a claim, search Wikipedia evidence and classify as
SUPPORTS / REFUTES / NOT ENOUGH INFO.

Data source: https://fever.ai or HuggingFace ``fever/fever``.

The environment provides the same search/lookup/finish tools as HotpotQA,
operating over evidence documents shipped with each claim.  The agent's
``finish`` action should output one of the three labels.
"""
from __future__ import annotations

import difflib
import json
import os
import random
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base_env import BaseEnv, ToolResult, score_answer  # noqa: F401
from ..utils.io import load_json, save_json


# =========================================================================== #
# Data loading
# =========================================================================== #
FEVER_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load FEVER dataset (our pre-processed JSON with evidence paragraphs)."""
    return load_json(path)


_FEVER_URLS = [
    "https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl",
    "https://fever.ai/download/fever/shared_task_dev.jsonl",
]


def _download_fever_jsonl(dst_jsonl: str) -> list:
    """Download FEVER shared_task_dev.jsonl from fever.ai / S3."""
    for url in _FEVER_URLS:
        try:
            print(f"  Trying: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode("utf-8")
            records = [json.loads(line) for line in raw.strip().split("\n") if line.strip()]
            # Cache the raw jsonl
            with open(dst_jsonl, "w", encoding="utf-8") as f:
                f.write(raw)
            return records
        except Exception as e:
            print(f"    failed: {e}")
    raise RuntimeError("Could not download FEVER dev set from any source.")


def _fever_jsonl_to_records(raw_records: list) -> list:
    """Convert FEVER shared_task_dev.jsonl rows to our standard format."""
    recs = []
    seen_ids = set()
    for ex in raw_records:
        claim_id = str(ex["id"])
        if claim_id in seen_ids:
            continue
        seen_ids.add(claim_id)

        label = ex.get("label", "NOT ENOUGH INFO")
        claim = ex.get("claim", "")

        # Parse evidence: list of annotation sets, each a list of
        # [annotation_id, evidence_id, wiki_title, sentence_id]
        context = []
        sf = []
        evidence_titles = set()
        for ev_set in (ex.get("evidence") or []):
            if not isinstance(ev_set, list):
                continue
            for ev in ev_set:
                if not isinstance(ev, list) or len(ev) < 4:
                    continue
                title_raw = ev[2]
                sent_id = ev[3]
                if title_raw is None:
                    continue
                title = str(title_raw).replace("_", " ")
                if title and title not in evidence_titles:
                    evidence_titles.add(title)
                    context.append([title, [claim]])
                    if isinstance(sent_id, int) and sent_id >= 0:
                        sf.append([title, sent_id])

        recs.append({
            "_id": claim_id,
            "question": claim,
            "answer": label if label in FEVER_LABELS else "NOT ENOUGH INFO",
            "type": "verification",
            "level": "claim",
            "context": context,
            "supporting_facts": sf,
        })
    return recs


def download_fever(dst: str) -> int:
    """Download FEVER dev set from fever.ai and reshape to our format.

    Each record gets: _id, question (=claim), answer (=label), context
    (list of [title, [sentences]]), supporting_facts, type, level.
    """
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    tmp.close()
    try:
        raw = _download_fever_jsonl(tmp.name)
        recs = _fever_jsonl_to_records(raw)
        save_json(recs, dst)
        return len(recs)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def download_fever_with_evidence(dst: str) -> int:
    """Download FEVER dev set with evidence — same as download_fever.

    The shared_task_dev.jsonl already contains evidence annotations.
    """
    return download_fever(dst)


def sample_pool(data: List[Dict[str, Any]], size: int,
                stratify_by: List[str], seed: int) -> List[Dict[str, Any]]:
    """Stratified sample balanced across label classes."""
    import collections
    rng = random.Random(seed)
    if size >= len(data):
        pool = list(data)
        rng.shuffle(pool)
        return pool

    buckets: Dict[tuple, List] = collections.defaultdict(list)
    for rec in data:
        key = tuple(rec.get(f, "NA") for f in stratify_by)
        buckets[key].append(rec)

    total = len(data)
    chosen = []
    for key, recs in buckets.items():
        rng.shuffle(recs)
        n = max(1, round(size * len(recs) / total))
        chosen.extend(recs[:n])

    rng.shuffle(chosen)
    return chosen[:size]


# =========================================================================== #
# Environment
# =========================================================================== #
@dataclass
class FEVEREnv(BaseEnv):
    """Per-claim environment for FEVER fact verification."""
    record: Dict[str, Any] = field(default_factory=dict)
    max_sentences_return: int = 5

    _titles: List[str] = field(default_factory=list)
    _para_by_title: Dict[str, List[str]] = field(default_factory=dict)
    _current_title: Optional[str] = None
    _lookup_pos: int = 0
    _lookup_keyword: Optional[str] = None
    retrieved_titles: List[str] = field(default_factory=list)

    def __post_init__(self):
        for title, sentences in self.record.get("context", []):
            self._titles.append(title)
            self._para_by_title[title] = sentences

    @property
    def question(self) -> str:
        return self.record["question"]  # = claim

    @property
    def gold_answer(self) -> str:
        return self.record["answer"]  # SUPPORTS / REFUTES / NOT ENOUGH INFO

    @property
    def gold_titles(self) -> List[str]:
        return sorted({t for t, _ in self.record.get("supporting_facts", [])})

    @property
    def qid(self) -> str:
        return self.record["_id"]

    def _best_title(self, query: str) -> Optional[str]:
        q = query.strip().lower()
        for t in self._titles:
            if t.lower() == q:
                return t
        for t in self._titles:
            if q in t.lower() or t.lower() in q:
                return t
        match = difflib.get_close_matches(query, self._titles, n=1, cutoff=0.3)
        return match[0] if match else None

    def search(self, entity: str) -> ToolResult:
        title = self._best_title(entity)
        if title is None:
            return ToolResult(
                observation=f"Could not find any page matching '{entity}'. "
                            f"Try a different search term.",
                retrieved_title=None)
        self._current_title = title
        self._lookup_pos = 0
        self._lookup_keyword = None
        self.retrieved_titles.append(title)
        summary = " ".join(self._para_by_title[title][:self.max_sentences_return])
        return ToolResult(observation=f"[{title}] {summary}", retrieved_title=title)

    def lookup(self, keyword: str) -> ToolResult:
        if self._current_title is None:
            return ToolResult(observation="No page is currently open. Use search first.")
        sentences = self._para_by_title[self._current_title]
        if self._lookup_keyword != keyword:
            self._lookup_keyword = keyword
            self._lookup_pos = 0
        kw = keyword.lower()
        while self._lookup_pos < len(sentences):
            sent = sentences[self._lookup_pos]
            self._lookup_pos += 1
            if kw in sent.lower():
                return ToolResult(observation=f"[{self._current_title}] {sent}",
                                  retrieved_title=self._current_title)
        return ToolResult(
            observation=f"No (more) sentences mentioning '{keyword}' "
                        f"in [{self._current_title}].",
            retrieved_title=self._current_title)

    def finish(self, answer: str) -> ToolResult:
        return ToolResult(observation=f"Episode finished with answer: {answer}",
                          finished=True, answer=answer)

    def step(self, action: str, action_input: str) -> ToolResult:
        action = (action or "").strip().lower()
        if action == "search":
            return self.search(action_input)
        if action == "lookup":
            return self.lookup(action_input)
        if action == "finish":
            return self.finish(action_input)
        return ToolResult(observation=f"Unknown action '{action}'. "
                                      f"Valid actions: search, lookup, finish.",
                          is_tool_call=False)


# =========================================================================== #
# FEVER-specific scoring
# =========================================================================== #
def score_answer_fever(prediction: Optional[str], ground_truth: str) -> Dict[str, float]:
    """FEVER uses label accuracy (exact match on normalized label)."""
    if prediction is None:
        return {"em": 0.0, "f1": 0.0, "correct": False}
    # Normalize prediction to one of the three labels
    pred = prediction.strip().upper()
    gold = ground_truth.strip().upper()
    # Handle common variations
    label_map = {
        "SUPPORTS": "SUPPORTS", "SUPPORT": "SUPPORTS", "TRUE": "SUPPORTS",
        "REFUTES": "REFUTES", "REFUTE": "REFUTES", "FALSE": "REFUTES",
        "NOT ENOUGH INFO": "NOT ENOUGH INFO", "NEI": "NOT ENOUGH INFO",
        "NOT ENOUGH INFORMATION": "NOT ENOUGH INFO",
    }
    pred = label_map.get(pred, pred)
    gold = label_map.get(gold, gold)
    em = int(pred == gold)
    return {"em": float(em), "f1": float(em), "correct": bool(em)}
