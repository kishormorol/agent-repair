"""2WikiMultiHopQA environment.

Task: 2-hop multi-hop QA over Wikipedia paragraphs.

Data source: https://github.com/Alab-NII/2wikimultihop or HuggingFace.

The environment uses the same search/lookup/finish tools as HotpotQA.
The data format is nearly identical: context paragraphs, supporting facts,
answer, and question type (bridge, comparison, etc.).
"""
from __future__ import annotations

import collections
import difflib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_env import BaseEnv, ToolResult, score_answer  # noqa: F401
from ..utils.io import load_json, save_json


# =========================================================================== #
# Data loading
# =========================================================================== #
def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load 2WikiMultiHopQA dataset (pre-processed JSON)."""
    return load_json(path)


def download_2wikimultihop(dst: str) -> int:
    """Download 2WikiMultiHopQA dev set from HuggingFace.

    Reshapes to HotpotQA-compatible format:
    _id, question, answer, context, supporting_facts, type, level.
    """
    from datasets import load_dataset as hf_load

    ds = hf_load("scholarly-shadows-syndicate/2WikiMultihopQA",
                 split="validation", trust_remote_code=True)

    recs = []
    for ex in ds:
        # 2WikiMultiHopQA has context as list of [title, [sentences]]
        context = []
        if "context" in ex and ex["context"]:
            ctx = ex["context"]
            if isinstance(ctx, dict) and "title" in ctx:
                for title, sents in zip(ctx["title"], ctx["sentences"]):
                    context.append([title, sents])
            elif isinstance(ctx, list):
                context = ctx

        sf = []
        if "supporting_facts" in ex and ex["supporting_facts"]:
            sfs = ex["supporting_facts"]
            if isinstance(sfs, dict) and "title" in sfs:
                for title, sid in zip(sfs["title"], sfs["sent_id"]):
                    sf.append([title, sid])
            elif isinstance(sfs, list):
                sf = sfs

        # Determine question type
        q_type = ex.get("type", "bridge")

        recs.append({
            "_id": ex.get("_id", ex.get("id", str(len(recs)))),
            "question": ex["question"],
            "answer": ex["answer"],
            "type": q_type,
            "level": "multi",
            "context": context,
            "supporting_facts": sf,
        })

    save_json(recs, dst)
    return len(recs)


def sample_pool(data: List[Dict[str, Any]], size: int,
                stratify_by: List[str], seed: int) -> List[Dict[str, Any]]:
    """Stratified sample balanced across question types."""
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
# Environment (identical tools to HotpotQA)
# =========================================================================== #
@dataclass
class WikiMultiHopEnv(BaseEnv):
    """Per-question environment for 2WikiMultiHopQA."""
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
        return self.record["question"]

    @property
    def gold_answer(self) -> str:
        return self.record["answer"]

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
