"""Lossless official-data adapters for the separately frozen extension study."""
from __future__ import annotations

import copy

from .base_env import exact_match, f1_score as squad_f1
from .hotpot_env import HotpotEnv, f1_score as hotpot_f1


def convert_musique(record):
    if record.get("answerable") is not True:
        raise ValueError("The extension uses the answerable MuSiQue-Ans split")
    paragraphs = record["paragraphs"]
    if not paragraphs or len({p["idx"] for p in paragraphs}) != len(paragraphs):
        raise ValueError("MuSiQue requires uniquely indexed paragraphs")
    pages, supporting = {}, []
    for paragraph in paragraphs:
        title, text = paragraph["title"], paragraph["paragraph_text"]
        if not isinstance(title, str) or not isinstance(text, str) or not title or not text:
            raise ValueError("Invalid MuSiQue title or paragraph")
        units = pages.setdefault(title, [])
        if paragraph["is_supporting"]:
            supporting.append([title, len(units)])
        # Preserve punctuation, whitespace, and repeated titles. A retrieval
        # unit is a complete paragraph, never a split on literal periods.
        units.append(text)
    hops = len(record["question_decomposition"])
    return {"_id": record["id"], "question": record["question"], "answer": record["answer"],
            "answer_aliases": list(record["answer_aliases"]), "scoring": "musique",
            "type": f"{hops}hop", "level": f"{hops}hop",
            "context": [[title, units] for title, units in pages.items()],
            "supporting_facts": supporting}


def convert_2wiki(record, aliases):
    result = copy.deepcopy(record)
    entry = aliases.get(record["answer_id"], {})
    result.update(answer_aliases=sorted(set(entry.get("aliases", []) + entry.get("demonyms", []))),
                  scoring="2wikimultihopqa", level="multi")
    return result


def score_record(prediction, record):
    if prediction is None:
        return {"em": 0.0, "f1": 0.0, "correct": False}
    dataset = record.get("scoring", "hotpotqa")
    if dataset not in {"hotpotqa", "musique", "2wikimultihopqa"}:
        raise ValueError("Unknown official answer scorer")
    answers = [record["answer"], *record.get("answer_aliases", [])]
    if any(not isinstance(answer, str) for answer in answers):
        raise ValueError("Answer aliases must be strings")
    scorer = squad_f1 if dataset == "musique" else hotpot_f1
    em = float(max(exact_match(prediction, answer) for answer in answers))
    f1 = float(max(scorer(prediction, answer) for answer in answers))
    return {"em": em, "f1": f1, "correct": bool(em or f1 >= 0.5)}


class ExtensionEnv(HotpotEnv):
    """Common offline tools, with each dataset's official answer/alias rules."""

    def score_answer(self, prediction, ground_truth):
        if ground_truth != self.record["answer"]:
            raise ValueError("Scoring reference differs from the environment record")
        return score_record(prediction, self.record)
