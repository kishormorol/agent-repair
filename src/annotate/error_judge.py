"""Ground-truth error annotation (Stage 3).

Produces, for each FAILED trajectory, the oracle 'broken step' + error type by
combining two signals:

  Signal A (programmatic): using gold supporting-fact titles, check whether/when
    the gold documents were retrieved. Objective for retrieval-type failures.
  Signal B (LLM judge): a strong model (Qwen2.5-72B) reads the question, gold
    answer, gold facts, and the numbered trajectory and names the earliest step
    responsible + an error type.

Also provides human-validation helpers (display formatter + agreement metrics)
so we can report judge<->human reliability on a hand-labeled subset.

Step indices are 0-based everywhere (matching Step.index). The judge prompt uses
1-based 'Step k' for readability; parsing converts back to 0-based.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Signal A — programmatic gold-retrieval check
# --------------------------------------------------------------------------- #
def programmatic_retrieval_check(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Return retrieval diagnostics from the trajectory's tool history.

    Uses each step's 'retrieved_title' (logged by the env) vs gold_titles.
    """
    gold = set(traj.get("gold_titles", []))
    first_step_for_title: Dict[str, int] = {}
    search_steps: List[int] = []
    for s in traj["steps"]:
        if s.get("action") == "search":
            search_steps.append(s["index"])
        rt = s.get("retrieved_title")
        if rt in gold and rt not in first_step_for_title:
            first_step_for_title[rt] = s["index"]
    missing = sorted(gold - set(first_step_for_title.keys()))
    all_retrieved = len(missing) == 0
    # Candidate broken step for a retrieval failure: the last search step
    # (the point by which retrieval should have succeeded but didn't).
    candidate = None
    if not all_retrieved and search_steps:
        candidate = search_steps[-1]
    return {
        "all_gold_retrieved": all_retrieved,
        "missing_titles": missing,
        "first_retrieval_step": first_step_for_title,
        "candidate_broken_step": candidate,
        "is_retrieval_failure": not all_retrieved,
    }


# --------------------------------------------------------------------------- #
# Signal B — LLM judge prompt + parsing
# --------------------------------------------------------------------------- #
def format_trajectory_for_judge(traj: Dict[str, Any]) -> str:
    lines = []
    for s in traj["steps"]:
        k = s["index"] + 1  # 1-based for the judge
        lines.append(f"[Step {k}]")
        lines.append(f"  Thought: {s['thought']}")
        lines.append(f"  Action: {s['action']}")
        lines.append(f"  Action Input: {s['action_input']}")
        lines.append(f"  Observation: {s['observation']}")
    return "\n".join(lines)


def build_judge_prompt(traj: Dict[str, Any], error_types: List[str],
                       retrieval_hint: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """Chat messages asking the judge for the earliest responsible step."""
    sys = (
        "You are an expert annotator of AI agent failures. Given a multi-hop "
        "question, its gold answer, the gold supporting facts, and a FAILED "
        "agent trajectory, identify the SINGLE EARLIEST step that is responsible "
        "for the wrong final answer. Attribute the failure to the earliest step "
        "where the agent first went wrong, even if the wrongness only became "
        "obvious later. Respond with STRICT JSON only."
    )
    n = len(traj["steps"])
    facts = traj.get("gold_supporting_facts_str", "")
    hint = ""
    if retrieval_hint and retrieval_hint.get("is_retrieval_failure"):
        hint = ("\nNote: a programmatic check found the agent never retrieved "
                f"these required pages: {retrieval_hint['missing_titles']}.")
    user = (
        f"QUESTION: {traj['question']}\n"
        f"GOLD ANSWER: {traj['gold_answer']}\n"
        f"GOLD SUPPORTING FACT TITLES: {traj.get('gold_titles', [])}\n"
        f"{('GOLD SUPPORTING FACTS: ' + facts) if facts else ''}\n"
        f"AGENT'S (WRONG) FINAL ANSWER: {traj.get('final_answer')}\n\n"
        f"TRAJECTORY ({n} steps):\n{format_trajectory_for_judge(traj)}{hint}\n\n"
        f"Return STRICT JSON with keys:\n"
        f'  "broken_step": integer from 1 to {n} (the earliest responsible step),\n'
        f'  "error_type": one of {error_types},\n'
        f'  "justification": one short sentence.\n'
        f"JSON:"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def parse_judge_output(text: str, n_steps: int,
                       error_types: List[str]) -> Optional[Dict[str, Any]]:
    """Parse the judge's JSON. Returns dict with 0-based 'broken_step' or None."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    raw = m.group(0) if m else text
    try:
        obj = json.loads(raw)
    except Exception:
        # last-resort regex extraction
        bs = re.search(r'"?broken_step"?\s*[:=]\s*(\d+)', text)
        et = re.search(r'"?error_type"?\s*[:=]\s*"?([a-z_]+)"?', text)
        if not bs:
            return None
        obj = {"broken_step": int(bs.group(1)),
               "error_type": et.group(1) if et else "faulty_reasoning",
               "justification": ""}
    try:
        step1 = int(obj["broken_step"])
    except Exception:
        return None
    step0 = max(0, min(n_steps - 1, step1 - 1))  # to 0-based, clamped
    et = str(obj.get("error_type", "")).strip().lower()
    if et not in error_types:
        et = "faulty_reasoning"
    return {"broken_step": step0, "error_type": et,
            "justification": str(obj.get("justification", "")).strip()}


# --------------------------------------------------------------------------- #
# Combine A + B into a final annotation
# --------------------------------------------------------------------------- #
def combine_annotation(qid: str, judge: Optional[Dict[str, Any]],
                       retrieval: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer the judge; fall back to the programmatic candidate if judge failed."""
    if judge is not None:
        label = {"broken_step": judge["broken_step"],
                 "error_type": judge["error_type"],
                 "justification": judge["justification"], "source": "judge"}
    elif retrieval.get("candidate_broken_step") is not None:
        label = {"broken_step": retrieval["candidate_broken_step"],
                 "error_type": "wrong_search_query",
                 "justification": "programmatic: required page never retrieved",
                 "source": "programmatic"}
    else:
        label = {"broken_step": 0, "error_type": "faulty_reasoning",
                 "justification": "fallback default", "source": "fallback"}
    label["qid"] = qid
    label["retrieval_check"] = retrieval
    return label


# --------------------------------------------------------------------------- #
# Human validation helpers
# --------------------------------------------------------------------------- #
def render_for_human(traj: Dict[str, Any]) -> str:
    """A compact display of a failed trajectory for hand-labeling."""
    head = (f"QUESTION: {traj['question']}\n"
            f"GOLD ANSWER: {traj['gold_answer']}\n"
            f"AGENT'S WRONG ANSWER: {traj.get('final_answer')}\n"
            f"GOLD TITLES: {traj.get('gold_titles', [])}\n"
            f"{'-'*70}")
    return head + "\n" + format_trajectory_for_judge(traj)


def compute_agreement(human: List[Dict[str, Any]],
                      judge: List[Dict[str, Any]]) -> Dict[str, float]:
    """Judge<->human agreement over aligned annotations (same order/qids).

    Returns exact step match, within-1 step match, and error-type Cohen's kappa.
    """
    assert len(human) == len(judge) and len(human) > 0
    exact = within1 = 0
    h_types, j_types = [], []
    for h, j in zip(human, judge):
        hs, js = int(h["broken_step"]), int(j["broken_step"])
        exact += int(hs == js)
        within1 += int(abs(hs - js) <= 1)
        h_types.append(h["error_type"]); j_types.append(j["error_type"])
    out = {"n": len(human), "step_exact": exact / len(human),
           "step_within1": within1 / len(human)}
    try:
        from sklearn.metrics import cohen_kappa_score
        out["error_type_kappa"] = float(cohen_kappa_score(h_types, j_types))
    except Exception:
        out["error_type_kappa"] = float("nan")
    return out
