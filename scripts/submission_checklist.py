"""Pre-submission gate for the ICLR AI-use disclosure.

The policy at https://iclr.cc/Conferences/2027/AIPolicyForAuthors requires a
mandatory in-paper disclosure section, and warns that a substantial falsehood
or misrepresentation produced by an LLM "might lead to desk rejection of the
paper". Two failure modes are cheap to check and expensive to miss: shipping
the statement with its author attestations still unwritten, and a statement
that omits a use the policy explicitly requires disclosing.

This is deliberately not part of the always-green test suite. The unfilled
brackets are correct until an author fills them, so a red suite would train
everyone to ignore it. Run this before submitting.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATEMENT = ROOT / "paper/ai_use_statement.tex"
MANUSCRIPT = ROOT / "paper/iclr2027.tex"

# Uses the policy lists as requiring disclosure, each paired with wording that
# shows this statement addresses it.
REQUIRED_USES = {
    "designing research methodology or experiments": "design of the length-matched swap control",
    "formulating mathematical claims": "formulation of the resolution-floor claim",
    "assisting with proofs": "brute-force enumeration used to check it",
    "proposing or refining hypotheses": "proposing and refining the hypotheses",
    "implementing methods": "implementation of the experiment",
    "dataset cleaning and reformatting": "reformatting and cleaning of per-trial records",
    "interpreting results": "interpretation of results",
    "generating synthetic datasets": "No synthetic dataset was generated",
    "literature analysis": "literature lookup and related-work analysis",
    "figure generation": "generation of the figures",
    "paper drafting": "drafting and revision of manuscript text",
}


def problems():
    """Yield one message per reason this submission is not ready."""
    if not STATEMENT.exists():
        yield f"{STATEMENT.relative_to(ROOT)} is missing entirely"
        return
    # The statement is hard-wrapped, so every evidence phrase spans line
    # breaks. Compare against a whitespace-normalised copy, never the raw text.
    raw = STATEMENT.read_text()
    text = " ".join(raw.split())

    if r"\input{ai_use_statement}" not in MANUSCRIPT.read_text():
        yield "the manuscript does not include the statement; the section would not appear"

    for bracket in re.findall(r"\[AUTHORS:.*?\]", raw, re.S):
        opening = " ".join(bracket.split())[:72]
        yield f"an author attestation is still unwritten: {opening}..."

    for use, evidence in sorted(REQUIRED_USES.items()):
        if evidence not in text:
            yield f"the statement no longer covers a disclosable use: {use}"

    if "take full responsibility" not in text:
        yield "the statement no longer accepts responsibility for AI-introduced errors"


def main():
    found = list(problems())
    if not found:
        print("AI-use disclosure: ready to submit.")
        return 0
    print("AI-use disclosure is NOT ready to submit:\n", file=sys.stderr)
    for item in found:
        print(f"  - {item}", file=sys.stderr)
    print("\nThe bracketed attestations are the author's to write. Nobody else "
          "can state what you verified.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
