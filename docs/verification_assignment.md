# Independent verification assignment

**Reviewers: Emam and Muna.** Assigned by the author, September 17, 2026.

The split below is by the two things most at risk, not by seniority: whether
the statistical argument holds, and whether the reported claims trace to real
evidence. Swap the assignments freely if they do not match your backgrounds.

## Why this review exists

Every number in the manuscript is generated from audited records, each study is
frozen by checksum before execution, and all four reported cells reproduce with
zero mismatches. Those safeguards are real, and they constrain arithmetic and
provenance tightly.

They do not constrain reasoning. Much of the design, analysis and prose was
produced with AI assistance and then checked largely by the same process that
produced it. **Self-consistency is not correctness.** This review exists to put
independent judgement where none has been applied.

Please record what you did *not* check, as well as what you did. The AI-use
statement needs an honest account of remaining gaps, and an unchecked area that
is named is far less dangerous than one silently assumed covered.

---

## Emam — statistical argument and inference

The claims here are load-bearing and were constructed, not inherited.

1. **The resolution floor.** The paper argues that a contrast with $m$ non-zero
   blocks cannot report a two-sided sign-flip $p$ below $2/2^{m}$, so cells with
   few non-zero blocks could not reach significance whatever the data showed.
   Two of four position-pair cells and two of six replication cells sit above
   0.05 on this floor. Check the claim is true, that it is applied correctly,
   and that it is not doing more rhetorical work than it should.
   See `scripts/paper_stats.py` and Appendix H.
2. **Ties.** Tied pairs stay in the point estimate and interval but leave the
   exact null. Confirm this is coherent and that the estimand is stated
   correctly in both places.
3. **The exact sign-flip test.** `src/repair/position_pairs.py` implements an
   integer dynamic program. A test checks it against brute-force enumeration;
   check that the null it assumes (within-block exchangeability) is the right
   one and is honestly described as an assumption.
4. **Families and adjustment.** Each study declares its own Holm family (4, 6,
   24 and 2 tests). Check no family was chosen after seeing results, and that
   zero-failure cells contributing $p=1$ is defensible.
5. **Intervals versus tests.** Qwen/HotpotQA has a bootstrap interval excluding
   zero and an exact $p$ of 0.125. The paper says this is not a positive
   finding. Check that framing is fair rather than convenient.
6. **The overall null.** With two of four cells unable to detect an effect, is
   this a rigorous null or an underpowered one? This is the question most
   likely to decide the paper's reception.

## Muna — evidence, reproduction and claim tracing

The goal is to find any claim that does not trace to evidence, and any evidence
that does not support its claim.

1. **Reproduce independently.** Extract
   `output/supplement/agent-repair-combined-supplement.zip` somewhere isolated
   and run `python reproduce.py`. It should reproduce the 4-, 6-, 24- and 2-test
   families with no workspace path or cloud credential. Report anything that
   needs local knowledge to work.
2. **Trace claims to sources.** Pick a dozen numbers from the manuscript at
   random and follow each back to the record that produced it. They should all
   come from generated assets. Flag any hand-typed figure.
3. **Check the exact-balance claim directly.** The central methodological claim
   is that the two arms carry identical origin multisets. Verify it from the
   raw trial CSVs rather than from the analyzer that asserts it, since the
   analyzer and the claim share an author.
4. **Cohort and independence.** The two position-pair cells share question
   identifiers. Confirm no pooled statistic is reported anywhere, and that the
   contrast is described as descriptive.
5. **Scope of the supplement.** It covers statistical reproduction only; raw
   replay and GPU regeneration are excluded and labelled. Check the labelling
   is accurate and that nothing implies more.
6. **Earlier work.** The historical appendix and pre-September material were not
   re-verified during recent work. Treat them as unreviewed and say so.

---

## Both reviewers

- **The AI-use statement** (`docs/ai_use_statement_draft.md`) is a draft with
  bracketed attestations. Neither reviewer should sign it; the author does. But
  tell the author whether its account of scope matches what you find in the
  repository and commit history.
- **Known errors caught during development** are listed at the end of that
  draft. They indicate where mistakes have historically occurred and may be
  worth probing.
- **Budget is exhausted** (~$0.75 of eligible credit). No new experiment can be
  run to answer a question raised by this review without a new allocation.

## Reporting

Please write findings to `docs/review_<name>_2026-09.md` with, for each item:
what you checked, how, what you found, and what you did not check. Verdicts
without method are hard to act on.
