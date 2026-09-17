# Review task — Emam

**Is the statistical argument sound?** Roughly half a day.

Full context, including what the paper studies and why this review exists, is in
[`verification_assignment.md`](verification_assignment.md). Read section 2 and 3
there first if the design is unfamiliar — the rest of this file assumes it.

## The one-paragraph version

An agent fails partway through answering a question. You can restart, or keep
the first *k* steps and regenerate the rest. The paper tests one rule for
choosing *k*: take the step with the highest token perplexity, back up two.
The result is null everywhere. Your job is to decide whether that null is
trustworthy or merely underpowered.

## Setup

```bash
git clone https://github.com/kishormorol/agent-repair.git
cd agent-repair && pip install -r requirements.txt
python -m pytest tests/test_position_pairs.py \
  tests/test_position_matched.py tests/test_run_monitor.py -q   # expect 29 passed
```

Those are the self-contained logic tests. **Do not run the whole suite** — most of
it replays raw experiment records too large to commit, so it fails on a clone by
design. Ask the author if you need those records.

Manuscript: `paper/iclr2027.tex`, built PDF at
`output/pdf/agent-repair-iclr2027-draft.pdf`. The relevant appendix is H.

## What to check

**1. The resolution floor — the most important item.**
An exact two-sided sign-flip test over $m$ non-zero blocks cannot report $p$
below $2/2^{m}$, since only two of the $2^m$ sign assignments are maximally
extreme. The paper concludes that cells with few non-zero blocks *could not have
reached significance whatever the data showed*. **Three of four** swap cells and
two of six replication cells sit above 0.05 on this floor.

- Is the bound correct?
- Is it applied where it belongs?
- **Is it doing more rhetorical work than it should?** "Could not have reached
  significance" can read as excusing a design that simply lacks power. Say so if
  you think it does.

Code: `scripts/paper_stats.py`. Tests: `tests/test_run_monitor.py` is unrelated;
see `tests/test_position_pair_paper.py`.

**2. Ties.** 83 of 102 swap pairs tie. Ties stay in the point estimate and
interval but leave the exact null, since a zero difference has no sign to flip.
Is that coherent, and is the estimand described consistently in both places?

**3. The exact test.** `src/repair/position_pairs.py` implements the null as an
integer dynamic program, checked against brute-force enumeration. The
implementation is probably fine. The question is the **assumption**:
within-block exchangeability. Is it plausible here, and is it described as an
assumption rather than a fact?

**4. Families.** Each study declares its own Holm family: 4, 6, 24 and 2 tests.
Was any family chosen after results were seen? Is a zero-failure cell
contributing $p=1$ defensible, or convenient?

**5. Interval versus test.** Qwen/HotpotQA has a bootstrap interval excluding
zero *and* an exact $p$ of 0.125 — because its floor is 0.125. The paper says
this is not a positive finding. Fair, or explaining away an inconvenient
interval?

**6. The bottom line.** Is this a rigorous null or an underpowered one? If
underpowered, what would fix it? Note that more questions demonstrably would
*not*: the cell with the most pairs (33) shares the highest floor with a
23-pair cell, because its extra pairs were ties.

## Please also record

- What you did **not** check. A named gap is safer than one assumed covered.
- Whether the AI-use statement draft (`docs/ai_use_statement_draft.md`)
  describes the AI involvement accurately, judged against the commit history.
  **Do not sign it** — that is the author's attestation.

## What a useful finding looks like

Not: *"The floor argument seems fine."*

But: *"I verified the $2/2^m$ bound by enumerating all sign assignments for m=4
and m=6; it is correct. But Appendix H uses it to argue the design 'could not
have reached significance', which reads as absolving an underpowering problem
the study does have. I would report it as a power limitation instead. I did not
check the replication cells' floors, only the swap cells'."*

## Reporting

Write to `docs/review_emam_2026-09.md`. Flag anything that would change a claim
as **blocking** and tell the author directly, not only in the file.

**Constraint:** the compute budget is exhausted (~$0.75 credit). No new
experiment can answer a question you raise without a new allocation. Saying so
is a legitimate conclusion.
