# Independent verification — brief for Emam and Muna

Assigned by the author, September 17, 2026. Written to be read cold: no prior
familiarity with the project is assumed.

---

## 1. What you are being asked to do

Decide whether the scientific argument in this paper holds, and whether its
claims trace to real evidence. You are **not** being asked to check arithmetic
— that is already machine-enforced and would waste your time. You are being
asked to apply judgement that no automated check can supply.

Expect roughly half a day each.

## 2. What the paper studies, in plain terms

A language-model agent answers a question by taking steps: search, look
something up, answer. Sometimes it fails. You can either **restart** the whole
attempt, or **keep the first k steps and regenerate the rest**. The question is
how to choose k.

The paper tests one specific rule: take the step where the model's own
token-level perplexity was highest, back up two steps, and regenerate from
there. Call this the *uncertainty policy*. The comparison is against restarting,
and against picking k at random.

**The result is null.** Across every study, the uncertainty policy shows no
clear advantage. The paper's contribution is that this null is carefully
controlled, not that a method works.

## 3. The design that matters most

An obvious objection to the earlier studies: maybe the uncertainty policy just
picks *earlier or later* values of k than the control, and any difference is
positional rather than informational. The earlier control tried to match
positions by fitting a distribution — imperfectly, as the paper measures.

The newest experiment removes that objection by construction:

> Take two failed questions whose original trajectories have **exactly the same
> length**. Pair them. Give each question **its partner's** uncertainty-chosen
> k. Now both arms use the identical multiset of k values — one arm just got
> them from the wrong question.

If uncertainty carries information beyond position, the real assignment should
beat the swapped one. It does not, in any of four cells (two models × two
datasets). **This is the central claim to scrutinise.**

## 4. Why an independent review

Every number is generated from audited records, each study is frozen by
checksum before execution, and all four cells reproduce with zero mismatches.
Those safeguards are real and they constrain arithmetic and provenance tightly.

They do not constrain reasoning. Much of the design, analysis and prose was
produced with AI assistance and then checked largely by the same process that
produced it. **Self-consistency is not correctness.** Errors that reached
execution during development are listed in
`docs/ai_use_statement_draft.md`; they indicate where mistakes have actually
occurred.

Please record what you did **not** check as well as what you did. A named gap
is far safer than one silently assumed covered.

## 5. Setup

```bash
git clone https://github.com/kishormorol/agent-repair.git
cd agent-repair
pip install -r requirements.txt      # numpy, pandas, pytest, matplotlib
python -m pytest -q                  # expect: 413 passed
```

The evidence you need is committed at
`output/supplement/agent-repair-combined-supplement.zip` (1.1 MB). It holds the
per-trial exports and reference analyses for every study, including both swap cells. Nothing in it
needs cloud access.

The manuscript is `paper/iclr2027.tex`; a built PDF is at
`output/pdf/agent-repair-iclr2027-draft.pdf`. The experiment is described in
Appendix H.

> **Note:** the full raw trajectories are *not* in the repository — only the
> per-trial exports. If you need raw records to settle something, ask the
> author rather than assuming they are unavailable.

---

## 6. Emam — is the statistical argument sound?

These claims were constructed for this paper, not inherited from established
practice. That is why they need you.

**6.1 The resolution floor.** The paper argues that an exact two-sided
sign-flip test over $m$ non-zero blocks cannot report $p$ below $2/2^{m}$,
because only two of the $2^m$ sign assignments are as extreme as possible. It
concludes that cells with few non-zero blocks *could not have reached
significance whatever the data showed*. Two of four position-pair cells and two
of six replication cells sit above 0.05 on this floor.

Check: is the bound correct? Is it applied where it belongs? And is it doing
more rhetorical work than it should — does "could not have reached
significance" excuse a design that simply lacks power?
Code: `scripts/paper_stats.py`. Reported in Appendix H.

**6.2 Ties.** Most pairs tie (e.g. 29 of 33 in one cell). Ties stay in the
point estimate and interval but drop out of the exact null, since a zero
difference has no sign to flip. Confirm this is coherent and that the estimand
is described consistently in both places.

**6.3 The exact test.** `src/repair/position_pairs.py` implements the sign-flip
null as an integer dynamic program, checked against brute-force enumeration in
`tests/test_position_pairs.py`. The implementation is likely fine. The question
is the assumption: within-block exchangeability. Is it plausible here, and is
it described honestly as an assumption rather than a fact?

**6.4 Families and adjustment.** Each study declares its own Holm family: 4, 6,
24 and 2 tests. Check that no family was chosen after results were seen, and
that zero-failure cells contributing $p=1$ is defensible rather than convenient.

**6.5 Interval versus test.** One cell (Qwen/HotpotQA) has a bootstrap interval
excluding zero *and* an exact $p$ of 0.125. The paper says this is not a
positive finding, because the floor for that cell is 0.125. Is that framing
fair, or is it explaining away an inconvenient interval?

**6.6 The bottom line.** With several cells unable to detect an effect, is this
a rigorous null or an underpowered one? If underpowered, what would fix it —
more questions, a different outcome measure, or a narrower claim? **This is the
question most likely to decide how the paper is received.**

---

## 7. Muna — do the claims trace to evidence?

The goal is to find any claim unsupported by evidence, or any evidence that
does not support its claim.

**7.1 Reproduce independently.**

```bash
mkdir /tmp/review && cd /tmp/review
unzip <repo>/output/supplement/agent-repair-combined-supplement.zip
cd combined-supplement
pip install -r requirements.txt
python reproduce.py
```

Expect `"status": "passed"` and exactly:

| study | primary tests | trial rows |
| --- | --- | --- |
| main | 4 | 2,034 |
| diagnosis | 6 | 3,186 |
| replication | 24 | 8,016 |
| position_pairs_qwen | 2 | 738 |
| position_pairs_mistral | 2 | 1,098 |

It recomputes each contrast's condition means and Holm adjustment from the
packaged CSVs. Report anything that needs local knowledge, a workspace path or
a credential — it should need none.

**7.2 Trace claims to sources.** Pick a dozen numbers from the manuscript at
random and follow each back to the record that produced it. All should come
from `paper/generated/`, which is machine-written. **Flag any hand-typed
figure** — that is precisely the failure mode the build is designed to prevent.

**7.3 Check exact balance yourself.** The central methodological claim is that
both arms carry identical multisets of k. Verify it from the raw trial CSV, not
from the analyzer that asserts it — **the analyzer and the claim share an
author.** In each of `position_pairs_qwen/trials.csv` and
`position_pairs_mistral/trials.csv`, group by dataset and compare the sorted
`origin` values for strategy `unc__perplexity__argmax__bt2` against
`length_matched_swap`. All four comparisons should match exactly.

**7.4 Independence.** The two position-pair cells use the *same* questions, so
their results are not independent. Confirm no pooled statistic appears anywhere
and that the cross-model contrast is described as descriptive only.

**7.5 Scope honesty.** The supplement reproduces statistics only; raw replay
and GPU regeneration are excluded and labelled as such. Check the labelling is
accurate and that nothing elsewhere implies more was verified than was.

**7.6 Earlier work.** The historical appendix and pre-September material were
not re-verified during recent work. Treat them as unreviewed and say so
explicitly, rather than letting silence imply coverage.

---

## 8. What a useful finding looks like

Not: *"The resolution floor argument seems fine."*

But: *"I confirmed the $2/2^m$ bound by enumerating all sign assignments for
m=4 and m=6. It is correct. However, Appendix H uses it to argue the design
'could not have reached significance', which reads as absolving the design of
an underpowering problem it does have. I would report the floor as a power
limitation rather than as a property that excuses the null. I did not check the
replication cells' floors, only the position-pair ones."*

State what you checked, how, what you found, and what you left. A verdict
without a method is hard to act on.

## 9. Reporting

Write to `docs/review_emam_2026-09.md` and `docs/review_muna_2026-09.md`.

Flag anything that would change a claim in the paper as **blocking**, and say
so to the author directly rather than only in the file.

## 10. Two constraints worth knowing

- **The compute budget is exhausted** (about $0.75 of eligible credit). No new
  experiment can answer a question this review raises without a new allocation
  decision. Say so if that is your conclusion — it is a legitimate finding.
- **Do not sign the AI-use statement.** `docs/ai_use_statement_draft.md` has
  bracketed attestations only the author can make. But do tell the author
  whether its account of AI involvement matches what you find in the repository
  and its commit history.
