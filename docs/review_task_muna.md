# Review task — Muna

**Do the claims trace to evidence?** Roughly half a day.

Full context, including what the paper studies and why this review exists, is in
[`verification_assignment.md`](verification_assignment.md). Read sections 2 and
3 there first if the design is unfamiliar.

## The one-paragraph version

Every number in this paper is supposed to be machine-generated from audited
records rather than typed by hand, and every study is supposed to reproduce from
a self-contained bundle. Your job is to find any claim that does not trace to
evidence, and any evidence that does not support its claim.

## Setup

```bash
git clone https://github.com/kishormorol/agent-repair.git
cd agent-repair && pip install -r requirements.txt
python -m pytest -q            # expect 415 passed
```

## What to check

**1. Reproduce independently.**

```bash
mkdir /tmp/review && cd /tmp/review
unzip ~/agent-repair/output/supplement/agent-repair-combined-supplement.zip
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

It recomputes each contrast's means and Holm adjustment from the packaged CSVs.
**Report anything that needs a workspace path, local knowledge or a
credential** — it should need none.

**2. Trace claims to sources.** Pick a dozen numbers from the manuscript at
random and follow each back to the record that produced it. All should come from
`paper/generated/`, which is machine-written. **Flag any hand-typed figure** —
that is exactly the failure the build is designed to prevent. (One such error
was already caught this way: a count of "two of four" cells that was actually
three.)

**3. Check exact balance yourself — the key item.**
The central methodological claim is that both arms use identical multisets of
*k*. **Verify it from the raw CSV, not from the analyzer that asserts it** — the
analyzer and the claim share an author.

```python
import pandas as pd
for cell in ["position_pairs_qwen", "position_pairs_mistral"]:
    t = pd.read_csv(f"{cell}/trials.csv")
    for ds, sub in t.groupby("dataset"):
        a = sorted(sub[sub.strategy == "unc__perplexity__argmax__bt2"].origin)
        b = sorted(sub[sub.strategy == "length_matched_swap"].origin)
        print(cell, ds, a == b)
```

All four should print `True`.

**4. Independence.** The two swap cells use the **same questions**, so their
results are not independent. Confirm no pooled statistic appears anywhere and
that the cross-model contrast is described as descriptive only.

**5. Scope honesty.** The supplement reproduces statistics only; raw replay and
GPU regeneration are excluded and labelled. Check the labelling is accurate and
that nothing elsewhere implies more was verified than was.

**6. Earlier work.** The historical appendix and pre-September material were
**not** re-verified alongside the recent work. Treat them as unreviewed and say
so explicitly rather than letting silence imply coverage.

## Please also record

- What you did **not** check.
- Whether the AI-use statement draft (`docs/ai_use_statement_draft.md`)
  describes the AI involvement accurately, judged against the commit history.
  **Do not sign it** — that is the author's attestation.

## What a useful finding looks like

Not: *"Reproduction worked."*

But: *"reproduce.py passed with the expected counts. I traced twelve numbers;
eleven came from paper/generated. The twelfth, the '3--23 questions per
contrast' range in the replication section, I could not locate a generator for
and it may be hand-typed. I did not check the historical appendix at all."*

## Reporting

Write to `docs/review_muna_2026-09.md`. Flag anything that would change a claim
as **blocking** and tell the author directly, not only in the file.

**Note:** the full raw trajectories are not in the repository, only per-trial
exports. If you need raw records to settle something, ask the author rather than
assuming they are unavailable.
