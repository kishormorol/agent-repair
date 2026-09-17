# Gap 1: replication position-control balance

## Resolution and remaining limit

The paper now reports the realized positional balance in every replication
cell. The comparator is called **development-fitted random**, retaining the
frozen `position_matched_random` identifier. Its development fit does not
guarantee a matching distribution on main failures.

This addresses the current-paper reporting and interpretation gap. It does
not remove the experimental imbalance. Stronger isolation of uncertainty
information from position would require more development failures and a new
frozen evaluation with declared position/length diagnostics. The completed
control and all primary results remain unchanged.

## Measured balance

`U` denotes the uncertainty policy; `C` denotes development-fitted random.
Each cell has 20 development questions, but only their failures enter the
fit. Main percentages give equal weight to each failed question and then
its three seeds. These are descriptive post-results diagnostics.

| Cell | Development failures | Main failures | Origin zero U / C (%) | Mean retained steps U / C | Maximum CDF gap | Shared executions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen / HotpotQA | 6 | 43 | 62.79 / 82.17 | 0.91 / 0.25 | 0.194 | 65/129 |
| Qwen / MuSiQue | 16 | 72 | 33.33 / 29.63 | 1.65 / 1.75 | 0.125 | 48/216 |
| Qwen / 2Wiki | 6 | 28 | 60.71 / 86.90 | 1.07 / 0.49 | 0.262 | 45/84 |
| Mistral / HotpotQA | 13 | 78 | 67.95 / 75.21 | 0.59 / 0.43 | 0.085 | 123/234 |
| Mistral / MuSiQue | 19 | 83 | 54.22 / 48.59 | 1.10 / 1.00 | 0.129 | 82/249 |
| Mistral / 2Wiki | 11 | 67 | 59.70 / 53.23 | 0.88 / 0.86 | 0.129 | 87/201 |

The origin `k` retains `k` original steps. The normalized origin is
`k / max(1, T - 1)`, where `T` is that question's original trajectory length.
Normalization occurs before averaging; a one-step trace has origin zero.
The maximum CDF gap is the largest absolute difference between the two
empirical cumulative distributions on this normalized scale. No
significance test or pass/fail balance threshold is attached to it.

Qwen/HotpotQA and Qwen/2Wiki each fit only six failed development traces,
five selecting zero. Their control-minus-treatment origin-zero gaps are
19.38 and 26.19 percentage points. The full curves also expose differences
away from zero: Qwen/MuSiQue has a 3.70-point zero-frequency difference but
a maximum cumulative gap of 0.125.

Shared executions require identical question, seed, origin, prompt and token
allowance. Shared seed pairs range from 22.2% to 53.6%; they are not
additional independent observations. The raw joins also retain counts of
questions sharing all seeds or differing in at least one seed.

## Reproduction and assets

Run from the repository root:

```sh
python scripts/build_extension_paper.py
make -C paper iclr-draft
```

The first command checks the existing independent local/remote reproduction,
then verifies original and uncertainty-record identities/checksums,
recomputes uncertainty from stored token probabilities, rebuilds each
development-only profile, and checks every displayed origin and execution
against the frozen policy. It performs no model generation. It reads 1,453
hashed inputs for the additional position diagnostics, including the
reproduced trial export, and writes 2,226 joined policy rows.

- [Diagnostic implementation](../scripts/extension_position_balance.py)
- [Paper asset builder](../scripts/build_extension_paper.py)
- [Summary table](../paper/generated/tables/iclr2027_extension_position_balance.tex)
- [Full normalized-origin curves](../paper/generated/figures/iclr2027_extension_position_ecdf.pdf)
- [Complete support counts and raw-source hashes](../paper/generated/iclr2027_extension_position_balance.json)
- [Row-level positions and execution identities](../paper/generated/iclr2027_extension_position_rows.csv)
- [Generated asset checksums](../paper/generated/iclr2027_extension_provenance.json)
- [Reporting tests](../tests/test_extension_paper.py)

The manuscript presents these in the replication section, limitations and
the appendix subsection **Realized Position Balance**. These exploratory
diagnostics leave the frozen 24-comparison family, outcomes and inference
unchanged. Gaps 2–6 from [the original review](paper_gap_review_2026-09-15.md)
remain queued for the user's one-at-a-time workflow.

## Verification

All 91 relevant reporting, position-control and documentation tests passed;
the 22 documentation tests also passed after final text edits. A separate
numeric check reproduced the six origin-zero counts, means, full-distribution
distances and shared-execution counts from the row-level export. Frozen
primary-input hashes still match the independent reproduction record.
The main text remains nine pages. Rendered PDF pages were inspected for
readability, clipping and table/figure layout; QA artifacts are under
`tmp/pdfs/position-balance-20260916/`.
