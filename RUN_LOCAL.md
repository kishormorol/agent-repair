# Running Locally (H100) — Full HotpotQA dev (7,405 questions)

The local pipeline is **CLI scripts** (not notebooks), so it can run unattended for
hours in `tmux`/`nohup`. Everything is **batched** (many rollouts on the GPU at
once) and **resumable** (checkpointed after every item — just re-run to continue).

---

## 1. Setup (once)

```bash
git clone <your repo>            # or copy the folder over
cd agent-repair

python -m venv .venv && source .venv/bin/activate      # or conda
pip install -r requirements.txt
```

> **Note on versions.** `requirements.txt` pins `torch==2.4.0` + `vllm==0.6.3.post1`
> + `transformers==4.45.2`, which supports H100 (`sm_90`). If the machine has a
> *newer* GPU (Blackwell / `sm_120`), that stack will NOT work — tell me and I'll
> re-pin to a newer torch/vLLM.

Download the data (auto-falls back to Hugging Face if the CMU host is down):

```bash
python scripts/run_setup.py --config config/config_local.yaml
```

---

## 2. Run the pipeline

All commands use the local profile: `--config config/config_local.yaml`
(full dev = 7,405 questions, batching on, 3 seeds).

```bash
# Stage 1 — generate trajectories        (~1-2 h on H100)
python scripts/run_generate.py    --config config/config_local.yaml

# Stage 2 — step-level uncertainty        (~1-2 h)
python scripts/run_uncertainty.py --config config/config_local.yaml

# Stage 3 — 72B judge annotation          (~1-2 h)
python scripts/run_annotate.py    --config config/config_local.yaml

# Stage 4 — localization scoring          (CPU, minutes)
python scripts/run_localize.py    --config config/config_local.yaml

# Stage 5 — repair, 18 strategies         (~2-4 h; batched + deduplicated)
python scripts/run_repair.py      --config config/config_local.yaml

# Stage 6 — tables, stats, figures        (CPU, minutes)
python scripts/run_eval.py        --config config/config_local.yaml
```

**Total: roughly one overnight run (~6–10 h) for the full 7,405 questions.**

### Run it unattended
```bash
tmux new -s repair
nohup python scripts/run_generate.py --config config/config_local.yaml \
      > outputs/logs/gen.out 2>&1 &
# detach with Ctrl-b d ; reattach with: tmux attach -t repair
```
Progress + ETA are logged to `outputs/logs/`.

---

## 3. The one manual step (do it any time after Stage 3)

Hand-label 50 failed trajectories to validate the judge. Interactive, resumable
(Ctrl-C any time — progress is saved):

```bash
python scripts/label_human.py --config config/config_local.yaml
```

It prints judge↔human agreement at the end and writes
`outputs/tables/judge_human_agreement.json`.

---

## 4. Smoke test first (strongly recommended)

Before committing to the full run, do a tiny pass to confirm the GPU + model work:

```bash
for s in run_generate run_uncertainty run_annotate run_localize run_repair run_eval; do
  python scripts/$s.py --config config/config_local.yaml --limit 20 || break
done
```
`--limit 20` processes only 20 items per stage. If that completes and writes
figures to `outputs/figures/`, remove `--limit` and launch the real run.

---

## 5. If you hit out-of-memory

Lower the batch sizes in `config/config_local.yaml`:

```yaml
runtime:
  gen_batch_size: 32        # from 64
  repair_batch_size: 32     # from 64
  uncertainty_batch_size: 16
  judge_batch_size: 16
```
The 72B judge (Stage 3) is the memory-heaviest; it needs ~40 GB at 4-bit and fits
an 80 GB H100 comfortably. If the GPU is smaller, it auto-falls back to a 32B judge.

---

## What's different from the Colab version

| | Colab | **Local (this)** |
|---|---|---|
| Interface | 7 notebooks | 8 CLI scripts |
| Execution | 1 question at a time | **batched (64 in flight)** |
| Dataset | 500–1,000 sampled | **full dev: 7,405** |
| Seeds | 1 | **3** (proper confidence intervals) |
| Storage | Google Drive | local `outputs/` |
| Stage 1 time | ~3 h for 500 | **~1–2 h for 7,405** |

Outputs land in `outputs/tables/` (8 tables) and `outputs/figures/` (4 figures),
exactly as before.
