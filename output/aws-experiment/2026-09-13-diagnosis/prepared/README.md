# Diagnosis/replay follow-up package

Prepared inputs only. No GPU validation or new study outcomes are included.
The frozen main cohort contains 250 new questions, excludes 799 documented
previous questions, and is split into five batches of 50. The development
check imports five known failed traces. Both use restart, uncertainty with
two-step backtracking, and same-model diagnosis/replay; three seeds; and
0.5x, 1x, and 2x original generated-token allowances. New-step allowance is
eight. The diagnosis policy is an adaptation, not a Doctor-RAG reproduction.

## Verify without inference

From this package directory, using Python's standard library:

```bash
python run_aws_diagnosis_study.py --session . --plan-only
```

The manifest covers every prepared input. Do not edit frozen files. Changed
code, policy or cohort needs a new package directory and run tag. Keep any
execution records separate from these original inputs.

## Execute on the retained instance

Upload the complete package contents to `/workspace/aws119-session/diagnosis-20260913-v1`. The cached model
and existing GPU environment must be available under `/workspace/agent-repair-iclr2027`. The
controller expects one GPU, the pinned model revision and vLLM 0.19.0.

Create `live-checks.json` by copying `live-checks.template.json`. Replace all
null fields with actual AWS observations and set a verification flag true
only after its check succeeds. All timestamp values need explicit timezones.
Observe the resource identity, running state, persistent storage and Stop
shutdown behavior; record current EC2-eligible credits and expiry, current
hourly rate, and cumulative gross allocation usage including unbilled usage.
Record an independent external stop and its deadline. `observed_utc` must
be within ten minutes of launch. `instance_start_utc` is the actual start of
this EC2 billing session, including setup and upload time.

The package limits this session to $25 gross, preserves $20 of the original
$119 allocation, allows at most 240 minutes from instance start and caps
the hourly compute input at $5.84531. That ceiling is a frozen guard, not a
current price quote or a provider billing cap. The external stop deadline
must be no later than `deadline_utc`. The template deliberately cannot pass
the live checks without measured values.

After the launch scope and live values are confirmed, on the instance run:

```bash
/workspace/jupyter-env/bin/python /workspace/aws119-session/diagnosis-20260913-v1/run_aws_diagnosis_study.py --session /workspace/aws119-session/diagnosis-20260913-v1 --checks /workspace/aws119-session/diagnosis-20260913-v1/live-checks.json
```

The controller performs GPU preflight, the development check and its audit,
then all five main batches and the pooled audit. It archives completed and
interrupted batches and schedules a guest shutdown when it exits. A failed
development audit prevents the main cohort from starting. An interrupted
cohort remains incomplete; do not report a smaller confirmatory study.

## Preserve and analyze the outputs

Download this entire session directory, including every `.tar.gz` and
`.tar.gz.sha256`, `live-checks-*.json`, `execution-window.json`, logs, batch
status/audit files, and `pilot/` and `main/` pooled outputs. Verify downloaded
archive hashes before treating retrieval as complete. Independently confirm
EC2 reaches stopped state. Preserve the retained volume and prior results.

`main/pooled-analysis.json` reports six primary contrasts in one Holm family,
twelve exploratory EM/F1 contrasts, nine policy/allowance summaries and
unique repair plus diagnosis usage. Diagnosis cost is charged in full to
each independently evaluated policy attempt and counted once per physical
diagnosis in study totals. Token counts are not a runtime or FLOPs claim.

For local re-auditing, extract `code.zip` into a separate source directory,
extract each main batch archive into its own directory, and use the frozen
`scripts/analyze_diagnosis_followup.py` with `--protocol main/protocol.json`,
`--bundle code.zip`, `--reference hotpot_evaluate_v1.reference.py`, `--output`
and `--runs` followed by all five extracted `run/` directories. Pass absolute
paths when executing outside this package. The analysis requires the full
cohort and the matching frozen source.
