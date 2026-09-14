"""Build the portable ICLR notebook and an allowlisted snapshot of the current code."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import textwrap
import tokenize
import zipfile
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "agent-repair-iclr2027-code.zip"


def build_bundle(root, destination):
    root, destination = Path(root), Path(destination)
    paths = [*root.glob("src/**/*.py"), *root.glob("scripts/*.py"),
             *root.glob("config/*.yaml"), *root.glob("tests/*.py"),
             *root.glob("pytest.ini"),
             root / "requirements.txt", root / "requirements-analysis.txt"]
    contents = {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(paths)}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    contents["bundle_manifest.json"] = json.dumps({"files": hashes, "code_sha256": digest},
                                                  sort_keys=True, indent=2).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 8, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return digest


def notebook(code_sha256):
    cells = []

    def md(source):
        cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source, tag=None):
        # Dedent notebook code without changing indentation inside embedded scripts.
        lines = source.strip("\n").splitlines()
        margin = len(lines[0]) - len(lines[0].lstrip())
        embedded = set()
        for token in tokenize.generate_tokens(io.StringIO("\n".join(lines)).readline):
            if token.type == tokenize.STRING:
                embedded.update(range(token.start[0] + 1, token.end[0]))
        source = "\n".join(line[margin:] if index not in embedded and line.startswith(" " * margin)
                           else line for index, line in enumerate(lines, 1))
        cell = nbformat.v4.new_code_cell(source.strip())
        if tag:
            cell.metadata["tags"] = [tag]
        cells.append(cell)

    md("""
    # Agent Repair: Controlled ICLR Experiments

    **Inference and evaluation, not training or fine-tuning.** This notebook uses
    the corrected local code snapshot, not the older GitHub notebook pipeline.
    Upload this notebook and `agent-repair-iclr2027-code.zip` into the same Jupyter
    folder. No repository push is needed. Existing runs are never deleted.

    **$119 AWS allocation:** one 80GB-class GPU, one model, one token budget, four
    default conditions, three seeds, no all-origin sweep or extra offset diagnostics.
    Supplying a frozen development position profile adds the position-matched
    random and uncertainty-without-backtracking controls, for six conditions.
    Keep the three QA datasets and run them sequentially. Extra datasets, model
    families and the 72B judge are deferred. This is a reduced study, not evidence
    that the original full experimental plan will finish within $119. Keep
    $20 in reserve and at most $99 for compute. Do not add the cash fallback
    budget to this allocation or count credit-covered usage as zero cost.

    Recommended first run: Linux x86-64, Python 3.10-3.13,
    persistent storage with at least 100 GB free (200 GB allocated is a starting
    point, not a guarantee for every study). Run one dataset per GPU process.
    A dedicated instance is more predictable than an availability-limited Colab
    runtime. No cloud instance is provisioned by this notebook.

    **Start with `EXECUTE_GPU = False` to check the package.** Then select the GPU
    runtime, confirm persistent storage, set it to `True`, and run cells in order.
    GPU rental continues to accrue while an instance is idle. Neither the budget
    worksheet nor the repair-count guard enforces a provider spending limit.
    AWS credits are not a spending cap: usage beyond eligible credits and taxes
    can reach your payment method. Verify credit eligibility, remaining balance,
    regional GPU quota and an independent stop before execution. AWS Budgets
    alerts can lag; they are not a hard stop. Back up results before stopping.

    The defaults are a development pilot, not publishable confirmatory results.
    Human annotation and the close-method/full-cost comparison are separate work.
    """)
    code('''
        from pathlib import Path
        import os

        EXECUTE_GPU = False
        PLATFORM = "aws"  # aws, runpod, colab, or other Linux Jupyter instance
        DATASET = "hotpotqa"  # hotpotqa, musique, 2wikimultihopqa
        PHASE = "pilot"       # pilot, development, test
        RUN_ID = "aws119-pilot-v1"  # new ID for changed study settings
        POOL_SIZE = 10        # initial questions, NOT number of failed trajectories
        BATCH_SIZE = 2        # benchmark 2/4/8 on separate pilots before freezing
        STRATEGY = "unc__perplexity__argmax__bt2"  # pilot candidate, not a selected winner
        ORIGIN_SWEEP = False
        INCLUDE_DIAGNOSTICS = False  # four core conditions only
        POSITION_PROFILE = None  # frozen development profile adds position-only and no-backtrack controls
        MULTIPLIERS = [1.0]
        MAX_REPAIR_EXECUTIONS = 120  # 10 questions * 4 conditions * 3 seeds, before reuse
        GPU_INDEX = "0"

        MODEL_ID = "Qwen/Qwen2.5-32B-Instruct-AWQ"
        MODEL_REVISION = None  # first pilot resolves a commit; copy it to every later run
        TEST_IDS = None        # absolute path to a JSON list, only for PHASE="test"
        DEVELOPMENT_IDS = None # frozen new-question list for PHASE="development"
        EXPLORED_IDS = None    # every previously explored/pilot/development question ID
        FROZEN_POLICY = None   # completed study-policy JSON, frozen BEFORE viewing test outcomes

        CODE_BUNDLE = Path("agent-repair-iclr2027-code.zip").resolve()
        STORAGE_ROOT = Path("/workspace/agent-repair-iclr2027")
        if PLATFORM == "colab":
            STORAGE_ROOT = Path("/content/drive/MyDrive/agent-repair-iclr2027")
        # Only set True after checking the provider's persistent volume/mount.
        PERSISTENT_STORAGE_CONFIRMED = False
        TOTAL_BUDGET_USD = 119.0
        NONCOMPUTE_RESERVE_USD = 20.0  # storage and contingency; tax may be charged separately
        SPENT_SO_FAR_USD = 0.0        # gross usage since allocating $119, BEFORE credit offsets
        SESSION_ALLOWANCE_USD = 10.0 # allocation only, NOT an automatic shutdown
        GPU_HOURLY_USD = None        # enter the live EC2 On-Demand quote, not Capacity Blocks
        BILLING_RATE_CONFIRMED = False
        AWS_CREDITS_CONFIRMED = False  # current eligible balance, expiry, coverage and other usage
        PROVIDER_BUDGET_CONTROLS_CONFIRMED = False  # independent stop; alerts alone are insufficient
        PLANNED_FAILED_QUESTIONS = None  # use development variance to set the study size
    ''', "parameters")
    md("""
    ## 1. Verify and Extract the Corrected Code

    For Colab, upload both files before this cell and select a GPU runtime.
    Drive mounting only occurs when `EXECUTE_GPU` is enabled. Store results on
    persistent storage, not the runtime's temporary disk. On Runpod, verify the
    volume attached at `/workspace`; a directory name alone does not prove persistence.
    On AWS use EBS-backed storage, not the instance's ephemeral NVMe disk.
    """)
    code('''
        import hashlib
        import json
        import sys
        import zipfile

        EXPECTED_CODE_SHA256 = "__CODE_SHA256__"
        if PLATFORM == "colab" and EXECUTE_GPU:
            from google.colab import drive
            drive.mount("/content/drive")
        # A plan-only check stays in the current directory and does not mount cloud storage.
        SOURCE_ROOT = STORAGE_ROOT if EXECUTE_GPU else Path.cwd() / ".iclr-notebook-check"
        REPO = SOURCE_ROOT / "code" / EXPECTED_CODE_SHA256[:12]
        if not CODE_BUNDLE.is_file():
            raise FileNotFoundError(f"Upload the matching code bundle: {CODE_BUNDLE}")
        with zipfile.ZipFile(CODE_BUNDLE) as archive:
            manifest = json.loads(archive.read("bundle_manifest.json"))
            hashes = manifest["files"]
            actual = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
            if actual != EXPECTED_CODE_SHA256:
                raise ValueError("Notebook and code bundle do not match; use the pair built together")
            if len(archive.namelist()) != len(hashes) + 1 or set(archive.namelist()) != {*hashes, "bundle_manifest.json"}:
                raise ValueError("Unexpected or duplicate bundle entries")
            verified = {}
            for name, expected in hashes.items():
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts or "\\\\" in name:
                    raise ValueError("Unsafe archive path")
                data = archive.read(name)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(f"Corrupt source file: {name}")
                target = REPO / relative
                if target.is_symlink() or not target.resolve().is_relative_to(REPO.resolve()):
                    raise ValueError("Extraction would leave the source directory")
                if target.exists() and target.read_bytes() != data:
                    raise ValueError(f"Extracted code changed: {target}; do not alter an active run")
                verified[target] = data
        for target, data in verified.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(data)
        print(f"Verified {len(hashes)} files. Code snapshot: {EXPECTED_CODE_SHA256}")
        print(f"Source: {REPO}")
        print(f"{PHASE}: {DATASET}, {POOL_SIZE} initial questions, {STRATEGY}")
        print("No experiments have run in this cell.")
    '''.replace("__CODE_SHA256__", code_sha256), "bootstrap")
    md("""
    ## 2. GPU and Environment Preflight

    The worksheet leaves $20 uncommitted and at most $99 for all compute,
    including installation, downloads, idle time, development and test execution.
    A session allowance applies from the provider's billing start, not from this
    cell. Subtract time already billed when arranging the provider stop. Enter
    gross usage since allocating $119, before credits, including other account
    workloads consuming the same credit pool. A $0 invoice after credits does
    not mean $0 compute use. The notebook cannot read your AWS account.
    Confirm the current rate and provider controls before enabling GPU work.

    A separate Python environment keeps vLLM out of the notebook kernel. The
    entry-point version is pinned to 0.19.0; resolved dependencies are recorded
    and checked on resume. This is not a claimed reproduction of the old environment.
    The installed driver must support the selected wheel stack. No GPU model is
    silently replaced with a smaller one if the machine is unsuitable.

    On dependency errors, inspect the complete exception and the
    [vLLM installation guide](https://docs.vllm.ai/en/v0.19.0/getting_started/installation/gpu/).
    Do not layer incompatible CUDA/PyTorch wheels into an existing environment.
    """)
    code('''
        import platform
        import runpy
        import shutil
        import subprocess

        budget_allocation = runpy.run_path(str(REPO / "scripts/estimate_budget.py"))["budget_allocation"]
        BUDGET = None
        if GPU_HOURLY_USD is None:
            if EXECUTE_GPU:
                raise ValueError("Enter the live quote in GPU_HOURLY_USD before GPU execution")
            print("QUOTE REQUIRED: no hourly cost or runtime allowance is estimated.")
        else:
            BUDGET = budget_allocation(total_usd=TOTAL_BUDGET_USD, spent_usd=SPENT_SO_FAR_USD,
                                       reserve_usd=NONCOMPUTE_RESERVE_USD,
                                       hourly_usd=GPU_HOURLY_USD, session_usd=SESSION_ALLOWANCE_USD)
            print(json.dumps(BUDGET, indent=2))
        print("Planning only. This notebook does not stop the instance or enforce cloud billing.")
        ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_INDEX, PYTHONUNBUFFERED="1")
        ENV["HF_HOME"] = str(STORAGE_ROOT / "huggingface")
        VENV = (Path("/content/agent-repair-venv") if PLATFORM == "colab"
                else STORAGE_ROOT / "venv-vllm019")
        PYTHON = VENV / "bin/python"
        if EXECUTE_GPU:
            if not BILLING_RATE_CONFIRMED or not PROVIDER_BUDGET_CONTROLS_CONFIRMED:
                raise ValueError("Confirm the live quote, funding allocation and independent stop first")
            if PLATFORM == "aws" and not AWS_CREDITS_CONFIRMED:
                raise ValueError("Confirm current AWS credit eligibility, balance, expiry and other account usage")
            if not PERSISTENT_STORAGE_CONFIRMED:
                raise ValueError("Verify your persistent mount, then set PERSISTENT_STORAGE_CONFIRMED=True")
            if platform.system() != "Linux" or platform.machine() != "x86_64":
                raise RuntimeError("This environment recipe targets Linux x86-64 NVIDIA instances")
            if not (3, 10) <= sys.version_info[:2] <= (3, 13):
                raise RuntimeError("Use a Python 3.10-3.13 notebook kernel")
            subprocess.run(["nvidia-smi", "-i", GPU_INDEX, "--query-gpu=name,memory.total,driver_version",
                            "--format=csv,noheader"], check=True)
            STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(STORAGE_ROOT).free < 100 * 1024**3 and not (STORAGE_ROOT / "huggingface").exists():
                raise RuntimeError("Provision at least 100 GiB free for model cache and initial results")
            if not PYTHON.exists():
                subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
            installed = subprocess.run([str(PYTHON), "-c", "import vllm; assert vllm.__version__ == '0.19.0'"],
                                       capture_output=True, text=True)
            if installed.returncode:
                subprocess.run([str(PYTHON), "-m", "pip", "install", "-r", str(REPO / "requirements.txt"),
                                "vllm==0.19.0", "pytest"], check=True)
            subprocess.run([str(PYTHON), "-m", "pip", "check"], check=True)
            subprocess.run([str(PYTHON), "-c",
                            "import torch,vllm; assert torch.cuda.is_available(); "
                            "g=torch.cuda.get_device_properties(0); print(g, torch.__version__, vllm.__version__); "
                            "assert g.total_memory >= 70 * 1024**3, 'Use the intended 80GB-class GPU for this pilot'"],
                           check=True, env=ENV)
        else:
            print("PLAN ONLY: GPU checks, installations, downloads and inference are disabled.")

        def call_python(source, settings):
            if not EXECUTE_GPU:
                raise RuntimeError("GPU execution is disabled")
            subprocess.run([str(PYTHON), "-c", "import json,sys; params=json.load(sys.stdin)\\n" + source],
                           input=json.dumps(settings), text=True, cwd=REPO, env=ENV, check=True)
    ''', "preflight")
    md("## 3. Run CPU Regressions Before Downloading Model Weights")
    code('''
        if EXECUTE_GPU:
            subprocess.run([str(PYTHON), "-m", "pytest", "-q",
                            "tests/test_controlled_repair.py", "tests/test_reviewer_fixes.py",
                            "tests/test_reviewer_analysis.py", "tests/test_vllm_limits.py",
                            "tests/test_cloud_runs.py", "tests/test_budget.py",
                            "tests/test_hotpot_scoring.py", "tests/test_prefix_replay.py",
                            "tests/test_repair_budget.py", "tests/test_aws_stop.py",
                            "tests/test_position_matched.py", "tests/test_study_batches.py",
                            "tests/test_aws_study.py"], cwd=REPO, check=True, env=ENV)
        else:
            print("CPU suite is available in the bundle; it runs before GPU inference when execution is enabled.")
    ''')
    md("""
    ## 4. Freeze the Run and Model Snapshot

    Each dataset gets its own pool, checkpoints, and outputs. For a study across
    several machines, use the same storage mount path, code snapshot, resolved
    model revision, environment, and study run ID. Only `DATASET` and its cohort
    files differ. Never let two workers write the same dataset/run directory.
    A kernel disconnect can interrupt a stage; rerun unchanged cells to resume.

    **For test runs:** supply frozen disjoint IDs and a completed policy manifest.
    No test sample size or selected policy is invented here. All pilot/development
    IDs belong in the explored-ID exclusion list. A manifest cannot prove that
    this list is complete or that selection really preceded evaluation.
    """)
    code('''
        SETTINGS = dict(dataset=DATASET, phase=PHASE, run_id=RUN_ID, strategy=STRATEGY,
                        pool_size=POOL_SIZE, batch_size=BATCH_SIZE, origin_sweep=ORIGIN_SWEEP,
                        include_diagnostics=INCLUDE_DIAGNOSTICS,
                        multipliers=MULTIPLIERS, code_sha256=EXPECTED_CODE_SHA256,
                        test_ids=TEST_IDS, development_ids=DEVELOPMENT_IDS,
                        explored_ids=EXPLORED_IDS, policy_file=FROZEN_POLICY,
                        position_profile=POSITION_PROFILE)
        if EXECUTE_GPU:
            if MODEL_ID != "Qwen/Qwen2.5-32B-Instruct-AWQ":
                raise ValueError("Use a separate reviewed profile for another model family")
            if PHASE == "test" and (not MODEL_REVISION or not TEST_IDS or not EXPLORED_IDS or not FROZEN_POLICY):
                raise ValueError("Supply a pinned model revision, both ID manifests, and frozen policy for test")
            call_python("""
from pathlib import Path
from src.utils.cloud_runs import run_directory, pin_model, prepare_config, write_once
import platform, subprocess
run_dir = run_directory(params['storage'], params['settings']['dataset'],
                        params['settings']['phase'], params['settings']['run_id'])
packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()
write_once(run_dir / 'environment_lock.json', {'python': platform.python_version(), 'packages': sorted(packages)})
pin = pin_model(run_dir, Path(params['storage']) / 'huggingface/hub', params['model_id'], params['revision'])
config = prepare_config(params['repo'], run_dir, model_pin=pin, **params['settings'])
print('Model/tokenizer commit:', pin['revision'])
print('Frozen configuration:', config)
""", dict(storage=str(STORAGE_ROOT), repo=str(REPO), model_id=MODEL_ID,
          revision=MODEL_REVISION, settings=SETTINGS))
            RUN_DIR = STORAGE_ROOT / "runs/qwen32b" / PHASE / RUN_ID / DATASET
            CONFIG = RUN_DIR / "config.yaml"
            MODEL_PIN = json.loads((RUN_DIR / "model_snapshot.json").read_text())
            print(json.dumps(MODEL_PIN, indent=2))
        else:
            print(json.dumps(SETTINGS, indent=2))
            print("No run config, model snapshot or results have been created.")
    ''')
    md("""
    ## 5. Generate Initial Trajectories and Score Stored Log Probabilities

    These stages use one model process at a time. The process exits between
    stages to release GPU memory. Stored-token uncertainty needs no sampling
    calls. There is no 72B judge in the primary gold-free comparison.
    If a download or inference stage fails, later stages must not be run until
    it is resolved. Preserve the failed attempt and its timing record.
    """)
    code('''
        def stage(script, *extra):
            call_python("""
from src.utils.cloud_runs import run_stage
run_stage(sys.executable, params['repo'], params['run_dir'], params['script'], params['config'], params['extra'])
""", dict(repo=str(REPO), run_dir=str(RUN_DIR), script=script,
          config=str(CONFIG), extra=list(extra)))

        if EXECUTE_GPU:
            stage("run_setup.py")
            stage("run_generate.py")
            stage("run_uncertainty.py")
        else:
            print("PLAN ONLY: setup -> initial generation -> stored-token uncertainty.")
    ''')
    md("""
    ## 6. Inspect the Repair Count, Then Execute

    Four conditions, three seeds and one token budget require at most
    `12 * N_failed` unique repair jobs. The ten-question pilot therefore has
    at most 120 repairs. Policies sharing an effective origin/prompt reuse one
    execution. The previous eight-origin/two-budget profile allowed 48 jobs
    per failed question; the new upper bound is 75% smaller, not a measured
    runtime saving. No all-origin curve or offset ablation is produced here.
    Keep all three seeds and freeze the affordable question cohort before test
    outcomes. An interrupted partial cohort is not a completed comparison.
    """)
    code('''
        if EXECUTE_GPU:
            stage("run_repair.py", "--dry-run")
            call_python("""
from pathlib import Path
from src.utils import load_config, save_json
from src.utils.cloud_runs import repair_plan
plan = repair_plan(load_config(params['config']))
save_json(plan, Path(params['run_dir']) / 'repair_plan.json')
print(json.dumps(plan, indent=2))
if plan['failed_questions'] == 0:
    raise ValueError('No failures in this cohort; preserve the pilot rather than fabricating repair trials')
if plan['missing_executions'] > params['maximum']:
    raise ValueError('Repair count exceeds the declared guard; inspect the plan and budget before proceeding')
""", dict(config=str(CONFIG), run_dir=str(RUN_DIR), maximum=MAX_REPAIR_EXECUTIONS))
        else:
            print(f"PLAN ONLY: maximum new repair jobs allowed = {MAX_REPAIR_EXECUTIONS}")
    ''')
    code('''
        if EXECUTE_GPU:
            # Recheck the guard so executing this cell alone cannot bypass it.
            plan = json.loads((RUN_DIR / "repair_plan.json").read_text())
            if plan["missing_executions"] > MAX_REPAIR_EXECUTIONS:
                raise RuntimeError("Review the repair-count guard before running")
            stage("run_repair.py")
        else:
            print("PLAN ONLY: repair generation is disabled.")
    ''')
    md("""
    ## 7. Paired Analysis and Pilot Throughput

    The analysis rejects incomplete question/seed coverage against the repair
    manifest. It compares the declared uncertainty policy with matched restart
    and random with the same backtracking offset, not a test-selected winner.
    Confidence intervals resample questions, not individual seed rows. Pilot
    p-values are exploratory, not confirmatory evidence.
    """)
    code('''
        if EXECUTE_GPU:
            call_python("""
from pathlib import Path
from src.utils.cloud_runs import primary_controls
import subprocess
_, baselines = primary_controls(params['strategy'], include_diagnostics=params['include_diagnostics'],
                               include_position_control=params['include_position_control'])
root = Path(params['run_dir'])
for multiplier in params['multipliers']:
    output = root / 'outputs/tables' / f'paired_m{float(multiplier):g}.json'
    subprocess.run([sys.executable, 'scripts/run_paired_analysis.py', '--results',
                    str(root / 'outputs/repairs/results.jsonl'), '--strategy', params['strategy'],
                    '--baselines', *baselines, '--seeds', '0', '1', '2', '--multiplier', str(multiplier),
                    '--output', str(output)], check=True)
    print(output)
""", dict(run_dir=str(RUN_DIR), strategy=STRATEGY, multipliers=MULTIPLIERS,
          include_diagnostics=INCLUDE_DIAGNOSTICS,
          include_position_control=POSITION_PROFILE is not None))
            for path in sorted((RUN_DIR / "outputs/tables").glob("paired_m*.json")):
                result = json.loads(path.read_text())
                print(path.name, PHASE.upper())
                for baseline, effect in result["primary_macro_contrasts"].items():
                    print(f"  vs {baseline}: {100*effect['delta']:+.2f} pp "
                          f"(95% CI {100*effect['delta_lo']:+.2f}, {100*effect['delta_hi']:+.2f}), "
                          f"N={effect['n_questions']} questions")
            attempts = [json.loads(line) for line in (RUN_DIR / "stage_attempts.jsonl").read_text().splitlines()]
            repairs = [a for a in attempts if a["script"] == "run_repair.py" and not a["extra"]]
            elapsed = sum(a["elapsed_seconds"] for a in repairs)
            count = sum(a["new_execution_files"] for a in repairs)
            if count and elapsed:
                seconds_per_execution = elapsed / count
                print(f"Observed repair wall time: {elapsed/3600:.3f} h for {count} new executions")
                print(f"Observed seconds/execution: {seconds_per_execution:.2f}, including model startup/retries")
                if PLANNED_FAILED_QUESTIONS:
                    if type(PLANNED_FAILED_QUESTIONS) is not int or PLANNED_FAILED_QUESTIONS < 1:
                        raise ValueError("Planned failure count must be a positive integer")
                    plan = json.loads((RUN_DIR / "repair_plan.json").read_text())
                    jobs = PLANNED_FAILED_QUESTIONS * plan["per_question_execution_upper_bound"]
                    hours = jobs * seconds_per_execution / 3600
                    print(f"Configured-profile upper-count projection: {hours:.2f} GPU-hours for {jobs} jobs")
                    if GPU_HOURLY_USD is not None:
                        print(f"Repair-only cost at your quote: ${hours*GPU_HOURLY_USD:.2f}")
            print("Projection excludes setup, initial generation, judging, other baselines, storage and idle billing.")
        else:
            print("No result table or runtime estimate is invented in plan-only mode.")
    ''')
    md("""
    ## 8. Export Raw Evidence Before Stopping the Instance

    This export includes the resolved configuration, IDs, model pin, environment,
    trajectories, raw repair executions, analysis and timing records. It excludes
    model weights, the environment directory, and credentials. Keep the matching
    source-code bundle alongside it. Download and verify the archive before
    terminating a pod; filesystem persistence differs by volume type.
    """)
    code('''
        if EXECUTE_GPU:
            import tarfile
            from datetime import datetime, timezone
            export_dir = STORAGE_ROOT / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            target = export_dir / f"{PHASE}-{RUN_ID}-{DATASET}-{stamp}.tar.gz"
            with tarfile.open(target, "w:gz") as archive:
                archive.add(RUN_DIR, arcname=DATASET, recursive=True)
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            target.with_suffix(target.suffix + ".sha256").write_text(digest.hexdigest() + "  " + target.name + "\\n")
            print("Result archive:", target)
            print("SHA256:", digest.hexdigest())
            print("Download via your provider's file interface or scp; this cell does not stop billing.")
        else:
            print("No experimental outputs exist to export in this plan-only run.")
    ''')
    md("""
    ## 9. Move From the Pilot to the Study

    1. Audit token/new-step allowances, prefix replay, scoring, missing-logprob rates
       and complete trial coverage on the actual GPU. Retain all pilot IDs as explored.
    2. Benchmark batch sizes on development-only runs with different run IDs. Do not
       change batching mid-study: batching and the serving stack can affect outputs.
    3. Select one policy on development data across QA3, choose sample sizes from
       paired-difference variance and a stated precision target, and freeze the policy
       below before any test outcomes are examined. Hash IDs with the same helper.
    4. Reuse one GPU sequentially with a different `DATASET`; keep the common policy,
       model revision, run ID and mount paths. Update actual spending before each
       session. Do not fund extra datasets or model families within this profile.
       Select sample sizes using development-only throughput and uncertainty;
       never stop or extend test collection based on whether an effect looks positive.
    5. Combine one completed raw file per dataset using `run_paired_analysis.py --results`
       followed by all three paths. Do not average pilot p-values or merge different models.

    The core notebook does not implement the development-fitted position-matched
    random baseline, diagnosis/replay comparator, repeated-restart answer selector,
    or blinded human labeling. These remain study requirements where their claims
    are retained. FEVER is deliberately excluded until evidence and prompting are fixed.
    """)
    md("""
    ### Optional: Freeze the Study Policy After Development

    This cell computes cohort hashes from your actual files; it does not select
    a winner or invent a sample size. Complete the fields and enable the switch
    only after the development selection and precision analysis are complete.
    The file cannot be overwritten with different settings. `n_questions` counts
    initial questions; the number of failures is observed after generation.
    Point `FROZEN_POLICY` to the resulting JSON in each test notebook.
    """)
    code('''
        FREEZE_STUDY_POLICY = False
        SELECTED_STRATEGY = ""  # selected using DEVELOPMENT outcomes only
        SELECTION_NOTE = ""    # development records, candidates and tie-breaking rule
        PRECISION_TARGET = ""  # interval-width target and sample-size rationale
        STUDY_MULTIPLIERS = [1.0]
        STUDY_ORIGIN_SWEEP = False
        STUDY_INCLUDE_DIAGNOSTICS = False
        STUDY_POSITION_PROFILE = None
        STUDY_BATCH_SIZE = 2
        COHORT_FILES = {
            "hotpotqa": {"test": None, "explored": None},
            "musique": {"test": None, "explored": None},
            "2wikimultihopqa": {"test": None, "explored": None},
        }
        if FREEZE_STUDY_POLICY:
            if not EXECUTE_GPU or "MODEL_PIN" not in globals():
                raise RuntimeError("Complete the environment and model-pin cells before freezing a policy")
            if PHASE == "test":
                raise ValueError("Do not create a study policy after opening test outcomes")
            if not SELECTED_STRATEGY or not SELECTION_NOTE or not PRECISION_TARGET:
                raise ValueError("Complete the development selection and precision rationale")
            call_python("""
from src.utils.cloud_runs import primary_controls, write_once
from src.utils.cohorts import read_ids
from src.repair.controlled import fingerprint
from src.repair.position_matched import validate_position_profile
from pathlib import Path
profile = json.loads(Path(params['position_profile']).read_text()) if params['position_profile'] else None
strategies, comparisons = primary_controls(params['strategy'], include_diagnostics=params['include_diagnostics'],
                                            include_position_control=profile is not None)
policy = {key: params[key] for key in ('strategy', 'selection_note', 'precision_target',
           'model_id', 'model_revision', 'code_sha256', 'multipliers', 'origin_sweep',
           'include_diagnostics', 'batch_size')}
policy.update(seeds=[0, 1, 2], max_steps=8, max_tokens_per_step=512, n_questions={}, cohort_sha256={})
if profile is not None:
    validate_position_profile(profile, strategy=params['strategy'])
    policy.update(position_profile_sha256=profile['sha256'], strategies=strategies,
                  primary_comparisons=comparisons)
for dataset, paths in params['cohorts'].items():
    if not paths['test'] or not paths['explored']:
        raise ValueError('Provide both ID files for every QA dataset')
    ids, explored = read_ids(paths['test']), read_ids(paths['explored'])
    if not ids or not explored or set(ids) & set(explored):
        raise ValueError('Cohorts must be nonempty and disjoint')
    if profile is not None:
        validate_position_profile(profile, dataset=dataset, excluded_ids=explored)
    policy['n_questions'][dataset] = len(ids)
    policy['cohort_sha256'][dataset] = {'test': fingerprint(ids), 'explored': fingerprint(explored)}
write_once(params['output'], policy)
print('Frozen policy:', params['output'])
print(json.dumps(policy, indent=2))
""", dict(strategy=SELECTED_STRATEGY, selection_note=SELECTION_NOTE,
          precision_target=PRECISION_TARGET, model_id=MODEL_ID,
          model_revision=MODEL_PIN["revision"], code_sha256=EXPECTED_CODE_SHA256,
          multipliers=STUDY_MULTIPLIERS, origin_sweep=STUDY_ORIGIN_SWEEP,
          include_diagnostics=STUDY_INCLUDE_DIAGNOSTICS,
          position_profile=STUDY_POSITION_PROFILE,
          batch_size=STUDY_BATCH_SIZE, cohorts=COHORT_FILES,
          output=str(STORAGE_ROOT / "study_policy.json")))
        else:
            print("No study policy has been selected or frozen automatically.")
    ''')
    result = nbformat.v4.new_notebook(cells=cells)
    result.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    result.metadata["language_info"] = {"name": "python", "version": "3.11"}
    result.metadata["agent_repair"] = {"code_sha256": code_sha256, "default_mode": "plan_only"}
    # Stable IDs avoid unrelated notebook churn when rebuilding the bundle.
    for index, cell in enumerate(result.cells):
        cell["id"] = f"iclr-{index:02d}"
    return result


def build_launch_folder(root, notebook_path, bundle_path):
    root = Path(root)
    destination = root / "output/aws-upload"
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for source in (Path(notebook_path), Path(bundle_path), root / "START_HERE.md"):
        target = destination / source.name
        shutil.copyfile(source, target)
        files[source.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {"files": files, "gpu_provisioned": False, "results_generated": False,
                "note": "Local launch package only; provider setup and GPU validation remain required."}
    (destination / "launch_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return destination


def main():
    bundle = ROOT / "output/notebooks" / BUNDLE_NAME
    digest = build_bundle(ROOT, bundle)
    result = notebook(digest)
    nbformat.validate(result)
    path = ROOT / "run_iclr2027.ipynb"
    nbformat.write(result, path)
    launch = build_launch_folder(ROOT, path, bundle)
    print(f"Notebook: {path}\nCode bundle: {bundle}\nCode SHA256: {digest}\nUpload folder: {launch}")


if __name__ == "__main__":
    main()
