"""Audit saved development evidence without modifying recorded outcomes."""
import ast
import collections
import hashlib
import json
import math
import re
import statistics
import string
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.env.hotpot_env import HotpotEnv

run, target = map(Path, sys.argv[1:3])
day = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
pool = read(run / 'data/processed/pool.json')
by_qid = {r['_id']: r for r in pool}
ids = read(day / 'development-ids.json')
excluded = read(day / 'explored-ids.json')
assert len(ids) == len(set(ids)) == len(pool) == 50
assert not set(ids).intersection(excluded)
assert [r['_id'] for r in pool] == ids
initial = {p.stem: read(p) for p in (run / 'outputs/trajectories').glob('*.json')}
assert set(initial) == set(ids)
failed = read(run / 'data/processed/failed_ids.json')
assert set(failed) == {q for q, t in initial.items() if not t['success']}
rows = [json.loads(s) for s in (run / 'outputs/repairs/results.jsonl').read_text().splitlines()]
executions = {p.stem: read(p)['trajectory'] for p in (run / 'outputs/repairs/executions').glob('*.json')}
strategies = ['full_restart', 'random_step__bt2', 'fixed_early', 'unc__perplexity__argmax__bt2']
expected = {(q, s, seed, 1.0) for q in failed for s in strategies for seed in (0, 1, 2)}
keys = [(r['qid'], r['strategy'], r['seed'], r['multiplier']) for r in rows]
assert len(keys) == len(set(keys)) and set(keys) == expected
assert {r['execution_id'] for r in rows} == set(executions)
assert len(executions) == read(run / 'repair_plan.json')['planned_unique_executions']

manifest = read(run / 'outputs/repairs/manifest.json')
assert hashlib.sha256(json.dumps(manifest['payload'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == manifest['sha256']
remote_root = Path(manifest['payload']['configuration']['paths']['local_base'])
for remote, digest in manifest['payload']['input_sha256'].items():
    local = run / Path(remote).relative_to(remote_root)
    assert sha(local) == digest, str(local)
assert all(r['manifest_sha256'] == manifest['sha256'] for r in rows)
with zipfile.ZipFile(day / 'agent-repair-iclr2027-code.zip') as bundle:
    for name in ('src/env/hotpot_env.py', 'src/agent/react_agent.py', 'src/agent/batch_runner.py'):
        assert bundle.read(name) == (Path(__file__).resolve().parents[3] / name).read_bytes()

# Execute only the three inspected, pure scoring functions from the official
# evaluator. Its CLI and optional ujson dependency are not needed for this audit.
reference = day / 'hotpot_evaluate_v1.reference.py'
tree = ast.parse(reference.read_text())
names = {'normalize_answer', 'f1_score', 'exact_match_score'}
defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert {node.name for node in defs} == names
scorer = {'re': re, 'string': string, 'Counter': collections.Counter}
exec(compile(ast.Module(body=defs, type_ignores=[]), str(reference), 'exec'), scorer)

observations = prefix_steps = invalid_at_cap = 0
for t in list(initial.values()) + list(executions.values()):
    q = t['qid']
    assert t['gold_answer'] == by_qid[q]['answer']
    env = HotpotEnv(by_qid[q])
    n_prefix = t['meta']['n_prefix_steps']
    assert n_prefix <= len(initial[q]['steps'])
    for index, step in enumerate(t['steps']):
        assert step['index'] == index
        if index < n_prefix:
            for field in ('index', 'thought', 'action', 'action_input', 'observation', 'is_tool_call', 'retrieved_title', 'n_gen_tokens'):
                assert step[field] == initial[q]['steps'][index][field], (q, index, field)
            prefix_steps += 1
        if step['action'] in ('search', 'lookup', 'finish'):
            result = env.step(step['action'], step['action_input'])
            for field in ('observation', 'is_tool_call', 'retrieved_title'):
                assert step[field] == getattr(result, field), (q, index, field)
            if result.finished:
                assert index == len(t['steps']) - 1 and t['final_answer'] == result.answer
        else:
            assert index == len(t['steps']) - 1
            assert step['observation'] == 'Invalid action. Use search, lookup, or finish.'
            assert step['is_tool_call'] is False and step['retrieved_title'] is None
            assert t['meta']['invalid_action_step'] == index
        observations += 1
    answer = t['final_answer']
    em = float(scorer['exact_match_score'](answer, t['gold_answer'])) if answer is not None else 0.0
    f1 = scorer['f1_score'](answer, t['gold_answer'])[0] if answer is not None else 0.0
    assert t['em'] == em and abs(t['f1'] - f1) < 1e-12, q
    assert bool(t['success']) == bool(em or f1 >= 0.5), q
    assert t['total_gen_tokens'] == sum(s['n_gen_tokens'] for s in t['steps'])
    assert t['meta']['recovery_gen_tokens'] == sum(s['n_gen_tokens'] for s in t['steps'][n_prefix:])

for r in rows:
    t = executions[r['execution_id']]
    assert t['qid'] == r['qid']
    assert r['recovery_gen_tokens'] <= r['budget'] == r['original_gen_tokens']
    assert r['step_budget_mode'] == 'new' and r['new_step_allowance'] == 8
    assert len(t['steps']) - t['meta']['n_prefix_steps'] <= 8
    assert r['target_step'] == t['meta']['n_prefix_steps']
    for field in ('em', 'f1', 'success', 'final_answer', 'terminated_reason'):
        assert r[field] == t[field]
    for field in ('recovery_gen_tokens', 'recovery_prompt_tokens', 'prompt_sha256'):
        assert r[field] == t['meta'][field]
    if 'invalid_action_step' in t['meta']:
        expected_reason = 'budget' if r['recovery_gen_tokens'] == r['budget'] else 'error'
        assert t['terminated_reason'] == expected_reason
for key, t in executions.items():
    if 'invalid_action_step' in t['meta'] and t['terminated_reason'] == 'budget':
        invalid_at_cap += 1

by_strategy = {}
means = {}
for name in strategies:
    group = [r for r in rows if r['strategy'] == name]
    means[name] = {q: statistics.mean(r['success'] for r in group if r['qid'] == q) for q in failed}
    by_strategy[name] = {
        'successful_seed_trials': sum(r['success'] for r in group), 'seed_trials': len(group),
        'success_rate': statistics.mean(r['success'] for r in group),
        'exact_match_rate': statistics.mean(r['em'] for r in group),
        'mean_f1': statistics.mean(r['f1'] for r in group),
        'target_step_counts': dict(collections.Counter(r['target_step'] for r in group)),
        'restart_origin_fraction': statistics.mean(r['target_step'] == 0 for r in group),
    }
variance = {}
for control in ('full_restart', 'random_step__bt2'):
    differences = [means[strategies[-1]][q] - means[control][q] for q in failed]
    sd = statistics.stdev(differences) if len(differences) > 1 else None
    variance[control] = {'n_failed_questions': len(failed), 'paired_question_sd': sd,
        'normal_approx_failed_n_for_95pct_halfwidth_0_05': math.ceil((1.96 * sd / 0.05) ** 2) if sd is not None else None,
        'note': 'Exploratory plug-in precision calculation; small-sample or zero variance is not a reliable sample-size guarantee.'}

summary = {
    'phase': 'development, not held out', 'dataset': 'hotpotqa',
    'initial_questions': len(initial), 'excluded_known_questions': len(excluded),
    'historical_exclusions_complete': False,
    'initial_successes': sum(t['success'] for t in initial.values()), 'failed_questions': len(failed),
    'success_definition': 'Exact match or official HotpotQA answer token F1 >= 0.5',
    'initial_exact_matches': sum(t['em'] for t in initial.values()),
    'initial_mean_f1': statistics.mean(t['f1'] for t in initial.values()),
    'unique_repair_executions': len(executions), 'complete_strategy_seed_rows': len(rows),
    'strategies': by_strategy, 'token_budget_violations': 0, 'new_step_budget_violations': 0,
    'initial_generated_tokens': sum(t['total_gen_tokens'] for t in initial.values()),
    'unique_repair_generated_tokens': sum(t['meta']['recovery_gen_tokens'] for t in executions.values()),
    'unique_repair_prompt_tokens': sum(t['meta']['recovery_prompt_tokens'] for t in executions.values()),
    'initial_termination_reasons': dict(collections.Counter(t['terminated_reason'] for t in initial.values())),
    'repair_termination_reasons': dict(collections.Counter(t['terminated_reason'] for t in executions.values())),
    'invalid_action_at_exact_token_budget': invalid_at_cap,
    'verified_input_hashes': len(manifest['payload']['input_sha256']),
    'trajectories_replayed_and_reference_scored': len(initial) + len(executions),
    'observations_verified': observations, 'retained_prefix_steps_verified': prefix_steps,
    'official_scorer_sha256': sha(reference), 'mismatches': 0,
    'model': read(run / 'model_snapshot.json'),
    'paired_analysis': read(run / 'outputs/tables/paired_m1.json'),
    'development_variance': variance,
    'stage_attempts': [json.loads(s) for s in (run / 'stage_attempts.jsonl').read_text().splitlines()],
}
target.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k not in ('paired_analysis', 'model', 'stage_attempts')}, indent=2))
