"""Replay downloaded evidence using its frozen source and the official scorer."""
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics
import string
import sys
import tempfile
import zipfile


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def mean(values):
    return statistics.mean(values) if values else None


def audit(run, bundle, reference, ids, excluded, strategies, expected_code, expected_policy=None,
          expected_model=None):
    with tempfile.TemporaryDirectory(prefix='agent-repair-audit-source-') as temp:
        with zipfile.ZipFile(bundle) as archive:
            sources = read_manifest = json.loads(archive.read('bundle_manifest.json'))
            assert sources['code_sha256'] == expected_code
            assert hashlib.sha256(json.dumps(sources['files'], sort_keys=True).encode()).hexdigest() == expected_code
            for name, digest in sources['files'].items():
                assert not Path(name).is_absolute() and '..' not in Path(name).parts
                assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
            archive.extractall(temp)
        sys.path.insert(0, temp)
        from src.env.hotpot_env import HotpotEnv
        from src.repair.controlled import plan_repairs
        pool = read(run / 'data/processed/pool.json')
        assert [record['_id'] for record in pool] == ids
        assert len(ids) == len(set(ids)) == len(pool) == 50
        assert not set(ids).intersection(excluded)
        by_qid = {record['_id']: record for record in pool}
        originals = {p.stem: read(p) for p in (run / 'outputs/trajectories').glob('*.json')}
        assert set(originals) == set(ids)
        failed = read(run / 'data/processed/failed_ids.json')
        assert set(failed) == {q for q, trace in originals.items() if not trace['success']}
        rows = [json.loads(line) for line in (run / 'outputs/repairs/results.jsonl').read_text().splitlines()]
        wrappers = {p.stem: read(p) for p in (run / 'outputs/repairs/executions').glob('*.json')}
        executions = {key: value['trajectory'] for key, value in wrappers.items()}
        expected_keys = {(q, name, seed, 1.0) for q in failed for name in strategies for seed in (0, 1, 2)}
        keys = [(r['qid'], r['strategy'], r['seed'], r['multiplier']) for r in rows]
        assert len(keys) == len(set(keys)) and set(keys) == expected_keys
        assert {r['execution_id'] for r in rows} == set(executions)
        assert len(executions) == read(run / 'repair_plan.json')['planned_unique_executions']
        manifest = read(run / 'outputs/repairs/manifest.json')
        assert fingerprint(manifest['payload']) == manifest['sha256']
        cfg = manifest['payload']['configuration']
        assert cfg['notebook_provenance']['code_sha256'] == expected_code
        assert cfg['repair']['strategies'] == strategies
        assert cfg['repair']['seeds'] == [0, 1, 2]
        assert cfg['repair']['budget']['multipliers'] == [1.0]
        assert cfg['repair']['step_budget_mode'] == 'new' and cfg['repair']['match_restart_hint']
        if expected_policy:
            assert fingerprint(read(run / 'study_policy.json')) == expected_policy
            assert cfg['notebook_provenance']['study_policy_sha256'] == expected_policy
            assert cfg['notebook_provenance']['phase'] == cfg['dataset']['split'] == 'test'
            assert read(run / 'test_ids.json') == ids
            assert read(run / 'explored_ids.json') == excluded
        model = read(run / 'model_snapshot.json')
        if expected_model:
            assert {k: model[k] for k in ('repo_id', 'revision')} == expected_model
        remote_root = Path(cfg['paths']['local_base'])
        for remote, digest in manifest['payload']['input_sha256'].items():
            assert sha(run / Path(remote).relative_to(remote_root)) == digest, remote
        assert all(r['manifest_sha256'] == manifest['sha256'] for r in rows)
        assert all(w['manifest_sha256'] == manifest['sha256'] for w in wrappers.values())
        planned = {}
        for q in failed:
            uncertainty = read(run / 'outputs/uncertainty' / (q + '.json'))
            for seed in (0, 1, 2):
                jobs = plan_repairs(cfg, by_qid[q], originals[q], uncertainty, None, seed, 1.0, strategies)
                for job in jobs:
                    for name in job['strategies']:
                        planned[q, name, seed, 1.0] = job
        assert set(planned) == expected_keys
        assert {job['meta']['execution_id'] for job in planned.values()} == set(executions)

        tree = ast.parse(reference.read_text())
        names = {'normalize_answer', 'f1_score', 'exact_match_score'}
        defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in defs} == names
        scorer = {'re': re, 'string': string, 'Counter': collections.Counter}
        exec(compile(ast.Module(body=defs, type_ignores=[]), str(reference), 'exec'), scorer)
        observations = prefixes = 0
        for trace in [*originals.values(), *executions.values()]:
            q = trace['qid']
            assert trace['gold_answer'] == by_qid[q]['answer']
            env = HotpotEnv(by_qid[q])
            n_prefix = trace['meta']['n_prefix_steps']
            assert n_prefix <= len(originals[q]['steps'])
            for index, step in enumerate(trace['steps']):
                assert step['index'] == index
                if index < n_prefix:
                    for field in ('index', 'thought', 'action', 'action_input', 'observation',
                                  'is_tool_call', 'retrieved_title', 'n_gen_tokens'):
                        assert step[field] == originals[q]['steps'][index][field], (q, index, field)
                    prefixes += 1
                if step['action'] in ('search', 'lookup', 'finish'):
                    replay = env.step(step['action'], step['action_input'])
                    for field in ('observation', 'is_tool_call', 'retrieved_title'):
                        assert step[field] == getattr(replay, field), (q, index, field)
                    if replay.finished:
                        assert index == len(trace['steps']) - 1 and trace['final_answer'] == replay.answer
                else:
                    assert index == len(trace['steps']) - 1
                    assert step['observation'] == 'Invalid action. Use search, lookup, or finish.'
                    assert step['is_tool_call'] is False and step['retrieved_title'] is None
                    assert trace['meta']['invalid_action_step'] == index
                observations += 1
            answer = trace['final_answer']
            em = float(scorer['exact_match_score'](answer, trace['gold_answer'])) if answer is not None else 0.0
            f1 = scorer['f1_score'](answer, trace['gold_answer'])[0] if answer is not None else 0.0
            assert trace['em'] == em and abs(trace['f1'] - f1) < 1e-12, q
            assert bool(trace['success']) == bool(em or f1 >= 0.5), q
            assert trace['total_gen_tokens'] == sum(step['n_gen_tokens'] for step in trace['steps'])
            assert trace['meta']['recovery_gen_tokens'] == sum(step['n_gen_tokens'] for step in trace['steps'][n_prefix:])
        for row in rows:
            job = planned[row['qid'], row['strategy'], row['seed'], row['multiplier']]
            trace = executions[row['execution_id']]
            assert trace['qid'] == row['qid']
            assert row['execution_id'] == job['meta']['execution_id']
            assert row['prompt_sha256'] == job['meta']['prompt_sha256']
            assert row['target_step'] == job['meta']['target_step'] == trace['meta']['n_prefix_steps']
            assert row['recovery_gen_tokens'] <= row['budget'] == job['token_budget'] == row['original_gen_tokens']
            assert row['step_budget_mode'] == 'new' and row['new_step_allowance'] == 8
            assert len(trace['steps']) - trace['meta']['n_prefix_steps'] <= 8
            for field in ('em', 'f1', 'success', 'final_answer', 'terminated_reason'):
                assert row[field] == trace[field]
            for field in ('recovery_gen_tokens', 'recovery_prompt_tokens', 'prompt_sha256'):
                assert row[field] == trace['meta'][field]
            if 'invalid_action_step' in trace['meta']:
                assert trace['terminated_reason'] == ('budget' if row['recovery_gen_tokens'] == row['budget'] else 'error')
        groups = {name: [row for row in rows if row['strategy'] == name] for name in strategies}
        return {
            'run_id': cfg['repair']['run_id'], 'code_sha256': expected_code,
            'source_files_verified': len(read_manifest['files']), 'bundle_sha256': sha(bundle),
            'model': model, 'initial_questions': len(originals), 'failed_questions': len(failed),
            'initial_successes': sum(t['success'] for t in originals.values()),
            'initial_exact_matches': sum(t['em'] for t in originals.values()),
            'initial_mean_f1': mean([t['f1'] for t in originals.values()]),
            'unique_repair_executions': len(executions), 'strategy_seed_rows': len(rows),
            'strategies': {name: {'successful_seed_trials': sum(r['success'] for r in group),
                                 'seed_trials': len(group), 'success_rate': mean([r['success'] for r in group]),
                                 'exact_match_rate': mean([r['em'] for r in group]),
                                 'mean_f1': mean([r['f1'] for r in group]),
                                 'target_step_counts': dict(collections.Counter(r['target_step'] for r in group))}
                           for name, group in groups.items()},
            'initial_generated_tokens': sum(t['total_gen_tokens'] for t in originals.values()),
            'unique_repair_generated_tokens': sum(t['meta']['recovery_gen_tokens'] for t in executions.values()),
            'unique_repair_prompt_tokens': sum(t['meta']['recovery_prompt_tokens'] for t in executions.values()),
            'input_hashes_verified': len(manifest['payload']['input_sha256']),
            'trajectories_replayed_and_reference_scored': len(originals) + len(executions),
            'observations_verified': observations, 'retained_prefix_steps_verified': prefixes,
            'prespecified_origin_and_prompt_checks': len(rows), 'official_scorer_sha256': sha(reference),
            'mismatches': 0, 'token_budget_violations': 0, 'new_step_budget_violations': 0,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--batch', help='Frozen main-study batch directory, such as batch01')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--development-check', action='store_true')
    args = parser.parse_args()
    day = Path(__file__).resolve().parent
    prior = day.parent / '2026-09-11-development'
    reference = prior / 'hotpot_evaluate_v1.reference.py'
    if args.development_check:
        cfg = read(args.run / 'outputs/repairs/manifest.json')['payload']['configuration']
        result = audit(args.run, prior / 'agent-repair-iclr2027-code.zip', reference,
                       read(prior / 'development-ids.json'), read(prior / 'explored-ids.json'),
                       cfg['repair']['strategies'], cfg['notebook_provenance']['code_sha256'])
    else:
        frozen = read(day / 'study-freeze.json')
        assert frozen['sha256'] == fingerprint(frozen['payload'])
        study = frozen['payload']
        batch = next(batch for batch in study['batches'] if batch['directory'] == args.batch)
        result = audit(args.run, day / 'agent-repair-iclr2027-code.zip', reference,
                       batch['ids'], study['explored_ids'], study['strategies'], study['code_sha256'],
                       batch['policy_sha256'], {'repo_id': study['model_id'], 'revision': study['model_revision']})
        assert result['run_id'] == batch['run_id']
        result.update(study_sha256=frozen['sha256'], batch=args.batch,
                      historical_exclusion_basis='reconstructed documented history')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        assert read(args.output) == result, 'Preserve a changed audit result separately'
    else:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('run_id', 'initial_questions', 'unique_repair_executions',
                                             'strategy_seed_rows', 'observations_verified', 'mismatches')}, indent=2))


if __name__ == '__main__':
    main()
