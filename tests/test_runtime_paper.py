import copy

import numpy as np
import pandas as pd
import pytest

from scripts.build_runtime_paper import load_runtime, validate_runtime, success_by_time, build_assets


def test_completed_runtime_can_be_reported_without_completing_replication(tmp_path):
    analysis, protocol, trials = load_runtime()
    assert analysis['complete'] is False
    runtime = validate_runtime(analysis, protocol, trials)
    assert len(runtime) == 225 and runtime.qid.nunique() == 25
    build_assets(tmp_path, analysis, protocol, runtime)
    table = (tmp_path/'tables/iclr2027_runtime_results.tex').read_text()
    assert 'Restart & 8/75 (10.7) & 13/75 (17.3)' in table
    assert (tmp_path/'figures/iclr2027_runtime_success.pdf').exists()


@pytest.mark.parametrize('defect', ['missing_trial', 'wrong_question', 'changed_cost', 'exploratory_holm'])
def test_runtime_rejects_changed_denominators_timing_and_inference(defect):
    analysis, protocol, trials = load_runtime()
    analysis = copy.deepcopy(analysis)
    runtime_rows = trials[trials['mode']=='runtime'].index
    if defect == 'missing_trial':
        trials = trials.drop(runtime_rows[0])
    elif defect == 'wrong_question':
        trials.loc[runtime_rows[:3], 'qid'] = 'not-in-frozen-cohort'
    elif defect == 'changed_cost':
        trials.loc[runtime_rows[0], 'incremental_wall_latency_s'] += 1
    else:
        next(r for r in analysis['runtime_comparisons'] if r['deadline_s']==5)['p_value_holm'] = 1.
    with pytest.raises(ValueError):
        validate_runtime(analysis, protocol, trials)


def test_runtime_curve_counts_only_successful_completions_and_keeps_all_attempts():
    frame = pd.DataFrame({'success':[True,False,True,True],
                          'incremental_wall_latency_s':[2.,1.,10.,25.]})
    np.testing.assert_array_equal(success_by_time(frame, [0.,2.,5.,10.,20.]), [0.,25.,25.,50.,50.])
