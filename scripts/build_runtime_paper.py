"""Report the runtime component from its independently reproduced Qwen records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.build_extension_paper import validate_cell_coverage, interval
from scripts.build_followup_paper import compare_values, require
from scripts.build_iclr_draft import write_table
from scripts.finalize_extension_study import split_audit_roundoff
from src.repair.controlled import fingerprint

BASE = ROOT/'output/aws-experiment/2026-09-15-extension-resume'
OUT = ROOT/'paper/generated'
LABELS = {'full_restart':'Restart', 'unc__perplexity__argmax__bt2':'Uncertainty, backtrack 2',
          'diagnosis_replay':'Diagnosis/replay'}
COLORS = ['#58636e', '#21618c', '#b96529']


def validate_runtime(analysis, protocol, trials):
    model = protocol['runtime']['model']
    cells = {(model, d) for d in protocol['cohorts']}
    validate_cell_coverage(analysis, protocol, trials, cells)
    runtime = trials[trials['mode']=='runtime'].copy()
    require(not runtime.empty, 'The runtime component has no completed attempts')
    require(set(runtime.strategy)==set(LABELS), 'Changed runtime policies')
    require(np.isfinite(runtime.incremental_wall_latency_s).all()
            and (runtime.incremental_wall_latency_s>=0).all(), 'Invalid measured runtime')
    return runtime


def load_runtime(base=BASE):
    base = Path(base)
    local = base.parent/'2026-09-14-extension/qwen-analysis-audited.json'
    remote = base/'retrieved/results/qwen32b-analysis.json'
    paths = [local, remote, local.with_suffix('.trials.csv'), base/'remote-qwen32b-analysis.trials.csv']
    reproduced = json.loads((base/'qwen-reproduction-check.json').read_text())
    require(reproduced.get('matched') is True and reproduced.get('model_key')=='qwen32b',
            'Qwen independent reproduction is incomplete')
    expected_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    require(reproduced['sha256']==expected_hashes, 'Evidence changed after runtime reproduction')
    analysis, counterpart = [json.loads(p.read_text()) for p in paths[:2]]
    frozen = json.loads((base/'retrieved/package/protocol.json').read_text())
    protocol = frozen['payload']
    require(frozen['sha256']==fingerprint(protocol)==analysis['protocol_sha256'], 'Runtime protocol mismatch')
    left, right = [split_audit_roundoff(a) for a in [analysis,counterpart]]
    compare_values(left[0],right[0])
    require([(r['model_key'],r['dataset']) for r in left[1]]==
            [(r['model_key'],r['dataset']) for r in right[1]], 'Changed runtime audit diagnostics coverage')
    keys = ['model_key','dataset','qid','mode','strategy','seed']
    frames = [pd.read_csv(p,keep_default_na=False).sort_values(keys).reset_index(drop=True) for p in paths[2:]]
    try: pd.testing.assert_frame_equal(*frames,check_dtype=False,check_exact=False,rtol=0,atol=1e-12)
    except AssertionError as error: raise ValueError('Local and remote runtime trials differ') from error
    require(reproduced['trial_rows_compared']==len(frames[0]) and
            reproduced['trial_columns_compared']==len(frames[0].columns), 'Runtime reproduction coverage changed')
    validate_runtime(analysis,protocol,frames[0])
    return analysis, protocol, frames[0]


def success_by_time(frame, deadlines):
    """A failed or late attempt stays in the denominator at every deadline."""
    require(len(frame)>0, 'Runtime success requires a nonempty attempt cohort')
    times = frame.incremental_wall_latency_s.to_numpy()
    successes = frame.success.to_numpy(dtype=bool)
    return np.array([100*np.mean(successes & (times<=t)) for t in deadlines])


def build_assets(out, analysis, protocol, runtime):
    out = Path(out)
    (out/'figures').mkdir(parents=True,exist_ok=True)
    rows, contrasts = [], []
    deadlines = protocol['runtime']['deadlines_s']
    primary = protocol['runtime']['primary_deadline_s']
    for strategy,label in LABELS.items():
        sub = runtime[runtime.strategy==strategy]
        rates=[]
        for seconds in deadlines:
            n=int((sub.success & (sub.incremental_wall_latency_s<=seconds)).sum())
            rates.append(f'{n}/{len(sub)} ({100*n/len(sub):.1f})')
        rows.append([label]+rates+[f'{sub.incremental_wall_latency_s.mean():.2f}'])
    for row in analysis['runtime_comparisons']:
        contrasts.append([f"{row['deadline_s']:g}",LABELS[row['strategy_b']],f"{100*row['delta']:+.2f}",
                          interval(row),f"{row['p_value']:.3f}",
                          '--' if row['p_value_holm'] is None else f"{row['p_value_holm']:.3f}"])
    write_table(out/'tables/iclr2027_runtime_results.tex', ['Policy']+
                [f'By {s:g} s (\\%)' for s in deadlines]+['Mean time (s)'],rows)
    write_table(out/'tables/iclr2027_runtime_contrasts.tex',
                ['Deadline (s)','Control','$\\Delta$ (pp)','95\\% interval (pp)','$p$','$p_H$'],contrasts)
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42}):
        fig, ax = plt.subplots(figsize=(6.4,2.6),layout='constrained')
        end=max(deadlines)
        grid=np.unique(np.concatenate(([0.,end],runtime.incremental_wall_latency_s.to_numpy())))
        grid=grid[(grid>=0)&(grid<=end)]
        for (strategy,label),color,style,marker in zip(LABELS.items(),COLORS,['--','-',':'],['s','o','^']):
            sub=runtime[runtime.strategy==strategy]
            ax.step(grid,success_by_time(sub,grid),where='post',label=label,color=color,linestyle=style,linewidth=1.8)
            ax.scatter([primary],success_by_time(sub,[primary]),color=color,marker=marker,s=28,zorder=4)
        ax.axvline(primary,color='#8c8c8c',linewidth=.8,linestyle='--',zorder=0)
        ax.text(primary+.35,1.2,f'{primary:g} s primary',color='#666666',fontsize=8)
        ax.set(xlim=(0,end),ylim=(0,22),xticks=[0,5,10,15,20],yticks=[0,5,10,15,20],
               xlabel='Measured time per attempt (seconds)',ylabel='Success by time (%)')
        ax.grid(axis='y',color='#e4e7ea',linewidth=.6)
        ax.set_axisbelow(True)
        ax.legend(loc='upper left',ncol=3,frameon=False,fontsize=8,handlelength=2,columnspacing=1.3)
        for extension in ['pdf','png']:
            fig.savefig(out/'figures'/f'iclr2027_runtime_success.{extension}',dpi=300)
        plt.close(fig)


def main():
    analysis,protocol,trials=load_runtime()
    runtime=validate_runtime(analysis,protocol,trials)
    build_assets(OUT,analysis,protocol,runtime)
    reproduced=json.loads((BASE/'qwen-reproduction-check.json').read_text())
    record={'scope':'Completed runtime component, independently reproduced from the preserved Qwen recovery records',
            'protocol_sha256':analysis['protocol_sha256'],'questions':runtime.qid.nunique(),'trials':len(runtime),
            'primary_deadline_s':protocol['runtime']['primary_deadline_s'],'primary_family_size':2,
            'source_sha256':reproduced['sha256'],
            'asset_sha256':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for folder in ['tables','figures'] for p in sorted((OUT/folder).glob('iclr2027_runtime_*'))}}
    (OUT/'iclr2027_runtime_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Built the independently reproduced runtime component: 25 questions, 225 attempts.')


if __name__=='__main__':
    main()
