"""Local independent recomputation from archived per-user sufficient statistics.

No project imports, feature arrays, prediction generation, fitting or network.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--snapshot-root', required=True, type=Path)
parser.add_argument('--out', required=True, type=Path)
args = parser.parse_args()
ROOT = args.snapshot_root.resolve()
OUT = args.out.resolve()
OUT.mkdir(parents=True, exist_ok=True)
BASE = ROOT/'results/infra/consolidation_long_20261105/execution'
report_path = BASE/'receipts/LONG-COMPARE-pose-20261105.json'
report = json.loads(report_path.read_text())
comparison = report['comparison']
analyses, selections = {}, {}
for dose in (19618, 39237, 49046):
    path = BASE/f'receipts/LONG-FIT-G{dose}-pose-20261105.json'
    raw = path.read_bytes()
    pin = report['fit_reports'][str(dose)]
    assert hashlib.sha256(raw).hexdigest() == pin['sha256'] and len(raw) == pin['bytes']
    fit = json.loads(raw)
    assert fit['source_sha256'] == report['source_sha256'] and fit['recipe_sha256'] == report['recipe_sha256']
    for reader, cell in fit['cells'].items():
        name = f'{dose}/{reader}'
        analyses[name] = cell['analysis']
        if reader.endswith('ridge'):
            selection = cell['fit']['selection']
            assert not selection['boundary_unresolved']
            selections[name] = {k: selection[k] for k in ('alpha','boundary_unresolved','extensions','cv_mse')}
reference = next(iter(analyses.values()))
for a in analyses.values():
    for key in ('support_sha256','targets_sha256','dynamic_threshold'):
        assert a[key] == reference[key]
    assert a['E2']['support'] == reference['E2']['support']

def metrics_group(s, n):
    return {'E_dyn': s['centered']/s['vary'], 'correlation': s['cov']/np.sqrt(s['varp']*s['vary']),
            'level_mse': s['mean_error']/n, 'total_mse': s['total']/n,
            'predicted_variance': s['varp']/n, 'target_variance': s['vary']/n,
            'variance_ratio': s['varp']/s['vary']}

def metrics_e2(s, n):
    return {'mae_aligned':s['aligned_sum_rad']/n, 'mae_misaligned':s['misaligned_sum_rad']/n,
            'gap_mae':(s['misaligned_sum_rad']-s['aligned_sum_rad'])/n,
            'misaligned_constant_ratio':s['misaligned_sum_rad']/s['constant_sum_rad']}

def interval(point, values):
    assert np.isfinite(values).all()
    return {'point':float(point), 'ci95':np.percentile(values, [2.5,97.5]).tolist()}

errors, draws_info, per_user, computed = [], {}, {}, {}
def verify(actual, published):
    errors.append(abs(actual['point']-published['point']))
    errors.extend(abs(a-b) for a,b in zip(actual['ci95'], published['ci95']))
    assert published['evaluable']

for group in ('dynamic','all','static','E2'):
    users = sorted(reference['E2']['per_user'] if group=='E2' else reference['groups'][group]['centered']['per_user'])
    idx = np.random.default_rng(20261102).integers(len(users),size=(2000,len(users)))
    # Explicit multiplicities, independent of the runtime's indexed summation.
    weights = np.stack([np.bincount(draw,minlength=len(users)) for draw in idx]).astype(float)
    points, samples = {}, {}
    for name,a in analyses.items():
        rows = a['E2']['per_user'] if group=='E2' else a['groups'][group]['centered']['per_user']
        assert sorted(rows)==users
        counts = np.array([rows[u]['n_valid_elements'] for u in users],float)
        if group=='E2':
            fields=('aligned_sum_rad','misaligned_sum_rad','constant_sum_rad')
            values={f:np.array([rows[u][f] for u in users]) for f in fields}
            fun=metrics_e2
        else:
            fields=('centered','vary','varp','cov','mean_error','total')
            values={f:np.array([rows[u]['sums'][f] for u in users]) for f in fields}
            fun=metrics_group
        points[name]=fun({f:v.sum() for f,v in values.items()},counts.sum())
        samples[name]=fun({f:weights@v for f,v in values.items()},weights@counts)
        calculated={f:interval(v,samples[name][f]) for f,v in points[name].items()}
        published=comparison['cells'][name]['E2'] if group=='E2' else comparison['cells'][name]['groups'][group]
        for metric,got in calculated.items(): verify(got,published[metric])
        computed.setdefault(name,{})[group]=calculated
        if group=='dynamic': per_user[name]={u:float(rows[u]['sums']['centered']/rows[u]['sums']['vary']) for u in users}
    for name,c in comparison['contrasts'].items():
        candidate,control=c['candidate'],c['control']
        published=c['E2'] if group=='E2' else c['groups'][group]
        for metric in points[candidate]:
            got=interval(points[candidate][metric]-points[control][metric],samples[candidate][metric]-samples[control][metric])
            verify(got,published['delta'][metric])
        if group!='E2':
            got=interval(100*(1-points[candidate]['E_dyn']/points[control]['E_dyn']),
                         100*(1-samples[candidate]['E_dyn']/samples[control]['E_dyn']))
            verify(got,published['relative_improvement_percent'])
    draws_info[group]={'n_users':len(users),'replicates':2000,'seed':20261102}

descriptive={}
for reader in ('post_ridge','trained_head','pre_ridge'):
    old,new=per_user[f'19618/{reader}'],per_user[f'49046/{reader}']
    rel=np.array([100*(1-new[u]/old[u]) for u in old])
    descriptive[reader]={'improved_users':int((rel>0).sum()),'worsened_users':int((rel<0).sum()),
                         'median_relative_improvement_percent':float(np.median(rel)),
                         'min_relative_improvement_percent':float(rel.min()),'max_relative_improvement_percent':float(rel.max())}

assert max(errors)<1e-11, max(errors)
result={'comparison_receipt_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'scope':'Local archived sufficient statistics only; no NPZ/checkpoint/cluster read or inference.',
        'implementation':'Plain NumPy; bootstrap through subject multiplicity matrix; no project imports.',
        'compared_scalar_values':len(errors),'max_absolute_difference':max(errors),'bootstrap':draws_info,
        'support_sha256':reference['support_sha256'],'targets_sha256':reference['targets_sha256'],
        'E2_support':reference['E2']['support'],
        'support':{g:{'n_selected_windows':v['n_selected_windows'],**{k:v['centered'][k] for k in ('n_users','n_windows','n_window_joints','n_valid_elements')}} for g,v in reference['groups'].items()},
        'E2_counts':{k:reference['E2'][k] for k in ('n_users','n_windows','n_valid_elements')},
        'selections':selections,'per_user_primary_descriptive':descriptive,
        'cells':computed,'contrasts':comparison['contrasts'],'references':comparison['references']}
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('compared_scalar_values','max_absolute_difference','bootstrap','support','E2_counts','per_user_primary_descriptive')},indent=2))
