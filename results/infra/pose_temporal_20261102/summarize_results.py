"""Summarize the frozen run report; no refit or new inferential contrast."""
import hashlib
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
source = root / 'execution/run.json'
raw = source.read_bytes()
report = json.loads(raw)
assert report['status'] == 'completed'
result = {
    'source_sha256': hashlib.sha256(raw).hexdigest(),
    'code_sha': report['runtime']['code_sha'],
    'plan_sha256': report['plan_sha256'],
    'slurm_job_id': report['slurm_job_id'],
    'scope': report['scope'],
    'cells': {},
    'paired': {},
}
for key, cell in report['cells'].items():
    fit = cell['fit']
    assert not fit['selection']['boundary_unresolved']
    result['cells'][key] = {
        'alpha': fit['selection']['alpha'],
        'n_features': fit['n_features'],
        'groups': {},
    }
    for group, values in cell['analysis']['groups'].items():
        centered = values['centered']
        result['cells'][key]['groups'][group] = {
            field: centered[field] for field in (
                'centered_mse_ratio', 'variance_ratio', 'pooled_centered_correlation',
                'n_users', 'n_windows', 'n_valid_elements')
        }
        result['cells'][key]['groups'][group].update(
            mae_rad=values['mae_rad'],
            ratio_ci95=report['paired'][group]['cell_ratio_ci95'][key],
        )
for group, paired in report['paired'].items():
    result['paired'][group] = {}
    for name, contrast in paired['contrasts'].items():
        users = [v['delta_centered_sse'] / v['target_variance_sum']
                 for v in contrast['per_user'].values()]
        result['paired'][group][name] = {
            'delta_ratio': contrast['delta_ratio'], 'ci95': contrast['ci95'],
            'median_user_delta_ratio': statistics.median(users),
            'negative_user_deltas': sum(v < 0 for v in users),
            'zero_user_deltas': sum(v == 0 for v in users),
            'positive_user_deltas': sum(v > 0 for v in users),
        }
(root / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, indent=2, allow_nan=False))
