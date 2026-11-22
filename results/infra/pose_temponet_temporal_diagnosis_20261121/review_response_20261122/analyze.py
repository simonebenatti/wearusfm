"""Verify Claude's additional arithmetic from archived JSON; no fitting."""
from pathlib import Path
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INPUT = ROOT / 'results/infra/pose_temponet_execution_20261121_A1/launcher'
inputs = {}


def read(name):
    p = INPUT / name
    inputs[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def edyn(s):
    return s['dynamic_sse'] / s['dynamic_target_ss']


def pool(rows):
    return {k: sum(r[k] for r in rows) for k in rows[0]}


gate = read('positive_controls_gate_20261121.json')['gate']
cal = [read(f'calibration_{f}.json') for f in (0, 1)]
heads = {h: read(f'selection_temporal_{h}.json') for h in ('TCN_post', 'L0', 'MLP')}
linear_tcn = read('selection_linear_TCN_post.json')
result = {'review_commit': 'f66691157fcd9f8588b3e58473e2fb1ac36c4dc1',
          'scope': 'Local arithmetic only; no training, cluster or new real-data inference.',
          'gate_scale': {}, 'cross_world_heuristic_not_causal': {}, 'fit_to_held': {}}
closed = {h: [gate['linear_diagnostics'][f'fold{f}'][h]['worlds']['temporal']['held']['statistics']
              for f in (0, 1)] for h in ('L0', 'L13')}
for panel, refs in gate['reference_gate']['temporal_records'].items():
    baseline, oracle = refs['clean_ls'], refs['oracle']
    margin = baseline - oracle
    observed = {h: edyn(d['selected_outer_statistics'][panel]) for h, d in heads.items()}
    for h, stats in closed.items():
        observed['closed_' + h] = edyn(pool(stats) if panel == 'pooled' else stats[int(panel[-1])])
    recovery = {h: (baseline - score) / margin for h, score in observed.items()}
    assert math.isclose(recovery['TCN_post'], gate['temporal_head_gate']['recovery'][panel], abs_tol=1e-14)
    result['gate_scale'][panel] = {'recovery': recovery,
        'TCN_fraction_L0_neural_to_oracle': (observed['L0'] - observed['TCN_post']) / (observed['L0'] - oracle),
        'required_fraction_L0_neural_to_oracle': (observed['L0'] - (baseline + oracle) / 2) / (observed['L0'] - oracle)}
for f in (0, 1):
    panel = f'fold{f}'
    refs = cal[f]['outer_reference_statistics']
    temp_margin = edyn(refs['temporal']['clean_ls']) - edyn(refs['temporal']['oracle'])
    excess = edyn(linear_tcn['selected_outer_statistics'][panel]) - edyn(refs['linear']['oracle'])
    result['cross_world_heuristic_not_causal'][panel] = {
        'linear_sigma': cal[f]['calibration']['selected']['linear']['sigma'],
        'temporal_sigma': cal[f]['calibration']['selected']['temporal']['sigma'],
        'linear_dynamic_target_ss': refs['linear']['oracle']['dynamic_target_ss'],
        'temporal_dynamic_target_ss': refs['temporal']['oracle']['dynamic_target_ss'],
        'linear_excess_in_units_of_temporal_margin': excess / temp_margin,
        'heuristic_sum_with_temporal_recovery': excess / temp_margin + result['gate_scale'][panel]['recovery']['TCN_post'],
        'linear_dynamic_threshold': refs['linear_threshold'],
        'temporal_dynamic_threshold': refs['temporal_threshold'],
        'caveat': 'Different targets, dynamic selectors, denominators, noise scales, fitted trajectories and selected exposures. Not an identified decomposition of the temporal learner error.'}
    d = gate['linear_diagnostics'][panel]
    fit_base = edyn(d['L0']['temporal_margin_clean_fit_statistics'])
    row = {'cleanLS': edyn(refs['temporal']['clean_ls']) - fit_base,
           'TCN': heads['TCN_post']['selected_outer_values'][panel]['E_dyn'] - edyn(heads['TCN_post']['selected_fitting_diagnostics'][str(f)])}
    for h in ('L0', 'L13'):
        row['closed_' + h] = edyn(d[h]['worlds']['temporal']['held']['statistics']) - edyn(d[h]['worlds']['temporal']['fitting']['statistics'])
    result['fit_to_held'][panel] = row

# Fixed-support identity. Centered residual and noise can have either sign
# of cross term; noisy error alone gives no deterministic clean-error bound.
examples = []
for residual in ([1., -1., 0., 0.], [-1., 1., 0., 0.]):
    noise = [0.5, -0.5, 0., 0.]
    clean_sse = sum(x*x for x in residual)
    oracle_sse = sum(x*x for x in noise)
    observed_sse = sum((x-n)**2 for x, n in zip(residual, noise))
    cross = sum(x*n for x, n in zip(residual, noise))
    assert observed_sse == clean_sse + oracle_sse - 2*cross
    examples.append({'clean_sse': clean_sse, 'oracle_sse': oracle_sse,
                     'observed_sse': observed_sse, 'cross': cross})
result['fixed_support_identity'] = {'formula': 'SSE_y = SSE_y_star + SSE_oracle - 2*<prediction-y_star, y-y_star>',
    'interpretation': 'All residuals centered identically on the same masked dynamic support. Cross term is not assumed zero.',
    'deterministic_examples': examples}
result['inputs'] = inputs
result['analysis_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(HERE / 'analysis.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
print(json.dumps({'gate_scale': result['gate_scale'], 'cross_world_heuristic_not_causal': result['cross_world_heuristic_not_causal'], 'fit_to_held': result['fit_to_held']}, indent=2))
