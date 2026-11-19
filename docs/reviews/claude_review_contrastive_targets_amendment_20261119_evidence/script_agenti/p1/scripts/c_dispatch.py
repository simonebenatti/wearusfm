import json, sys, copy, subprocess, tempfile, os
from pathlib import Path
ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
import contrastive_prepare as P
import contrastive_run as CR
R = P.R
new = {n: P._read(ROOT / f'results/infra/contrastive_targets_20261118/review_configs/{n}.config.json') for n in ('C0','CN','CR','calibration')}
old = {n: P._read(ROOT / f'results/infra/contrastive_comparison_20261116/configs/{n}.config.json') for n in ('C0','CN','CR','calibration')}
def tryit(label, fn):
    try:
        r = fn(); print(f'{label:70s} OK', r if r is not None else '')
    except Exception as e:
        print(f'{label:70s} REJECT {type(e).__name__}: {str(e)[:80]}')
# validate_recipe calibration stage
tryit('validate_recipe(new calibration, calibration)', lambda: (lambda c: (c.loader.rms_window_ms, c.loader.ms_fast_window_ms, c.target_version))(CR.validate_recipe(new['calibration'], 'calibration')))
tryit('validate_recipe(old calibration, calibration)', lambda: CR.validate_recipe(old['calibration'], 'calibration'))
d = copy.deepcopy(new['calibration']); del d['loader']['rms_window_ms']; del d['loader']['ms_fast_window_ms']
tryit('validate_recipe(new calibration senza le due finestre)', lambda: CR.validate_recipe(d, 'calibration'))
d = copy.deepcopy(new['calibration']); del d['loader']['ms_fast_window_ms']
tryit('validate_recipe(new calibration senza ms_fast)', lambda: CR.validate_recipe(d, 'calibration'))
d = copy.deepcopy(new['calibration']); del d['target_version']
tryit('validate_recipe(new calibration senza target_version)', lambda: (lambda c: c.target_version)(CR.validate_recipe(d, 'calibration')))
d = copy.deepcopy(new['calibration']); d['target_version'] = 'step1_psd_power_shape_energy_v1'
tryit('validate_recipe(new calibration con tv v1)', lambda: CR.validate_recipe(d, 'calibration'))
# technical_config of CUDA check
import contrastive_cuda_check as CC
base = P.config_from_dict(new['calibration'])
for arm in ('CN', 'CR'):
    t = CC.technical_config(base, arm)
    print('technical_config', arm, (t.loader.rms_window_ms, t.loader.ms_fast_window_ms, t.target_version), 'min/max window', t.loader.min_window_s, t.loader.max_window_s)
    # stop-check configs derived from technical config
    from dataclasses import replace
    cov = replace(t, datasets=None, max_steps=2, contrastive={**t.contrastive, 'arm': 'C0', 'coverage_block': 2, 'coverage_min_queries': 1000})
    print('   stop-check cfg', (cov.loader.rms_window_ms, cov.loader.ms_fast_window_ms, cov.target_version))
# calibrated-configs generation path with a synthetic authenticated receipt
weights = {'nce': 0.0123, 'rnc': 0.0456}
P.authenticate_calibration = lambda receipt, run_dir: {'weights': weights}
for endpoint in (576, 5000, 19618, 544):
    cfgs = P.calibrated_configs({}, Path('x'), endpoint)
    print('calibrated_configs endpoint', endpoint, {a: (c['loader']['rms_window_ms'], c['loader']['ms_fast_window_ms'], c['target_version'], c['max_steps'], c['contrastive']['nce_weight'], c['contrastive']['rnc_weight']) for a, c in cfgs.items()})
# run main() calibrated-configs writing to OUT/cohort with patched auth
receipt_path = OUT / 'receipt.json'
receipt_path.write_text(json.dumps({'initial_state_sha256': '0'*64}))
sys.argv = ['contrastive_prepare.py', 'calibrated-configs', '--receipt', str(receipt_path), '--run-dir', str(OUT), '--endpoint', '576', '--out', str(OUT / 'cohort')]
P.main()
for f in sorted((OUT / 'cohort').glob('*.config.json')):
    c = json.loads(f.read_text())
    print('  file', f.name, (c['loader']['rms_window_ms'], c['loader']['ms_fast_window_ms'], c['target_version'], c['max_steps']))
