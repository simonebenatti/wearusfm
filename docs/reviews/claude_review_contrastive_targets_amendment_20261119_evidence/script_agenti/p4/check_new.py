import sys, json, importlib.util, shutil
from pathlib import Path
root = Path(sys.argv[1]); work = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location('tp', root/'tests/cpu/test_contrastive_prepare.py')
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
P = T.P
if work.exists(): shutil.rmtree(work)
work.mkdir(parents=True)
run = work/'run'
T.fixture_run(run)
rec = P.calibration_report(run)
print('status', rec['status'], 'weights', rec['weights'], 'k', rec['historical_gradient_multiplier'])
print('plan pinned in receipt', rec['reader_provenance']['plan']['path'], rec['reader_provenance']['plan']['sha256'][:12])
cfgs = P.calibrated_configs(rec, run, 576)
for a, c in cfgs.items():
    cc = c['contrastive']
    from wearusfm.training.contrastive_objectives import ContrastiveConfig
    eff = ContrastiveConfig(**cc).weights(600)
    print(a, c['loader']['rms_window_ms'], c['loader']['ms_fast_window_ms'], c['target_version'], 'stored', (cc['nce_weight'], cc['rnc_weight']), 'effective@600', eff, 'max_steps', c['max_steps'])
# lambda exactly at bounds / just outside
import copy
for base_over_term, label in ((1/(.1*P.historical_gradient_multiplier()), 'lambda=1 (boundary)'), (1.0001/(.1*P.historical_gradient_multiplier()), 'lambda>1')):
    r2 = work/('run_'+label.split()[0].replace('=','').replace('>','gt'))
    cfg, summary, records = T.fixture_run(r2)
    for r in records[256:]:
        r['contrast_global_grad_base'] = base_over_term; r['contrast_global_grad_nce'] = 1.; r['contrast_global_grad_rnc'] = 1.
    T.write_run(r2, cfg, summary, records)
    rep = P.calibration_report(r2)
    print(label, rep['status'], rep['failure_reasons'], rep['weights'])
    try:
        P.calibrated_configs(rep, r2, 576); print('  calibrated_configs accepted')
    except ValueError as e:
        print('  calibrated_configs rejected:', str(e)[:80])
