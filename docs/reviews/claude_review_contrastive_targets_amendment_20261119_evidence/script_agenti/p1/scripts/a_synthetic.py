import json, sys, copy
from pathlib import Path
ROOT = Path(sys.argv[1])
import contrastive_prepare as P
R = P.R
base = json.loads((ROOT / 'results/infra/tokenizer_execution_20261103/19618/T100-G/config.json').read_text())
def run(name, d):
    try:
        cfg = P.config_from_dict(d)
        out = R.config_to_dict(cfg)
        res = (out['loader']['rms_window_ms'], out['loader']['ms_fast_window_ms'], out['target_version'])
        # does trainer start-check accept it?
        from wearusfm.model.spectral_targets import target_version_for_fast_window, RETIRED_TARGET_VERSIONS
        try:
            ok = cfg.target_version == target_version_for_fast_window(cfg.loader.ms_fast_window_ms) and cfg.target_version not in RETIRED_TARGET_VERSIONS
        except Exception as e:
            ok = 'ERR ' + str(e)[:60]
        print(f'{name:45s} -> {res} trainer_guard_ok={ok}')
    except Exception as e:
        print(f'{name:45s} -> EXC {type(e).__name__}: {str(e)[:100]}')
# 1. historical without target_version
d = copy.deepcopy(base); del d['target_version']; run('storico senza target_version', d)
# 2. only rms
d = copy.deepcopy(base); d['loader']['rms_window_ms'] = 100.; run('solo rms=100', d)
d = copy.deepcopy(base); d['loader']['rms_window_ms'] = 100.; d['target_version'] = 'step1_psd_power_shape_energy_fast125_v3'; run('solo rms=100 + v3', d)
# 3. only fast
d = copy.deepcopy(base); d['loader']['ms_fast_window_ms'] = 125.; run('solo fast=125, tv v1', d)
d = copy.deepcopy(base); d['loader']['ms_fast_window_ms'] = 125.; d['target_version'] = 'step1_psd_power_shape_energy_fast125_v3'; run('solo fast=125 + v3', d)
# 4. retired
d = copy.deepcopy(base); d['loader']['ms_fast_window_ms'] = 100.; d['loader']['rms_window_ms'] = 100.; d['target_version'] = 'step1_psd_power_shape_energy_fast100_v2'; run('fast=100 v2 ritirato', d)
# 5. new explicit
d = copy.deepcopy(base); d['loader']['ms_fast_window_ms'] = 125.; d['loader']['rms_window_ms'] = 100.; d['target_version'] = 'step1_psd_power_shape_energy_fast125_v3'; run('nuovo esplicito', d)
# 6. no loader dict
d = copy.deepcopy(base); d['loader'] = None; run('loader None', d)
# 7. build_review_configs on a baseline already 100/125 -> must reject
d = copy.deepcopy(base); d['loader']['ms_fast_window_ms'] = 125.; d['loader']['rms_window_ms'] = 100.; d['target_version'] = 'step1_psd_power_shape_energy_fast125_v3'
try:
    P.build_review_configs(d); print('build_review_configs(base nuova) ACCETTATA')
except Exception as e: print('build_review_configs(base nuova) ->', type(e).__name__, str(e)[:90])
d = copy.deepcopy(base); del d['target_version']
try:
    P.build_review_configs(d); print('build_review_configs(base senza tv) ACCETTATA')
except Exception as e: print('build_review_configs(base senza tv) ->', type(e).__name__, str(e)[:90])
