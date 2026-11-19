import sys, json, copy
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import test_contrastive_benchmark_readout as TB
B = TB.B
folder = Path(sys.argv[2])
cfg, provenance, cohort = TB._run_folder(folder)
print('fixture cfg windows:', cfg['loader']['rms_window_ms'], cfg['loader']['ms_fast_window_ms'], cfg['target_version'])
B._run_evidence(folder, 'C0', cohort, cfg, provenance)
print('matching recipe accepted')
# Historical-style C0 benchmark: fields absent + v1 (expected is the new explicit recipe)
hist = copy.deepcopy(cfg)
del hist['loader']['rms_window_ms']; del hist['loader']['ms_fast_window_ms']
hist['target_version'] = 'step1_psd_power_shape_energy_v1'
(folder / 'config.json').write_text(json.dumps(hist))
try:
    B._run_evidence(folder, 'C0', cohort, cfg, provenance)
    print('UNEXPECTED: historical recipe accepted')
except ValueError as e:
    print('historical 25/50 C0 benchmark rejected:', e)
