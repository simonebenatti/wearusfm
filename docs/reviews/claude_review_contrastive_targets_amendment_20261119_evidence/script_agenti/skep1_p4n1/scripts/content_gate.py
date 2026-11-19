"""Reproduce: old calibration evidence is rejected by content, wherever it is placed (incl. under fresh_root)."""
import json, shutil, sys
from pathlib import Path
import contrastive_prepare as P
work = Path(sys.argv[1])
hist = P.ROOT / 'results/infra/contrastive_comparison_20261116/calibration_completed_20261117/run'
receipt = json.loads((P.ROOT / 'results/infra/contrastive_comparison_20261116/calibration_completed_20261117/calibration_report.json').read_text())
fresh = work / 'wearusfm_runs/results/contrastive_targets_20261118/calibration'
if fresh.exists(): shutil.rmtree(fresh)
shutil.copytree(hist, fresh)
for label, folder in (('posizione storica', hist), ('copiata sotto fresh_root', fresh)):
    for name, fn in (('calibration_report', lambda f: P.calibration_report(f)),
                     ('authenticate_calibration', lambda f: P.authenticate_calibration(receipt, f)),
                     ('calibrated_configs', lambda f: P.calibrated_configs(receipt, f))):
        try:
            fn(folder); print(label, name, '-> ACCETTATO')
        except Exception as e:
            print(label, name, '-> RIFIUTATO:', type(e).__name__, str(e)[:110])
# Old config with the two windows rewritten to 100/125/v3: still blocked by source digest/compat
cfg = json.loads((fresh/'config.json').read_text())
cfg['loader']['rms_window_ms'] = 100.; cfg['loader']['ms_fast_window_ms'] = 125.
from wearusfm.model.spectral_targets import target_version_for_fast_window
cfg['target_version'] = target_version_for_fast_window(125.)
(fresh/'config.json').write_text(json.dumps(cfg))
try:
    P.calibration_report(fresh); print('config riscritto 100/125/v3 -> ACCETTATO')
except Exception as e:
    print('config riscritto 100/125/v3 -> RIFIUTATO:', type(e).__name__, str(e)[:140])
