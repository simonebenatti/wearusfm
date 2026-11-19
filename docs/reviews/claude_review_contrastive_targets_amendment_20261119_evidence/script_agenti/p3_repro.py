import json, sys
from pathlib import Path
import contrastive_prepare as P
from wearusfm.training import run as R
base = P._read(P.BASELINE)
print('baseline sha ok', P._pin(P.BASELINE)['sha256'] == P.BASELINE_SHA256)
print('baseline raw loader has fields:', 'rms_window_ms' in base['loader'], 'ms_fast_window_ms' in base['loader'], 'target_version raw:', base.get('target_version'))
b = R.config_to_dict(P.config_from_dict(base))
print('baseline restored', b['loader']['rms_window_ms'], b['loader']['ms_fast_window_ms'], b['target_version'])
cfgs = P.build_review_configs(base)
d = Path('results/infra/contrastive_targets_20261118/review_configs')
for n, c in cfgs.items():
    committed = json.loads((d / (n + '.config.json')).read_text())
    print(n, 'equal to committed:', committed == c, c['loader']['rms_window_ms'], c['loader']['ms_fast_window_ms'], c['target_version'])
# Round trip of new configs
for n, c in cfgs.items():
    rt = R.config_to_dict(P.config_from_dict(c))
    print(n, 'roundtrip equal:', rt == c)
print('PLAN path', P.PLAN.relative_to(P.ROOT))
print('preparation.json pins:')
prep = json.loads((d / 'preparation.json').read_text())
import hashlib
for n, pin in prep['configs'].items():
    print(' ', n, pin['sha256'] == hashlib.sha256((d / (n + '.config.json')).read_bytes()).hexdigest(), pin['path'])
print(' baseline', prep['baseline'])
