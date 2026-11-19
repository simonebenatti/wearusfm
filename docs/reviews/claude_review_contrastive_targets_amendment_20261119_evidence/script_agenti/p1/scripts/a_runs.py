import json, sys
from pathlib import Path
ROOT = Path(sys.argv[1])
import contrastive_prepare as P
R = P.R
paths = sorted(ROOT.glob('results/infra/contrastive_comparison_20261116/**/config.json'))
counts = {}
for p in paths:
    raw = json.loads(p.read_text())
    if not isinstance(raw, dict) or 'loader' not in raw: continue
    out = R.config_to_dict(P.config_from_dict(raw))
    key = (('rms_window_ms' in raw['loader']), ('ms_fast_window_ms' in raw['loader']), raw.get('target_version'),
           out['loader']['rms_window_ms'], out['loader']['ms_fast_window_ms'], out['target_version'])
    counts.setdefault(key, []).append(str(p.relative_to(ROOT)))
for k, v in counts.items():
    print(k, len(v)); [print('   ', x) for x in v[:12]]
