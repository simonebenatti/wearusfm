import json, sys, glob, copy
from pathlib import Path
ROOT = Path(sys.argv[1])
import contrastive_prepare as P
R = P.R
def show(path):
    raw = json.loads(Path(path).read_text())
    has = ('rms_window_ms' in raw.get('loader', {}), 'ms_fast_window_ms' in raw.get('loader', {}), 'target_version' in raw)
    cfg = P.config_from_dict(raw)
    out = R.config_to_dict(cfg)
    # what changed relative to raw?
    diffs = []
    def walk(a, b, p):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k not in a: diffs.append(('+', p + '.' + k, b[k]))
                elif k not in b: diffs.append(('-', p + '.' + k, a[k]))
                else: walk(a[k], b[k], p + '.' + k)
        elif a != b:
            diffs.append(('~', p, (a, b)))
    walk(raw, out, '')
    print(Path(path).relative_to(ROOT), 'raw has (rms, fast, tv)=', has, '->', (out['loader']['rms_window_ms'], out['loader']['ms_fast_window_ms'], out['target_version']),
          'diffs:', diffs if len(diffs) < 8 else (len(diffs), diffs[:8]))
paths = [ROOT / 'results/infra/tokenizer_execution_20261103/19618/T100-G/config.json']
paths += sorted(ROOT.glob('results/infra/contrastive_comparison_20261116/configs/*.config.json'))
paths += sorted(ROOT.glob('results/infra/contrastive_comparison_20261116/calibrated_cohort_20261118_bf16/*.config.json'))
paths += sorted(ROOT.glob('results/infra/contrastive_comparison_20261116/calibrated_cohort_20261117/*.config.json'))
paths += sorted(ROOT.glob('results/infra/contrastive_comparison_20261116/calibrated_cohort_20261118_mmap/*.config.json'))
paths += sorted(ROOT.glob('results/infra/contrastive_targets_20261118/review_configs/*.config.json'))
for p in paths:
    show(p)
