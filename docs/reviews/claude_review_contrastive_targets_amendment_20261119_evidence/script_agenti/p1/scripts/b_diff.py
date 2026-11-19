import json, sys
from pathlib import Path
ROOT = Path(sys.argv[1])
def walk(a, b, p, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a: out.append(('+new', p + '.' + k, b[k]))
            elif k not in b: out.append(('-old', p + '.' + k, a[k]))
            else: walk(a[k], b[k], p + '.' + k, out)
    elif a != b or type(a) != type(b):
        out.append(('~', p, a, b))
for name in ('C0', 'CN', 'CR', 'calibration'):
    old = json.loads((ROOT / f'results/infra/contrastive_comparison_20261116/configs/{name}.config.json').read_text())
    new = json.loads((ROOT / f'results/infra/contrastive_targets_20261118/review_configs/{name}.config.json').read_text())
    out = []
    walk(old, new, '', out)
    print(name, len(out))
    for row in out: print('   ', row)
# also compare against the bf16 calibrated cohort 576 configs (lambda filled)
for name in ('C0', 'CN', 'CR'):
    old = json.loads((ROOT / f'results/infra/contrastive_comparison_20261116/calibrated_cohort_20261118_bf16/{name}.config.json').read_text())
    new = json.loads((ROOT / f'results/infra/contrastive_targets_20261118/review_configs/{name}.config.json').read_text())
    out = []
    walk(old, new, '', out)
    print('bf16cohort vs new', name, len(out))
    for row in out: print('   ', row)
