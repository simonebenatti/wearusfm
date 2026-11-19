import json, sys, hashlib
from pathlib import Path
ROOT = Path(sys.argv[1])
old = json.loads((ROOT / 'results/infra/contrastive_comparison_20261116/plan.json').read_text())
newp = ROOT / 'results/infra/contrastive_targets_20261118/plan.json'
new = json.loads(newp.read_text())
print('new plan sha', hashlib.sha256(newp.read_bytes()).hexdigest())
pre = json.loads((ROOT / 'results/infra/contrastive_targets_20261118/preflight_plan.json').read_text())
print('preflight pins plan', pre['science_plan'])
def walk(a, b, p, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a: out.append(('+', p + '.' + k, json.dumps(b[k])[:110]))
            elif k not in b: out.append(('-', p + '.' + k, json.dumps(a[k])[:110]))
            else: walk(a[k], b[k], p + '.' + k, out)
    elif a != b:
        out.append(('~', p, json.dumps(a)[:80], json.dumps(b)[:80]))
out = []
walk(old, new, '', out)
for row in out: print(row)
