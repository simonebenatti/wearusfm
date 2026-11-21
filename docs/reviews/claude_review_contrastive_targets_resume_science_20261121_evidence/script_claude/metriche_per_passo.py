import json, sys, math
R = sys.argv[1]
def load(p):
    return {r['step']: r for r in (json.loads(l) for l in open(p))}
runs = {'CN': load(f'{R}/benchmark_execution_20261121/CN/metrics.jsonl'),
        'CN_A': load(f'{R}/reloads_execution_20261121/CN_A/metrics.jsonl'),
        'CN_B': load(f'{R}/reloads_execution_20261121/CN_B/metrics.jsonl'),
        'CR': load(f'{R}/benchmark_execution_20261121/CR/metrics.jsonl'),
        'CR_A': load(f'{R}/reloads_execution_20261121/CR_A/metrics.jsonl'),
        'CR_B': load(f'{R}/reloads_execution_20261121/CR_B/metrics.jsonl')}
for k, v in runs.items(): print(k, 'steps', min(v), max(v), len(v))
def rel(a, b):
    if a == b: return 0.0
    return abs(a - b) / max(abs(a), 1e-300)
fields = ['grad_norm', 'grad_clip_factor', 'total', 'jepa', 'contrast_nce', 'contrast_rnc', 'contrast_global_grad_base',
          'contrast_global_grad_nce', 'contrast_global_grad_rnc', 'contrast_projector_grad_norm', 'contrast_global_cos_nce_base', 'pose', 'log_rms']
for a, b in (('CN', 'CN_A'), ('CR', 'CR_A'), ('CN_A', 'CN_B'), ('CR_A', 'CR_B')):
    print('=====', a, 'vs', b)
    print('step ' + ' '.join(f'{f[:14]:>14}' for f in fields))
    for s in range(545, 577):
        ra, rb = runs[a][s], runs[b][s]
        print(f'{s:4d} ' + ' '.join(f'{rel(ra[f], rb[f]):14.3e}' if f in ra and f in rb else f'{"-":>14}' for f in fields))
