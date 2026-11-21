import json,sys
E=sys.argv[1]
def load(p): return {r['step']:r for r in map(json.loads,open(p))}
def num(v): return isinstance(v,(int,float)) and not isinstance(v,bool)
for arm in ['CN','CR']:
  c=load(f'{E}/benchmark_execution_20261121/{arm}/metrics.jsonl')
  a=load(f'{E}/reloads_execution_20261121/{arm}_A/metrics.jsonl')
  b=load(f'{E}/reloads_execution_20261121/{arm}_B/metrics.jsonl')
  steps=sorted(a); print('==',arm,steps[0],steps[-1])
  # A vs B identical?
  diffAB=[(s,k) for s in steps for k in a[s] if num(a[s][k]) and a[s][k]!=b[s][k] and not k.startswith('t_')]
  print(' A!=B fields (non-timing):',len(diffAB),diffAB[:5])
  keys=['total','grad_norm','grad_clip_factor','contrast_projector_grad_norm','contrast_nce','contrast_rnc','jepa','pose','contrast_global_grad_base','contrast_global_grad_nce','contrast_global_grad_rnc','contrast_global_cos_nce_base']
  print(' step '+' '.join(f'{k[:14]:>14s}' for k in keys))
  for s in steps:
    row=[]
    for k in keys:
      x=c[s].get(k);y=a[s].get(k)
      if num(x) and num(y) and x!=0: row.append(f'{(y-x)/abs(x):14.2e}')
      else: row.append(f'{str(x)[:6]:>14s}')
    print(f' {s} '+' '.join(row))
  # exact-equal count fields (non-loss) between c and a
  cnt=[k for k in a[steps[0]] if num(a[steps[0]][k]) and ('count' in k or 'eligible' in k or 'replaced' in k or 'age' in k or 'excluded' in k or 'current' in k or 'same_' in k or 'queries' in k or 'discarded' in k or 'pairs' in k or 'anchors' in k or 'windows' in k)]
  bad=[(s,k,c[s][k],a[s][k]) for s in steps for k in cnt if c[s][k]!=a[s][k]]
  print(' count-like fields',len(cnt),'differing:',len(bad),bad[:8])
  print(' grad_clip_factor cont', [round(c[s]['grad_clip_factor'],4) for s in steps[:6]], 'clip cfg?')
