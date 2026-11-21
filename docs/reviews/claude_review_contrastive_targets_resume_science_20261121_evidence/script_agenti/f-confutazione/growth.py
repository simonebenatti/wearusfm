import json,sys,math,re,statistics as st
E=sys.argv[1]
def load(p): return {r['step']:r for r in map(json.loads,open(p))}
for arm in ['CN','CR']:
  c=load(f'{E}/benchmark_execution_20261121/{arm}/metrics.jsonl'); a=load(f'{E}/reloads_execution_20261121/{arm}_A/metrics.jsonl')
  for k in ['grad_norm','contrast_projector_grad_norm','contrast_nce','contrast_rnc','jepa','total','pose']:
    out=[]
    for lo in (545,553,561,569):
      v=[((a[s][k]-c[s][k])/c[s][k])**2 for s in range(lo,lo+8) if c[s].get(k)]
      out.append('%.1e'%math.sqrt(sum(v)/len(v)) if v else '-')
    print(arm,f'{k:30s}',out)
  # per-dataset nce loss rel diffs at 547,548
  ds=sorted(k for k in a[548] if k.startswith('contrast_nce_loss__'))
  for s in (547,548,549):
    vals={k.split('__')[1]:(a[s][k]-c[s][k])/c[s][k] for k in ds if c[s].get(k)}
    print(arm,s,'nce per-dataset |rel| median %.1e max %.1e'%(st.median(map(abs,vals.values())),max(map(abs,vals.values()))) if vals else '', 'n nonzero',sum(1 for v in vals.values() if v!=0),'/',len(vals))
