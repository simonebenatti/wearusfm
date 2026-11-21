import json,sys
E=sys.argv[1]
def load(p): return {r['step']:r for r in map(json.loads,open(p))}
def num(v): return isinstance(v,(int,float)) and not isinstance(v,bool)
for arm in ['CN','CR']:
  c=load(f'{E}/benchmark_execution_20261121/{arm}/metrics.jsonl')
  a=load(f'{E}/reloads_execution_20261121/{arm}_A/metrics.jsonl')
  b=load(f'{E}/reloads_execution_20261121/{arm}_B/metrics.jsonl')
  for s in [545,546,547,548]:
    ks=sorted(k for k in a[s] if num(a[s][k]) and not k.startswith('t_') and a[s][k]!=b[s][k])
    print(arm,s,'A!=B',len(ks),[(k,'%.2e'%((a[s][k]-b[s][k])/abs(b[s][k]) if b[s][k] else a[s][k])) for k in ks][:12])
    ks2=sorted(k for k in a[s] if num(a[s][k]) and not k.startswith('t_') and k in c[s] and a[s][k]!=c[s][k])
    print(arm,s,'C!=A',len(ks2),[(k,'%.2e'%((a[s][k]-c[s][k])/abs(c[s][k]) if c[s][k] else a[s][k])) for k in ks2][:40])
  kk=[k for k in a[545] if num(a[545][k]) and not k.startswith('t_')]
  print(arm,'n numeric fields',len(kk), 'missing in cont', [k for k in kk if k not in c[545]][:5])
