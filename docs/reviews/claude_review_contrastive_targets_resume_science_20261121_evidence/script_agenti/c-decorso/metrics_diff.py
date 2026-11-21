import json,sys,math
import numpy as np
E=sys.argv[1]
B=E+'/benchmark_execution_20261121'; RL=E+'/reloads_execution_20261121'
def load(p): return {d['step']:d for d in map(json.loads,open(p))}
skip={'t_data_s','t_step_s','t_data_s_mean'}
out={}
for arm in ['CN','CR']:
    C=load(f'{B}/{arm}/metrics.jsonl'); A=load(f'{RL}/{arm}_A/metrics.jsonl'); Bb=load(f'{RL}/{arm}_B/metrics.jsonl')
    steps=sorted(A)
    keys=[k for k in A[steps[0]] if isinstance(A[steps[0]][k],(int,float)) and not isinstance(A[steps[0]][k],bool) and k not in skip]
    # A vs B differences
    ab=set()
    for s in steps:
        for k in A[s]:
            if A[s][k]!=Bb[s].get(k): ab.add(k)
    print(arm,'A/B differing keys:',sorted(ab))
    # non-numeric equality cont vs A
    for k in ['origin','classes']:
        print(arm,k,'cont==A all steps:',all(C[s][k]==A[s][k] for s in steps))
    res={}
    for k in keys:
        r=[]
        for s in steps:
            c=C[s].get(k); a=A[s].get(k)
            if c is None or a is None: r.append(None); continue
            d=abs(c-a); den=abs(c)
            r.append(0.0 if d==0 else (d/den if den>0 else float('inf')))
        res[k]=r
    out[arm]=res
    # first nonzero step per key
    print(arm,'first differing step (cont vs A) per key:')
    for k in keys:
        r=res[k]
        nz=[i for i,x in enumerate(r) if x]
        if nz: print("  ",k,"first",steps[nz[0]],"%.3e"%r[nz[0]], " ".join("%d:%s"%(steps[i],"%.1e"%r[i] if r[i] is not None else "NA") for i in (0,1,2,3,7,11,15,21,31)))
    print(arm,'keys identical all 32 steps:', [k for k in keys if not any(res[k])])
json.dump(out,open(sys.argv[2],'w'))
