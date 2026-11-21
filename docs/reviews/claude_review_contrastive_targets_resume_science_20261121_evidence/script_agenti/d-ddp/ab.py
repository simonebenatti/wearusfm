import json
def load(p):
    return {x["step"]:x for x in map(json.loads,filter(str.strip,open(p)))}
R="reloads_execution_20261121"
for arm in ("CN","CR"):
    a=load(f"{R}/{arm}_A/metrics.jsonl"); b=load(f"{R}/{arm}_B/metrics.jsonl")
    tot=0
    for s in range(545,577):
        d=[(k,"%.1e"%(abs(a[s][k]-b[s][k])/max(abs(a[s][k]),1e-300))) for k in a[s] if isinstance(a[s][k],float) and not k.startswith("t_") and a[s][k]!=b[s].get(k)]
        tot+=len(d)
        if s<=547 or s==576: print(arm,s,"A_vs_B n_differ",len(d),d[:8], "grad_norm_eq",a[s]["grad_norm"]==b[s]["grad_norm"])
    print(arm,"total differing float fields A/B over 32 steps",tot, "grad_norm all equal", all(a[s]["grad_norm"]==b[s]["grad_norm"] for s in range(545,577)))
