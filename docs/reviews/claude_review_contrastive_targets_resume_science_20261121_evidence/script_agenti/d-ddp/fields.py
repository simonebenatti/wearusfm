import json
def load(p):
    return {x["step"]:x for x in map(json.loads,filter(str.strip,open(p)))}
B="benchmark_execution_20261121"; R="reloads_execution_20261121"
for arm in ("CN","CR"):
    c=load(f"{B}/{arm}/metrics.jsonl"); r=load(f"{R}/{arm}_A/metrics.jsonl")
    ks=sorted(k for k in r[545] if isinstance(r[545][k],(int,float)) and not isinstance(r[545][k],bool))
    print(arm, "numeric keys:", len(ks))
    for s in (545,546,547):
        row=[]
        for k in ks:
            a,b=c[s].get(k),r[s].get(k)
            if a is None or b is None: continue
            if k.startswith("t_"): continue
            rel=abs(a-b)/max(abs(a),1e-300)
            (row.append((k,"%.2e"%rel)) if a!=b else None)
        print(" ",s,"n_differ",len(row),row)
