import json,sys
def load(p):
    d={}
    for l in open(p):
        if l.strip():
            x=json.loads(l); d[x["step"]]=x
    return d
B="benchmark_execution_20261121"; R="reloads_execution_20261121"
for arm in ("CN","CR"):
    c=load(f"{B}/{arm}/metrics.jsonl")
    for ph in "AB":
        r=load(f"{R}/{arm}_{ph}/metrics.jsonl")
        print(arm,ph,"steps",min(r),max(r))
        for s in range(545,577):
            if s in r:
                keys=[k for k in r[s] if isinstance(r[s][k],float) and k.startswith(("grad","loss","contrast_global"))][:0]
                a,b=c[s]["grad_norm"],r[s]["grad_norm"]
                gc=(c[s].get("grad_clip_factor"),r[s].get("grad_clip_factor"))
                if s<=552 or s in (560,576): print(" ",s,repr(a),repr(b),"rel=%.3e"%(abs(a-b)/abs(a)), "clip",gc)
