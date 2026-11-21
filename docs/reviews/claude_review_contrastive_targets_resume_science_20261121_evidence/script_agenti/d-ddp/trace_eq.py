import json
B="benchmark_execution_20261121"; R="reloads_execution_20261121"
for arm in ("CN","CR"):
  for r in range(4):
    c=[json.loads(l) for l in open(f"{B}/{arm}/trace-{r}.jsonl")]
    for ph in "AB":
        x=[json.loads(l) for l in open(f"{R}/{arm}_{ph}/trace-{r}.jsonl")]
        keys=[k for k in c[0] if k.endswith("sha256") or k in("boundary",) or "rng" in k or k.startswith("queue")]
        diffs={}
        for a,b in zip(c,x):
            assert a["update"]==b["update"]
            d=[k for k in keys if a.get(k)!=b.get(k)]
            if d: diffs[a["update"]]=d
        print(arm,r,ph,"n",len(c),len(x),"keys",len(keys),"boundary_eq",c[0].get("boundary")==x[0].get("boundary"),"diffs",diffs if diffs else "none")
print(sorted(c[0].keys()))
