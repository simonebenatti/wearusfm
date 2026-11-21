import json, sys
from pathlib import Path
R = Path(sys.argv[1])
B = R/"benchmark_execution_20261121"; L = R/"reloads_execution_20261121"
def rows(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
for arm in ("CN","CR"):
    dirs = {"cont": B/arm, "A": L/f"{arm}_A", "Bb": L/f"{arm}_B"}
    print("=== ARM", arm)
    # traces
    for rank in range(4):
        t = {k: rows(d/f"trace-{rank}.jsonl") for k,d in dirs.items()}
        assert [r["update"] for r in t["cont"]] == list(range(545,577)), [r["update"] for r in t["cont"]][:3]
        diffs = {}
        for k in ("A","Bb"):
            for a,b in zip(t["cont"], t[k]):
                for f in set(a)|set(b):
                    if f in ("phase",): continue
                    if a.get(f)!=b.get(f):
                        diffs.setdefault((k,f),[]).append(a["update"])
        ab = [(a["update"],f) for a,b in zip(t["A"],t["Bb"]) for f in set(a)|set(b) if f!="phase" and a.get(f)!=b.get(f)]
        print(" rank",rank,"cont vs reload diffs:",{kf:(len(v),v[:4]) for kf,v in diffs.items()}, "A vs B diffs:", ab[:5])
        # boundary equality across ranks
        print("   boundary equal cont/A/B:", t["cont"][0].get("boundary")==t["A"][0].get("boundary")==t["Bb"][0].get("boundary"))
    # ddp traces
    for rank in range(4):
        d = {k: rows(dd/f"trace-{rank}.ddp.jsonl") for k,dd in dirs.items()}
        print(" ddp rank",rank,"first cont:",{x:d["cont"][0].get(x) for x in ("update","iteration","has_rebuilt_buckets","bucket_sizes")},
              "first A:",{x:d["A"][0].get(x) for x in ("update","iteration","has_rebuilt_buckets","bucket_sizes")})
        same_rebuilt = d["cont"][0].get("rebuilt_per_bucket_param_indices")==d["A"][-1].get("rebuilt_per_bucket_param_indices")
        print("   cont rebuilt order == A rebuilt order (final):", same_rebuilt, "| cont buckets at 545:", d["cont"][0].get("rebuilt_bucket_sizes"))
