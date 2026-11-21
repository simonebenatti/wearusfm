import json, sys, math
from pathlib import Path
R = Path(sys.argv[1])
B = R/"benchmark_execution_20261121"; L = R/"reloads_execution_20261121"
def rows(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
TIMING = ("t_data_s","t_step_s","t_data_s_mean")
def isnum(v): return isinstance(v,(int,float)) and not isinstance(v,bool)
for arm in ("CN","CR"):
    c = {r["step"]: r for r in rows(B/arm/"metrics.jsonl")}
    A = rows(L/f"{arm}_A"/"metrics.jsonl"); Bb = rows(L/f"{arm}_B"/"metrics.jsonl")
    print("=== ARM", arm, "reload steps", A[0]["step"], A[-1]["step"])
    abdiff = sorted({k for a,b in zip(A,Bb) for k in set(a)|set(b) if k not in TIMING and a.get(k)!=b.get(k)})
    print(" A vs B differing non-timing fields:", abdiff[:20])
    for a in A[:4] + A[-1:]:
        s = a["step"]; x = c[s]
        keys = sorted(set(x)|set(a))
        diff = [k for k in keys if k not in TIMING and x.get(k)!=a.get(k)]
        same = [k for k in keys if k not in TIMING and x.get(k)==a.get(k)]
        print(f" step {s}: {len(diff)} differ / {len(same)} equal")
        for k in ("grad_norm","grad_clip_factor","total","contrast_nce","contrast_rnc","jepa","pose","log_rms","ms_fast","contrast_projector_grad_norm","lr"):
            if k in x or k in a:
                xv, av = x.get(k), a.get(k)
                rel = abs(xv-av)/max(abs(xv),1e-30) if isnum(xv) and isnum(av) else None
                print(f"    {k}: cont={xv!r} reload={av!r} rel={rel}")
        if s==545:
            print("   differing fields at 545:", diff[:40])
