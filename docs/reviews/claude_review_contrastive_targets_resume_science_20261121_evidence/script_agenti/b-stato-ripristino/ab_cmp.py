import json, sys
from pathlib import Path
R = Path(sys.argv[1]); B = R/"benchmark_execution_20261121"; L = R/"reloads_execution_20261121"
def rows(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
for arm in ("CN","CR"):
    A = rows(L/f"{arm}_A"/"metrics.jsonl"); Bb = rows(L/f"{arm}_B"/"metrics.jsonl")
    c = {r["step"]: r for r in rows(B/arm/"metrics.jsonl")}
    print("==", arm)
    for k in ('band_shape','contrast_rnc','jepa','log_env','log_rms','ms_fast','ms_slow','total','contrast_nce','pose','grad_norm','contrast_projector_grad_norm'):
        d=[(a['step'], abs(a[k]-b[k])/max(abs(a[k]),1e-30)) for a,b in zip(A,Bb) if a.get(k)!=b.get(k)]
        print(f" A/B {k}: n_diff={len(d)} max_rel={max([x for _,x in d] or [0]):.2e}")
    # first step where cont vs reload differs per field
    for k in ('contrast_nce','contrast_rnc','pose','grad_norm','contrast_projector_grad_norm','jepa'):
        first=[a['step'] for a in A if c[a['step']].get(k)!=a.get(k)]
        print(f" cont/A {k}: first differing step {first[:1]}, n={len(first)}")
