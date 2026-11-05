"""Numero minimo di shard (rettangoli dataset x fold) per cella/compito sotto un tetto di walltime, con la formula congelata."""
import json, sys, itertools
from pathlib import Path
root = Path(sys.argv[1])
CELLS = ("T100-G-30M-natural", "T100-G-300M-natural", "T100-G-300M-matched")
TASKS = {"p1": ("ninapro_db2", "ninapro_db3", "ninapro_db6"), "p2": ("epn612", "uci_emg")}
def weights(cell, ds):
    b = root / cell / ds / "benchmark.json"
    if b.exists():
        r = json.loads(b.read_text()); el = [f["elapsed_s"] for f in r["fits"]]; c = r["fold_train_counts"]
        return [1.5 * (sum(el) + max(el)) * max(1.0, c[f] / c[0]) for f in range(4)], "misurato"
    return None, "mancante"
def subsets(xs):
    for n in range(1, len(xs) + 1):
        yield from itertools.combinations(xs, n)
def min_cover(w, cap):
    cells = [(d, f) for d in w for f in range(4)]
    rects = []
    for D in subsets(list(w)):
        for F in subsets(range(4)):
            tot = sum(w[d][f] for d in D for f in F)
            if 120 + tot <= cap: rects.append((frozenset((d, f) for d in D for f in F), D, F, tot))
    best = [None]
    def dfs(unc, chosen):
        if best[0] is not None and len(chosen) >= len(best[0]): return
        if not unc: best[0] = list(chosen); return
        first = min(unc)
        for cellset, D, F, tot in rects:
            if first in cellset and cellset <= unc:
                dfs(unc - cellset, chosen + [(D, F, tot)])
    dfs(frozenset(cells), [])
    return best[0]
# EPN612 300M naturale: cinque C misurati nell'audit parziale; C=100 ignoto
part = [json.loads(l) for l in (root / "T100-G-300M-natural/epn612/audit_progress.jsonl").read_text().splitlines() if l.strip()]
t5 = [r["elapsed_s"] for r in part]; s5, m5 = sum(t5), max(t5)
ref = json.loads((root / "T100-G-30M-natural/epn612/benchmark.json").read_text())["fold_train_counts"]
ratios = [max(1.0, ref[f] / ref[0]) for f in range(4)]
print(f"EPN612 300M naturale: cinque C = {s5:.1f}s, massimo {m5:.1f}s; rapporti dei fold {['%.4f' % r for r in ratios]}")
for label, t100 in (("limite inferiore C=100 >= 559 s", 559.0), ("C=100 = 1,5 x C=10", 1.5 * m5), ("C=100 = 2 x C=10", 2 * m5), ("C=100 a 12.000 iterazioni al ritmo di C=10", 12000 * part[-1]["elapsed_s"] / part[-1]["n_iter"])):
    pf = 1.5 * (s5 + t100 + max(m5, t100))
    print(f"   {label:45s}: t100={t100:7.1f}s  per fold={pf:7.1f}s  quattro fold={pf*sum(ratios):8.1f}s = {pf*sum(ratios)*8/3600:5.2f} core-ore  (un fold solo entra in 7200 s: {120+pf*max(ratios) <= 7200})")
for cap in (7200, 10800, 14400):
    print("=" * 90); print(f"tetto per shard {cap} s")
    total_n = 0; total_s = 0.0
    for cell in CELLS:
        for task, dss in TASKS.items():
            w = {}; missing = []
            for ds in dss:
                ww, st = weights(cell, ds)
                if ww is None: missing.append(ds)
                else: w[ds] = ww
            if missing:
                # limite inferiore: EPN612 con C=100 = 559 s, UCI = 0
                pf = 1.5 * (s5 + 559.0 + max(m5, 559.0)); w = {"epn612": [pf * r for r in ratios]}
            sol = min_cover(w, cap)
            n = len(sol) if sol else None
            tot = sum(120 + s[2] for s in sol) if sol else float("nan")
            total_n += n or 0; total_s += tot
            tag = " (LIMITE INFERIORE: EPN612 parziale, UCI escluso)" if missing else ""
            print(f"  {cell:22s} {task}: shard minimi = {n}; walltime sommato = {tot:8.1f}s{tag}")
            if cap == 7200 and sol:
                for D, F, t in sol: print(f"        {'+'.join(D):28s} fold {list(F)}  {120+t:7.1f}s")
    print(f"  TOTALE: shard minimi = {total_n}; walltime sommato >= {total_s:9.1f}s = {total_s*8/3600:6.2f} core-ore")
