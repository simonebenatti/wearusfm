"""Review Claude del pacchetto contrastivo db5ac41: la regola di lettura applicata ai bracci STORICI a 19.618 passi.

Usa le funzioni del comparatore consegnato (scripts/contrastive_compare.py: intervals, MARGINS, livello simultaneo) e il suo stesso
schema di bootstrap (un generatore col seme 20261116, dataset in ordine alfabetico, soggetti entro dataset, stesse estrazioni per i
bracci) sui conteggi per soggetto e classe pubblicati dalla campagna log-quadratica:
results/infra/logsquare_evaluation_20261107/final/classification/report.json.gz (nel repository, ramo di GPT).
Domanda: con le ampiezze vere, la non inferiorita' su P1 e P2 (margine 1 punto, IC simultaneo 99,4444%) e' sostenibile?
Uso: PYTHONPATH=<albero db5ac41>/src:<albero>/scripts python rule_real_widths.py <albero db5ac41> [repliche]
"""
import gzip, itertools, json, sys
from pathlib import Path
import numpy as np
import contrastive_compare as CC
from wearusfm.harness import downstream_ab as DA

tree = Path(sys.argv[1]); n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else CC.N_BOOT
rep = json.loads(gzip.decompress((tree / "results/infra/logsquare_evaluation_20261107/final/classification/report.json.gz").read_bytes()))
arms = ("T25-L", "T100-L", "T100-G", "T100-LQ")
rng = np.random.default_rng(CC.SEED)
points, samples = {}, {}
for ds in sorted(CC.DATASETS):
    stats = [rep["datasets"][ds]["paired_counts"][a] for a in arms]
    users, den, _ = DA._arrays(stats[0])
    hits = [DA._arrays(s)[2] for s in stats]
    draws = rng.integers(0, len(users), size=(n_boot, len(users)))
    vals = np.empty((n_boot, len(arms)))
    for start in range(0, n_boot, 256):
        ch = draws[start:start + 256]; d = den[ch].sum(1); present = d > 0; nc = present.sum(1)
        for k, h in enumerate(hits):
            vals[start:start + len(ch), k] = np.divide(h[ch].sum(1), d, out=np.zeros_like(d, dtype=float), where=present).sum(1) / nc
    dd = den.sum(0)
    points[ds] = np.array([np.mean(h.sum(0)[dd > 0] / dd[dd > 0]) for h in hits]); samples[ds] = vals
    published = [rep["datasets"][ds]["cells"][a].get("balanced_accuracy") for a in arms]
    print(f"{ds:12s} soggetti {len(users):3d} | balanced accuracy ricalcolata {np.round(100 * points[ds], 3)} | pubblicata {np.round(100 * np.array(published, dtype=float), 3) if all(p is not None for p in published) else published}")
tp = {t: sum(w * points[d] for w, d in zip(CC.WEIGHTS[t], CC.TASKS[t])) for t in CC.TASKS}
ts = {t: sum(w * samples[d] for w, d in zip(CC.WEIGHTS[t], CC.TASKS[t])) for t in CC.TASKS}
print(f"\nrepliche {n_boot}, seme {CC.SEED}, livello simultaneo {CC.SIMULTANEOUS_LEVEL:.6f}, margini {CC.MARGINS}")
ni = {"p1": [0, 0], "p2": [0, 0]}; useful = {"p1": 0, "p2": 0}; half = {"p1": [], "p2": []}
for i, j in itertools.permutations(range(len(arms)), 2):
    for t in ("p1", "p2"):
        point = 100 * (tp[t][i] - tp[t][j]); draw = 100 * (ts[t][:, i] - ts[t][:, j]); ci = CC.intervals(draw)
        lo, hi = ci["ci_simultaneous"]; ok = lo > -CC.MARGINS[t]["ni"]; use = point >= CC.MARGINS[t]["gain"] and lo > 0
        ni[t][0] += ok; ni[t][1] += 1; useful[t] += use; half[t].append((hi - lo) / 2)
        if t == "p2" or (arms[i], arms[j]) == ("T100-G", "T100-L"):
            print(f"{t} {arms[i]:8s} meno {arms[j]:8s}: {point:+.3f} punti | IC95 [{ci['ci95'][0]:+.3f}; {ci['ci95'][1]:+.3f}] | IC simultaneo [{lo:+.3f}; {hi:+.3f}] | non inferiorita' {'SI' if ok else 'no'} | guadagno utile {'SI' if use else 'no'}")
for t in ("p1", "p2"):
    print(f"\n{t}: semiampiezza dell'IC simultaneo da {min(half[t]):.3f} a {max(half[t]):.3f} punti; non inferiorita' sostenuta in {ni[t][0]} contrasti ordinati su {ni[t][1]}; guadagno utile in {useful[t]} su {ni[t][1]}")
    print(f"   stima minima per sostenere la non inferiorita': da {min(half[t]) - CC.MARGINS[t]['ni']:+.3f} a {max(half[t]) - CC.MARGINS[t]['ni']:+.3f} punti (semiampiezza meno margine)")
