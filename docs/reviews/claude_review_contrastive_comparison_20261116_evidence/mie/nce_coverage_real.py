"""Review Claude del pacchetto contrastivo db5ac41 (16/11/2026): copertura InfoNCE sul flusso VERO di 19.618 passi.

Riproduce, con i soli indici del log (nessun segnale, nessun encoder), la politica della memoria di db5ac41
(`training/contrastive_memory.py`): FIFO per dataset con capienza 1.024, eta' massima 256 update, chiave piu' recente per
identita' fisica, candidati = memoria precedente + batch globale corrente; negativo ammissibile = stesso dataset, non la stessa
finestra, e nella stessa registrazione solo stesso trial con distanza >= 4 s; query ammissibile con almeno 32 negativi.
Cancello di `ContrastiveRuntime.coverage_report`: dopo 256 update di riempimento, blocchi da 1.024 update; quota complessiva
>= 0,9, e per OGNI dataset atteso almeno 32 query e quota >= 0,8.

Ingressi FUORI dal repository (solo indici e conteggi): log del controllo 30M con `origin` e manifest v1.1 (vedi sotto).
Approssimazione dichiarata: il log non ha la fine della finestra; uso inizio + DURATA s (4 s = massimo; 1 s come sensibilita').
Uso: python nce_coverage_real.py [durata_s]
"""
import gzip, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
CAP, AGE, MINNEG, GAP, FILL, BLOCK = 1024, 256, 32, 4.0, 256, 1024
man = Path.home() / "wearusfm_local/manifests_derived_20261015/derived_final/manifest_v1_1_nested_1.json.gz"
doc = json.loads(gzip.open(man, "rb").read())
col = {c: i for i, c in enumerate(doc["columns"])}
rows = [r for r in doc["rows"] if r[col["split"]] == "pretraining"]
ds_of = [r[col["dataset"]] for r in rows]
fs_of = [float(r[col["fs_hz"]]) for r in rows]
expected = sorted({r[col["dataset"]] for r in rows if r[col["weight"]] > 0})
log = Path.home() / "wearusfm_local/logs/review6_remote/control30m/metrics.jsonl"

mem = defaultdict(dict)            # dataset -> {ident: (step, rec, trial, t0, t1)}  (dict mantiene l'ordine d'inserimento)
tot = defaultdict(lambda: [0, 0]); blk = defaultdict(lambda: [0, 0]); negs = defaultdict(list)
reports = []; upd = 0; rnc_active = 0; blk_rnc = 0
for step, line in enumerate(open(log)):
    rec_ = json.loads(line); o = rec_["origin"]
    cur = {}
    for s, tr, st in o:                                   # deduplica: vince l'indice globale piu' basso
        ident = (s, tr, st)
        if ident not in cur:
            cur[ident] = (step, s, -1 if tr is None else int(tr), st / fs_of[s], st / fs_of[s] + DUR)
    by_ds = defaultdict(list)
    for ident, e in cur.items():
        by_ds[ds_of[ident[0]]].append((ident, e))
    arrays = {}
    for d in {ds_of[s] for s, _, _ in o}:
        ents = [e for i, e in mem[d].items() if step - e[0] <= AGE and i not in cur] + [e for _, e in by_ds.get(d, [])]
        arrays[d] = np.array([(e[1], e[2], e[3], e[4]) for e in ents], dtype=np.float64).reshape(-1, 4)
    for s, tr, st in o:
        d = ds_of[s]; a = arrays[d]; t0 = st / fs_of[s]; t1 = t0 + DUR; trial = -1 if tr is None else int(tr)
        same = a[:, 0] == s
        gap = np.maximum(a[:, 2], t0) - np.minimum(a[:, 3], t1)
        ok = (~same) | (same & (a[:, 1] == trial) & (gap >= GAP))
        n = int(ok.sum()); good = n >= MINNEG
        if step >= FILL:
            tot[d][0] += 1; tot[d][1] += good; blk[d][0] += 1; blk[d][1] += good; negs[d].append(n)
    # commit: eta', deduplica (la piu' recente vince), capienza per dataset (le piu' recenti restano)
    for d in list(mem):
        mem[d] = {i: e for i, e in mem[d].items() if step - e[0] <= AGE and i not in cur}
    for d, items in by_ds.items():
        for ident, e in items:
            mem[d][ident] = e
        if len(mem[d]) > CAP:
            keep = list(mem[d].items())[-CAP:]
            mem[d] = dict(keep)
    if step >= FILL:
        upd += 1; active = rec_.get("pose_windows", 0) >= 1; rnc_active += active; blk_rnc += active
        if upd % BLOCK == 0:
            total = sum(v[0] for v in blk.values()); good = sum(v[1] for v in blk.values())
            per = {d: (blk[d][0], blk[d][1] / max(blk[d][0], 1)) if d in blk else (0, 0.0) for d in set(expected) | set(blk)}
            failing = sorted(d for d, (q, f) in per.items() if q < 32 or f < 0.8)
            reports.append({"through_step": step, "nce": good / max(total, 1), "rnc": blk_rnc / BLOCK, "failing": failing,
                            "passed": total > 0 and good / total >= .9 and blk_rnc / BLOCK >= .9 and not failing})
            blk = defaultdict(lambda: [0, 0]); blk_rnc = 0

print(f"durata assunta della finestra {DUR} s; passi {step + 1}; update dopo il riempimento {upd}; blocchi completi {len(reports)}")
T = sum(v[0] for v in tot.values()); G = sum(v[1] for v in tot.values())
print(f"InfoNCE, quota complessiva di query ammissibili: {G / T:.4f} ({G} su {T}); RnC, update con almeno una finestra di posa valida: {rnc_active / upd:.4f}")
print(f"{'dataset':16s} {'quota finestre':>14s} {'query':>8s} {'ammissibili':>12s} {'negativi: mediana':>18s} {'min':>6s} {'max':>6s}")
for d in sorted(tot, key=lambda k: -tot[k][0]):
    q, g = tot[d]; v = np.array(negs[d])
    print(f"{d:16s} {100 * q / T:13.3f}% {q:8d} {100 * g / q:11.2f}% {int(np.median(v)):18d} {int(v.min()):6d} {int(v.max()):6d}")
print("blocchi superati:", sum(r["passed"] for r in reports), "su", len(reports))
for r in reports[:3] + reports[-1:]:
    print(f"  blocco fino al passo {r['through_step']}: nce {r['nce']:.4f} rnc {r['rnc']:.4f} dataset sotto soglia {r['failing']} superato {r['passed']}")
fails = defaultdict(int)
for r in reports:
    for d in r["failing"]:
        fails[d] += 1
print("dataset sotto soglia (numero di blocchi su", len(reports), "):", dict(fails))
