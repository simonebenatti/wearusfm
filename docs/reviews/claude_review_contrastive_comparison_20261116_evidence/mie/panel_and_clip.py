"""Review Claude del pacchetto contrastivo db5ac41 (16/11/2026): tre controlli di persona sui soli indici del log vero.

(1) Finestre per dataset nel pannello di calibrazione (batch 256..511, base 0), nel tratto dell'audit (256..1.023) e nel primo
    blocco del cancello di runtime (256..1.279): il cancello chiede almeno 32 query per dataset in ciascuno.
(2) Update in cui il clipping del gradiente non e' attivo (norma prima del clipping <= 1).
(3) Aritmetica float32 del peso delle ancore: fl(0,2/n) contro fl(0,2 * fl(1/n)).
Ingressi FUORI dal repository: log del controllo 30M (solo indici) e manifest v1.1.
"""
import gzip, json
from pathlib import Path
from collections import Counter
import numpy as np

man = Path.home() / "wearusfm_local/manifests_derived_20261015/derived_final/manifest_v1_1_nested_1.json.gz"
doc = json.loads(gzip.open(man, "rb").read())
col = {c: i for i, c in enumerate(doc["columns"])}
rows = [r for r in doc["rows"] if r[col["split"]] == "pretraining"]
ds_of = [r[col["dataset"]] for r in rows]
log = Path.home() / "wearusfm_local/logs/review6_remote/control30m/metrics.jsonl"
spans = {"pannello calibrazione 256..511": (256, 512), "audit 256..1023": (256, 1024), "primo blocco 256..1279": (256, 1280)}
cnt = {k: Counter() for k in spans}
total = Counter(); noclip = []; steps = 0
for step, line in enumerate(open(log)):
    r = json.loads(line); steps += 1
    for s, tr, st in r["origin"]:
        d = ds_of[s]; total[d] += 1
        for k, (a, b) in spans.items():
            if a <= step < b:
                cnt[k][d] += 1
    g = r.get("grad_norm")
    if g is not None and g <= 1.0:
        noclip.append((step + 1, round(g, 3)))
print("passi", steps, "finestre", sum(total.values()))
for k in spans:
    c = cnt[k]; n = sum(c.values())
    low = sorted(c.items(), key=lambda kv: kv[1])[:3]
    print(f"{k}: finestre {n}; i tre dataset piu' piccoli: {low}; dataset sotto 32: {[d for d in total if c[d] < 32]}")
print("quota capgmyo sul flusso:", total["capgmyo"], "su", sum(total.values()), f"= {100 * total['capgmyo'] / sum(total.values()):.4f}%")
print("update con norma prima del clipping <= 1 (update a base 1, norma):", noclip, "; quindi clipping attivo in", steps - len(noclip), "su", steps)
f32 = np.float32
diff = [n for n in range(1, 65) if f32(0.2) / f32(n) != f32(0.2) * (f32(1) / f32(n))]
print("float32, n in 1..64 con fl(0,2/n) != fl(0,2*fl(1/n)):", diff)
print("n = 32 uguale:", bool(f32(0.2) / f32(32) == f32(0.2) * (f32(1) / f32(32))), "; n = 4 uguale:", bool(f32(0.2) / f32(4) == f32(0.2) * (f32(1) / f32(4))), "; n = 8 uguale:", bool(f32(0.2) / f32(8) == f32(0.2) * (f32(1) / f32(8))))
