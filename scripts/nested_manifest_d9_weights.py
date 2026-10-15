"""Proposta per D16 (15/10/2026, da concordare con GPT): pesi dei sottoinsiemi annidati con le regole firmate di D9 rifatte sul sottoinsieme (allocate: quote di classe, ore^alpha, tetto di passaggi)."""
import gzip
import json
import sys

sys.path.insert(0, "src")
from wearusfm.data import pretraining_manifest as M  # noqa: E402

S = (sys.argv[1] if len(sys.argv) > 1 else "manifest").rstrip("/") + "/"  # cartella con manifest_v1_1.json.gz (copia locale, non nel repo)
doc = json.loads(gzip.decompress(open(S + "manifest_v1_1.json.gz", "rb").read()))
rows = [dict(zip(doc["columns"], r)) for r in doc["rows"] if r[doc["columns"].index("split")] == "pretraining"]
p = doc["params"]
for f in ("0.125", "0.25", "0.5", "1"):
    sub = [r for r in rows if f == "1" or f in r["nested"]]
    units = {}
    for r in sub:
        c, h = units.get(r["unit"], (r["quota_class"], 0.0))
        units[r["unit"]] = (c, h + r["n_samples"] / r["fs_hz"] / 3600)
    alloc = M.allocate(units, p["quota"], p["alpha"], p["max_passes"], p["epochs"])
    cls = {}
    for u, (share, passes) in alloc["per_unit"].items():
        cls[units[u][0]] = cls.get(units[u][0], 0.0) + share
    tot = sum(cls.values())
    capped = [u for u, (s, ps) in alloc["per_unit"].items() if ps >= p["max_passes"] - 1e-9]
    print(f, "classi", {k: round(100 * v / tot, 2) for k, v in sorted(cls.items())}, "quota non assegnabile", {k: round(v, 4) for k, v in alloc["unused_quota"].items() if v > 1e-9},
          "passaggi max", round(max(ps for _, ps in alloc["per_unit"].values()), 3), "unita' al tetto", len(capped))
