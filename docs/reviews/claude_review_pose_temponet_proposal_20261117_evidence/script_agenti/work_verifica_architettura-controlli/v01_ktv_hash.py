# Verifica indipendente: key_time_valid_sha256 delle ricevute = hash di (n,160) tutta vera?
import hashlib, json, sys, glob, os
import numpy as np
SNAP = sys.argv[1]
def my_hash(arr):
    # reimplementazione letta da scripts/pose_temporal_readout.py righe 41-47 (non importata)
    d = hashlib.sha256()
    a = np.ascontiguousarray(arr)
    d.update(json.dumps([a.dtype.str, a.shape]).encode())
    d.update(memoryview(a).cast("B"))
    return d.hexdigest()
def find(obj, key, path=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = path + "/" + k
            if k == key: out.append((p, v))
            out += find(v, key, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += find(v, key, path + f"[{i}]")
    return out
rec = sorted(glob.glob(os.path.join(SNAP, "results/infra/consolidation_long_20261105/execution/receipts/LONG-G*-pose-*-20261105.json")))
for f in rec:
    j = json.load(open(f))
    hs = find(j, "key_time_valid_sha256")
    nw = find(j, "n_windows")
    print(os.path.basename(f), "n_windows:", [v for _, v in nw][:3], "ktv entries:", len(hs))
    for p, h in hs:
        n_candidates = sorted(set(v for _, v in nw if isinstance(v, int)))
        res = []
        for n in n_candidates:
            full = np.ones((n, 160), dtype=bool)
            res.append((n, my_hash(full) == h))
            # controprova: una patch falsa
            alt = full.copy(); alt[0, 159] = False
            res.append((n, "una_falsa", my_hash(alt) == h))
        print("   ", p, h[:16], res)
