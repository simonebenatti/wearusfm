"""Misura SINTETICA (Mac): lettura di NPZ compressi con la stessa struttura delle cache della posa.

Struttura come save_cache in scripts/consolidation_long_pose.py (snapshot c15aac9, righe 159-164):
np.savez_compressed con pre, lat (FP16 N x 160 x 384), key_time_valid (bool N x 160), pred e target
(FP32 N x 160 x 20), mask (bool N x 160 x 20), std (FP32 N x 20), pose_var (FP32 N), scalari e JSON.
Dati casuali (non i latenti veri), scala della cache train (N = 1907).
Uso: python bench_npz2.py write <file> | read-one <file> | read-all <file>
Ogni lettura gira in un processo nuovo, cosi' il picco RSS (ru_maxrss) misura solo la lettura.
"""
import json
import os
import resource
import sys
import time

import numpy as np

mode, path = sys.argv[1], sys.argv[2]
N, P, D, J = 1907, 160, 384, 20


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # byte su macOS


if mode == "write":
    rng = np.random.default_rng(0)
    arrays = {}
    for k in ("pre", "lat"):
        a = np.empty((N, P, D), dtype=np.float16)
        for i in range(0, N, 256):  # a blocchi, per non allocare il float32 intero
            a[i:i + 256] = rng.standard_normal((min(256, N - i), P, D), dtype=np.float32)
        arrays[k] = a
    lengths = rng.integers(20, P + 1, size=N)
    valid = np.arange(P)[None, :] < lengths[:, None]
    arrays["key_time_valid"] = valid
    arrays["pred"] = rng.standard_normal((N, P, J), dtype=np.float32) * valid[..., None]
    arrays["target"] = rng.standard_normal((N, P, J), dtype=np.float32) * valid[..., None]
    arrays["mask"] = valid[..., None] & (rng.random((N, P, J)) < 0.95)
    arrays["std"] = rng.random((N, J), dtype=np.float32) + 0.1
    arrays["pose_var"] = rng.random(N, dtype=np.float32)
    t = time.perf_counter()
    with open(path, "xb") as f:
        np.savez_compressed(f, **arrays, has_head=np.asarray(True), dim=np.asarray(D), n_recordings=np.asarray(10),
                            meta_json=np.asarray(json.dumps([{"user": "u", "n_patches": int(x)} for x in lengths])),
                            config_json=np.asarray("{}"))
    raw = sum(a.nbytes for a in arrays.values())
    print(json.dumps({"mode": mode, "numeric_bytes_uncompressed": int(raw), "file_bytes": os.path.getsize(path),
                      "ratio": os.path.getsize(path) / raw, "write_s": time.perf_counter() - t}))
else:
    base = rss()
    t = time.perf_counter()
    with np.load(path, allow_pickle=False) as a:
        keys = ["lat"] if mode == "read-one" else [k for k in a.files]
        out = {k: a[k] for k in keys}
    dt = time.perf_counter() - t
    loaded = sum(v.nbytes for v in out.values())
    print(json.dumps({"mode": mode, "keys": keys, "loaded_bytes": int(loaded), "read_s": dt,
                      "MB_per_s_uncompressed": loaded / 1e6 / dt, "rss_before": base, "peak_rss_after": rss(),
                      "peak_minus_before": rss() - base}))
