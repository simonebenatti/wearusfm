"""Misura SINTETICA (Mac, 1 thread): costo e memoria della lettura di un membro FP16 da un NPZ compresso.
Forma del membro = vista `lat` della cache train (1907 x 160 x 384, FP16). Dati casuali, non i latenti veri."""
import os, resource, sys, time, json
import numpy as np
path = sys.argv[1]
rng = np.random.default_rng(0)
N, P, D = 1907, 160, 384
t = time.perf_counter()
lat = rng.standard_normal((N, P, D), dtype=np.float32).astype(np.float16)
pre = rng.standard_normal((N, P, D), dtype=np.float32).astype(np.float16)
gen = time.perf_counter() - t
t = time.perf_counter()
with open(path, "xb") as f:
    np.savez_compressed(f, pre=pre, lat=lat)
write = time.perf_counter() - t
size = os.path.getsize(path)
raw = lat.nbytes + pre.nbytes
del lat, pre
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t = time.perf_counter()
with np.load(path, allow_pickle=False) as a:
    one = a["lat"]          # solo la vista che serve
read_one = time.perf_counter() - t
rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t = time.perf_counter()
x32 = one[:32].astype(np.float32)
cast = time.perf_counter() - t
print(json.dumps({"raw_bytes_two_views": raw, "npz_bytes": size, "ratio": size / raw, "write_compressed_s": write,
                  "read_one_view_s": read_one, "one_view_bytes": int(one.nbytes),
                  "read_MB_per_s_uncompressed": one.nbytes / 1e6 / read_one,
                  "peak_rss_before_read": rss0, "peak_rss_after_read": rss1, "cast_batch32_s": cast}, indent=1))
os.remove(path)
