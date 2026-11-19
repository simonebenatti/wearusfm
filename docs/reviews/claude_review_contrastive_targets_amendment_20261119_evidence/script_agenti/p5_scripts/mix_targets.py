"""Weighted estimate of anchor_targets CPU for the audit stream mix: historical 25/50 vs new 100/125.
Counts: old audit 64652559 coverage_partial_block (768 batches, 24576 windows); fs/channels from docs/inventario_corpus.md.
Class C (hyser, csl_hdemg) half full / half virtual (~24 ch); window length 2.5 s (loader draws 1-4 s)."""
import time, json, sys
import numpy as np
from wearusfm.model import anchor_targets as AT
counts = {"camargo2021": 876, "capgmyo": 45, "csl_hdemg": 551, "emg2pose": 6049, "emg2qwerty": 5991, "grabmyo": 1422,
          "hyser": 1310, "ninapro_db10": 2103, "ninapro_db2": 995, "ninapro_db3": 237, "ninapro_db4": 490, "ninapro_db5": 513,
          "ninapro_db6": 716, "ninapro_db7": 717, "ninapro_db8": 484, "putemg": 1012, "zhang2026": 1065}
props = {"camargo2021": [(1000., 11, 1.)], "capgmyo": [(1000., 128, 1.)], "csl_hdemg": [(2048., 168, .5), (2048., 24, .5)],
         "emg2pose": [(2000., 16, 1.)], "emg2qwerty": [(2000., 32, 1.)], "grabmyo": [(2048., 28, 1.)],
         "hyser": [(2048., 256, .5), (2048., 24, .5)], "ninapro_db10": [(1926., 12, 1.)], "ninapro_db2": [(2000., 12, 1.)],
         "ninapro_db3": [(2000., 12, 1.)], "ninapro_db4": [(2000., 12, 1.)], "ninapro_db5": [(200., 16, 1.)],
         "ninapro_db6": [(2000., 16, 1.)], "ninapro_db7": [(2000., 12, 1.)], "ninapro_db8": [(2000., 16, 1.)],
         "putemg": [(5120., 24, 1.)], "zhang2026": [(2000., 8, 1.)]}
T = float(sys.argv[1]) if len(sys.argv) > 1 else 2.5
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
rng = np.random.default_rng(0)
scale = 32768 / sum(counts.values())
tot = {"old": 0., "new": 0.}
rows = []
for d, n in counts.items():
    per = {"old": 0., "new": 0.}
    for fs, c, w in props[d]:
        x = rng.standard_normal((c, int(fs * T))).astype(np.float32)
        for label, rms, fast in (("old", 25., 50.), ("new", 100., 125.)):
            kw = dict(patch_ms=25., guard=(int(.1 * fs), int(.1 * fs)), multiscale=True, rms_window_ms=rms, ms_fast_window_ms=fast)
            AT.anchor_targets(x, fs, 450., **kw)
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter(); AT.anchor_targets(x, fs, 450., **kw); ts.append(time.perf_counter() - t0)
            per[label] += w * float(np.median(ts))
    for k in tot: tot[k] += per[k] * n * scale
    rows.append((d, n, per["old"] * 1e3, per["new"] * 1e3, per["new"] / per["old"]))
for r in sorted(rows, key=lambda r: -r[1] * r[2]):
    print(f"{r[0]:13s} n={r[1]:5d} old={r[2]:7.2f} ms new={r[3]:7.2f} ms ratio={r[4]:.2f}")
print(json.dumps({"window_s": T, "windows": 32768, "mac_target_cpu_s_old": round(tot["old"], 1),
                  "mac_target_cpu_s_new": round(tot["new"], 1), "delta_s": round(tot["new"] - tot["old"], 1),
                  "ratio": round(tot["new"] / tot["old"], 3)}))
