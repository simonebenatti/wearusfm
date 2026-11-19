"""Relative cost of anchor_targets for historical 25/50 vs new 100/125 windows (synthetic signals)."""
import sys, time
import numpy as np
from wearusfm.model import anchor_targets as AT

shapes = [("emg2pose-like", 16, 2000.0, 4.0), ("emg2qwerty-like", 32, 2000.0, 4.0),
          ("db2-like", 12, 2000.0, 4.0), ("grabmyo-like", 28, 2048.0, 4.0),
          ("csl-like", 192, 2048.0, 4.0), ("hyser-like", 256, 2048.0, 4.0),
          ("db5-like", 16, 200.0, 4.0), ("1s-2k", 16, 2000.0, 1.0), ("fs1926", 16, 1926.0, 4.0)]
rng = np.random.default_rng(0)
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
tot = {"old": 0.0, "new": 0.0}
for name, c, fs, sec in shapes:
    x = rng.standard_normal((c, int(fs * sec))).astype(np.float32)
    res = {}
    for label, rms, fast in (("old", 25.0, 50.0), ("new", 100.0, 125.0)):
        kw = dict(patch_ms=25.0, guard=(200, 200), multiscale=True, rms_window_ms=rms, ms_fast_window_ms=fast)
        AT.anchor_targets(x, fs, 450.0, **kw)  # warm-up
        best = []
        for _ in range(reps):
            t0 = time.perf_counter(); AT.anchor_targets(x, fs, 450.0, **kw); best.append(time.perf_counter() - t0)
        res[label] = float(np.median(best))
        tot[label] += res[label]
    print(f"{name:16s} C={c:3d} fs={fs:6.0f} T={sec}s old={res['old']*1e3:8.2f} ms new={res['new']*1e3:8.2f} ms ratio={res['new']/res['old']:.3f}")
print(f"sum old={tot['old']*1e3:.1f} ms new={tot['new']*1e3:.1f} ms ratio={tot['new']/tot['old']:.3f}")
