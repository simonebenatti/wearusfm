import time, numpy as np
from wearusfm.model import anchor_targets as AT
rng = np.random.default_rng(0)
for fs, c, secs in ((2000., 16, 10.), (1000., 8, 10.), (4000., 16, 5.)):
    x = rng.standard_normal((c, int(fs*secs)))
    res = {}
    for rms, fast in ((25., 50.), (100., 125.)):
        AT.anchor_targets(x, fs, fs/2, multiscale=True, rms_window_ms=rms, ms_fast_window_ms=fast)
        t = time.perf_counter()
        for _ in range(5):
            AT.anchor_targets(x, fs, fs/2, multiscale=True, rms_window_ms=rms, ms_fast_window_ms=fast)
        res[(rms, fast)] = (time.perf_counter() - t) / 5
    a, b = res[(25., 50.)], res[(100., 125.)]
    print(f"fs={fs:.0f} C={c} T={secs}s: 25/50 {a*1e3:.1f} ms, 100/125 {b*1e3:.1f} ms, ratio {b/a:.2f}")
