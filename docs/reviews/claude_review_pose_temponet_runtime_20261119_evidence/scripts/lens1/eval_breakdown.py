import time, numpy as np, torch
exec(open('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/poseheads/agents/lens1/eval_overhead.py').read().split("for name, fn in")[0])
def t(fn, reps=5):
    fn(); ts=[]
    for _ in range(reps):
        t0=time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    return np.median(ts)*1e3
batches = [held[s:s+32] for s in range(0, len(held), 32)]
pos = [np.arange(s, min(s+32, len(held))) for s in range(0, len(held), 32)]
print("prod view_inputs        %.1f ms" % t(lambda: [D.view_inputs(bundle.cache, "post", r) for r in batches]))
print("prod _metric_panel      %.1f ms" % t(lambda: [W._metric_panel(bundle, 0, r) for r in batches]))
print("prod selection_stats    %.1f ms" % t(lambda: [metrics.selection_statistics(pred148[:len(r)], W._metric_panel(bundle, 0, r), thr) for r in batches]))
print("bench cast              %.1f ms" % t(lambda: [validation.features[p[0]:p[-1]+1].to(dtype=torch.float32) for p in pos]))
print("bench subset            %.1f ms" % t(lambda: [D.subset(panel, p) for p in pos]))
print("bench selection_stats   %.1f ms" % t(lambda: [metrics.selection_statistics(pred160[:len(p)], D.subset(panel, p), thr) for p in pos]))
