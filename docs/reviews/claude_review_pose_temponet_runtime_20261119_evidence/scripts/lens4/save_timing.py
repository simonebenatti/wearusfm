"""Lens 4 check: does the first benchmark 'save' sample carry the one-off dataset identity hash?"""
import time, sys, tempfile
from pathlib import Path
import numpy as np, torch
from pose_temponet import train, heads, resources
torch.set_num_threads(1)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 954
rng = np.random.default_rng(0)
x = torch.from_numpy(rng.standard_normal((n,160,384)).astype(np.float16))
y = torch.from_numpy(rng.standard_normal((n,160,20)).astype(np.float32))
s = torch.ones((n,148,20), dtype=torch.bool)
ds = train.TensorTrainingSet(x, y, s, [f"w{i:05d}" for i in range(n)])
t = train.Trajectory(lambda seed, f: heads.make_head("L0", seed), ds, lr=1e-3, seed=0, fold=0, device="cpu")
t.train_to_step(1)
out = Path(tempfile.mkdtemp(dir=sys.argv[2]))
samples = []
for k in range(10):
    t0 = time.perf_counter(); train.save_checkpoint(out/"c.pt", t.state()); samples.append(time.perf_counter()-t0)
print("n_windows", n, "save samples (s):", [round(v,4) for v in samples])
print("median", round(float(np.median(samples)),4), "p95(linear)", round(resources.percentile95(samples),4),
      "p95 without first sample-type outlier ~", round(resources.percentile95(samples[1:]+[samples[1]]),4))
t0=time.perf_counter(); _=ds.identity; print("identity (cached) s", round(time.perf_counter()-t0,6))
ds2 = train.TensorTrainingSet(x, y, s, [f"w{i:05d}" for i in range(n)])
t0=time.perf_counter(); _=ds2.identity; print("identity (fresh) s", round(time.perf_counter()-t0,4))
