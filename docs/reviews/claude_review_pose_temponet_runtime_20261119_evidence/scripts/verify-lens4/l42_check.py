"""L4-2: mirror benchmark_gpu's order for one cell (CPU, production fold shape) and
time the ten save samples exactly as benchmark.py:272 does; then repeat with the
dataset identity touched beforehand (the proposed warm-up)."""
import sys, time, tempfile
from pathlib import Path
import numpy as np, torch
from pose_temponet import benchmark as B, train, heads, resources, metrics, data as D
torch.set_num_threads(1)
n = 954
rng = np.random.default_rng(0)
target = rng.standard_normal((n, 160, 20)).astype(np.float32)
data = {"lat": rng.standard_normal((n, 160, 384)).astype(np.float16), "target": target,
        "mask": np.ones((n, 160, 20), bool), "std": np.ones((n, 20), np.float32),
        "meta": [{"user": f"u{i % 20:02d}", "n_patches": 160, "w": i} for i in range(n)]}
ix = np.arange(n)
out = Path(tempfile.mkdtemp(dir=sys.argv[1]))
for warm in (False, True):
    fitting = B._dataset(data, ix, "post")
    traj = train.Trajectory(lambda seed, f: heads.make_head("L0", seed), fitting, lr=1e-3, seed=0, fold=0, device="cpu")
    traj.train_to_step(1)                       # (warm-up train; eval does not touch identity)
    if warm:
        _ = fitting.identity                    # proposed fix: one untimed identity/state access
    ck = out / f"c{int(warm)}.pt"
    save = []
    for epoch in range(10):
        traj.train_to_step(epoch + 2)
        t0 = time.perf_counter(); digest = train.save_checkpoint(ck, traj.state()); save.append(time.perf_counter() - t0)
        traj.restore(train.load_checkpoint(ck, expected_sha256=digest))
    print("identity pre-touched" if warm else "as in benchmark.py ", [round(v, 4) for v in save],
          "median", round(float(np.median(save)), 4), "p95", round(resources.percentile95(save), 4))
t0 = time.perf_counter(); _ = B._dataset(data, ix, "post").identity; print("fresh identity hash s", round(time.perf_counter() - t0, 4))
