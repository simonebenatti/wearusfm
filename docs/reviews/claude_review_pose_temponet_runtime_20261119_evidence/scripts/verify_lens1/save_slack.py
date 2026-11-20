"""Gate charges 2*MAX_EVALS*save per fold/LR, but production on_checkpoint (_progress) never writes
a checkpoint; only Trajectory.state() on improvements. Measure both on the Mac for TCN_post (CPU)."""
import time, tempfile
from pathlib import Path
import numpy as np, torch
from pose_temponet import train as T, heads
n = 953
g = torch.Generator().manual_seed(0)
ds = T.TensorTrainingSet(torch.randn(n, 160, 384, generator=g).half(), torch.randn(n, 160, 20, generator=g),
                         torch.ones(n, 148, 20, dtype=torch.bool), [f"w{i:04d}" for i in range(n)])
tr = T.Trajectory(lambda s, f: heads.make_head("TCN_post", s), ds, lr=1e-3, seed=0, fold=0)
tr.train_to_step(3)
d = Path(tempfile.mkdtemp(dir="/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/poseheads/agents/verify_lens1"))
def med(fn, reps=10):
    fn(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    return np.median(ts)*1e3
print("state() only (production, on improvement): %.1f ms" % med(lambda: tr.state()))
print("save_checkpoint(state()) (benchmark 'save'): %.1f ms" % med(lambda: T.save_checkpoint(d/"x.pt", tr.state())))
print("checkpoint bytes:", (d/"x.pt").stat().st_size)
