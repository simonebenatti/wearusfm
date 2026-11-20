import sys, time
from pathlib import Path
import numpy as np
ROOT = Path("/Users/simonebenatti/dev/wearusfm/.claude/worktrees/review-pose-heads-2f98211")
sys.path.insert(0, str(ROOT / "tests" / "cpu"))
from pose_temponet import metrics as M
from test_pose_temponet_data import fixture
users = tuple(f"u{i:02d}" for i in range(40))
data, _ = fixture(users=users, windows=24)   # 960 windows, ~ train-like shape
data.pop("lat"); data.pop("pre")
rng = np.random.default_rng(1)
data["pose_var"][:] = rng.random(len(data["meta"])).astype(np.float32)*0.5
pred = (data["target"]*.75).astype(np.float32)
for bw in (64, 32, 64, 32):
    t = time.perf_counter(); M.evaluate(pred, data, per_joint=True, batch_windows=bw); dt = time.perf_counter()-t
    print("per_joint batch", bw, "windows", len(data["meta"]), "seconds", round(dt, 3), "s/window", dt/len(data["meta"]))
for bw in (64, 32):
    t = time.perf_counter(); M.evaluate(pred[:, 6:-6].astype(np.float64), data, batch_windows=bw); dt = time.perf_counter()-t
    print("no per_joint batch", bw, "seconds", round(dt, 3))
