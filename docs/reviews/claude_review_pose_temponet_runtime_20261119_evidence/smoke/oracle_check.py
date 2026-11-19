"""Independent check of locked predictions and one report cell from the run2 tree."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(sys.argv[1])
plan = json.loads(Path(sys.argv[2]).read_text())
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("E2E_OUT", "/tmp")
from pose_temponet import heads, metrics as M
import test_e2e_chain as drv

rng = np.random.default_rng(20261119)
W = rng.normal(size=(384, 20))/np.sqrt(384)
train = drv.build_cache(plan["producers"]["train"]["users"], 7, "train", rng, W)
val = drv.build_cache(plan["producers"]["val"]["users"], 2, "val", rng, W)
lengths = np.asarray([m["n_patches"] for m in val["meta"]])
available = np.arange(160)[None, :] < lengths[:, None]

for head in ():
    states = torch.load(root/"refit"/head/"refit_states.pt", map_location="cpu", weights_only=False)
    for seed in (0, 1, 2):
        model = heads.make_head(head, seed=seed)
        model.load_state_dict(states["refit_states"][seed]["model"])
        model.eval()
        x = (val["pre"] if head == "TCN_pre" else val["lat"]).astype(np.float32)
        x = np.where(available[..., None], x, 0).astype(np.float32)
        with torch.no_grad():
            pred = model(torch.from_numpy(x))[:, 6:-6].numpy()
        with np.load(root/"locked_inference"/f"{head}_s{seed}.npz") as z:
            saved = z["prediction"]
        print(head, seed, "max |locked - direct| =", float(np.abs(saved-pred).max()))

# Independent pooled dynamic E_dyn for TCN_post seed 0
with np.load(root/"locked_inference"/"TCN_post_s0.npz") as z:
    p = z["prediction"].astype(np.float64)
y = val["target"][:, 6:-6].astype(np.float64)
tt = np.arange(6, 154)
m = val["mask"][:, 6:-6] & ((tt[None, :] + 6) < lengths[:, None])[..., None]
dyn = np.isfinite(val["pose_var"]) & (val["pose_var"] >= M.THRESHOLD)
err = tgt = 0.
for i in np.flatnonzero(dyn):
    for j in range(20):
        t = m[i, :, j]
        if t.sum() < 2:
            continue
        pc, yc = p[i, t, j]-p[i, t, j].mean(), y[i, t, j]-y[i, t, j].mean()
        err += ((pc-yc)**2).sum(); tgt += (yc**2).sum()
report = json.loads((root/"metrics"/"report.json").read_text())
print("oracle E_dyn TCN_post s0", err/tgt, "report", report["heads"]["TCN_post"]["dynamic"]["per_seed"]["0"]["E_dyn"])
print("dynamic windows", int(dyn.sum()), "of", len(dyn), "static group", report["heads"]["TCN_post"]["static"]["per_seed"]["0"]["E_dyn"])
