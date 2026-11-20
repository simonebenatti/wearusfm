"""Lens 4 smoke test: benchmark.benchmark_cpu end to end at production geometry (160x384, L13=4992)
on a tiny 80-window/40-user train fixture; then gate_and_pack with the stubbed GPU fixture receipt."""
import sys, copy, json, tempfile, time
from pathlib import Path
import numpy as np, torch
from pose_temponet import benchmark, train, resources
sys.path.insert(0, "tests/cpu")
from test_pose_temponet_resources import receipts
profile = {"torch": str(torch.__version__), "precision": "float32_no_autocast", "deterministic_algorithms": True,
           "cudnn_deterministic": True, "cudnn_benchmark": False, "cudnn_tf32": False, "matmul_tf32": False,
           "threads": 8, "interop_threads": 1, "cublas_workspace": ":4096:8"}
train.configure_numerics = lambda: dict(profile)
benchmark._hardware = lambda: {"fixture": "cpu"}
out_root = Path(tempfile.mkdtemp(dir=sys.argv[1]))
rng = np.random.default_rng(1)
n = 80
target = rng.standard_normal((n,160,20)).astype(np.float32)
data = {"config": {"split": "train", "which": "teacher", "checkpoint_step": 49046},
        "lat": rng.standard_normal((n,160,384)).astype(np.float16), "target": target,
        "mask": np.ones_like(target, dtype=bool), "std": np.ones((n,20), np.float32),
        "pose_var": rng.uniform(0,.6,n), "meta": [{"user": f"u{i//2:02d}", "n_patches": 160, "window": i} for i in range(n)]}
users = sorted({m["user"] for m in data["meta"]})
t0 = time.time()
r = benchmark.benchmark_cpu(data, input_loader=lambda: copy.deepcopy(data), output_dir=out_root/"cost_only",
                            allowed_train_users=users, input_identity="pin")
print("cpu benchmark wall s", round(time.time()-t0,1), "measured", r.get("measured"), "status", r.get("status"))
print("fold projections s", r.get("fold_projection_seconds"), "report s", r.get("report_projection_seconds"))
print("patterns per design", [[len(d["patterns"]) for d in f["designs"]] for f in r.get("folds", [])])
json.dumps(benchmark_plain := json.loads(json.dumps(r, default=float)), allow_nan=False)
_, gpu = receipts()
gpu["population"] = r["population"]
for head in resources.HEADS:
    for part, c in gpu["cells"][head].items():
        nf = r["population"]["fit_fold_windows"]; fw = r["population"]["fold_windows"]
        if part == "full": c.update(train_windows=r["population"]["fit_train_windows"], eval_windows=r["population"]["train_windows"])
        else:
            f = int(part[-1]); c.update(train_windows=nf[f], eval_windows=fw[1-f])
g = resources.gate_and_pack(benchmark_plain, gpu, work_free_bytes=20*2**30, pending_write_bytes=0, persistent_output_bytes=7*2**30)
print("gate accepted CPU receipt; failing gates:", [k for k, v in g["gates"].items() if not v])
