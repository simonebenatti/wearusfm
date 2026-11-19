"""Lens 4 smoke test: run benchmark.benchmark_gpu end to end with CUDA calls stubbed to CPU,
then feed its receipt to resources.gate_and_pack (structure check only, fixture numbers)."""
import sys, copy, json, tempfile
from pathlib import Path
import numpy as np, torch
from pose_temponet import benchmark, train, resources
out_root = Path(tempfile.mkdtemp(dir=sys.argv[1]))
profile = {"torch": str(torch.__version__), "precision": "float32_no_autocast", "deterministic_algorithms": True,
           "cudnn_deterministic": True, "cudnn_benchmark": False, "cudnn_tf32": False, "matmul_tf32": False,
           "threads": 8, "interop_threads": 1, "cublas_workspace": ":4096:8"}
train.configure_numerics = lambda: dict(profile)
torch.cuda.is_available = lambda: True
torch.cuda.synchronize = lambda *a, **k: None
torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
torch.cuda.max_memory_reserved = lambda *a, **k: 1
benchmark._hardware = lambda: {"fixture": "cpu-stubbed"}
Orig = train.Trajectory
class CpuTrajectory(Orig):
    def __init__(self, *a, device="cpu", **k): super().__init__(*a, device="cpu", **k)
benchmark.train.Trajectory = CpuTrajectory
orig_eval = benchmark._eval_model
benchmark._eval_model = lambda m, d, p, t, device="cuda": orig_eval(m, d, p, t, device="cpu")
orig_batch = train.TensorTrainingSet.batch
train.TensorTrainingSet.batch = lambda self, pos, device: orig_batch(self, pos, torch.device("cpu"))
rng = np.random.default_rng(0)
n = 80
target = rng.standard_normal((n,160,20)).astype(np.float32)
data = {"config": {"split": "train", "which": "teacher", "checkpoint_step": 49046},
        "lat": rng.standard_normal((n,160,384)).astype(np.float16), "pre": rng.standard_normal((n,160,384)).astype(np.float16),
        "target": target, "mask": np.ones_like(target, dtype=bool), "std": np.ones((n,20), np.float32),
        "pose_var": rng.uniform(0,.6,n), "meta": [{"user": f"u{i//2:02d}", "n_patches": 160, "window": i} for i in range(n)]}
users = sorted({m["user"] for m in data["meta"]})
proxy = out_root/"train-proxy.bin"; proxy.write_bytes(b"x"*(1<<20))
r = benchmark.benchmark_gpu(data, input_loader=lambda: copy.deepcopy(data), output_dir=out_root/"cost_only",
                            reference_byte_count=652548847, reference_proxy_path=proxy,
                            allowed_train_users=users, input_identity="pin")
print("gpu receipt keys:", sorted(r))
print("population:", r["population"])
print("storage projected GiB:", r["storage"]["projected_bytes"]/2**30)
cell = r["cells"]["TCN_post"]["fold0"]
print("TCN_post fold0 save samples:", [round(x,4) for x in cell["save_seconds"]], "p95", round(cell["statistics"]["save"]["p95"],4), "median", round(cell["statistics"]["save"]["median"],4))
cpu = {"schema": "pose-temponet-cost-v1", "kind": "cpu", "train_only": True, "measured": True, "test_only": False,
       "input_identity": "pin", "numeric_profile": dict(profile), "hardware": {"fixture": 1}, "population": r["population"],
       "fold_projection_seconds": [400., 420.], "report_projection_seconds": 100., "all_patterns_measured": True,
       "all_alphas_measured": True, "grid_combinations_measured": [1936, 1936], "elapsed_seconds": 100.,
       "memory": {"peak_rss_bytes": 2**30, "projected_peak_bytes": 2*2**30}}
g = resources.gate_and_pack(cpu, json.loads(json.dumps(r)), work_free_bytes=20*2**30, pending_write_bytes=0,
                            persistent_output_bytes=r["persistent_output_bytes"])
print("gate structure OK; all_passed", g["all_passed"], {k: v for k, v in g["gates"].items() if not v})
print("projection", {k: round(v,3) if isinstance(v,float) else v for k, v in g["projection"].items()})
