"""Unmocked benchmark.benchmark_gpu logic at the REAL train size with all five heads, device redirected to CPU
(as tests/cpu/test_pose_temponet_runtime_fixes.py does). Timings mean nothing; wiring and structural gates do."""
import json, resource, sys, time, traceback
from pathlib import Path
import numpy as np, torch
sys.path[:0] = ["scripts", "src"]
from pose_temponet import provenance as P, data as D, benchmark as B, runner as R, train as T, resources

root = Path(sys.argv[1]).resolve()
plan = json.loads((root/"plan.json").read_text())
allowed = plan["producers"]["train"]["users"]
identity = {"execution_sha256": "e"*64}
T.configure_numerics = lambda: T.numerical_profile()
torch.cuda.is_available = lambda: True
torch.cuda.synchronize = lambda *a, **k: None
torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
torch.cuda.max_memory_reserved = lambda *a, **k: 1
original_trajectory, original_score, original_batch = T.Trajectory, B._production_score, B._prediction_batch
T.Trajectory = lambda *a, **k: original_trajectory(*a, **{**k, "device": "cpu"})
B._production_score = lambda *a, **k: original_score(*a, **{**k, "device": "cpu"})
B._prediction_batch = lambda *a, **k: original_batch(*a, **{**k, "device": "cpu"})
B._validation_surrogate = lambda d: (18548*160*(384*4+20*4+20*4+20), B._rss())      # size only: no 4.6 GB allocation on the Mac
loader = lambda: P.read_verified_inputs(plan, "train", stage="gpu_benchmark")
cache, first = B._load_measurement(loader, identity)
out = root/"bench_gpu"
started = time.perf_counter()
try:
    result = B.benchmark_gpu(cache, input_loader=loader, input_measurement=first, output_dir=out/"cost_only",
                             reference_byte_count=652549847, reference_proxy_path=plan["inputs"]["cache_train"]["path"],
                             allowed_train_users=allowed, input_identity=identity)
    text = json.dumps(R._plain(result), sort_keys=True, indent=2, allow_nan=False)
    (out/"benchmark.json").write_text(text)
    print("benchmark_gpu logic OK | elapsed s", round(result["elapsed_seconds"], 1), "| population", result["population"],
          "| files", len(list((out/"cost_only").iterdir())), "| strict JSON bytes", len(text), flush=True)
    for head, parts in result["cells"].items():
        print("  ", head, {part: (c["train_windows"], c["eval_windows"], c["excluded_fitting_windows"], c["steps_per_epoch"], c["checkpoint_file_bytes"]) for part, c in parts.items()}, flush=True)
    print("   storage projected GiB", round(result["storage"]["projected_bytes"]/2**30, 3), "| io", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in result["io"].items() if k != "source_identity"}, flush=True)
    cpu_path = root/"bench_cpu_repair"/"benchmark.json"
    if cpu_path.exists():
        cpu = json.loads(cpu_path.read_text()); gpu = json.loads(text)
        for receipt in (cpu, gpu):       # the Mac ran one thread: only this field is edited to reach the structural checks
            receipt["numeric_profile"]["threads"] = 8; receipt["numeric_profile"]["interop_threads"] = 1
            for key, value in {"deterministic_algorithms": True, "cudnn_deterministic": True, "cudnn_benchmark": False,
                               "cudnn_tf32": False, "matmul_tf32": False, "cublas_workspace": ":4096:8"}.items():
                receipt["numeric_profile"][key] = value
        try:
            gate = resources.gate_and_pack(cpu, gpu, work_free_bytes=44460445696, pending_write_bytes=25769803776,
                                           persistent_output_bytes=gpu["persistent_output_bytes"])
            print("gate_and_pack structural checks passed; gates:", gate["gates"], flush=True)
            print("   matrix:", gate["matrix"], flush=True)
        except Exception as error:
            print("gate_and_pack raised:", type(error).__name__, error, flush=True)
except Exception as error:
    frame = traceback.extract_tb(error.__traceback__)[-1]
    print("benchmark_gpu logic FAILED after", round(time.perf_counter()-started, 1), "s:", type(error).__name__, error,
          "| at", Path(frame.filename).name, frame.lineno, frame.name, flush=True)
print("process peak RSS GiB:", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30, 2), flush=True)
