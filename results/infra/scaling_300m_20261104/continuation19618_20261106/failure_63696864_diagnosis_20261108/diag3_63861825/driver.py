import hashlib, importlib.util, json, os, platform, sys
from dataclasses import fields
from pathlib import Path
code = Path(os.environ["CODE_DIR"]).resolve()
sys.path.insert(0, str(code / "src"))
OUT, MODE, argv = Path(sys.argv[1]), sys.argv[2], json.loads(Path(sys.argv[3]).read_text())
rank = int(os.environ.get("RANK", "0"))
import numpy as np
import torch
from wearusfm.training import run as R
from wearusfm.training import pairing as PR


def components(batch):
    out = {}
    for f in fields(batch):
        v = getattr(batch, f.name)
        if f.name == "pose" and v is None:
            continue
        dg = hashlib.sha256(); PR._update(dg, v); out[f.name] = dg.hexdigest()
    return out


def libs():
    names = set()
    for line in open("/proc/self/maps"):
        parts = line.split()
        b = os.path.basename(parts[-1]) if len(parts) >= 6 else ""
        if any(k in b.lower() for k in ("omp", "blas", "mkl", "lapack", "cuda", "nccl", "cudnn", "fft", "triton", "libm.so", "libm-")):
            names.add(b)
    return sorted(names)


def denormal_flushed():
    a = np.array([1e-40], dtype=np.float32)
    return bool((a * np.float32(1.0))[0] == 0.0)


class Done(Exception):
    pass


orig = R._fixed_eval_set


def patched(index, cfg, scales, window_s):
    batches, ident = orig(index, cfg, scales, window_s)
    rec = {"mode": MODE, "rank": rank, "node": platform.node(), "ident": ident, "components": [components(b) for b in batches],
           "libs": libs(), "denormal_flushed": denormal_flushed(), "torch_threads": torch.get_num_threads(),
           "transformers_loaded": "transformers" in sys.modules, "cuda_initialized": torch.cuda.is_initialized(),
           "run_file": R.__file__, "numpy": np.__version__, "torch": torch.__version__,
           "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RANK", "WORLD_SIZE", "LOCAL_RANK")}}
    (OUT / f"{MODE}_rank{rank}.json").write_text(json.dumps(rec, indent=1))
    raise Done()


R._fixed_eval_set = patched
spec = importlib.util.spec_from_file_location("train_jepa_cli", code / "scripts/train_jepa.py")
CLI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CLI)
try:
    CLI.main(argv)
except Done:
    print(f"DIAG {MODE} rank {rank}: insieme fisso calcolato e salvato", flush=True)
    os._exit(3)
print(f"DIAG {MODE} rank {rank}: il trainer e' andato oltre l'insieme fisso senza calcolarlo?", flush=True)
os._exit(4)
