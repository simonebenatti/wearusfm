"""Diagnosi 4: identita' dell'insieme fisso e del pannello in un dato ordine di import (torch_first | numpy_first | run_first)."""
import sys
MODE = sys.argv[1]
if MODE == "torch_first":          # come `python3 scripts/train_jepa.py` (launcher di GPT): torch prima di tutto
    import torch  # noqa: F401
    import numpy as np
elif MODE == "numpy_first":        # come i miei driver di diagnosi: numpy, poi torch
    import numpy as np
    import torch  # noqa: F401
elif MODE == "run_first":          # come scaling_continue.py worker: run.py per primo (dentro: numpy, poi torch)
    from wearusfm.training import run as R  # noqa: F401
    import numpy as np
    import torch  # noqa: F401
else:
    raise SystemExit(f"modo sconosciuto: {MODE}")
import hashlib, importlib.util, io, json, os, platform, time
from contextlib import redirect_stdout
from dataclasses import fields, is_dataclass
from pathlib import Path
from wearusfm.training import run as R
from wearusfm.data import pretraining_loader as L
from wearusfm.training import pairing as PR

OUT = Path(os.environ["DIAG_OUT"])
code = Path(os.environ["CODE_DIR"]).resolve()
S = Path("/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs")
PREV = S / "runs/scaling_300m_20261104/T100-G-300M-5000"
B = Path(os.environ["DIAG_B"])
argv = json.loads((OUT / "argv_plain_cli.json").read_text())

spec = importlib.util.spec_from_file_location("train_jepa_cli_" + MODE, code / "scripts/train_jepa.py")
CLI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CLI)
ap = CLI.build_parser()
args = ap.parse_args(argv)
R.check_input_files(args.manifest, args.window_seconds, args.nested, args.manifest_sha256)
window_s = json.loads(args.window_seconds.read_text())["window_s"] if args.window_seconds else None
cfg = CLI.build_config(args, ap.error)
scales = json.loads(args.scales.read_text())["scales"] if args.scales else {}

skips = {}
_orig_skip = L.PretrainLoader._skip
def _skip(self, key, why):
    skips[key] = why
    return _orig_skip(self, key, why)
L.PretrainLoader._skip = _skip

def ftz():
    d = np.float32(1e-39)
    return {"ftz": bool(d * np.float32(1.0) == 0), "daz": bool(d + np.float32(0.0) == 0)}

def libs():
    out = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 6 and ".so" in parts[5]:
            out.add(os.path.basename(parts[5]))
    return sorted(out)

def components(batch):
    out = {}
    for f in fields(batch):
        dg = hashlib.sha256(); PR._update(dg, getattr(batch, f.name)); out[f.name] = dg.hexdigest()
    return out

def walk_arrays(obj, prefix, sink, depth=0):
    """Salva ogni ndarray/tensore raggiungibile (dataclass, dict, list/tuple) come .npy: per il confronto numerico fra varianti."""
    if depth > 4 or obj is None:
        return
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        np.save(sink / f"{prefix}.npy", obj); return
    if is_dataclass(obj):
        for f in fields(obj):
            walk_arrays(getattr(obj, f.name), f"{prefix}.{f.name}", sink, depth + 1)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            walk_arrays(v, f"{prefix}.{k}", sink, depth + 1)
    elif isinstance(obj, (list, tuple)) and obj and not isinstance(obj[0], (str, int, float, bool)):
        for i, v in enumerate(obj[:64]):
            walk_arrays(v, f"{prefix}[{i}]", sink, depth + 1)

def stats(batch):
    out = {"skipped_sessions": getattr(batch, "skipped_sessions", None)}
    try:
        s = np.asarray(batch.signals)
        nz = np.abs(s[s != 0])
        out.update({"signals_dtype": str(s.dtype), "signals_shape": list(s.shape), "zeros": int((s == 0).sum()),
                    "nonfinite": int((~np.isfinite(s)).sum()), "min_abs_nonzero": float(nz.min()) if nz.size else None,
                    "subnormal_f32": int((nz < 1.1754944e-38).sum()) if nz.size else 0})
    except Exception as e:  # noqa: BLE001
        out["signals_error"] = f"{type(e).__name__}: {e}"
    return out

t0 = time.time()
index = R.load_index(args.manifest, args.root, cfg.datasets, cfg.nested)
if cfg.nested is not None:
    R.check_derived_binding(index.params, cfg, window_s)
scales = dict(scales)
batches, ident = R._fixed_eval_set(index, cfg, scales, window_s)
t1 = time.time()
panel, pident = R._pose_panel_set(index, cfg, scales, window_s)
t2 = time.time()
(OUT / f"{MODE}_fixed_ident.json").write_text(json.dumps(ident))
(OUT / f"{MODE}_panel_ident.json").write_text(json.dumps(pident))
dump = OUT / f"{MODE}_arrays"; dump.mkdir(exist_ok=True)
for i in (0, 1):
    walk_arrays(batches[i], f"fixed{i}", dump)
    walk_arrays(panel[i], f"panel{i}", dump)

A_fixed = json.loads((PREV / "fixed_eval_set.json").read_text())
A_panel = json.loads((PREV / "pose_panel_set.json").read_text())
B_fixed = json.loads((B / "ident_run1.json").read_text())
B_panel = json.loads((B / "pose_panel_ident_now.json").read_text())
def cmp(x, ref):
    eq = [a == b for a, b in zip(x["batch_sha256"], ref["batch_sha256"])]
    return {"identical": x == ref, "windows_equal": x.get("windows") == ref.get("windows"), "hash_equal": eq, "n_hash_equal": sum(eq),
            "n": len(eq)}
buf = io.StringIO()
try:
    with redirect_stdout(buf):
        np.show_runtime()
except Exception as e:  # noqa: BLE001
    buf.write(f"show_runtime non disponibile: {e}")
report = {"protocol": "scaling-diag-importorder-20261108-v4", "mode": MODE, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
          "node": platform.node(), "code_sha": os.environ.get("EXPECTED_SHA"), "run_file": R.__file__, "numpy_file": np.__file__,
          "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
          "ftz": ftz(), "torch_threads": torch.get_num_threads(), "omp": os.environ.get("OMP_NUM_THREADS"), "libs": libs(),
          "n_modules": len(sys.modules), "numpy_runtime": buf.getvalue(), "skips": skips,
          "timing_s": {"index_and_fixed": round(t1 - t0, 1), "panel": round(t2 - t1, 1)},
          "fixed_vs_A": cmp(ident, A_fixed), "fixed_vs_B": cmp(ident, B_fixed),
          "panel_vs_A": cmp(pident, A_panel), "panel_vs_B": cmp(pident, B_panel),
          "fixed_components": [components(b) for b in batches], "fixed_stats": [stats(b) for b in batches],
          "panel_stats": [stats(b) for b in panel],
          "panel_unchanged_A_vs_B": [a == b for a, b in zip(A_panel["batch_sha256"], B_panel["batch_sha256"])]}
(OUT / f"{MODE}_report.json").write_text(json.dumps(report, indent=1))
print(f"DIAG4 {MODE}: insieme fisso = A {report['fixed_vs_A']['n_hash_equal']}/8, = B {report['fixed_vs_B']['n_hash_equal']}/8; "
      f"pannello = A {report['panel_vs_A']['n_hash_equal']}/64, = B {report['panel_vs_B']['n_hash_equal']}/64; "
      f"finestre uguali ad A {report['fixed_vs_A']['windows_equal']}/{report['panel_vs_A']['windows_equal']}; ftz {report['ftz']}; "
      f"salti {len(skips)}; thread torch {report['torch_threads']}; librerie {len(report['libs'])}; moduli {report['n_modules']}", flush=True)
