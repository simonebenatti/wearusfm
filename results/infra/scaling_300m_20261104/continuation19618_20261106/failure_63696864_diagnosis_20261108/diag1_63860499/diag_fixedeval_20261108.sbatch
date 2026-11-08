#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --ntasks=1
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_diag_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_diag_%j.out
# Diagnosi (sola lettura dei dati, 0 GPU) del fallimento del job 63696864: ricalcola l'identita' dell'insieme fisso della
# diagnostica di stabilita' (training/run._fixed_eval_set) con il codice congelato del checkout e la confronta con
# fixed_eval_set.json del run a 5.000, seminato nella cartella della continuazione. Non tocca run, checkpoint o ricevute.
set -euo pipefail
: "${WORK:?WORK}" "${CODE_DIR:?CODE_DIR}" "${EXPECTED_SHA:?EXPECTED_SHA}" "${DIAG_OUT:?DIAG_OUT}"
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
echo "nodo $(hostname); cpu: $(grep -m1 'model name' /proc/cpuinfo)"
python3 - <<'PY'
import importlib.util, json, os, platform, sys, time
from pathlib import Path
code = Path(os.environ["CODE_DIR"]).resolve()
sys.path.insert(0, str(code / "src"))
import numpy as np
import torch
from wearusfm.training import run as R
from wearusfm.data import pretraining_loader as L

S = Path("/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs")
REC = S / "results/scaling_300m_20261104/continuation19618_v1"
NEW = S / "runs/scaling_300m_20261104/T100-G-300M-19618-from5000"
PREV = S / "runs/scaling_300m_20261104/T100-G-300M-5000"
OUT = Path(os.environ["DIAG_OUT"])
OUT.mkdir(parents=True, exist_ok=False)
launch = json.loads((REC / "launch.json").read_text())
argv = list(launch["trainer_argv"])
assert argv[0] == "scripts/train_jepa.py", argv[:2]
argv = argv[1:]
clean = []
i = 0
while i < len(argv):
    a = argv[i]
    if a == "--distributed":
        i += 1; continue
    if a in ("--out-dir", "--num-workers", "--device"):
        i += 2; continue
    clean.append(a); i += 1
clean += ["--out-dir", str(OUT / "cli_out"), "--num-workers", "0", "--device", "cpu"]
seeded = json.loads((NEW / "fixed_eval_set.json").read_text())
prev = json.loads((PREV / "fixed_eval_set.json").read_text())
report = {"protocol": "scaling-diag-fixedeval-20261108-v1", "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "failed_job": "63696864",
          "code_sha": os.environ.get("EXPECTED_SHA"), "node": platform.node(), "cpu": platform.processor(),
          "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
          "threads": {"omp": os.environ.get("OMP_NUM_THREADS"), "torch": torch.get_num_threads()},
          "trainer_argv_used": clean, "seeded_equals_previous": seeded == prev}
skips = {}
_orig_skip = L.PretrainLoader._skip
def _skip(self, key, why):
    skips[key] = why
    return _orig_skip(self, key, why)
L.PretrainLoader._skip = _skip

def diag(cfg, manifest, roots, out_dir, *, scales=None, window_s=None, **kw):
    t0 = time.time()
    index = R.load_index(manifest, roots, cfg.datasets, cfg.nested)
    if cfg.nested is not None:
        R.check_derived_binding(index.params, cfg, window_s)
    scales = dict(scales or {})
    report["index"] = {"rows": len(index.rows), "roots": [str(r) for r in index.roots], "params": index.params,
                       "datasets": sorted({r["dataset"] for r in index.rows}), "scales": len(scales), "window_s": None if window_s is None else len(window_s)}
    t1 = time.time()
    batches, ident = R._fixed_eval_set(index, cfg, scales, window_s)
    report["timing_s"] = {"index": round(t1 - t0, 2), "fixed_set": round(time.time() - t1, 2)}
    (OUT / "ident_now.json").write_text(json.dumps(ident))
    sw, nw = seeded["windows"], ident["windows"]
    first = next((k for k, (a, b) in enumerate(zip(sw, nw)) if a != b), None)
    report["compare"] = {
        "seed_equal": seeded["seed"] == ident["seed"], "seed_seeded": seeded["seed"], "seed_now": ident["seed"],
        "windows": {"n_seeded": len(sw), "n_now": len(nw), "equal_positions": sum(a == b for a, b in zip(sw, nw)),
                    "first_difference": first, "first_seeded": sw[first] if first is not None else None,
                    "first_now": nw[first] if first is not None else None,
                    "sessions_only_seeded": sorted({w[0] for w in sw} - {w[0] for w in nw}),
                    "sessions_only_now": sorted({w[0] for w in nw} - {w[0] for w in sw})},
        "batch_sha256": {"n_seeded": len(seeded["batch_sha256"]), "n_now": len(ident["batch_sha256"]),
                         "equal": [a == b for a, b in zip(seeded["batch_sha256"], ident["batch_sha256"])]},
        "identical": seeded == ident}
    report["skipped_sessions"] = skips
    files = {}
    by_key = {f"{r['dataset']}/{r['subject']}/{r['session']}": r for r in index.rows}
    for key in sorted({w[0] for w in sw} | {w[0] for w in nw}):
        row = by_key.get(key)
        entry = {"in_index": row is not None, "in_seeded": any(w[0] == key for w in sw), "in_now": any(w[0] == key for w in nw)}
        if row is not None:
            try:
                p = index.path_of(row)
                st = p.stat() if p.exists() else None
                entry.update({"path": str(p), "exists": st is not None, "bytes": st.st_size if st else None,
                              "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(st.st_mtime)) if st else None})
            except Exception as e:  # noqa: BLE001
                entry["path_error"] = f"{type(e).__name__}: {e}"
        files[key] = entry
    report["session_files"] = files
    return {"diagnosis": "done"}

R.train = diag
spec = importlib.util.spec_from_file_location("train_jepa_cli", code / "scripts/train_jepa.py")
CLI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CLI)
status = None
try:
    status = CLI.main(clean)
except BaseException as e:  # noqa: BLE001
    report["cli_error"] = f"{type(e).__name__}: {e}"
    raise
finally:
    report["cli_status"] = status
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    c = report.get("compare")
    if c:
        print("DIAGNOSI: identico" if c["identical"] else "DIAGNOSI: diverso", json.dumps({k: v for k, v in c.items() if k != "windows"}),
              json.dumps({k: v for k, v in c["windows"].items() if not k.startswith("sessions_only")}), "salti:", len(skips), flush=True)
PY
echo "diagnosi scritta in $DIAG_OUT"
