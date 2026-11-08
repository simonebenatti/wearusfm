#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --ntasks=1
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_diag_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_diag_%j.out
# Diagnosi 2 (sola lettura dei dati, 0 GPU) del fallimento del job 63696864. Col codice congelato del checkout:
#  a) l'insieme fisso ricalcolato due volte nello stesso processo e confrontato con la diagnosi 1 (determinismo);
#  b) hash per componente di ogni batch dell'insieme fisso (segnali, maschere, target, posa...);
#  c) il pannello della posa ricalcolato e confrontato con pose_panel_set.json del run a 5.000 (riferimento del 05/11);
#  d) i primi batch dello stream del training del processo 0 ricalcolati e confrontati con metrics.jsonl del run a 5.000.
# Non tocca run, checkpoint o ricevute.
set -euo pipefail
: "${WORK:?WORK}" "${CODE_DIR:?CODE_DIR}" "${EXPECTED_SHA:?EXPECTED_SHA}" "${DIAG_OUT:?DIAG_OUT}" "${DIAG_PREV:?DIAG_PREV}"
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
echo "nodo $(hostname); cpu: $(grep -m1 'model name' /proc/cpuinfo); flags avx512: $(grep -m1 -o 'avx512f' /proc/cpuinfo || echo no)"
python3 - <<'PY'
import hashlib, importlib.util, json, os, platform, sys, time
from dataclasses import fields
from pathlib import Path
code = Path(os.environ["CODE_DIR"]).resolve()
sys.path.insert(0, str(code / "src"))
import numpy as np
import scipy
import torch
from wearusfm.training import run as R
from wearusfm.training import pairing as PR
from wearusfm.data import pretraining_loader as L

S = Path("/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs")
REC = S / "results/scaling_300m_20261104/continuation19618_v1"
NEW = S / "runs/scaling_300m_20261104/T100-G-300M-19618-from5000"
PREV = S / "runs/scaling_300m_20261104/T100-G-300M-5000"
OUT = Path(os.environ["DIAG_OUT"])
OUT.mkdir(parents=True, exist_ok=False)
diag1 = json.loads((Path(os.environ["DIAG_PREV"]) / "ident_now.json").read_text())
launch = json.loads((REC / "launch.json").read_text())
argv = list(launch["trainer_argv"])
assert argv[0] == "scripts/train_jepa.py", argv[:2]
argv = argv[1:]
clean, i = [], 0
while i < len(argv):
    a = argv[i]
    if a == "--distributed":
        i += 1; continue
    if a in ("--out-dir", "--num-workers", "--device"):
        i += 2; continue
    clean.append(a); i += 1
clean += ["--out-dir", str(OUT / "cli_out"), "--num-workers", "0", "--device", "cpu"]
seeded = json.loads((NEW / "fixed_eval_set.json").read_text())
seeded_panel = json.loads((PREV / "pose_panel_set.json").read_text())
metrics = []
with open(PREV / "metrics.jsonl") as fh:
    for line in fh:
        metrics.append(json.loads(line))
        if len(metrics) >= 12:
            break
report = {"protocol": "scaling-diag-fixedeval-20261108-v2", "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "failed_job": "63696864",
          "code_sha": os.environ.get("EXPECTED_SHA"), "node": platform.node(),
          "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
          "threads": {"omp": os.environ.get("OMP_NUM_THREADS"), "torch": torch.get_num_threads()}, "trainer_argv_used": clean,
          "metrics_first_records": [{k: r.get(k) for k in ("step", "batch_sha256", "batch_index", "index")} for r in metrics],
          "metrics_record_keys": sorted(metrics[0].keys()) if metrics else None}
skips = {}
_orig_skip = L.PretrainLoader._skip
def _skip(self, key, why):
    skips[key] = why
    return _orig_skip(self, key, why)
L.PretrainLoader._skip = _skip

def components(batch):
    out = {}
    for f in fields(batch):
        v = getattr(batch, f.name)
        if f.name == "pose" and v is None:
            continue
        dg = hashlib.sha256(); PR._update(dg, v); out[f.name] = dg.hexdigest()
    return out

def wins(batch):
    return [[R._session_key(r), None if w.trial is None else int(w.trial), int(w.start), int(w.stop)] for r, w in zip(batch.rows, batch.windows)]

def diag(cfg, manifest, roots, out_dir, *, scales=None, window_s=None, **kw):
    t0 = time.time()
    index = R.load_index(manifest, roots, cfg.datasets, cfg.nested)
    if cfg.nested is not None:
        R.check_derived_binding(index.params, cfg, window_s)
    scales = dict(scales or {})
    report["cfg"] = {"seed": cfg.seed, "batch_size": cfg.batch_size, "fixed_eval_batches": cfg.fixed_eval_batches, "stream_segment": cfg.stream_segment,
                     "pose_panel": [cfg.pose_panel_users, cfg.pose_panel_recordings, cfg.pose_panel_windows], "rows": len(index.rows), "scales": len(scales)}
    # a) insieme fisso due volte
    t1 = time.time()
    b1, ident1 = R._fixed_eval_set(index, cfg, scales, window_s)
    t2 = time.time()
    b2, ident2 = R._fixed_eval_set(index, cfg, scales, window_s)
    t3 = time.time()
    report["fixed"] = {"twice_identical": ident1 == ident2, "equals_diag1": ident1 == diag1, "equals_seeded": ident1 == seeded,
                       "windows_equal_seeded": ident1["windows"] == seeded["windows"],
                       "hash_equal_seeded": [a == b for a, b in zip(ident1["batch_sha256"], seeded["batch_sha256"])],
                       "hash_equal_diag1": [a == b for a, b in zip(ident1["batch_sha256"], diag1["batch_sha256"])],
                       "timing_s": [round(t2 - t1, 1), round(t3 - t2, 1)]}
    # b) componenti
    c1, c2 = [components(b) for b in b1], [components(b) for b in b2]
    report["fixed_components"] = c1
    report["fixed_components_twice_equal"] = [x == y for x, y in zip(c1, c2)]
    report["fixed_components_differing_between_runs"] = sorted({k for x, y in zip(c1, c2) for k in x if x[k] != y.get(k)})
    report["fixed_batch_fields"] = {k: (type(getattr(b1[0], k)).__name__) for k in c1[0]}
    (OUT / "ident_run1.json").write_text(json.dumps(ident1))
    (OUT / "ident_run2.json").write_text(json.dumps(ident2))
    # c) pannello della posa
    t4 = time.time()
    panel_err = None
    try:
        panel, pident = R._pose_panel_set(index, cfg, scales, window_s)
        (OUT / "pose_panel_ident_now.json").write_text(json.dumps(pident))
        report["panel"] = {"identical": pident == seeded_panel, "keys_now": sorted(pident), "keys_seeded": sorted(seeded_panel),
                           "key_equal": {k: pident.get(k) == seeded_panel.get(k) for k in sorted(set(pident) | set(seeded_panel))},
                           "n_batches": [len(pident.get("batch_sha256", [])), len(seeded_panel.get("batch_sha256", []))],
                           "hash_equal_count": sum(a == b for a, b in zip(pident.get("batch_sha256", []), seeded_panel.get("batch_sha256", []))),
                           "timing_s": round(time.time() - t4, 1)}
        report["panel_components_first"] = components(panel[0])
    except Exception as e:  # noqa: BLE001
        panel_err = f"{type(e).__name__}: {e}"
        report["panel"] = {"error": panel_err}
    # d) stream del training: processo 0, 6 worker, 8 campioni per processo (32 globali su 4 processi), segmento cfg.stream_segment
    t5 = time.time()
    world, per_proc, n_workers = 4, cfg.batch_size // 4, 6
    stream = []
    for w in range(n_workers):
        loader = L.PretrainLoader(index, cfg.loader, dict(scales), window_s)
        gen = L.batch_stream(loader, per_proc, seed=cfg.seed, start=0, worker=w, n_workers=n_workers, segment=cfg.stream_segment, rank=0)
        b = next(gen)
        stream.append({"index": b.index, "sha256": PR.batch_sha256(b), "windows": wins(b), "skipped": b.skipped_sessions})
    known = {r.get("batch_sha256"): r.get("step") for r in metrics}
    report["stream"] = {"batches": stream, "matches_metrics": [{"index": s["index"], "metrics_step": known.get(s["sha256"])} for s in stream],
                        "timing_s": round(time.time() - t5, 1), "assumed": {"world_size": world, "batch_per_process": per_proc, "num_workers": n_workers}}
    report["skipped_sessions"] = skips
    report["timing_total_s"] = round(time.time() - t0, 1)
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
    f = report.get("fixed")
    if f:
        print("FISSO: due volte identico", f["twice_identical"], "| uguale a diag1", f["equals_diag1"], "| uguale al seme", f["equals_seeded"],
              "| hash uguali al seme", sum(f["hash_equal_seeded"]), "/ 8", flush=True)
        print("COMPONENTI diverse fra le due esecuzioni:", report.get("fixed_components_differing_between_runs"), flush=True)
    p = report.get("panel")
    if p:
        print("PANNELLO:", json.dumps({k: v for k, v in p.items() if k not in ("keys_now", "keys_seeded")}), flush=True)
    s = report.get("stream")
    if s:
        print("STREAM:", json.dumps(s["matches_metrics"]), flush=True)
PY
echo "diagnosi 2 scritta in $DIAG_OUT"
