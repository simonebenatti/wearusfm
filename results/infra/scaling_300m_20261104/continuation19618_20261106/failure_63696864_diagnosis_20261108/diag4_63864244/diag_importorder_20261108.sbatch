#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --ntasks=1
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_diag_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_diag_%j.out
# Diagnosi 4 (sola lettura dei dati, 0 GPU) del fallimento del job 63696864. Ipotesi: il contenuto dell'insieme fisso dipende dal percorso
# di lancio. Il launcher di GPT (scaling_train.py run -> torchrun scripts/train_jepa.py) importa torch PRIMA di numpy; il mio wrapper
# (scaling_continue.py worker) e tutti i miei driver di diagnosi importano numpy (o run.py) PRIMA di torch. Prove, tutte con il codice
# congelato del checkout e gli stessi argomenti del run: (1) ricostruzione dell'insieme fisso e del pannello della posa in tre ordini di
# import; (2) la CLI pura `python3 scripts/train_jepa.py ...` su CPU, fermata appena scrive fixed_eval_set.json e pose_panel_set.json;
# (3) impronte sha256 e stat dei file di dati delle sessioni dell'insieme fisso e delle registrazioni del pannello; (4) differenze
# numeriche fra le varianti sui primi due batch. Confronto con A = identita' del run a 5.000 (05/11) e B = ricalcolo di ieri (DIAG_B).
# Non tocca run, checkpoint, ricevute o dati: scrive solo in DIAG_OUT (nuova).
set -euo pipefail
: "${WORK:?WORK}" "${CODE_DIR:?CODE_DIR}" "${EXPECTED_SHA:?EXPECTED_SHA}" "${DIAG_OUT:?DIAG_OUT}" "${DIAG_B:?DIAG_B}"
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir "$DIAG_OUT"
echo "diag4 job $SLURM_JOB_ID su $(hostname), checkout $CODE_DIR @ $EXPECTED_SHA, uscite $DIAG_OUT"

# --- argomenti del trainer: quelli della continuazione (launch.json), senza --distributed, con uscita/worker/device per la CPU ---------
python3 - <<'PYA'
import json, os
from pathlib import Path
OUT = Path(os.environ["DIAG_OUT"])
REC = Path("/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/scaling_300m_20261104/continuation19618_v1")
argv = list(json.loads((REC / "launch.json").read_text())["trainer_argv"])
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
clean += ["--out-dir", str(OUT / "plain_cli_run"), "--num-workers", "0", "--device", "cpu"]
(OUT / "argv_plain_cli.json").write_text(json.dumps(clean, indent=1))
(OUT / "argv_lines.txt").write_text("\n".join(clean) + "\n")
print("argomenti:", len(clean))
PYA

# --- (1) tre ordini di import -----------------------------------------------------------------------------------------------------------
cat > "$DIAG_OUT/variant.py" <<'PYV'
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
PYV

for MODE in numpy_first torch_first run_first; do
  echo "=== variante $MODE ($(date +%H:%M:%S)) ==="
  if ! python3 "$DIAG_OUT/variant.py" "$MODE" > "$DIAG_OUT/$MODE.log" 2>&1; then
    echo "variante $MODE FALLITA (rc=$?)"; tail -n 20 "$DIAG_OUT/$MODE.log"
  else
    grep "^DIAG4" "$DIAG_OUT/$MODE.log" || tail -n 5 "$DIAG_OUT/$MODE.log"
  fi
done

# --- (2) la CLI pura, come la avvia il launcher di GPT, su CPU: fermata appena esistono i due file dell'identita' ----------------------
echo "=== CLI pura ($(date +%H:%M:%S)) ==="
mapfile -t ARGS < "$DIAG_OUT/argv_lines.txt"
RUN="$DIAG_OUT/plain_cli_run"
python3 scripts/train_jepa.py "${ARGS[@]}" > "$DIAG_OUT/plain_cli.log" 2>&1 &
PID=$!
STOP="in corso"
for i in $(seq 1 600); do
  if [ -s "$RUN/fixed_eval_set.json" ] && [ -s "$RUN/pose_panel_set.json" ]; then sleep 5; STOP="file presenti dopo $((i*2)) s"; break; fi
  if ! kill -0 "$PID" 2>/dev/null; then STOP="processo terminato da solo dopo $((i*2)) s"; break; fi
  sleep 2
done
kill "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
echo "CLI pura: $STOP"; ls -la "$RUN" 2>/dev/null || true; tail -n 8 "$DIAG_OUT/plain_cli.log" || true

# --- (3) impronte dei file di dati ---------------------------------------------------------------------------------------------------
cat > "$DIAG_OUT/fingerprints.py" <<'PYF'
"""Diagnosi 4: sha256, dimensione, mtime/ctime e inode dei file delle sessioni dell'insieme fisso e delle registrazioni del pannello."""
import hashlib, importlib.util, json, os, time
from pathlib import Path
import numpy as np  # noqa: F401
from wearusfm.training import run as R

OUT = Path(os.environ["DIAG_OUT"])
code = Path(os.environ["CODE_DIR"]).resolve()
S = Path("/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs")
PREV = S / "runs/scaling_300m_20261104/T100-G-300M-5000"
B = Path(os.environ["DIAG_B"])
argv = json.loads((OUT / "argv_plain_cli.json").read_text())
spec = importlib.util.spec_from_file_location("train_jepa_cli_fp", code / "scripts/train_jepa.py")
CLI = importlib.util.module_from_spec(spec); spec.loader.exec_module(CLI)
ap = CLI.build_parser(); args = ap.parse_args(argv); cfg = CLI.build_config(args, ap.error)
index = R.load_index(args.manifest, args.root, cfg.datasets, cfg.nested)
by_key = {f"{r['dataset']}/{r['subject']}/{r['session']}": r for r in index.rows}
A_fixed = json.loads((PREV / "fixed_eval_set.json").read_text())
A_panel = json.loads((PREV / "pose_panel_set.json").read_text())
B_panel = json.loads((B / "pose_panel_ident_now.json").read_text())
unchanged = {rec["recording"]: (a == b) for rec, a, b in zip(A_panel["recordings"], A_panel["batch_sha256"], B_panel["batch_sha256"])}
fixed_keys = {w[0] for w in A_fixed["windows"]}
keys = sorted(fixed_keys | set(unchanged))

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()
def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t))
def describe(p):
    st = p.stat()
    return {"bytes": st.st_size, "mtime": iso(st.st_mtime), "ctime": iso(st.st_ctime), "inode": st.st_ino, "sha256": sha(p)}

pose_root = Path(cfg.loader.pose_root) if getattr(cfg.loader, "pose_root", None) else None
t0 = time.time(); out = {}; n_files = 0; n_bytes = 0
for k in keys:
    row = by_key.get(k)
    e = {"in_index": row is not None, "in_fixed": k in fixed_keys, "in_panel": k in unchanged, "panel_unchanged_A_vs_B": unchanged.get(k)}
    if row is not None:
        try:
            d = index.path_of(row)
            e["dir"] = str(d); e["files"] = {}
            for name in ("metadata.json", "data_int16.npy"):
                p = d / name
                if p.exists():
                    e["files"][name] = describe(p); n_files += 1; n_bytes += e["files"][name]["bytes"]
            if pose_root is not None and row["dataset"] == "emg2pose":
                for cand in (pose_root / row["subject"] / row["session"], pose_root / ("u" + row["subject"]) / row["session"]):
                    if cand.exists():
                        e["pose_dir"] = str(cand)
                        for name in ("pose_200hz_f16.npy", "pose_valid.npy", "pose_meta.json"):
                            p = cand / name
                            if p.exists():
                                e["files"][name] = describe(p); n_files += 1; n_bytes += e["files"][name]["bytes"]
                        break
        except Exception as ex:  # noqa: BLE001
            e["error"] = f"{type(ex).__name__}: {ex}"
    out[k] = e
summary = {"protocol": "scaling-diag-importorder-20261108-v4", "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "n_sessions": len(keys),
           "n_fixed_sessions": len(fixed_keys), "n_panel_recordings": len(unchanged), "n_files": n_files, "n_bytes": n_bytes,
           "elapsed_s": round(time.time() - t0, 1), "pose_root": str(pose_root), "roots": [str(r) for r in index.roots]}
(OUT / "fingerprints.json").write_text(json.dumps({"summary": summary, "sessions": out}, indent=1))
print("IMPRONTE:", json.dumps(summary), flush=True)
PYF
echo "=== impronte ($(date +%H:%M:%S)) ==="
if ! timeout 1200 python3 "$DIAG_OUT/fingerprints.py" > "$DIAG_OUT/fingerprints.log" 2>&1; then
  echo "impronte FALLITE o oltre il tempo (rc=$?)"; tail -n 10 "$DIAG_OUT/fingerprints.log"
else
  grep "^IMPRONTE" "$DIAG_OUT/fingerprints.log"
fi

# --- (4) sintesi: identita' per percorso e differenze numeriche fra varianti --------------------------------------------------------
echo "=== sintesi ($(date +%H:%M:%S)) ==="
python3 - <<'PYS'
import json, os
from pathlib import Path
import numpy as np
OUT = Path(os.environ["DIAG_OUT"])
S = Path("/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs")
PREV = S / "runs/scaling_300m_20261104/T100-G-300M-5000"
B = Path(os.environ["DIAG_B"])
A_fixed = json.loads((PREV / "fixed_eval_set.json").read_text()); A_panel = json.loads((PREV / "pose_panel_set.json").read_text())
B_fixed = json.loads((B / "ident_run1.json").read_text()); B_panel = json.loads((B / "pose_panel_ident_now.json").read_text())
def n_eq(x, ref):
    return sum(a == b for a, b in zip(x["batch_sha256"], ref["batch_sha256"]))
rows = {}
for mode in ("numpy_first", "torch_first", "run_first"):
    fp, pp = OUT / f"{mode}_fixed_ident.json", OUT / f"{mode}_panel_ident.json"
    if fp.exists() and pp.exists():
        f, p = json.loads(fp.read_text()), json.loads(pp.read_text())
        rows[mode] = {"fixed_A": n_eq(f, A_fixed), "fixed_B": n_eq(f, B_fixed), "panel_A": n_eq(p, A_panel), "panel_B": n_eq(p, B_panel),
                      "windows_A": f["windows"] == A_fixed["windows"]}
    else:
        rows[mode] = {"errore": "identita' mancante"}
run = OUT / "plain_cli_run"
if (run / "fixed_eval_set.json").exists() and (run / "pose_panel_set.json").exists():
    f, p = json.loads((run / "fixed_eval_set.json").read_text()), json.loads((run / "pose_panel_set.json").read_text())
    rows["plain_cli"] = {"fixed_A": n_eq(f, A_fixed), "fixed_B": n_eq(f, B_fixed), "panel_A": n_eq(p, A_panel), "panel_B": n_eq(p, B_panel),
                         "windows_A": f["windows"] == A_fixed["windows"]}
else:
    rows["plain_cli"] = {"errore": "file dell'identita' non scritti"}
print("TABELLA (hash uguali ad A = run 5000 del 05/11, a B = ricalcolo di ieri; insieme fisso su 8, pannello su 64):")
for k, v in rows.items():
    print(f"  {k:12s} {json.dumps(v)}")
# differenze numeriche fra le varianti (primi due batch dell'insieme fisso e del pannello)
diffs = {}
ref = OUT / "numpy_first_arrays"
for mode in ("torch_first", "run_first"):
    d = OUT / f"{mode}_arrays"
    if not (ref.exists() and d.exists()):
        continue
    per = {}
    for p in sorted(ref.glob("*.npy")):
        q = d / p.name
        if not q.exists():
            per[p.stem] = "assente"; continue
        a, b = np.load(p, allow_pickle=False), np.load(q, allow_pickle=False)
        if a.shape != b.shape or a.dtype != b.dtype:
            per[p.stem] = f"forma/dtype diversi {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"; continue
        if a.dtype.kind in "fc":
            same = np.array_equal(a, b, equal_nan=True)
            if same:
                per[p.stem] = "uguale"
            else:
                m = ~(np.isnan(a) & np.isnan(b)); delta = np.abs(a[m].astype(np.float64) - b[m].astype(np.float64))
                nd = int((a[m] != b[m]).sum())
                per[p.stem] = {"n_diff": nd, "n": int(a.size), "max_abs_diff": float(delta.max()) if delta.size else 0.0,
                               "max_abs_ref": float(np.nanmax(np.abs(a))) if a.size else 0.0,
                               "max_rel_diff": float((delta / np.maximum(np.abs(a[m].astype(np.float64)), 1e-30)).max()) if delta.size else 0.0}
        else:
            per[p.stem] = "uguale" if np.array_equal(a, b) else {"n_diff": int((a != b).sum()), "n": int(a.size)}
    diffs[mode] = per
(OUT / "summary.json").write_text(json.dumps({"table": rows, "array_diffs_vs_numpy_first": diffs}, indent=1))
for mode, per in diffs.items():
    changed = {k: v for k, v in per.items() if v != "uguale"}
    print(f"DIFFERENZE numeriche {mode} vs numpy_first: {len(per)} array, {len(changed)} diversi")
    for k, v in list(changed.items())[:40]:
        print("   ", k, json.dumps(v) if isinstance(v, dict) else v)
PYS
echo "diag4 conclusa ($(date +%H:%M:%S)); uscite in $DIAG_OUT"
