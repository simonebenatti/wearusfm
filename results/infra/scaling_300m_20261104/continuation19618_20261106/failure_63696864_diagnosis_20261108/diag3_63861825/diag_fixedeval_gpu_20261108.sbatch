#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_diag_gpu_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_diag_gpu_%j.out
# Diagnosi 3 (GPU, contesto reale) del fallimento del job 63696864: ricalcola l'insieme fisso della diagnostica di stabilita'
# con il codice congelato del checkout nel contesto dei job veri (torchrun, 4 processi, CUDA, DDP) in quattro modi:
#   resume     = ripresa dal checkpoint a 5.000 (copia del checkpoint in una cartella di diagnosi, mai il run vero);
#   fresh      = run nuovo (cartella vuota);
#   plain_cuda = un processo solo su una GPU, senza torchrun;
#   plain_cpu  = un processo solo su CPU, senza torchrun (come le diagnosi 1 e 2 su login08).
# In ogni modo il trainer si ferma subito dopo il calcolo dell'insieme fisso (monkeypatch), senza un passo di training.
# Non tocca run, checkpoint o ricevute esistenti; scrive solo sotto DIAG_OUT (scratch).
set -uo pipefail
: "${WORK:?WORK}" "${CODE_DIR:?CODE_DIR}" "${EXPECTED_SHA:?EXPECTED_SHA}" "${DIAG_OUT:?DIAG_OUT}" "${DIAG_LOGIN08:?DIAG_LOGIN08}"
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export WEARUSFM_CODE_SHA="$EXPECTED_SHA" WEARUSFM_CODE_DIR="$CODE_DIR"
S=/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs
PREV="$S/runs/scaling_300m_20261104/T100-G-300M-5000"
REC="$S/results/scaling_300m_20261104/continuation19618_v1"
mkdir -p "$DIAG_OUT" || exit 2
[ -z "$(ls -A "$DIAG_OUT")" ] || { echo "DIAG_OUT non vuota: $DIAG_OUT"; exit 2; }
echo "nodo $(hostname); cpu: $(grep -m1 'model name' /proc/cpuinfo); gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1)"
mkdir -p "$DIAG_OUT/resume_run" "$DIAG_OUT/fresh_run" "$DIAG_OUT/plain_cuda_run" "$DIAG_OUT/plain_cpu_run"
t0=$(date +%s)
for f in checkpoint.pt fixed_eval_set.json pose_panel_set.json origin_sessions.json; do cp "$PREV/$f" "$DIAG_OUT/resume_run/$f" || exit 2; done
echo "copia del checkpoint: $(( $(date +%s) - t0 )) s; sha256 della copia: $(sha256sum "$DIAG_OUT/resume_run/checkpoint.pt" | cut -c 1-16)"
cat > "$DIAG_OUT/driver.py" <<'PY'
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
PY
python3 - "$REC" "$DIAG_OUT" <<'PY'
import json, sys
from pathlib import Path
rec, out = Path(sys.argv[1]), Path(sys.argv[2])
argv = json.loads((rec / "launch.json").read_text())["trainer_argv"]
assert argv[0] == "scripts/train_jepa.py"
argv = argv[1:]
def strip(a, drop_flags=(), drop_opts=()):
    r, i = [], 0
    while i < len(a):
        if a[i] in drop_flags:
            i += 1; continue
        if a[i] in drop_opts:
            i += 2; continue
        r.append(a[i]); i += 1
    return r
base = strip(argv, drop_opts=("--out-dir",))
(out / "argv_resume.json").write_text(json.dumps(base + ["--out-dir", str(out / "resume_run")]))
(out / "argv_fresh.json").write_text(json.dumps(base + ["--out-dir", str(out / "fresh_run")]))
single = strip(base, drop_flags=("--distributed",), drop_opts=("--num-workers", "--device"))
(out / "argv_plain_cuda.json").write_text(json.dumps(single + ["--out-dir", str(out / "plain_cuda_run"), "--num-workers", "0", "--device", "cuda"]))
(out / "argv_plain_cpu.json").write_text(json.dumps(single + ["--out-dir", str(out / "plain_cpu_run"), "--num-workers", "0", "--device", "cpu"]))
print("argv scritti; --distributed presente nel base:", "--distributed" in base)
PY
cd "$CODE_DIR" || exit 2
for mode in resume fresh; do
  echo "=== $mode (torchrun, 4 processi) $(date +%T) ==="
  python3 -m torch.distributed.run --standalone --nproc_per_node=4 "$DIAG_OUT/driver.py" "$DIAG_OUT" "$mode" "$DIAG_OUT/argv_$mode.json" > "$DIAG_OUT/$mode.log" 2>&1
  echo "uscita torchrun $mode: $? (attesa: non zero, il processo 0 si ferma apposta); file: $(ls "$DIAG_OUT" | grep -c "^${mode}_rank")"
  grep -h "DIAG\|Error\|error" "$DIAG_OUT/$mode.log" | grep -v "Warning\|warn" | head -n 6
done
for mode in plain_cuda plain_cpu; do
  echo "=== $mode (processo singolo) $(date +%T) ==="
  python3 "$DIAG_OUT/driver.py" "$DIAG_OUT" "$mode" "$DIAG_OUT/argv_$mode.json" > "$DIAG_OUT/$mode.log" 2>&1
  echo "uscita $mode: $? (attesa 3)"
  grep -h "DIAG\|Error" "$DIAG_OUT/$mode.log" | grep -v "Warning\|warn" | head -n 4
done
echo "=== confronto $(date +%T) ==="
python3 - "$DIAG_OUT" "$PREV" "$DIAG_LOGIN08" <<'PY'
import json, sys
from pathlib import Path
out, prev, login = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
seed = json.loads((prev / "fixed_eval_set.json").read_text())
l08 = json.loads((login / "ident_now.json").read_text())
recs = {}
for mode in ("resume", "fresh", "plain_cuda", "plain_cpu"):
    p = out / f"{mode}_rank0.json"
    if p.exists():
        recs[mode] = json.loads(p.read_text())
summary = {}
for mode, r in recs.items():
    ident = r["ident"]
    summary[mode] = {"node": r["node"], "equals_seed": ident == seed, "hash_equal_seed": sum(a == b for a, b in zip(ident["batch_sha256"], seed["batch_sha256"])),
                     "windows_equal_seed": ident["windows"] == seed["windows"], "equals_login08": ident == l08,
                     "hash_equal_login08": sum(a == b for a, b in zip(ident["batch_sha256"], l08["batch_sha256"])),
                     "denormal_flushed": r["denormal_flushed"], "transformers_loaded": r["transformers_loaded"], "cuda_initialized": r["cuda_initialized"],
                     "torch_threads": r["torch_threads"], "n_libs": len(r["libs"]), "libs": r["libs"]}
modes = list(recs)
pairs = {}
for i in range(len(modes)):
    for j in range(i + 1, len(modes)):
        a, b = recs[modes[i]]["components"], recs[modes[j]]["components"]
        diff = sorted({k for x, y in zip(a, b) for k in x if x[k] != y.get(k)})
        pairs[f"{modes[i]} vs {modes[j]}"] = {"identical": a == b, "components_differing": diff}
report = {"summary": summary, "component_pairs": pairs}
(out / "comparison.json").write_text(json.dumps(report, indent=1))
for mode, s in summary.items():
    print(f"{mode:11s} nodo {s['node']}: uguale al seme {s['equals_seed']} (hash {s['hash_equal_seed']}/8, finestre {s['windows_equal_seed']}); "
          f"uguale a login08 {s['equals_login08']} (hash {s['hash_equal_login08']}/8); denormali azzerati {s['denormal_flushed']}; "
          f"transformers {s['transformers_loaded']}; cuda {s['cuda_initialized']}; thread torch {s['torch_threads']}; librerie {s['n_libs']}")
for k, v in pairs.items():
    print(f"componenti {k}: identiche {v['identical']}; diverse: {v['components_differing']}")
PY
echo "diagnosi GPU scritta in $DIAG_OUT; durata totale $(( $(date +%s) - t0 )) s"
