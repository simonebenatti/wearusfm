#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=05:30:00
#SBATCH -J scaling_train
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_train_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_train_%j.out

# Fresh scientific300M5000 only. 22GPUh /176coreh cap; checkout/evaluation separate.
# No recipe overrides, checkpoint initialization, resume or automatic continuation.
# Submit only through submit_job.py after the wrapper's four real gates are ready.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "uso: scaling_train.sbatch <piano relativo al checkout> <directory ricevute nuova>" >&2
  exit 2
fi
: "${WORK:?WORK}" "${CODE_DIR:?CODE_DIR}" "${EXPECTED_SHA:?EXPECTED_SHA}" "${SLURM_JOB_ID:?SLURM_JOB_ID}"
PLAN_REL="$1"
OUT="$2"
case "$PLAN_REL" in /*|"") echo "richiesto piano relativo al checkout" >&2; exit 2;; esac
if [ "${SLURM_JOB_NUM_NODES:-}" != 1 ] || [ "${SLURM_NTASKS:-}" != 1 ] || \
   [ "${SLURM_CPUS_PER_TASK:-}" != 32 ] || [ "${SLURM_MEM_PER_NODE:-}" != 491520 ] || \
   [ "${SLURM_JOB_PARTITION:-}" != boost_usr_prod ]; then
  echo "richiesti boost_usr_prod, 1 nodo, 1 task, 32 CPU e 480G" >&2
  exit 2
fi
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

python3 - "$CODE_DIR" "$PLAN_REL" "$OUT" "$WORK" <<'PY'
import importlib.util
import json
from pathlib import Path
import sys

code = Path(sys.argv[1]).resolve(strict=True)
plan_path = (code / sys.argv[2]).resolve(strict=True)
out_arg, work = Path(sys.argv[3]), Path(sys.argv[4]).resolve()
if not code.is_relative_to(work / "wearusfm_campaigns"):
    raise SystemExit("richiesto checkout di campagna congelato")
if not plan_path.is_relative_to(code) or not plan_path.is_file():
    raise SystemExit("piano esterno al checkout")
if not out_arg.is_absolute() or out_arg.exists() or out_arg.is_symlink() or out_arg.resolve() != out_arg:
    raise SystemExit("directory ricevute assoluta, canonica e nuova richiesta")
if any(out_arg.is_relative_to(root) for root in (code, work / "wearusfm", work / "wearusfm_campaigns")):
    raise SystemExit("ricevute dentro il codice")
plan = json.loads(plan_path.read_text())
if plan.get("protocol") != "scaling-training-20261104-v1" or plan.get("arm") != "T100-G-300M" or \
        type(plan.get("checkpoint_step")) is not int or plan["checkpoint_step"] != 5000:
    raise SystemExit("consentito solo fresh300M5000")
if plan.get("allocation") != {"nodes": 1, "gpus": 4, "cpus": 32, "memory": "480G", "walltime_seconds": 19800}:
    raise SystemExit("allocazione richiesta: 4GPU, 1 nodo, 32CPU, 480G, 19800s")
if plan.get("receipt_dir") != str(out_arg):
    raise SystemExit("directory ricevute diversa dal piano")
spec = importlib.util.spec_from_file_location("scaling_train_launcher", code / "scripts/scaling_train.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)
wrapper.verify_allocation(plan)
PY

module list 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# The frozen wrapper verifies all real gate/source/input pins, derives the exact
# fresh recipe, and refuses existing output directories or incomplete endpoints.
exec python3 scripts/scaling_train.py run \
  --plan "$CODE_DIR/$PLAN_REL" --expected-code-sha "$EXPECTED_SHA" --out "$OUT"
