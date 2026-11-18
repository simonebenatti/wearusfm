#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH -J scaling_memory
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_memory_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_memory_%j.out

# S3 only: fresh500 updates with the unchanged full production trainer.
# 4GPUh +32coreh ceiling; the separate checkout costs at most0.333334coreh.
# No scientific endpoint, no resume, no submission of a subsequent phase.
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "uso: scaling_production_memory.sbatch <piano relativo al checkout>" >&2
  exit 2
fi
: "${WORK:?WORK}" "${CODE_DIR:?CODE_DIR}" "${EXPECTED_SHA:?EXPECTED_SHA}" "${SLURM_JOB_ID:?SLURM_JOB_ID}"
PLAN_REL="$1"
case "$PLAN_REL" in /*|"") echo "richiesto piano relativo al checkout" >&2; exit 2;; esac
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

python3 - "$CODE_DIR" "$PLAN_REL" "$WORK" <<'PY'
from pathlib import Path
import sys

code = Path(sys.argv[1]).resolve(strict=True)
plan = (code / sys.argv[2]).resolve(strict=True)
if not code.is_relative_to(Path(sys.argv[3]).resolve() / "wearusfm_campaigns"):
    raise SystemExit("richiesto checkout di campagna congelato")
if not plan.is_relative_to(code) or not plan.is_file():
    raise SystemExit("piano esterno al checkout")
PY

module list 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# The Python supervisor checks exact real SLURM resources, all source/input pins,
# fresh external output directories and complete production evidence.
exec python3 scripts/scaling_production_memory.py run \
  --plan "$CODE_DIR/$PLAN_REL" --expected-code-sha "$EXPECTED_SHA"
