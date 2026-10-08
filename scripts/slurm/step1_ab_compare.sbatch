#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH -J step1_ab_compare
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_ab_compare_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_ab_compare_%j.out
# Submit after all FOUR successful ridge jobs. No training or GPU allocation.
set -euo pipefail
EXPECTED_COMMIT="${1:?frozen commit}"
RUNS="${2:?runs root}"
EVAL="${3:?evaluation root}"
OUT="${4:?NEW absolute report}"
case "$OUT" in /*) ;; *) echo "absolute output required"; exit 2;; esac
cd "$WORK/wearusfm"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python scripts/step1_ab.py compare --expected-commit "$EXPECTED_COMMIT" \
  --runs-root "$RUNS" --eval-root "$EVAL" --out "$OUT"
sha256sum "$OUT"
