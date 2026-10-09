#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH -J step1_downstream_probe
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_downstream_probe_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_downstream_probe_%j.out
# Arguments as extract; optional sixth arg: spectral comparison JSON.
# Only last probe gets it, with afterok on its extract AND all other seven probes.
# Comparison uses the SAME allocation, within the 128 core-hour total cap.
set -euo pipefail
EXPECTED="${1:?commit}"; TASK="${2:?p1/p2}"; ARM="${3:?arm}"; RUNS="${4:?runs}"; EVAL="${5:?eval}"
EXTRA=()
if [ "$#" -eq 6 ]; then EXTRA=(--compare-after --spectral-report "$6"); fi
cd "$WORK/wearusfm"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python scripts/step1_downstream.py --stage probe --expected-commit "$EXPECTED" --task "$TASK" --arm "$ARM" \
  --runs-root "$RUNS" --eval-root "$EVAL" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" "${EXTRA[@]}"
