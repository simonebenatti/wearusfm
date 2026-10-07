#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -J step1_calibrate
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_calibrate_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_calibrate_%j.out
# TRAIN-only calibration; seed 0; 100 windows per train subject. 0 GPU-hour, <=1 core-hour.
set -euo pipefail
MANIFEST="${1:?manifest}"; SCALES="${2:?scales}"; WINDOW_SECONDS="${3:?window_seconds}"; OUT="${4:?new output directory}"
case "$OUT" in /*) ;; *) echo "absolute output required"; exit 2;; esac
case "$OUT" in "$WORK/wearusfm"|"$WORK/wearusfm"/*) echo "output inside repo forbidden"; exit 2;; esac
if [ -e "$OUT" ]; then echo "NEW output directory required"; exit 2; fi
cd "$WORK/wearusfm"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT"
git rev-parse HEAD
python scripts/spectral_step1.py split --manifest "$MANIFEST" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
  --datasets emg2qwerty --seed 0 --out "$OUT/splits.json"
python scripts/spectral_step1.py calibrate --manifest "$MANIFEST" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
  --datasets emg2qwerty --seed 0 --splits "$OUT/splits.json" --scales "$SCALES" --window-seconds "$WINDOW_SECONDS" \
  --windows-per-subject 100 --out "$OUT/calibration.npz"
