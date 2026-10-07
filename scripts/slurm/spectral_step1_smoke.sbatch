#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH -J step1_smoke
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_smoke_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_smoke_%j.out
# <=0.5 GPU-hour =4 local hours. Functional test, NOT a quality A/B experiment.
set -euo pipefail
MANIFEST="${1:?manifest}"; SCALES="${2:?scales}"; WINDOW_SECONDS="${3:?window_seconds}"
CALIBRATION="${4:?calibration NPZ}"; RUN_DIR="${5:?NEW run directory}"
case "$RUN_DIR" in /*) ;; *) echo "absolute output required"; exit 2;; esac
case "$RUN_DIR" in "$WORK/wearusfm"|"$WORK/wearusfm"/*) echo "output inside repo forbidden"; exit 2;; esac
if [ -e "$RUN_DIR" ]; then echo "NEW run directory required"; exit 2; fi
cd "$WORK/wearusfm"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONDONTWRITEBYTECODE=1
git rev-parse HEAD
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/check_step1_cuda.py
COMMON=(--preset sanity --manifest "$MANIFEST" --root "$WORK/data/processed" --root "$SCRATCH/data/processed"
  --scales "$SCALES" --window1-rules --window-seconds "$WINDOW_SECONDS" --datasets emg2qwerty --out-dir "$RUN_DIR"
  --num-workers 6 --seed 0 --keep-calibration "$CALIBRATION" --keep-weight 0.05 --keep-grad-every 100)
python scripts/train_jepa.py "${COMMON[@]}" --max-steps 200 --time-limit-s 1200
# Resume identity/statistics test in SAME allocation, at most one more update.
python scripts/train_jepa.py "${COMMON[@]}" --max-steps 201 --time-limit-s 90
