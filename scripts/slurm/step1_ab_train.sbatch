#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH -J step1_ab
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_ab_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_ab_%j.out
# A: 4h cap; B: submit with --time=06:00:00. No automatic resume/chaining.
# Usage: sbatch [--time=06:00:00] scripts/slurm/step1_ab_train.sbatch \
#   <expected commit> <A|B> <0|1> <NEW absolute run directory>
set -euo pipefail
EXPECTED_COMMIT="${1:?frozen commit}"
ARM="${2:?A or B}"
SEED="${3:?model seed 0 or 1}"
OUT="${4:?NEW absolute run directory}"
case "$OUT" in /*) ;; *) echo "absolute output required"; exit 2;; esac
case "$ARM" in A|B) ;; *) echo "arm must be A or B"; exit 2;; esac
case "$SEED" in 0|1) ;; *) echo "seed must be 0 or 1"; exit 2;; esac
# Verify the scheduler's allocation, not a user-supplied time estimate.
EXPECTED_LIMIT=04:00:00
if [ "$ARM" = B ]; then EXPECTED_LIMIT=06:00:00; fi
ALLOC_LIMIT=$(scontrol show job -o "$SLURM_JOB_ID" | tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2}')
if [ "$ALLOC_LIMIT" != "$EXPECTED_LIMIT" ]; then
  echo "allocation must be exactly ${EXPECTED_LIMIT} for arm ${ARM}"; exit 2
fi
cd "$WORK/wearusfm"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/step1_ab.py train --arm "$ARM" --seed "$SEED" --expected-commit "$EXPECTED_COMMIT" \
  --manifest "$WORK/wearusfm_runs/results/passo6/capgmyo_grid_fix/manifest_v1_1.json.gz" \
  --scales "$WORK/wearusfm_runs/results/passo6/loader_59623700/session_scales.json" \
  --window-seconds "$WORK/wearusfm_runs/results/passo6/window_seconds_59425316.json" \
  --calibration "$WORK/wearusfm_runs/results/passo6/step1_0710/calibration.npz" \
  --splits "$WORK/wearusfm_runs/results/passo6/step1_0710/splits.json" \
  --root "$WORK/data/processed" --root "$SCRATCH/data/processed" --out-dir "$OUT"
