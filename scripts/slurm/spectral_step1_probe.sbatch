#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH -J step1_probe
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_probe_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_probe_%j.out

# Prima sonda spettrale frozen, ridge CPU. Lanciare in dipendenza afterok dal
# job di estrazione. Uso: sbatch [--dependency=afterok:<job>] \
#   scripts/slurm/spectral_step1_probe.sbatch \
#   <features.npz> <sha256 checkpoint> <new report.json>
set -euo pipefail
FEATURES="${1:?features .npz}"
EXPECTED_CKPT_SHA="${2:?expected checkpoint sha256}"
OUT="${3:?new report .json}"

case "$OUT" in /*) ;; *) echo "absolute output required"; exit 2;; esac
case "$OUT" in "$WORK/wearusfm"|"$WORK/wearusfm"/*) echo "output inside repo forbidden"; exit 2;; esac
if [ -e "$OUT" ]; then echo "NEW report output required"; exit 2; fi

cd "$WORK/wearusfm"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$(dirname "$OUT")" "$WORK/wearusfm_runs/logs"
git rev-parse HEAD
sha256sum "$FEATURES"
python scripts/spectral_step1.py probe --data "$FEATURES" --out "$OUT" \
  --expected-checkpoint-sha256 "$EXPECTED_CKPT_SHA"
sha256sum "$OUT"
