#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH -J step1_extract
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_extract_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_extract_%j.out

# Prima sonda spettrale frozen, estrazione GPU. Protocollo:
# docs/step1_frozen_spectral_probe.md. Tutte le uscite devono essere nuove e
# fuori dal repository.
# Uso: sbatch scripts/slurm/spectral_step1_extract.sbatch \
#   <checkpoint.pt> <sha256 checkpoint> <manifest.json.gz> <session_scales.json> \
#   <window_seconds.json> <splits.json> <features.npz>
set -euo pipefail
CKPT="${1:?checkpoint}"
EXPECTED_CKPT_SHA="${2:?expected checkpoint sha256}"
MANIFEST="${3:?manifest}"
SCALES="${4:?scales}"
WINDOW_SECONDS="${5:?window_seconds}"
SPLITS="${6:?splits}"
OUT="${7:?new features .npz}"
EXPECTED_SPLIT_SHA="e064a1102bb4a7dd1b01a70caefa8fa2c58d8c9c751bafeafe6bd2892fff7a30"

case "$OUT" in /*) ;; *) echo "absolute output required"; exit 2;; esac
case "$OUT" in "$WORK/wearusfm"|"$WORK/wearusfm"/*) echo "output inside repo forbidden"; exit 2;; esac
if [ -e "$OUT" ]; then echo "NEW feature output required"; exit 2; fi
test "$(sha256sum "$CKPT" | awk '{print $1}')" = "$EXPECTED_CKPT_SHA"
test "$(sha256sum "$SPLITS" | awk '{print $1}')" = "$EXPECTED_SPLIT_SHA"

cd "$WORK/wearusfm"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$(dirname "$OUT")" "$WORK/wearusfm_runs/logs"
git rev-parse HEAD
python scripts/spectral_step1.py extract \
  --manifest "$MANIFEST" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
  --datasets emg2qwerty --seed 0 --splits "$SPLITS" --scales "$SCALES" \
  --window-seconds "$WINDOW_SECONDS" --windows-per-subject 100 \
  --checkpoint "$CKPT" --which teacher --device cuda --out "$OUT"
sha256sum "$OUT"
