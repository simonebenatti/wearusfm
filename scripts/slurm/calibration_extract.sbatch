#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH -J calibration_extract
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/calibration_extract_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/calibration_extract_%j.out

# Calibrazione C1-C4 (docs/calibrazione_c1_c4_selezione.md): feature dei soli soggetti di pretraining di NinaPro DB2, DB3, DB6 dal checkpoint
# post-decay di un run (scripts/calibration_probe.py --stage extract; stessa estrazione della P1). 1 GPU; la durata la fissa il piano
# (--time: 30M, 100M, 300M). Nodo della catena di valutazione (scripts/evaluation_chain.py), dopo la guardia del checkpoint.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/calibration_extract.sbatch <checkpoint.pt> <cartella feature>

set -uo pipefail
CKPT="${1:?checkpoint}"; FEAT="${2:?cartella delle feature}"; shift 2
case "$FEAT" in /*) ;; *) echo "cartella delle feature relativa: rifiutata"; exit 2;; esac
case "$FEAT" in "$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm/*|"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*)
  echo "feature dentro il codice: rifiutate"; exit 2;; esac
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
mkdir -p "$FEAT" "$WORK/wearusfm_runs/logs"
python3 scripts/calibration_probe.py --stage extract --checkpoint "$CKPT" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
  --features "$FEAT" "$@"
