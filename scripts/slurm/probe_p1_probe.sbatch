#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH -J probe_p1_lin
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/probe_p1_probe_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/probe_p1_probe_%j.out

# Metrica P1, fase delle sonde (CPU, 0 GPU-ora). Uso: sbatch [--dependency=afterok:<job>] probe_p1_probe.sbatch <checkpoint.pt> <cartella delle
# feature> <report.json> (il checkpoint serve solo al report: la fase non carica il modello)
# Con la catena di campagna: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> ... (scripts/slurm/lib_code_dir.sh).

set -uo pipefail
CKPT="${1:?checkpoint}"; FEAT="${2:?cartella delle feature}"; OUT="${3:?report .json}"
case "$FEAT" in /*) ;; *) echo "cartella delle feature relativa: rifiutata"; exit 2;; esac
case "$FEAT" in "$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm/*|"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*)
  echo "feature dentro il codice: rifiutate"; exit 2;; esac
case "$OUT" in /*) ;; *) echo "rapporto relativo: rifiutato"; exit 2;; esac
case "$OUT" in "$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm/*|"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*)
  echo "rapporto dentro il codice: rifiutato"; exit 2;; esac
# codice: checkout di campagna (CODE_DIR + EXPECTED_SHA, SHA e pulizia verificati) o il canonico (critica del punto 5, 16/10/2026)
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
mkdir -p "$(dirname "$OUT")"
python3 scripts/probe_p1.py --stage probe --checkpoint "$CKPT" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
  --features "$FEAT" --out "$OUT"
