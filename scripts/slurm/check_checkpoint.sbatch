#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH -J check_checkpoint
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/check_checkpoint_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/check_checkpoint_%j.out

# Guardia della catena di valutazione (scripts/evaluation_chain.py, accordo del 15/10/2026): il checkpoint atteso e' stato pubblicato da un
# run finito senza STOP e ha l'identita' dichiarata (scripts/check_checkpoint.py). Esce con 3 se no: la valutazione in afterok non parte.
# CPU seriale, 1 core, minuti (lo SHA-256 di un checkpoint da 0,5-4,7 GB); 0 GPU-ora. Il rapporto va fuori dal codice.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/check_checkpoint.sbatch <run_dir> <checkpoint> <rapporto.json> \
#        [--min-steps N] [--sha256 HEX]

set -uo pipefail
RUN_DIR="${1:?cartella del run}"; CKPT="${2:?nome del checkpoint}"; REPORT="${3:?rapporto}"; shift 3
case "$REPORT" in /*) ;; *) echo "rapporto con percorso relativo: rifiutato"; exit 2;; esac
case "$REPORT" in "$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm/*|"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*)
  echo "rapporto dentro il codice: rifiutato"; exit 2;; esac
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
# il python3 di sistema del nodo e' troppo vecchio (SyntaxError su `from __future__ import annotations`, 01/10/2026; ultrareview del 15/10)
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
python3 scripts/check_checkpoint.py "$RUN_DIR" "$CKPT" --report "$REPORT" "$@"
