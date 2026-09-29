#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH -J ingest_kaifosh
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_kaifosh_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_kaifosh_%j.out

# Passo 2: ingest di Kaifosh et al. 2025 (Discrete Gestures): 33 GB di HDF5 dentro full_data.tar,
# 16 canali a 2 kHz, un file per (utente, dataset). Solo EMG (le etichette servono all'harness).
# Uso: sbatch scripts/slurm/ingest_kaifosh.sbatch --max-recordings 1   (collaudo)
#      sbatch scripts/slurm/ingest_kaifosh.sbatch                       (pieno)
#
# Raw letto da $SCRATCH (copia verificata per checksum il 29/09/2026). Il processato (~17 GB int16)
# va in $SCRATCH e NON in $WORK, per non aggravare la quota di progetto (vedi CLAUDE.md): la
# data di modifica va rinfrescata prima del purge (25/10/2026). Report fuori dal repo.

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

RAW="$SCRATCH/data/raw/kaifosh_discrete_gestures"
mkdir -p "$WORK/wearusfm_runs/results/passo2" "$WORK/wearusfm_runs/logs" "$SCRATCH/data/processed/kaifosh"
python3 scripts/ingest_kaifosh.py \
  --tar "$RAW/full_data.tar" \
  --csv "$RAW/discrete_gestures_corpus.csv" \
  --out-root "$SCRATCH/data/processed/kaifosh" \
  --report "$WORK/wearusfm_runs/results/passo2/kaifosh_ingest_report_${SLURM_JOB_ID}.json" \
  "$@"
