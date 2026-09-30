#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH -J ingest_hyser
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_hyser_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_hyser_%j.out

# Passo 2: ingest di Hyser (HD-sEMG, 256 canali, 2048 Hz): SOLO i file `*_raw_*` (76 GB su 143), letti
# due volte (picchi per canale, poi scrittura). Uso: sbatch scripts/slurm/ingest_hyser.sbatch --max-groups 3
# (collaudo) oppure senza argomenti (pieno: 200 gruppi = 20 soggetti x 2 sessioni x 5 sotto-dataset, con
# `pr` diviso in dynamic e maintenance). Raw da $SCRATCH (scaricato il 23/09/2026 con SHA256SUMS del
# dataset); processato su $SCRATCH (~76 GB int16); report fuori dal repo.

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

mkdir -p "$WORK/wearusfm_runs/results/passo2" "$WORK/wearusfm_runs/logs" "$SCRATCH/data/processed/hyser"
python3 scripts/ingest_hyser.py \
  --raw-root "$SCRATCH/data/raw/hyser" \
  --out-root "$SCRATCH/data/processed/hyser" \
  --report "$WORK/wearusfm_runs/results/passo2/hyser_ingest_report_${SLURM_JOB_ID}.json" \
  "$@"
