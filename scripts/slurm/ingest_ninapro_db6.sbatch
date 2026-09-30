#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH -J ingest_ninapro_db6
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_ninapro_db6_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_ninapro_db6_%j.out

# Passo 2: ingest di NinaPro DB6 (21 GB di zip: 10 soggetti x 2 zip, 10 sessioni giorno/ora ciascuno).
# Uso: sbatch scripts/slurm/ingest_ninapro_db6.sbatch [--subjects 1]
# Raw da $SCRATCH (copia verificata per checksum il 29/09/2026); processato su $SCRATCH (~4-5 GB);
# report fuori dal repo.

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

mkdir -p "$WORK/wearusfm_runs/results/passo2" "$WORK/wearusfm_runs/logs" "$SCRATCH/data/processed/ninapro_db6"
python3 scripts/ingest_ninapro_db6.py \
  --raw-root "$SCRATCH/data/raw/ninapro" \
  --out-root "$SCRATCH/data/processed/ninapro_db6" \
  --report "$WORK/wearusfm_runs/results/passo2/ninapro_db6_ingest_report_${SLURM_JOB_ID}.json" \
  "$@"
