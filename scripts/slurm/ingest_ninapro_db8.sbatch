#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -J ingest_ninapro_db8
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_ninapro_db8_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_ninapro_db8_%j.out

# Passo 2: ingest di NinaPro DB8 (36 .mat, 24 GB: 12 soggetti x 3 acquisizioni), opzione A (Simone, 01/10/2026).
# Uso: sbatch scripts/slurm/ingest_ninapro_db8.sbatch --subjects 1 11     (collaudo: un normodotato e un amputato)
#      sbatch scripts/slurm/ingest_ninapro_db8.sbatch                     (tutti)
# Raw da $SCRATCH (scaricato il 01/10/2026); processato su $SCRATCH (~2 GB in int16); report fuori dal repo.
# Memoria: un soggetto alla volta, ~5 M campioni x 16 canali in float64 (~0,7 GB) piu' le copie: 8 GB bastano.

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

mkdir -p "$WORK/wearusfm_runs/results/passo2" "$WORK/wearusfm_runs/logs" "$SCRATCH/data/processed/ninapro_db8"
python3 scripts/ingest_ninapro_db8.py \
  --raw-root "$SCRATCH/data/raw/ninapro" \
  --out-root "$SCRATCH/data/processed/ninapro_db8" \
  --report "$WORK/wearusfm_runs/results/passo2/ninapro_db8_ingest_report_${SLURM_JOB_ID}.json" \
  "$@"
