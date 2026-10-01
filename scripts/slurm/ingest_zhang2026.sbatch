#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH -J ingest_zhang2026
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_zhang2026_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_zhang2026_%j.out

# Passo 2: ingest di Zhang 2026 (64 soggetti, 621 sequenze, 37 GB di CSV in $WORK/data/raw/zhang2026), opzione B e modo random sparso (decisioni.md).
# Uso: sbatch scripts/slurm/ingest_zhang2026.sbatch --max-sessions 2      (collaudo: primo soggetto, anatomical e random)
#      sbatch scripts/slurm/ingest_zhang2026.sbatch --skip-existing       (pieno, riprendibile)
# Budget di tempo 2 h 40 min su 3 h di limite; processato su $SCRATCH (~3 GB in int16); report fuori dal repo.

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

mkdir -p "$WORK/wearusfm_runs/results/passo2" "$WORK/wearusfm_runs/logs" "$SCRATCH/data/processed/zhang2026"
python3 scripts/ingest_zhang2026.py \
  --raw-root "$WORK/data/raw/zhang2026" \
  --out-root "$SCRATCH/data/processed/zhang2026" \
  --report "$WORK/wearusfm_runs/results/passo2/zhang2026_ingest_report_${SLURM_JOB_ID}.json" \
  --time-budget-s 9600 \
  "$@"
