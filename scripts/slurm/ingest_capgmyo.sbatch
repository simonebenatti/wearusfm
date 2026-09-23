#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -J ingest_capgmyo
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_capgmyo_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_capgmyo_%j.out

# Passo 2 (D7a sbloccato, piano_operativo_v10.md): primo ingest reale, CapgMyo-DBa, il
# piu' piccolo fra i dataset di pretraining. Primo lancio: SOLO soggetto 1, collaudo
# prima di lanciare su tutti e 18 (v10 §11: "un dataset alla volta, dal piu' piccolo").

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

python3 scripts/ingest_capgmyo.py \
  --raw-root "$WORK/data/raw/capgmyo" \
  --out-root "$WORK/data/processed/capgmyo" \
  --subjects 1 \
  --report "$WORK/wearusfm_runs/results/passo2/capgmyo_ingest_report_s1.json"
