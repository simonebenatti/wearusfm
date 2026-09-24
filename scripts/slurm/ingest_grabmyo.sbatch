#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -J ingest_grabmyo
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_grabmyo_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_grabmyo_%j.out

# Passo 2: ingest GRABMyo. Primo lancio: SOLO partecipante 1, sessione 1 (119 record),
# collaudo prima di lanciare su tutti i 43 partecipanti x 3 sessioni.

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

python3 scripts/ingest_grabmyo.py \
  --raw-root "$WORK/data/raw/grabmyo" \
  --out-root "$WORK/data/processed/grabmyo" \
  --participants 1 \
  --sessions 1 \
  --report "$WORK/wearusfm_runs/results/passo2/grabmyo_ingest_report_p1s1.json"
