#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -J ingest_camargo
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_camargo_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_camargo_%j.out

# Passo 2: ingest Camargo 2021 dal formato convertito con MATLAB
# (scripts/convert_camargo_mcos_tables.m, conversione verificata 3147/3147 file).
# Collaudo su AB06 riuscito (job 59010112: 143 trial, 0 canali scartati, 0 assi
# temporali anomali) - questo lancio copre tutti i 22 soggetti.

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

python3 scripts/ingest_camargo.py \
  --raw-root "$WORK/data/raw/camargo2021_emg_converted" \
  --out-root "$WORK/data/processed/camargo2021" \
  --report "$WORK/wearusfm_runs/results/passo2/camargo_ingest_report.json"
