#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -J ingest_csl_hdemg
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_csl_hdemg_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_csl_hdemg_%j.out

# Passo 2: ingest CSL-hdemg. Primo lancio: SOLO soggetto 5, sessione 3 (lo stesso
# usato nell'esempio ufficiale del dataset, src/example.py), collaudo prima di
# lanciare su tutti i 5 soggetti x 5 sessioni.

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

python3 scripts/ingest_csl_hdemg.py \
  --zip-path "$WORK/data/raw/csl_hdemg/csl_hdemg.zip" \
  --out-root "$WORK/data/processed/csl_hdemg" \
  --subjects 5 \
  --sessions 3 \
  --report "$WORK/wearusfm_runs/results/passo2/csl_hdemg_ingest_report_s5session3.json"
