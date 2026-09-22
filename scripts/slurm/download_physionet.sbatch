#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -J dl_physionet
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_physionet_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_physionet_%j.out
#SBATCH --dependency=singleton

# Passo 1 (v10 §2.5): GRABMyo e Hyser da PhysioNet, open access.
#
# Corretto due volte il 22/09/2026:
# 1. wget -r ricorsivo sul server web physionet.org era troppo lento (~150KB/s,
#    17h stimate solo per GRABMyo).
# 2. Lo zip unico ("Download the ZIP file") passa dallo stesso server web lento -
#    nessun miglioramento, il collo di bottiglia e' physionet.org da Leonardo, non
#    il numero di richieste.
# Fix: il dataset e' mirrorato sul bucket S3 pubblico physionet-open (Open Data on
# AWS), che risponde velocemente da Leonardo. Niente aws-cli disponibile: si usa
# scripts/s3_bucket_download.py (solo libreria standard, REST API S3 diretta),
# paginato e parallelo (8 worker).

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

RAW="$WORK/data/raw"

echo "=== GRABMyo (via S3 physionet-open, ~10 GB, migliaia di file WFDB) ==="
python3 scripts/s3_bucket_download.py --bucket physionet-open --prefix "grabmyo/1.1.0/" \
  --dest "$RAW/grabmyo" --strip-prefix --workers 8

echo "=== Hyser (HD-sEMG, via S3, ~143 GB - probabile piu' sottomissioni) ==="
python3 scripts/s3_bucket_download.py --bucket physionet-open --prefix "hd-semg/2.0.0/" \
  --dest "$RAW/hyser" --strip-prefix --workers 8

echo "=== riepilogo ==="
du -sh "$RAW/grabmyo" "$RAW/hyser" 2>&1
