#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH -J dl_db10
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_db10_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/dl_db10_%j.out

# NinaPro DB10 = MeganePro (Harvard Dataverse, CC BY 4.0, nessun guestbook): SOLO i .mat di MDS1, MDS2, MDS4 (Simone, 01/10/2026; niente video
# .mp4, niente MDSInfo). 243 file, 78,0 GB (elenco dall'API il 01/10/2026). Ogni file e' verificato per dimensione e MD5 contro l'API prima di
# essere rinominato da .part; un file gia' completo si salta. Il budget di 3 h 45 min lascia chiudere il job prima del limite: se il report dice
# "stopped": "time-budget", si risottomette lo stesso comando e riprende. Velocita' misurata dal login node il 01/10/2026: ~25 MB/s (78 GB in ~1 h).
#
# Uso: sbatch scripts/slurm/download_db10.sbatch
# Destinazione $SCRATCH ($WORK e' pieno; purge a 40 giorni: rinfrescare la data di modifica); report fuori dal repo.

set -uo pipefail
cd "$WORK/wearusfm"

# il python3 di sistema del nodo e' troppo vecchio per lo script (01/10/2026: SyntaxError su `from __future__ import annotations`)
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

mkdir -p "$WORK/wearusfm_runs/results/passo2" "$WORK/wearusfm_runs/logs" "$SCRATCH/data/raw/ninapro/DB10"
python3 scripts/dataverse_download.py \
  --server https://dataverse.harvard.edu \
  --dataset doi:10.7910/DVN/1Z3IOM=MDS1 \
  --dataset doi:10.7910/DVN/78QFZH=MDS2 \
  --dataset doi:10.7910/DVN/F9R33N=MDS4 \
  --suffix .mat \
  --dest "$SCRATCH/data/raw/ninapro/DB10" \
  --report "$WORK/wearusfm_runs/results/passo2/download_db10_${SLURM_JOB_ID}.json" \
  --time-budget-s 13500 \
  "$@"
