#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH -J ingest_emg2pose
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_emg2pose_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_emg2pose_%j.out

# Passo 2: ingest di emg2pose (462 GB tar, 25.253 registrazioni di 193 utenti a 2 kHz, un bracciale da 16 canali: solo EMG (~100 GB su $SCRATCH)).
# Uso: sbatch scripts/slurm/ingest_emg2pose.sbatch --max-recordings 3   (collaudo)
#      sbatch scripts/slurm/ingest_emg2pose.sbatch --skip-existing         (pieno; RIPRENDIBILE: se il budget di tempo
#      finisce prima, si rilancia con lo stesso comando e si riparte da dove si era arrivati)
# Budget di tempo 2h40 su 3h di limite (<= 6 ore locali): lo script si ferma in ordine e scrive il report.
# Processato su $SCRATCH (no quota; rinfrescare la data di modifica prima del purge: 25/10/2026); report fuori dal repo.

set -uo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

mkdir -p "$WORK/wearusfm_runs/results/passo2" "$WORK/wearusfm_runs/logs" "$SCRATCH/data/processed/emg2pose"
python3 scripts/ingest_meta_bracelet.py \
  --dataset emg2pose \
  --tar "$SCRATCH/data/raw/emg2pose/emg2pose_dataset.tar" \
  --csv "$WORK/data/raw/emg2pose_mini/emg2pose_metadata.csv" \
  --out-root "$SCRATCH/data/processed/emg2pose" \
  --report "$WORK/wearusfm_runs/results/passo2/emg2pose_ingest_report_${SLURM_JOB_ID}.json" \
  --time-budget-s 9600 \
  "$@"
