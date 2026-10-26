#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -J ingest_emg2pose_pose
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_emg2pose_pose_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/ingest_emg2pose_pose_%j.out

# Seconda ondata del pilot degli obiettivi (docs/pilot_posa_v1.md §6, punti 2, 5, 6): ingest della POSA di emg2pose (solo `joint_angles`) dal tar
# gia' su $SCRATCH, in un albero separato da quello processato dell'EMG. Dal checkout di campagna (CODE_DIR, mai da $WORK/wearusfm).
# CPU seriale, 2 core, 0 GPU-ora. Budget di tempo 3 h 40 su 4 h: si ferma in ordine e scrive il report; si riprende con --skip-existing.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/ingest_emg2pose_pose.sbatch --limit 40 --both-sides   (collaudo)
#      sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/ingest_emg2pose_pose.sbatch --skip-existing --zero-margin-samples <M> --no-measure --stats
#      sbatch ... scripts/slurm/ingest_emg2pose_pose.sbatch --stats-only

set -uo pipefail
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
OUT_ROOT="$SCRATCH/data/processed/emg2pose_pose"
REPORT="$WORK/wearusfm_runs/results/passo2/emg2pose_pose_ingest_report_${SLURM_JOB_ID}.json"
mkdir -p "$WORK/wearusfm_runs/results/passo2" "$WORK/wearusfm_runs/logs" "$OUT_ROOT"
echo "== $(date -Is) codice $WEARUSFM_CODE_SHA in $WEARUSFM_CODE_DIR; out $OUT_ROOT; report $REPORT"
python3 scripts/ingest_emg2pose_pose.py \
  --tar "$SCRATCH/data/raw/emg2pose/emg2pose_dataset.tar" \
  --csv "$WORK/data/raw/emg2pose_mini/emg2pose_metadata.csv" \
  --out-root "$OUT_ROOT" \
  --report "$REPORT" \
  --time-budget-s 13200 \
  "$@"
