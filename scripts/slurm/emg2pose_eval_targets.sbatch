#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH -J emg2pose_eval_targets
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/emg2pose_eval_targets_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/emg2pose_eval_targets_%j.out
# Passo 0 di TEMPONet su emg2pose (docs/temponet_emg2pose_plan.md; regole ed emendamento in docs/decisioni.md, 10/11/2026). CPU, 0 GPU-ore,
# al massimo 2 core-ore (1 h su 2 CPU; il primo tentativo, job 64096497, e' fallito per I/O, vedi lo script). Dal tar: finestre
# ufficiali e angoli esatti di validazione e test, letti con pread dei soli membri necessari; dal processato: scala dell'ingresso.
# Le uscite vanno in una cartella NUOVA su $SCRATCH (rigenerabili); il manifest si copia sul Mac.
#   sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> emg2pose_eval_targets.sbatch <cartella nuova> [--limit N]
set -uo pipefail
OUT="${1:?cartella delle uscite}"; shift
case "$OUT" in /*) ;; *) echo "percorso relativo rifiutato: $OUT"; exit 2;; esac
case "$OUT" in "$WORK/wearusfm"|"$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm|/leonardo_work/IscrB_WearUsFM/wearusfm/*|\
"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*|"$SCRATCH/data"/*) echo "uscite dentro il codice o nei dati: rifiutato"; exit 2;; esac
if [ "${SLURM_RESTART_COUNT:-0}" != 0 ]; then echo "riavvio automatico non ammesso"; exit 2; fi
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
echo "== $(date -Is) codice $WEARUSFM_CODE_SHA in $WEARUSFM_CODE_DIR; uscite $OUT"
python3 scripts/emg2pose_eval_targets.py \
  --tar "$SCRATCH/data/raw/emg2pose/emg2pose_dataset.tar" \
  --csv "$WORK/data/raw/emg2pose_mini/emg2pose_metadata.csv" \
  --processed-root "$SCRATCH/data/processed/emg2pose" \
  --pose-root "$SCRATCH/data/processed/emg2pose_pose" \
  --out "$OUT" --time-budget-s 3300 --io-threads 4 "$@" || exit 1
du -sh "$OUT"
