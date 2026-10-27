#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -J pose_audit
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/pose_audit_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/pose_audit_%j.out

# Controllo 1 della diagnosi di GPT del 27/10/2026 (collaudo dbg di R3 non superato): audit della catena dei bersagli di posa su 20 registrazioni
# vere di emg2pose (identita', ricostruzione dal tar grezzo, oracolo delle patch, supporto) e ridge da log-RMS alla posa con scansione del
# ritardo. CPU seriale, 4 core, 16 GB, 1 h di tetto, 0 GPU-ora. Dal checkout di campagna (CODE_DIR). Uscita fuori dal repo.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/pose_audit.sbatch [altri argomenti di scripts/pose_audit.py]

set -uo pipefail
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
OUT="$WORK/wearusfm_runs/results/pilot_posa_20261026/pose_audit/pose_audit_${SLURM_JOB_ID}.json"
mkdir -p "$(dirname "$OUT")" "$WORK/wearusfm_runs/logs"
echo "== $(date -Is) codice $WEARUSFM_CODE_SHA in $WEARUSFM_CODE_DIR; uscita $OUT"
python3 scripts/pose_audit.py \
  --csv "$WORK/data/raw/emg2pose_mini/emg2pose_metadata.csv" \
  --processed-root "$SCRATCH/data/processed/emg2pose" --pose-root "$SCRATCH/data/processed/emg2pose_pose" \
  --tar "$SCRATCH/data/raw/emg2pose/emg2pose_dataset.tar" \
  --out "$OUT" "$@"
