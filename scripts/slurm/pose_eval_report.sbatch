#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH -J pose_eval_report
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/pose_eval_report_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/pose_eval_report_%j.out

# Emendamento 1 (E1-E3): il rapporto dalle estrazioni (val e, per E3, il sottoinsieme del train). CPU seriale, 8 core, 2 h di tetto.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/pose_eval_report.sbatch <val.npz> <rapporto.json> [--train <train_subset.npz>]

set -uo pipefail
VAL="${1:?val.npz}"; OUT="${2:?rapporto.json}"; shift 2
case "$OUT" in /*) ;; *) echo "rapporto relativo: rifiutato"; exit 2;; esac
case "$OUT" in "$WORK/wearusfm"|"$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm|/leonardo_work/IscrB_WearUsFM/wearusfm/*|\
"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*) echo "rapporto dentro il codice: rifiutato"; exit 2;; esac
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$(dirname "$OUT")"
python3 scripts/pose_eval.py --stage report --val "$VAL" --out "$OUT" "$@"
