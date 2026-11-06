#!/bin/bash
# Frozen point-6 diagnostic; GPU resources explicitly supplied for extraction only.
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/pose_slots_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/pose_slots_%j.out
set -euo pipefail
: "${WORK:?}" "${CODE_DIR:?}" "${EXPECTED_SHA:?}"
stage="${1:?stage required}"
case "$stage" in
  preflight|benchmark|run) [ "$#" -eq 1 ] || exit 2 ;;
  extract) [ "$#" -eq 2 ] || exit 2; case "$2" in B|R3c) ;; *) exit 2 ;; esac ;;
  *) echo 'usage: pose_slots.sbatch preflight|benchmark|run|extract MODEL' >&2; exit 2 ;;
esac
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTHONDONTWRITEBYTECODE=1
plan="$CODE_DIR/results/infra/pose_slots_20261102/plan.json"
output_root='/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/pose_slots_20261102'
case "$stage" in
  preflight)
    python3 "$CODE_DIR/scripts/pose_slot_extract.py" preflight --plan "$plan" \
      --expected-code-sha "$EXPECTED_SHA" --out "$output_root/preflight" ;;
  extract)
    python3 "$CODE_DIR/scripts/pose_slot_extract.py" extract --model "$2" --plan "$plan" \
      --expected-code-sha "$EXPECTED_SHA" --out "$output_root/$2" ;;
  benchmark|run)
    python3 "$CODE_DIR/scripts/pose_slot_readout.py" "$stage" --plan "$plan" \
      --expected-code-sha "$EXPECTED_SHA" --out "$output_root/$stage.json" ;;
esac
code_dir_setup
