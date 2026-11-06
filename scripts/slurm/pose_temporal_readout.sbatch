#!/bin/bash
# CPU-only frozen readout. Resources are explicit in the committed job plan.
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/pose_temporal_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/pose_temporal_%j.out
set -euo pipefail
: "${WORK:?}" "${CODE_DIR:?}" "${EXPECTED_SHA:?}"
if [ "$#" -ne 2 ]; then
  echo 'usage: pose_temporal_readout.sbatch {validate|benchmark|run} OUTPUT_JSON' >&2
  exit 2
fi
stage="$1"
output="$2"
case "$stage" in validate|benchmark|run) ;; *) echo 'invalid stage' >&2; exit 2 ;; esac
case "$output" in
  "$WORK/wearusfm_runs/results/pose_temporal_20261102/"*.json) ;;
  *) echo 'output must be a JSON in the approved external result directory' >&2; exit 2 ;;
esac
case "/$output/" in */../*|*/./*) echo 'noncanonical output path' >&2; exit 2 ;; esac
if [ -e "$output" ]; then echo 'output already exists; no overwrite' >&2; exit 2; fi
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTHONDONTWRITEBYTECODE=1
python3 "$CODE_DIR/scripts/pose_temporal_readout.py" "$stage" \
  --plan "$CODE_DIR/results/infra/pose_temporal_20261102/plan.json" \
  --expected-code-sha "$EXPECTED_SHA" --out "$output"
# Also check checkout cleanliness/read-only status after the computation.
code_dir_setup
