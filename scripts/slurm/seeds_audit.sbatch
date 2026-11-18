#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -J seeds_audit
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/seeds_audit_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/seeds_audit_%j.out

# Structural checks only, on the fixed public inventory. No predictive test metrics.
set -euo pipefail
source "${CODE_DIR:?}/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RESULT="$WORK/wearusfm_runs/results/seeds_20261118"
mkdir -p "$RESULT"
echo "code=$WEARUSFM_CODE_SHA job=$SLURM_JOB_ID"
python scripts/seeds_audit.py \
  --manifest results/infra/seeds_20261118/source_manifest.json \
  --raw-root "$SCRATCH/data/external_benchmarks/seeds" \
  --report "$RESULT/audit_${SLURM_JOB_ID}.json" --check-duplicates
