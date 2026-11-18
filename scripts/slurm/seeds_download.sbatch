#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH -J seeds_download
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/seeds_download_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/seeds_download_%j.out

# Official public payload only. No preprocessing, pretraining registration or fit.
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
echo "code=$WEARUSFM_CODE_SHA job=$SLURM_JOB_ID raw=$SCRATCH/data/external_benchmarks/seeds"
python scripts/seeds_download.py \
  --manifest results/infra/seeds_20261118/source_manifest.json \
  --destination "$SCRATCH/data/external_benchmarks/seeds" \
  --receipt "$RESULT/download_${SLURM_JOB_ID}.json" \
  --max-bytes 59737275276 --workers 4 --timeout 120 --retries 2 \
  --free-margin-bytes 5368709120
