#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH -J step1_random_probe
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_random_probe_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_random_probe_%j.out
# Args as extract. Optional sixth arg: closed learned eval root, ONLY on last
# probe depending afterok on its extract AND the other three probes.
set -euo pipefail
export EXPECTED_SHA="${1:?commit}"
TASK="${2:?p1/p2}"; SEED="${3:?seed}"; RUNS="${4:?runs}"; EVAL="${5:?eval}"
: "${CODE_DIR:?frozen CODE_DIR required}"
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
EXTRA=()
if [ "$#" -eq 6 ]; then EXTRA=(--compare-after --learned-root "$6"); fi
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python scripts/step1_random_downstream.py --stage probe --expected-commit "$EXPECTED_SHA" --task "$TASK" --model-seed "$SEED" \
  --runs-root "$RUNS" --eval-root "$EVAL" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" "${EXTRA[@]}"
