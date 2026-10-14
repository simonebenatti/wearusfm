#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH -J step2c_probe
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step2c_probe_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step2c_probe_%j.out
# Step 2C (docs/step2c_frozen_branches.md): sonde P1/P2 ufficiali su due varianti di un teacher, in sequenza nella stessa allocazione
# (RPW,RPW0 oppure RP,RW). P2 va sottomesso con --time=02:30:00. Con COMPARE=1 (solo l'ultimo job, dopo tutti gli altri) il confronto
# finale; con STEP2A=<cartella del test 2A concluso> anche il confronto con le statistiche fisse. Nessun retry.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha>[,COMPARE=1][,STEP2A=<cartella>] scripts/slurm/step2c_probe.sbatch \
#        <p1|p2> <arm> <varianti, es. RPW,RPW0> <runs A/B> <runs dei rami> <downstream> <eval>
set -euo pipefail
TASK="${1:?p1/p2}"; ARM="${2:?braccio}"; VARIANTS="${3:?varianti}"; RUNS="${4:?runs A/B}"; BRANCH_RUNS="${5:?runs dei rami}"
DOWN="${6:?downstream}"; EVAL="${7:?eval}"
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
IFS=',' read -r -a LIST <<< "$VARIANTS"
EXTRA=()
if [ "${COMPARE:-0}" = "1" ]; then
  EXTRA=(--compare-after)
  if [ -n "${STEP2A:-}" ]; then EXTRA+=(--step2a-root "$STEP2A"); fi
fi
python scripts/step2c.py --stage probe --task "$TASK" --arm "$ARM" --variant "${LIST[@]}" --expected-commit "$WEARUSFM_CODE_SHA" \
  --runs-root "$RUNS" --branch-runs "$BRANCH_RUNS" --downstream-root "$DOWN" --eval-root "$EVAL" \
  --root "$WORK/data/processed" --root "$SCRATCH/data/processed" "${EXTRA[@]}"
