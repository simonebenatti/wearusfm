#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH -J step2a_probe
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step2a_probe_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step2a_probe_%j.out
# Test 2A dello step 2 (docs/step2a_stats_probe.md): sonde P1/P2 ufficiali su una o piu' varianti, in sequenza nella stessa allocazione.
# Varianti per teacher (ARM = B_seed0 o B_seed1): M, MS; senza teacher (ARM = -): S, K. Con COMPARE=1 (solo l'ultimo job, dopo tutti gli
# altri) il confronto finale nella stessa allocazione. Tempi massimi nel protocollo (--time al momento della sottomissione). Nessun retry.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha>[,COMPARE=1] scripts/slurm/step2a_probe.sbatch <p1|p2> <M,MS|S,K> <arm|->
#        <runs> <downstream> <eval>
set -euo pipefail
TASK="${1:?p1/p2}"; VARIANTS="${2:?varianti}"; ARM="${3:?braccio o -}"; RUNS="${4:?runs}"; DOWN="${5:?downstream}"; EVAL="${6:?eval}"
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
ARM_ARGS=()
if [ "$ARM" != "-" ]; then ARM_ARGS=(--arm "$ARM"); fi
IFS=',' read -r -a LIST <<< "$VARIANTS"
for i in "${!LIST[@]}"; do
  EXTRA=()
  if [ "${COMPARE:-0}" = "1" ] && [ "$i" -eq $(( ${#LIST[@]} - 1 )) ]; then EXTRA=(--compare-after); fi
  python scripts/step2a.py --stage probe --task "$TASK" --variant "${LIST[$i]}" "${ARM_ARGS[@]}" --expected-commit "$WEARUSFM_CODE_SHA" \
    --runs-root "$RUNS" --downstream-root "$DOWN" --eval-root "$EVAL" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
    "${EXTRA[@]}"
done
