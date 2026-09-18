#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -J step0_asse_l
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step0_asse_l_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step0_asse_l_%j.out

# Passo 0, asse L (layout: padding / packing / bucketing) - PRIMA VALIDAZIONE, scala
# ridotta (non ancora la misura statistica finale a 500 batch, spec §10): serve prima
# accertare che il codice giri senza errori su hardware reale (mai testato senza CUDA).
#
# Output SEMPRE fuori dalla working tree del repo (CLAUDE.md, "Progetto FM sEMG").

set -euo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

RESULTS="$WORK/wearusfm_runs/results/step0/asse_l"
mkdir -p "$RESULTS" "$WORK/wearusfm_runs/logs"

COMMON="--d-model 384 --n-backbone-layers 10 --batch-size 32 --n-batches 15 --warmup-batches 3"

for arm in padding packing bucketing; do
  echo "=== asse L: $arm ==="
  python bench/measure_asse_l.py --arm "$arm" $COMMON --seed 0 \
    --out "$RESULTS/${arm}_${SLURM_JOB_ID}.json"
done

echo "risultati in $RESULTS"
