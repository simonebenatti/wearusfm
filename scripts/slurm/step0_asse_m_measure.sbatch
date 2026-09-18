#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -J step0_asse_m
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step0_asse_m_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step0_asse_m_%j.out

# Passo 0, asse M (montaggi: al volo vs precompute), parametri D2 congelati
# (docs/decisioni.md). Scrive il pool su $WORK, poi misura a cache fredda e calda per
# entrambi gli arm - D6a (docs/decisioni.md) decide fra i due col risultato.
#
# Output SEMPRE fuori dalla working tree del repo (CLAUDE.md, "Progetto FM sEMG").

set -euo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

RUNS="$WORK/wearusfm_runs"
DATA="$RUNS/data/step0_${SLURM_JOB_ID}"
RESULTS="$RUNS/results/step0/asse_m"
mkdir -p "$DATA" "$RESULTS" "$RUNS/logs"

# D2: quote 40/35/25, p_piena 0.5, classe C pessimistica (default dello script),
# contesto 4s, batch 64, 8 worker per GPU (qui: 8 core, nessuna GPU su questa QoS).
COMMON="--quote-a 40 --quote-b 35 --quote-c 25 --p-piena 0.5 --context-s 4.0"
MEASURE_COMMON="--n-batches 500 --warmup-batches 50 --batch-size 64 --n-workers 8 --n-params 30e6"

for arm in al_volo precompute; do
  root="$DATA/$arm"
  echo "=== scrittura pool: $arm ==="
  python bench/bench_dataloader.py write-shards --root "$root" --arm "$arm" \
    --n-shards 3000 --seed 0 $COMMON

  echo "=== misura a cache FREDDA: $arm ==="
  python bench/bench_dataloader.py measure --root "$root" --arm "$arm" $MEASURE_COMMON \
    --seed 1 --out "$RESULTS/${arm}_cold_${SLURM_JOB_ID}.json"

  echo "=== misura a cache CALDA: $arm ==="
  python bench/bench_dataloader.py measure --root "$root" --arm "$arm" $MEASURE_COMMON \
    --seed 2 --out "$RESULTS/${arm}_warm_${SLURM_JOB_ID}.json"
done

echo "risultati in $RESULTS"
