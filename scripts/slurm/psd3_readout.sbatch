#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -J psd3_readout
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/psd3_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/psd3_%j.out

# PSD-3, lettori a encoder congelato sulle cache di B19618 (docs/psd3_readers_20261029.md, preparazione GPT d4092b8; esecuzione affidata a
# Claude il 29/10, docs/prompt_claude_psd3_dopo_sonde_20261029.md): CPU seriale, 0 GPU-ora, nessuna estrazione nuova.
# Il codice gira SOLO da un checkout di campagna congelato (CODE_DIR + EXPECTED_SHA obbligatori: lib_code_dir.sh verifica SHA, pulizia e sola
# lettura). Il piano scientifico e' quello congelato nel checkout: il launcher lo fissa, l'operatore non puo' indicarne un altro.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> [--cpus-per-task=N --mem=.. --time=..] \
#        scripts/slurm/psd3_readout.sbatch <uscita assoluta, fuori dal codice> <fase: inventory|benchmark|run|compare> [argomenti]
# I thread BLAS/OpenMP (e quelli di torch, che legge OMP_NUM_THREADS) seguono i core allocati: la firma numerica dei rapporti li registra, e il
# confronto esige la stessa firma per tutti e venti gli shard: stessi core per tutti gli shard.

set -uo pipefail
OUT="${1:?uscita (file), assoluta e fuori dal codice}"; STAGE="${2:?fase: inventory|benchmark|run|compare}"; shift 2
: "${CODE_DIR:?CODE_DIR del checkout di campagna congelato}" "${EXPECTED_SHA:?EXPECTED_SHA del checkout di campagna}"
case "$OUT" in /*) ;; *) echo "uscita relativa: rifiutato"; exit 2;; esac
case "$OUT" in "$WORK/wearusfm"|"$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm|/leonardo_work/IscrB_WearUsFM/wearusfm/*|\
"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*|"$CODE_DIR"|"$CODE_DIR"/*) echo "uscita dentro il codice: rifiutato"; exit 2;; esac
case "$STAGE" in inventory|benchmark|run|compare) ;; *) echo "fase sconosciuta: $STAGE"; exit 2;; esac
for a in "$@"; do case "$a" in --plan|--plan=*|--out|--out=*|--archive|--archive=*) echo "argomento non ammesso: $a (piano e uscita li fissa il launcher)"; exit 2;; esac; done
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$(dirname "$OUT")"
python3 scripts/psd3_readout.py "$STAGE" --plan "$CODE_DIR/results/infra/psd3_preparation_20261029/plan.json" --out "$OUT" "$@"
