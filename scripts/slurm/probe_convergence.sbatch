#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH -J probe_convergence
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/probe_convergence_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/probe_convergence_%j.out

# Controllo della convergenza delle sonde P1/P2 di B19618 contro R3c (docs/sonde_convergenza_20261029.md): CPU seriale, 0 GPU-ora.
# Il codice di analisi gira SOLO da un checkout di campagna congelato (CODE_DIR + EXPECTED_SHA obbligatori: lib_code_dir.sh verifica SHA, pulizia e
# sola lettura ed esporta WEARUSFM_CODE_SHA, che lo script registra come codice di analisi, distinto dal codice 24139ab che ha prodotto le cache).
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> [--cpus-per-task=N --mem=.. --time=..] \
#        scripts/slurm/probe_convergence.sbatch <uscita assoluta, fuori dal codice> <fase: probe|aggregate|parity|gate|compare-phases> [argomenti]
# I thread BLAS/OpenMP seguono i core allocati (le sonde originali giravano cosi', a 8 core); lo script li registra nell'audit.

set -uo pipefail
OUT="${1:?uscita (file o cartella), assoluta e fuori dal codice}"; STAGE="${2:?fase: probe|aggregate|parity|gate|compare-phases}"; shift 2
: "${CODE_DIR:?CODE_DIR del checkout di campagna congelato}" "${EXPECTED_SHA:?EXPECTED_SHA del checkout di campagna}"
case "$OUT" in /*) ;; *) echo "uscita relativa: rifiutato"; exit 2;; esac
case "$OUT" in "$WORK/wearusfm"|"$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm|/leonardo_work/IscrB_WearUsFM/wearusfm/*|\
"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*|"$CODE_DIR"|"$CODE_DIR"/*) echo "uscita dentro il codice: rifiutato"; exit 2;; esac
case "$STAGE" in probe|aggregate|parity|gate|compare-phases) ;; *) echo "fase sconosciuta: $STAGE"; exit 2;; esac
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$(dirname "$OUT")"
python3 scripts/probe_convergence.py --stage "$STAGE" --out "$OUT" "$@"
