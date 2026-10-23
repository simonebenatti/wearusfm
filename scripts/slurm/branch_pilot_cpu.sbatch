#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH -J branch_pilot_cpu
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/branch_pilot_cpu_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/branch_pilot_cpu_%j.out
# Pilot dei rami (scripts/branch_pilot.py), fasi su CPU: ibrido PWh (minuti), uscite dei rami di un modello su P1 o P2 (P2 con
# --time=03:00:00, P1 1 h 30), confronto finale o del solo primario. Radici fisse (A/B, 2C, tranche downstream, diagnostica); la cartella
# del pilot e' il primo argomento. Nessuna ripresa, nessun retry. 0 GPU-ora.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/branch_pilot_cpu.sbatch <cartella del pilot> \
#        hybrid --teacher T  |  extract --task p1|p2 --arm PWl|PWf|PWm|PWh --teacher T  |  compare [--primary-only]
set -euo pipefail
PILOT="${1:?cartella del pilot}"; STAGE="${2:?fase}"; shift 2
DIAGNOSTIC_DIR=/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/rami_diagnostica_20261021  # fissa: i rapporti PW0/PW/RPW0/RPW
case "$PILOT" in /*) ;; *) echo "cartella del pilot relativa: rifiutata"; exit 2;; esac
case "$PILOT" in "$WORK/wearusfm"|"$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm|/leonardo_work/IscrB_WearUsFM/wearusfm/*|\
"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*) echo "cartella del pilot dentro il codice: rifiutata"; exit 2;; esac
for a in "$@"; do
  case "$a" in --out|--out=*|--pilot-runs*|--eval-root*|--downstream-root*|--step2c-runs*|--runs-root*|--diagnostic-dir*|--reports-dir*|\
--expected-commit*)
    echo "argomento $a non ammesso: radici e uscite le fissa il launcher"; exit 2;; esac
done
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
COMMON=(--expected-commit "$WEARUSFM_CODE_SHA" --pilot-runs "$PILOT/runs" --step2c-runs /leonardo_work/IscrB_WearUsFM/wearusfm_runs/runs/step2c_1410)
case "$STAGE" in
  hybrid) python scripts/branch_pilot.py --stage hybrid "${COMMON[@]}" "$@" ;;
  extract) python scripts/branch_pilot.py --stage extract "${COMMON[@]}" --eval-root "$PILOT/cache" \
             --downstream-root /leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/step1_downstream_0910 \
             --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
             --epn-root "$WORK/data/raw/epn612/EMG-EPN612 Dataset/trainingJSON" \
             --epn-root "$WORK/data/raw/epn612/EMG-EPN612 Dataset/testingJSON" --uci-root "$WORK/data/raw/uci_emg" "$@" ;;
  compare)
    OUT="$PILOT/esito.json"
    for a in "$@"; do [ "$a" = --primary-only ] && OUT="$PILOT/esito_primario.json"; done
    python scripts/branch_pilot.py --stage compare --expected-commit "$WEARUSFM_CODE_SHA" --pilot-runs "$PILOT/runs" \
      --diagnostic-dir "$DIAGNOSTIC_DIR" --reports-dir "$PILOT/sonde" --out "$OUT" "$@" ;;
  *) echo "fase $STAGE sconosciuta"; exit 2 ;;
esac
