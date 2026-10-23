#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH -J classical_baseline
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/classical_baseline_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/classical_baseline_%j.out
# Baseline classica (feature nel tempo per canale, nessun modello; scripts/classical_baseline.py): per un compito, la validazione interna
# (stessa sonda della calibrazione) e il test ufficiale (probe_p1/probe_p2 --feature-set td --checkpoint none). CPU seriale, 0 GPU-ora.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/classical_baseline.sbatch <p1|p2> <cartella dei rapporti>
set -uo pipefail  # come i launcher delle sonde (module non c'e' fuori dal cluster); ogni comando Python esce con || exit
TASK="${1:?p1 o p2}"; OUT="${2:?cartella dei rapporti}"
case "$TASK" in p1|p2) ;; *) echo "compito: p1 o p2"; exit 2;; esac
case "$OUT" in /*) ;; *) echo "cartella dei rapporti relativa: rifiutata"; exit 2;; esac
case "$OUT" in "$WORK/wearusfm"|"$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm|/leonardo_work/IscrB_WearUsFM/wearusfm/*|\
"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*) echo "rapporti dentro il codice: rifiutato"; exit 2;; esac
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "cartella dei rapporti $OUT non vuota: ne serve una NUOVA"; exit 2; fi
mkdir -p "$OUT"
EPN=(--epn-root "$WORK/data/raw/epn612/EMG-EPN612 Dataset/trainingJSON" --epn-root "$WORK/data/raw/epn612/EMG-EPN612 Dataset/testingJSON")
ROOTS=(--root "$WORK/data/processed" --root "$SCRATCH/data/processed")
if [ "$TASK" = p1 ]; then
  python3 scripts/classical_baseline.py --task p1 "${ROOTS[@]}" --out "$OUT/interna_p1.json" || exit 1
  python3 scripts/probe_p1.py --checkpoint none --feature-set td --device cpu "${ROOTS[@]}" --window-s 1 --val-frac 0.1 \
    --weighting subjects --stage all --features "$OUT/feature_ufficiali_p1" --audit-inputs --out "$OUT/ufficiale_p1.json" || exit 1
else
  python3 scripts/classical_baseline.py --task p2 "${EPN[@]}" --uci-root "$WORK/data/raw/uci_emg" --out "$OUT/interna_p2.json" || exit 1
  python3 scripts/probe_p2.py --checkpoint none --feature-set td --device cpu "${EPN[@]}" --uci-root "$WORK/data/raw/uci_emg" \
    --stage all --features "$OUT/feature_ufficiali_p2" --audit-inputs --out "$OUT/ufficiale_p2.json" || exit 1
fi
