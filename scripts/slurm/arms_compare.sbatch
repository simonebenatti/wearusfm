#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -J arms_compare
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/arms_compare_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/arms_compare_%j.out

# Bracci dell'obiettivo (docs/bracci_obiettivo_finestra1.md): confronto con le regole firmate il 16/10/2026 (scripts/arms_compare.py). CPU
# seriale, minuti; 0 GPU-ora. Ultimo nodo della catena di valutazione di un gruppo di bracci, dopo tutte le sonde.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/arms_compare.sbatch <spec.json (relativo al checkout)> <esito.json>

set -uo pipefail
SPEC="${1:?spec del confronto}"; OUT="${2:?esito}"
case "$OUT" in /*) ;; *) echo "uscita relativa: rifiutata"; exit 2;; esac
case "$OUT" in "$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm/*|"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*)
  echo "uscita dentro il codice: rifiutata"; exit 2;; esac
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python3 scripts/arms_compare.py "$SPEC" --out "$OUT"
