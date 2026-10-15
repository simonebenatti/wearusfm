#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH -J calibration_probe
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/calibration_probe_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/calibration_probe_%j.out

# Calibrazione C1-C4 (docs/calibrazione_c1_c4_selezione.md): sonde a fold di soggetti sulle feature di un checkpoint (scripts/calibration_probe.py
# --stage probe): metrica S fuori fold e predizioni per i confronti accoppiati. CPU seriale, 8 core; 0 GPU-ora. Nodo della catena di
# valutazione, dopo l'estrazione.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/calibration_probe.sbatch <cartella feature> <rapporto.json>

set -uo pipefail
FEAT="${1:?cartella delle feature}"; OUT="${2:?rapporto}"; shift 2
case "$OUT" in /*) ;; *) echo "rapporto relativo: rifiutato"; exit 2;; esac
case "$OUT" in "$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm/*|"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*)
  echo "rapporto dentro il codice: rifiutato"; exit 2;; esac
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$(dirname "$OUT")"
python3 scripts/calibration_probe.py --stage probe --features "$FEAT" --out "$OUT" "$@"
