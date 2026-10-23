#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH -J channel_readout_probe
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/channel_readout_probe_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/channel_readout_probe_%j.out

# Readout per canale (docs/readout_per_canale_v1.md): la sonda della validazione interna (stessi fold, C e pesi di calibration_probe) su uno
# strato della cache, a larghezza pari (PCA sul train del fold a k = 2d, default) o intera (--k 0): scripts/channel_readout.py --stage probe.
# CPU seriale, 8 core, 0 GPU-ora. Uno strato per job.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/channel_readout_probe.sbatch <cartella cache> <rapporto.json> --layer <strato> [--task p2] [--k N]

set -uo pipefail
FEAT="${1:?cartella delle cache}"; OUT="${2:?rapporto}"; shift 2
case "$OUT" in /*) ;; *) echo "rapporto relativo: rifiutato"; exit 2;; esac
case "$OUT" in "$WORK/wearusfm"|"$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm|/leonardo_work/IscrB_WearUsFM/wearusfm/*|\
"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*) echo "rapporto dentro il codice: rifiutato"; exit 2;; esac
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$(dirname "$OUT")"
python3 scripts/channel_readout.py --stage probe --features "$FEAT" --out "$OUT" "$@"
