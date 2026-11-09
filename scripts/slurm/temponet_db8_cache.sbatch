#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000M
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH -J temponet_db8_cache
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/temponet_db8_cache_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/temponet_db8_cache_%j.out
# Cache .npy di NinaPro DB8 per la replica di TEMPONet (docs/temponet_db8_replication.md): legge una volta i 36 .mat ufficiali da
# $SCRATCH/data/raw/ninapro/DB8 (sola lettura) e scrive emg/DoA/etichette e manifest.json in una cartella NUOVA. CPU, 0 GPU-ora,
# al massimo 8 core-ore. Uso:
#   sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/temponet_db8_cache.sbatch <cartella della cache>
set -uo pipefail
OUT="${1:?cartella della cache}"
case "$OUT" in /*) ;; *) echo "cartella della cache relativa: rifiutata"; exit 2;; esac
case "$OUT" in "$WORK/wearusfm"|"$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm|/leonardo_work/IscrB_WearUsFM/wearusfm/*|\
"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*|"$SCRATCH/data/raw"/*) echo "cache dentro il codice o nel raw: rifiutato"; exit 2;; esac
if [ "${SLURM_RESTART_COUNT:-0}" != 0 ]; then echo "riavvio automatico non ammesso"; exit 2; fi
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/temponet_db8_cache.py --raw-root "$SCRATCH/data/raw/ninapro" --out "$OUT" --workers "${SLURM_CPUS_PER_TASK:-8}" || exit 1
ls -la "$OUT" | tail -5
du -sh "$OUT"
