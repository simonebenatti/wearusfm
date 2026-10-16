#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH -J probe_p2
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/probe_p2_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/probe_p2_%j.out

# Metrica P2 di D12 (EPN-612 e UCI-EMG, stesse finestre e split della replica di NeuroRVQ), fase di estrazione delle feature (GPU).
# Uso: sbatch scripts/slurm/probe_p2.sbatch <checkpoint.pt> <cartella delle feature> [args extra]
# Le sonde (CPU): probe_p2_probe.sbatch, in dipendenza afterok. Uscite fuori dal repo ($SCRATCH/wearusfm_runs/p2/...).

# Con la catena di campagna: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> ... (scripts/slurm/lib_code_dir.sh).

set -uo pipefail
CKPT="${1:?checkpoint}"; FEAT="${2:?cartella delle feature}"; shift 2
case "$FEAT" in /*) ;; *) echo "cartella delle feature relativa: rifiutata"; exit 2;; esac
case "$FEAT" in "$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm/*|"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*)
  echo "feature dentro il codice: rifiutate"; exit 2;; esac
# codice: checkout di campagna (CODE_DIR + EXPECTED_SHA, SHA e pulizia verificati) o il canonico (critica del punto 5, 16/10/2026)
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
mkdir -p "$FEAT" "$WORK/wearusfm_runs/logs"
RAW="${RAW_ROOT:-$WORK/data/raw}"  # EPN-612 e UCI-EMG restano in $WORK (delete_verified_raw.sbatch non li tocca)
python3 scripts/probe_p2.py --stage extract --checkpoint "$CKPT" --epn-root "$RAW/epn612/EMG-EPN612 Dataset/trainingJSON" \
  --epn-root "$RAW/epn612/EMG-EPN612 Dataset/testingJSON" --uci-root "$RAW/uci_emg" --features "$FEAT" "$@"
