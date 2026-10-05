#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH -J probe_p1
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/probe_p1_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/probe_p1_%j.out

# Metrica P1 di D12 (cross-soggetto sul ramo sparso, NinaPro DB2, DB3, DB6), fase di estrazione delle feature (GPU; protocollo in
# docs/foglio_p1_cross_soggetto.md). Uso: sbatch scripts/slurm/probe_p1.sbatch <checkpoint.pt> <cartella delle feature> [args extra]
# Le sonde (CPU): probe_p1_probe.sbatch, in dipendenza afterok. Uscite fuori dal repo ($SCRATCH/wearusfm_runs/p1/...).

set -uo pipefail
CKPT="${1:?checkpoint}"; FEAT="${2:?cartella delle feature}"; shift 2
cd "$WORK/wearusfm"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
mkdir -p "$FEAT" "$WORK/wearusfm_runs/logs"
python3 scripts/probe_p1.py --stage extract --checkpoint "$CKPT" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
  --features "$FEAT" "$@"
