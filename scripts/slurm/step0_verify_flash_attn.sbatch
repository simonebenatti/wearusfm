#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH -J step0_flash_attn_check
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step0_verify_flash_attn_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step0_verify_flash_attn_%j.out

# Passo 0: prova funzionale di flash_attn_varlen_func in bf16 su A100.
# Precondizione del braccio L2 (dataloader_bench_spec.md §6). ~10 min su boost_qos_dbg.
#
# Gli output (log, risultati) vanno SEMPRE fuori dalla working tree del repo
# ($WORK/wearusfm_runs/, non $WORK/wearusfm/): altrimenti un job puo' creare file
# non tracciati che confliggono col prossimo `git push leonardo main` dal Mac
# (CLAUDE.md, "Progetto FM sEMG"). Un risultato entra nel repo solo quando lo si
# copia deliberatamente e lo si committa dal Mac.

set -euo pipefail
cd "$WORK/wearusfm"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"

mkdir -p "$WORK/wearusfm_runs/results/step0" "$WORK/wearusfm_runs/logs"
python bench/verify_flash_attn_varlen.py --out "$WORK/wearusfm_runs/results/step0/flash_attn_varlen_check_${SLURM_JOB_ID}.json"
