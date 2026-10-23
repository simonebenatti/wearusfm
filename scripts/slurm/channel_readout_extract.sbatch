#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH -J channel_readout_extract
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/channel_readout_extract_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/channel_readout_extract_%j.out

# Readout per canale (docs/readout_per_canale_v1.md): dal checkpoint post-decay di un run, gli strati local / dec per canale e il readout
# medio (scripts/channel_readout.py --stage extract) dei soli soggetti della validazione interna (P1: pretraining di DB2, DB3, DB6; P2 con
# `--task p2 --epn-root ... --uci-root ...`: soggetti fuori dai test dei tre split). 1 GPU; la durata la fissa il piano (--time).
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/channel_readout_extract.sbatch <checkpoint.pt> <cartella cache> [altri argomenti]

set -uo pipefail
CKPT="${1:?checkpoint}"; FEAT="${2:?cartella delle cache}"; shift 2
case "$FEAT" in /*) ;; *) echo "cartella delle cache relativa: rifiutata"; exit 2;; esac
case "$FEAT" in "$WORK/wearusfm"|"$WORK/wearusfm"/*|/leonardo_work/IscrB_WearUsFM/wearusfm|/leonardo_work/IscrB_WearUsFM/wearusfm/*|\
"$WORK/wearusfm_campaigns"/*|/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/*) echo "cache dentro il codice: rifiutate"; exit 2;; esac
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
mkdir -p "$FEAT" "$WORK/wearusfm_runs/logs"
python3 scripts/channel_readout.py --stage extract --checkpoint "$CKPT" --root "$WORK/data/processed" --root "$SCRATCH/data/processed" \
  --features "$FEAT" "$@"
