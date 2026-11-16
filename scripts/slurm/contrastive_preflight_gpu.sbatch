#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=00:30:00
#SBATCH -J contrastive_gpu
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/contrastive_gpu_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/contrastive_gpu_%j.out

set -euo pipefail
if [ "$#" -ne 1 ]; then echo 'usage: launcher <relative reviewed job.json>' >&2; exit 2; fi
: "${WORK:?}" "${CODE_DIR:?}" "${EXPECTED_SHA:?}" "${SLURM_JOB_ID:?}"
case "$1" in /*|*..*) echo 'contained relative job specification required' >&2; exit 2;; esac
case "$CODE_DIR" in "$WORK"/wearusfm_campaigns/*) ;; *) echo 'frozen campaign checkout required' >&2; exit 2;; esac
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
JOB_INFO=$(scontrol show job -o "$SLURM_JOB_ID")
ALLOC=$(printf '%s\n' "$JOB_INFO" | tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2}')
SECS=$(printf '%s\n' "$ALLOC" | awk -F'[-:]' 'NF==4 {print $1*86400+$2*3600+$3*60+$4; next} NF==3 {print $1*3600+$2*60+$3; next} NF==2 {print $1*60+$2; next} {print 0}')
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec python3 scripts/contrastive_job.py --job "$CODE_DIR/$1" --kind gpu --allocated-seconds "$SECS"
