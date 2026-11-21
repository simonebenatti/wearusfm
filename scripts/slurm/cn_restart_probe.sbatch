#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=00:15:00
#SBATCH -J cn_restart_probe
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/cn_restart_probe_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/cn_restart_probe_%j.err
set -euo pipefail
: "${WORK:?}" "${CODE_DIR:?}" "${EXPECTED_SHA:?}" "${SLURM_JOB_ID:?}"
[[ "$#" == 0 && "$CODE_DIR" == "$WORK/wearusfm_campaigns/cn-restart545-20261121" ]]
[[ "${TORCH_DIST_INIT_BARRIER:-}" == 1 ]]
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
info=$(scontrol show job -o "$SLURM_JOB_ID")
[[ " $info " == *" TimeLimit=00:15:00 "* ]]
# SLURM_JOB_END_TIME availability differs between Slurm installations.
if [[ -z "${SLURM_JOB_END_TIME:-}" ]]; then
    end=$(awk '{for(i=1;i<=NF;i++)if($i~/^EndTime=/){sub(/^EndTime=/,"",$i); print $i}}' <<< "$info")
    SLURM_JOB_END_TIME=$(date -d "$end" +%s)
    export SLURM_JOB_END_TIME
fi
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec python3 scripts/diagnostics/contrastive_cn_restart.py --run
