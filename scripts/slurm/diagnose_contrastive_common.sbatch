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
#SBATCH -J contrast_diagnostic
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/contrastive_diag_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/contrastive_diag_%j.out

set -eu -o pipefail
: "${WORK:?}" "${CODE_DIR:?}" "${EXPECTED_SHA:?}" "${SLURM_JOB_ID:?}" "${DIAGNOSTIC_CELL:?}"
[[ "$#" == 0 ]]
[[ "$CODE_DIR" == "$WORK/wearusfm_campaigns/contrastive-common-20261118" ]]
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
[[ "$SLURM_JOB_NUM_NODES" == 1 && "$SLURM_NTASKS" == 1 && "$SLURM_CPUS_PER_TASK" == 32 ]]
[[ "$SLURM_MEM_PER_NODE" == 245760 ]]
scontrol show job -o "$SLURM_JOB_ID" | tr ' ' '\n' | grep -Fx 'TimeLimit=00:30:00'
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python3 -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 4'
output="$WORK/wearusfm_runs/results/contrastive_common_diagnostics_20261118/$DIAGNOSTIC_CELL"
case "$DIAGNOSTIC_CELL" in
    cuda)
        export TMPDIR="${SLURM_TMPDIR:-/tmp}"
        python3 -c 'import shutil, sys, tempfile; sys.exit(0 if shutil.disk_usage(tempfile.gettempdir()).free >= 1024**3 else "CUDA diagnostic requires 1 GiB of temporary space")'
        program=(scripts/diagnostics/contrastive_common_state_cuda.py)
        ;;
    resume_CN|resume_CR)
        program=(scripts/diagnostics/contrastive_common_checkpoint.py --arm "${DIAGNOSTIC_CELL#resume_}")
        ;;
    *) echo "Unknown diagnostic cell $DIAGNOSTIC_CELL" >&2; exit 2 ;;
esac
[[ ! -e "$output" && ! -L "$output" ]]
mkdir -p "$(dirname -- "$output")"
echo "job=$SLURM_JOB_ID cell=$DIAGNOSTIC_CELL code=$EXPECTED_SHA output=$output"
exec python3 "${program[@]}" --out-dir "$output"
