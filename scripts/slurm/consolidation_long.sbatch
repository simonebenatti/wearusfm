#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH -J consolidation_long_eval
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/consolidation_long_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/consolidation_long_%j.out
set -euo pipefail
: "${CODE_DIR:?}" "${EXPECTED_SHA:?}" "${WORK:?}"
[ "${SLURM_RESTART_COUNT:-0}" = 0 ] || { echo 'No automatic restart/requeue' >&2; exit 2; }
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 - "$@" <<'PY'
import sys,os,json,subprocess
from pathlib import Path
sys.path[:0]=['src','scripts']
from consolidation_long import parser
args=parser().parse_args(sys.argv[1:])
plan=json.loads(args.plan.read_text())
key={'extract-classification':'extract','extract-pose-train':'extract','extract-pose-val':'extract',
     'benchmark-classification':'classification_benchmark','benchmark-pose':'pose_benchmark',
     'fit-classification':'classification_fit','fit-pose':'pose_fit',
     'compare-classification':'classification_compare','compare-pose':'pose_compare','summary':'summary'}[args.stage]
policy=plan['resources'][key]
if args.stage=='fit-classification' and args.dataset=='ninapro_db2':
 policy=plan['resources']['classification_fit_db2']
info=subprocess.check_output(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']],text=True)
def hhmmss(seconds):return '%02d:%02d:%02d'%(seconds//3600,seconds//60%60,seconds%60)
if ('TimeLimit='+hhmmss(policy['seconds'])) not in info.split():raise SystemExit('wrong walltime')
if int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=policy['cpus']:raise SystemExit('wrong CPUs')
if int(os.environ.get('SLURM_MEM_PER_NODE','0'))!=policy['memory_gib']*1024:raise SystemExit('wrong RAM')
if int(os.environ.get('SLURM_JOB_NUM_NODES','0'))!=1:raise SystemExit('one node required')
if policy['gpus']:
 import torch
 if torch.cuda.device_count()!=1:raise SystemExit('one GPU required')
elif 'gres/gpu=' in info:raise SystemExit('CPU stage must not allocate GPUs')
print(json.dumps({'event':'allocation_accepted','stage':args.stage,'allocation':policy}),flush=True)
PY
exec python3 scripts/consolidation_long.py "$@"
