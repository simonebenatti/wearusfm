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
# Set numeric threads before importing any scientific module. The historical
# extraction uses one thread; both historical CPU readers use eight. Summary
# has only two allocated CPUs and does not fit a reader.
NUMERIC_THREADS=$(python3 - "$@" <<'THREAD_POLICY'
import argparse
threads = {
    'extract-classification': 1, 'extract-pose-train': 1, 'extract-pose-val': 1,
    'benchmark-classification': 8, 'benchmark-pose': 8,
    'fit-classification': 8, 'fit-pose': 8,
    'compare-classification': 8, 'compare-pose': 8, 'summary': 1,
}
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--stage', required=True, choices=threads)
args, _ = parser.parse_known_args()
print(threads[args.stage])
THREAD_POLICY
)
export OMP_NUM_THREADS="$NUMERIC_THREADS" OPENBLAS_NUM_THREADS="$NUMERIC_THREADS" MKL_NUM_THREADS="$NUMERIC_THREADS"
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
if args.stage=='fit-classification':
 # Amendment A1: each registered fold shard has its own walltime, frozen in the plan before any fit.
 shard='%s/%s/%s/folds%s'%(args.step,args.view,args.dataset,'-'.join(map(str,args.folds or [])))
 table=plan['amendment']['fit_shards']
 if shard not in table:raise SystemExit('unregistered amendment fit shard')
 policy={**policy,'seconds':table[shard]['walltime_seconds']}
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
print(json.dumps({'event':'allocation_accepted','stage':args.stage,'allocation':policy,
                  'numeric_threads':int(os.environ['OMP_NUM_THREADS'])}),flush=True)
PY
exec python3 scripts/consolidation_long.py "$@"
