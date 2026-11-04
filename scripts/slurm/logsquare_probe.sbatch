#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH -J logsquare_probe
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/logsquare_probe_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/logsquare_probe_%j.out

# Operational wrapper for the frozen teacher/pool768 endpoint reader.
# Args: <plan relative to CODE_DIR> <plan SHA256> <stage> <new absolute output>
#   fit-shard: <db2-01|db2-23|db3|db6|p2>
#   aggregate: <p1|p2> <shard.json...> (four P1 shards, one P2 shard)
#   compare: <two candidate aggregate.json paths>
# Default resources cover DB2/P2 fits. Override sbatch CPU/time for DB3/DB6
# (8 CPU/30min), aggregate (2 CPU/20min), compare (2 CPU/30min).
# CPU gate reuses pinned historical per-shard measured resources; no new benchmark run.
# No science overrides, cache generation or submission here.
set -euo pipefail
if [ "$#" -lt 4 ]; then
  echo 'usage: logsquare_probe.sbatch <relative plan> <plan SHA256> <fit-shard|aggregate|compare> <new absolute output> [stage arguments]' >&2
  exit 2
fi
: "${WORK:?WORK}" "${CODE_DIR:?CODE_DIR}" "${EXPECTED_SHA:?EXPECTED_SHA}" "${SLURM_JOB_ID:?SLURM_JOB_ID}"
PLAN_REL="$1"
PLAN_SHA="$2"
STAGE="$3"
OUT="$4"
shift 4
case "$PLAN_REL" in /*|"") echo 'plan must be relative to code' >&2; exit 2;; esac
[[ "$PLAN_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo 'invalid plan SHA256' >&2; exit 2; }
EXTRA=()
INPUTS=()
case "$STAGE" in
  fit-shard)
    [ "$#" -eq 1 ] || { echo 'fit-shard requires shard key' >&2; exit 2; }
    CPUS_EXPECTED=8
    SECONDS_CAP=3600
    case "$1" in
      db2-01) EXTRA=(--shard db2-01);;
      db2-23) EXTRA=(--shard db2-23);;
      db3) EXTRA=(--shard db3); SECONDS_CAP=1800;;
      db6) EXTRA=(--shard db6); SECONDS_CAP=1800;;
      p2) EXTRA=(--shard p2);;
      *) echo 'unregistered shard key' >&2; exit 2;;
    esac
    ;;
  aggregate)
    [ "$#" -ge 2 ] || { echo 'aggregate requires task and shard reports' >&2; exit 2; }
    TASK="$1"
    shift 1
    case "$TASK:$#" in p1:4|p2:1) ;; *) echo 'aggregate requires four P1 shards or one P2 shard' >&2; exit 2;; esac
    INPUTS=("$@")
    EXTRA=(--task "$TASK" --shards "${INPUTS[@]}")
    CPUS_EXPECTED=2
    SECONDS_CAP=1200
    ;;
  compare)
    [ "$#" -eq 2 ] || { echo 'compare requires two candidate aggregate reports' >&2; exit 2; }
    INPUTS=("$@")
    EXTRA=(--reports "${INPUTS[@]}")
    CPUS_EXPECTED=2
    SECONDS_CAP=1800
    ;;
  *) echo 'unregistered stage' >&2; exit 2;;
esac
if [ "${SLURM_JOB_PARTITION:-}" != lrd_all_serial ] || [ "${SLURM_JOB_NUM_NODES:-0}" -ne 1 ] \
    || [ "${SLURM_NTASKS:-1}" -ne 1 ] || [ "${SLURM_CPUS_PER_TASK:-0}" -ne "$CPUS_EXPECTED" ]; then
  echo "lrd_all_serial, one node/task and $CPUS_EXPECTED CPUs required for $STAGE" >&2; exit 2
fi
if [ "${SLURM_MEM_PER_NODE:-0}" -le 0 ] || [ "${SLURM_MEM_PER_NODE:-0}" -gt 30720 ]; then
  echo 'positive node memory not exceeding 30G required' >&2; exit 2
fi
JOB_INFO=$(scontrol show job -o "$SLURM_JOB_ID")
ALLOC=$(printf '%s\n' "$JOB_INFO" | tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2}')
SECS=$(printf '%s\n' "$ALLOC" | awk -F'[-:]' '
  NF==4 {print $1*86400+$2*3600+$3*60+$4; next}
  NF==3 {print $1*3600+$2*60+$3; next}
  NF==2 {print $1*60+$2; next}
  {print 0}')
if [ "$SECS" -le 0 ] || [ "$SECS" -gt "$SECONDS_CAP" ]; then
  echo "walltime $ALLOC exceeds the stage/shard ceiling of $SECONDS_CAP seconds or is invalid" >&2; exit 2
fi
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS="$CPUS_EXPECTED" OPENBLAS_NUM_THREADS="$CPUS_EXPECTED" MKL_NUM_THREADS="$CPUS_EXPECTED"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""

CHECK_ARGS=("$CODE_DIR" "$PLAN_REL" "$PLAN_SHA" "$OUT" "$WORK" "$STAGE")
if [ "$STAGE" != fit-shard ]; then CHECK_ARGS+=("${INPUTS[@]}"); fi
python3 - "${CHECK_ARGS[@]}" <<'PY'
import hashlib
from pathlib import Path
import sys

code = Path(sys.argv[1]).resolve(strict=True)
plan = (code / sys.argv[2]).resolve(strict=True)
out_arg = Path(sys.argv[4])
out = out_arg.resolve()
work = Path(sys.argv[5]).resolve()
if not code.is_relative_to(work / 'wearusfm_campaigns') or not plan.is_relative_to(code) or not plan.is_file():
    raise SystemExit('frozen campaign checkout and contained plan required')
if hashlib.sha256(plan.read_bytes()).hexdigest() != sys.argv[3]:
    raise SystemExit('plan file hash mismatch')
if not out_arg.is_absolute() or out_arg != out or out_arg.exists() or out_arg.is_symlink():
    raise SystemExit('canonical absolute new output required')
if any(out.is_relative_to(root) for root in (code, work / 'wearusfm', work / 'wearusfm_campaigns')):
    raise SystemExit('output inside code')
if sys.argv[6] == 'aggregate':
    for path in (out.with_suffix('.oof.npz'), out.with_suffix('.audit.jsonl')):
        if path.exists() or path.is_symlink():
            raise SystemExit('aggregate sidecar output already exists')
for raw in sys.argv[7:]:
    path = Path(raw)
    if not path.is_absolute() or path != path.resolve(strict=True) or not path.is_file():
        raise SystemExit('canonical absolute completed input reports required')
# logsquare_probe validates five candidate cache pins and six original historical reports,
# source identity, fold roles and complete dataset coverage before publishing.
PY

module list 2>&1
python3 scripts/logsquare_probe.py "$STAGE" --plan "$CODE_DIR/$PLAN_REL" \
  --expected-plan-sha256 "$PLAN_SHA" --expected-code-sha "$EXPECTED_SHA" \
  --out "$OUT" "${EXTRA[@]}"
code_dir_setup
