#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH -J scaling_identity_precheck
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_identity_precheck_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_identity_precheck_%j.out

# Identity pre-check of the 300M continuation 5000 -> 19618 (v2 gate, 09/11/2026): CPU only, 0 GPU-hours, 4 CPU x 20 min cap
# (1.33 core-hours; 24G as the 4-CPU diagnosis job 63861166 that computed the same sets).
# Recomputes the three identity files of the 5000 run on the worker's own import path (torch first) and writes a receipt the
# GPU job requires; refuses to pass if any bit differs. Submit only through submit_job.py, from a frozen campaign checkout,
# after the review. Partition, CPUs and walltime of the real allocation are checked here and by the wrapper with scontrol.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "uso: scaling_identity_precheck.sbatch <piano relativo al checkout> <directory ricevuta nuova>" >&2
  exit 2
fi
: "${WORK:?WORK}" "${CODE_DIR:?CODE_DIR}" "${EXPECTED_SHA:?EXPECTED_SHA}" "${SLURM_JOB_ID:?SLURM_JOB_ID}"
PLAN_REL="$1"
OUT="$2"
case "$PLAN_REL" in /*|"") echo "richiesto piano relativo al checkout" >&2; exit 2;; esac
if [ "${SLURM_JOB_NUM_NODES:-}" != 1 ] || [ "${SLURM_NTASKS:-}" != 1 ] || \
   [ "${SLURM_CPUS_PER_TASK:-}" != 4 ] || [ "${SLURM_MEM_PER_NODE:-}" != 24576 ] || \
   [ "${SLURM_JOB_PARTITION:-}" != lrd_all_serial ]; then
  echo "richiesti lrd_all_serial, 1 nodo, 1 task, 4 CPU e 24G" >&2
  exit 2
fi
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

python3 - "$CODE_DIR" "$PLAN_REL" "$OUT" "$WORK" <<'PY'
import importlib.util
import json
from pathlib import Path
import sys

code = Path(sys.argv[1]).resolve(strict=True)
plan_path = (code / sys.argv[2]).resolve(strict=True)
out_arg, work = Path(sys.argv[3]), Path(sys.argv[4]).resolve()
if not code.is_relative_to(work / "wearusfm_campaigns"):
    raise SystemExit("richiesto checkout di campagna congelato")
if not plan_path.is_relative_to(code) or not plan_path.is_file():
    raise SystemExit("piano esterno al checkout")
if not out_arg.is_absolute() or out_arg.exists() or out_arg.is_symlink() or out_arg.resolve() != out_arg:
    raise SystemExit("directory ricevuta assoluta, canonica e nuova richiesta")
if any(out_arg.is_relative_to(root) for root in (code, work / "wearusfm", work / "wearusfm_campaigns")):
    raise SystemExit("ricevuta dentro il codice")
plan = json.loads(plan_path.read_text())
if plan.get("protocol") != "scaling-continuation-training-20261106-v1" or plan.get("arm") != "T100-G-300M" or \
        type(plan.get("start_step")) is not int or type(plan.get("end_step")) is not int or \
        (plan["start_step"], plan["end_step"]) != (5000, 19618):
    raise SystemExit("consentito solo il pre-controllo della continuazione 300M 5000 -> 19618")
precheck = plan.get("identity_precheck")
if not isinstance(precheck, dict) or precheck.get("protocol") != "scaling-continuation-identity-precheck-20261109-v1" or \
        precheck.get("allocation") != {"partition": "lrd_all_serial", "cpus": 4, "memory": "24G", "walltime_seconds": 1200}:
    raise SystemExit("pre-controllo richiesto: lrd_all_serial, 4 CPU, 24G, 1200 s")
if precheck.get("receipt_dir") != str(out_arg):
    raise SystemExit("directory della ricevuta diversa dal piano")
spec = importlib.util.spec_from_file_location("scaling_continue_precheck_launcher", code / "scripts/scaling_continue.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)
wrapper.verify_precheck_allocation(plan)
PY

module list 2>&1
# The frozen wrapper verifies sources and inputs, recomputes the identity sets with torch imported first and refuses
# any difference; the receipt identity_precheck.json is required by the training job of the same plan and code.
exec python3 scripts/scaling_continue.py identity-precheck \
  --plan "$CODE_DIR/$PLAN_REL" --expected-code-sha "$EXPECTED_SHA" --out "$OUT"
