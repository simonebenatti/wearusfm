#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=00:30:00
#SBATCH -J logsquare_preflight
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/logsquare_preflight_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/logsquare_preflight_%j.out

# Collaudo train-only appaiato T100-L/T100-LQ: 64 batch globali da 32 condivisi, nessuna
# valutazione/checkpoint/ripresa. Il piano congelato fissa configurazioni e input.
# Costo massimo: 4 GPU x 30 min = 2 GPUh, 32 CPU x 30 min = 16 coreh allocati.
# Sottomettere tramite scripts/submit_job.py dopo il checkout di campagna.
# Argomenti: <plan relativo al checkout> <directory output assoluta e nuova>.
# Ambiente obbligatorio: CODE_DIR, EXPECTED_SHA. Nessun EXTRA_ARGS.
# Vedi docs/tokenizer_logsquare_protocol_20261104.md per provenienze e condizioni successive.

set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "uso: logsquare_preflight.sbatch <plan relativo al checkout> <output assoluto nuovo>" >&2
  exit 2
fi
: "${WORK:?WORK}" "${CODE_DIR:?CODE_DIR}" "${EXPECTED_SHA:?EXPECTED_SHA}" "${SLURM_JOB_ID:?SLURM_JOB_ID}"
PLAN_REL="$1"
OUT="$2"
case "$PLAN_REL" in /*|"") echo "il piano deve essere relativo al checkout" >&2; exit 2;; esac
case "$OUT" in /*) ;; *) echo "output relativo: rifiutato" >&2; exit 2;; esac
if [ "${SLURM_JOB_NUM_NODES:-0}" -ne 1 ] || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 32 ]; then
  echo "richiesti un nodo e 32 CPU per task" >&2; exit 2
fi
JOB_INFO=$(scontrol show job -o "$SLURM_JOB_ID")
ALLOC=$(printf '%s\n' "$JOB_INFO" | tr ' ' '\n' | awk -F= '$1=="TimeLimit" {print $2}')
SECS=$(printf '%s\n' "$ALLOC" | awk -F'[-:]' '
  NF==4 {print $1*86400+$2*3600+$3*60+$4; next}
  NF==3 {print $1*3600+$2*60+$3; next}
  NF==2 {print $1*60+$2; next}
  {print 0}')
if [ "$SECS" -le 0 ] || [ "$SECS" -gt 1800 ]; then
  echo "TimeLimit non valido o oltre il tetto di 30 minuti: $ALLOC" >&2; exit 2
fi
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

# Risoluzione canonica anche dei link: piano nel codice congelato, risultati fuori
# dai checkout e assenti. La creazione atomica della directory spetta al rank 0.
python3 - "$CODE_DIR" "$PLAN_REL" "$OUT" "$WORK" <<'PY'
from pathlib import Path
import sys
import torch

code = Path(sys.argv[1]).resolve(strict=True)
plan = (code / sys.argv[2]).resolve(strict=True)
out_arg = Path(sys.argv[3])
out = out_arg.resolve()
work = Path(sys.argv[4]).resolve()
if not code.is_relative_to(work / "wearusfm_campaigns"):
    raise SystemExit("richiesto un checkout di campagna congelato")
if not plan.is_relative_to(code) or not plan.is_file():
    raise SystemExit("piano esterno al checkout: rifiutato")
if out_arg.exists() or out_arg.is_symlink():
    raise SystemExit("output gia esistente: nessuna ripresa o sovrascrittura")
if any(out.is_relative_to(root) for root in (code, work / "wearusfm", work / "wearusfm_campaigns")):
    raise SystemExit("output dentro il codice: rifiutato")
if torch.cuda.device_count() != 4:
    raise SystemExit("richieste esattamente 4 GPU CUDA visibili")
PY

module list 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec torchrun --standalone --nproc_per_node=4 scripts/logsquare_preflight.py \
  --plan "$CODE_DIR/$PLAN_REL" --expected-code-sha "$EXPECTED_SHA" --out "$OUT"
