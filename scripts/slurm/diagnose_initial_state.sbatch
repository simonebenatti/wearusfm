#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH -J diagnose_initial_state
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/diagnose_initial_state_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/diagnose_initial_state_%j.out
# Diagnostica dello stato iniziale dei teacher dell'A/B per la baseline casuale (scripts/diagnose_initial_state.py): ricostruzioni in processi
# nuovi con ambienti diversi (thread all'avvio, import, seme), confronto con pairing.json e col teacher finale. Su un nodo Booster, come il
# training dell'A/B e gli extract r2 della baseline; la GPU non si usa (nessun CUDA). Sola lettura dei run; rapporto in una cartella nuova.
# Uso: sbatch --export=ALL,CODE_DIR=<checkout>,EXPECTED_SHA=<sha> scripts/slurm/diagnose_initial_state.sbatch <runs A/B> <rapporto NUOVO .json>
set -euo pipefail
RUNS="${1:?runs A/B}"; OUT="${2:?rapporto nuovo}"
case "$OUT" in /*) ;; *) echo "percorso assoluto richiesto"; exit 2;; esac
source "${CODE_DIR:-$WORK/wearusfm}/scripts/slurm/lib_code_dir.sh" && code_dir_setup || exit 2
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
# il processo principale a un thread; ogni ricostruzione imposta il proprio ambiente all'avvio del suo processo
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
lscpu | head -20
python scripts/diagnose_initial_state.py --runs-root "$RUNS" --seeds 0 1 --out "$OUT"
