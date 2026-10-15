#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH -J step1_random_probe_group
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_random_probe_group_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/step1_random_probe_group_%j.out
# Same four probes as step1_random_downstream_4000_v1, grouped by task to
# respect lrd_all_serial's ten-submission limit alongside eight 2A jobs.
# Args: full SHA, p1/p2, historical training root, NEW eval root, learned root.
# P1 depends on both P1 extracts. P2 depends on both P2 extracts AND P1 group;
# the comparison runs after its seed1 probe. Each seed retains a 2h cap.
set -euo pipefail
SHA="${1:?commit}"; TASK="${2:?p1/p2}"; RUNS="${3:?runs}"; EVAL="${4:?eval}"; DOWN="${5:?learned root}"
: "${CODE_DIR:?frozen CODE_DIR required}"
case "$TASK" in p1|p2) ;; *) echo "task must be p1 or p2"; exit 2;; esac
command -v timeout >/dev/null || { echo "GNU timeout required for the per-seed 2h cap"; exit 2; }
for SEED in 0 1; do
  EXTRA=()
  if [ "$TASK" = p2 ] && [ "$SEED" = 1 ]; then EXTRA=("$DOWN"); fi
  timeout 7200s bash "$CODE_DIR/scripts/slurm/step1_random_probe.sbatch" \
    "$SHA" "$TASK" "$SEED" "$RUNS" "$EVAL" "${EXTRA[@]}"
done
