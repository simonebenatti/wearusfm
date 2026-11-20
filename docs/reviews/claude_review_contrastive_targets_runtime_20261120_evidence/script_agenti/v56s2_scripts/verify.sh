#!/bin/bash
# usage: verify.sh <worktree> [receipt]
W="$1"; RCPT="${2:-results/infra/contrastive_targets_20261118/amendment_20261119/validation_final/local_validation.json}"
cd "$W" || exit 9
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$W/src:$W/tests/cpu:$W/scripts" \
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python -B scripts/contrastive_local_validation.py --verify "$RCPT" 2>&1 | tail -1
echo "rc=${PIPESTATUS[0]}"
