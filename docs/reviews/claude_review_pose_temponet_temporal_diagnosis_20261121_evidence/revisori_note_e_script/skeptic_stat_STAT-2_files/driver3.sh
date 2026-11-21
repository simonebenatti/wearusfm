#!/bin/zsh
W=/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/wf3_skeptic_stat2_work
cd /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/review_diag || exit 1
export PYTHONPATH=src:scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/Users/simonebenatti/.venvs/wearusfm-tok/bin/python
TOY_DROPOUT=0 $PY $W/toy_nd.py run s8_truth_clean_nodrop S8 clean truth 40 200 > $W/log_s8_truth_clean_nodrop.txt 2>&1 &
A=$!
TOY_DROPOUT=0 $PY $W/toy_nd.py run s8_truth_noisy_nodrop S8 noisy truth 40 200 > $W/log_s8_truth_noisy_nodrop.txt 2>&1 &
B=$!
wait $A $B
echo "nodrop done $(date)"
