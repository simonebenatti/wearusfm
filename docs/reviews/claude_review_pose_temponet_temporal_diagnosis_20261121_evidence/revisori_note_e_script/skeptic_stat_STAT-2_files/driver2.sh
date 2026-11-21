#!/bin/zsh
W=/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/wf3_skeptic_stat2_work
cd /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/review_diag || exit 1
export PYTHONPATH=src:scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/Users/simonebenatti/.venvs/wearusfm-tok/bin/python
$PY $W/toy.py run f_wide_noisy TCN_post noisy random 28 400 > $W/log_f_wide_noisy.txt 2>&1 &
A=$!
$PY $W/toy.py run f_wide_clean TCN_post clean random 28 400 > $W/log_f_wide_clean.txt 2>&1 &
B=$!
$PY $W/toy.py run f_wide_truthfree TCN_post noisy truthfree 23 400 > $W/log_f_wide_truthfree.txt 2>&1 &
C=$!
$PY $W/toy.py run f_s8_truth S8 noisy truth 94 400 > $W/log_f_s8_truth.txt 2>&1 &
D=$!
wait $A $B $C $D
echo "full repro done $(date)"
