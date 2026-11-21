#!/bin/zsh
W=/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/wf3_skeptic_stat2_work
cd /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/review_diag || exit 1
export PYTHONPATH=src:scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/Users/simonebenatti/.venvs/wearusfm-tok/bin/python
# long robustness runs: small head, LR 0.01 (the recipe's grid contains it), noisy and clean target
$PY $W/toy.py run s8_noisy_lr01 S8 noisy random 150 400 0.01 > $W/log_s8_noisy_lr01.txt 2>&1 &
P1=$!
$PY $W/toy.py run s8_clean_lr01 S8 clean random 150 400 0.01 > $W/log_s8_clean_lr01.txt 2>&1 &
P2=$!
# reproduction of the first epochs of the reviewer's six runs
$PY $W/toy.py run r_s8_noisy S8 noisy random 10 10000 > $W/log_r_s8_noisy.txt 2>&1 &
A=$!
$PY $W/toy.py run r_s8_clean S8 clean random 10 10000 > $W/log_r_s8_clean.txt 2>&1 &
B=$!
$PY $W/toy.py run r_s8_truth S8 noisy truth 10 10000 > $W/log_r_s8_truth.txt 2>&1 &
C=$!
wait $A $B $C
$PY $W/toy.py run r_wide_noisy TCN_post noisy random 3 10000 > $W/log_r_wide_noisy.txt 2>&1 &
A=$!
$PY $W/toy.py run r_wide_clean TCN_post clean random 3 10000 > $W/log_r_wide_clean.txt 2>&1 &
B=$!
$PY $W/toy.py run r_wide_truthfree TCN_post noisy truthfree 3 10000 > $W/log_r_wide_truthfree.txt 2>&1 &
C=$!
wait $A $B $C
echo "short runs done $(date)"
wait $P1 $P2
echo "all done $(date)"
