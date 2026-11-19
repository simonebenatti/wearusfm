#!/bin/zsh
T=/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_targets/tmp/verify-p1n4
C=$T/clone
for M in "$@"; do
  cd $C
  /usr/bin/python3 $T/mutate.py $C $M || exit 3
  git -C $C status --short
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$C/src:$C/tests/cpu:$C/scripts \
    /Users/simonebenatti/.venvs/wearusfm-tok/bin/python -B -m pytest -q -p no:cacheprovider --basetemp=$T/bt-$M -rf \
    tests/cpu/test_contrastive_*.py > $T/$M.out 2>&1
  echo "$M exit=$?"
  grep -E "^FAILED|^ERROR|passed|failed" $T/$M.out | tail -25
  rm -rf $T/bt-$M
  git -C $C checkout -q -- results
  git -C $C status --short | wc -l
done
