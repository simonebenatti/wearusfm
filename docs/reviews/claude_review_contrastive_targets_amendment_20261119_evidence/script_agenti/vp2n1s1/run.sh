#!/bin/zsh
# Usage: run.sh <label> <mutation|NONE> [test files...]  (default: every contrastive test file)
set -u
S=/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_targets
W=$S/tmp/vp2n1s1
M=$W/mut
PY=/Users/simonebenatti/.venvs/wearusfm-tok/bin/python
LABEL=$1; MUT=$2; shift 2
OUT=$W/out_$LABEL; mkdir -p $OUT
git -C $M checkout -q -- . && git -C $M clean -fdq
rm -f $OUT/mark.txt
if [ "$MUT" != "NONE" ]; then $PY -B $W/mut.py $M $MUT $OUT/mark.txt > $OUT/mutate.txt 2>&1 || { cat $OUT/mutate.txt; exit 1; }; fi
git -C $M diff > $OUT/diff.txt
cd $M
if [ $# -eq 0 ]; then set -- tests/cpu/test_contrastive_*.py tests/cpu/test_rank_contrast.py; fi
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$M/src:$M/tests/cpu:$M/scripts \
  $PY -B -m pytest -q -p no:cacheprovider -rf --basetemp=$W/bt_$LABEL "$@" > $OUT/pytest.txt 2>&1
echo "pytest_rc=$?" >> $OUT/pytest.txt
rm -rf $W/bt_$LABEL
git -C $M checkout -q -- . && git -C $M clean -fdq
grep -E "^FAILED|^ERROR| passed| failed|pytest_rc" $OUT/pytest.txt | sed 's/ - .*//' | head -40
