#!/bin/zsh
# usage: run_one.sh <mutation-id|none> <test files...>
MUT=$1; shift
RT=/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp
CL=$RT/v53b_clone
OUT=$RT/v53b_out; mkdir -p $OUT
BT=$RT/v53b_bt_$MUT
cd $CL
if [ "$MUT" = none ]; then unset V53B_MUT; else export V53B_MUT=$MUT; fi
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=$CL/src:$CL/tests/cpu:$CL/scripts /Users/simonebenatti/.venvs/wearusfm-tok/bin/python -B -m pytest -q -p no:cacheprovider \
  --basetemp=$BT "$@" > $OUT/$MUT.log 2>&1
echo "$MUT rc=$? :: $(tail -1 $OUT/$MUT.log)"
rm -rf $BT
