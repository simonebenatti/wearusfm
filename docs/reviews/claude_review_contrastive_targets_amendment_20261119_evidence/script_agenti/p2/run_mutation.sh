#!/bin/zsh
# Usage: run_mutation.sh <label> <comma-separated mutations or NONE>
set -u
S=/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_targets
M=$S/tmp/p2/mut
PY=/Users/simonebenatti/.venvs/wearusfm-tok/bin/python
LABEL=$1
MUTS=$2
OUTD=$S/tmp/p2/results/$LABEL
mkdir -p $OUTD
git -C $M checkout -q -- . && git -C $M clean -fdq
git -C $M status --short > $OUTD/status_before.txt
if [ "$MUTS" != "NONE" ]; then
  PYTHONDONTWRITEBYTECODE=1 $PY -B $S/tmp/p2/mutate.py $M $MUTS > $OUTD/mutate.txt 2>&1 || { echo "mutation failed"; cat $OUTD/mutate.txt; exit 1; }
fi
git -C $M diff --stat > $OUTD/diffstat.txt
cd $M
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$M/src:$M/tests/cpu:$M/scripts \
  $PY -B -m pytest -q -p no:cacheprovider -rf --basetemp=$S/tmp/p2/bt_$LABEL \
  tests/cpu/test_contrastive_historical_calibration.py tests/cpu/test_contrastive_historical_callsites.py \
  tests/cpu/test_contrastive_prepare.py tests/cpu/test_contrastive_run.py tests/cpu/test_contrastive_job.py \
  tests/cpu/test_contrastive_source_compat.py tests/cpu/test_contrastive_benchmark_readout.py tests/cpu/test_contrastive_audit.py \
  > $OUTD/pytest.txt 2>&1
echo "pytest_rc=$?" >> $OUTD/pytest.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$M/src:$M/scripts \
  $PY -B $S/tmp/p2/probe_mut.py $M $S/tmp/p2/work_$LABEL > $OUTD/probe.json 2> $OUTD/probe.err
rm -rf $S/tmp/p2/bt_$LABEL $S/tmp/p2/work_$LABEL
cd $S
grep -E "^FAILED|passed|failed|pytest_rc" $OUTD/pytest.txt | sed 's/ - .*//' | head -40
cat $OUTD/probe.json
