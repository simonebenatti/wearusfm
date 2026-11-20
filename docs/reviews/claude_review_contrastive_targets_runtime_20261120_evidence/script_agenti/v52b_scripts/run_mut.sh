#!/bin/bash
# usage: run_mut.sh <clone> <mutation|NONE> <label> <test files...>
set -u
WT="$1"; MUT="$2"; LABEL="$3"; shift 3
T=/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp
OUT=$T/v52b_scripts/results.tsv
PY=/Users/simonebenatti/.venvs/wearusfm-tok/bin/python
cd "$WT" || exit 2
git checkout -q -- . && git clean -fdq
if [ "$MUT" != "NONE" ]; then $PY -B $T/v52b_scripts/mut.py . "$MUT" >/dev/null || { echo -e "$MUT\t$LABEL\tAPPLY_FAILED" >> $OUT; exit 3; }; fi
BT=$T/v52b_bt_${LABEL}_$$
start=$(date +%s)
LOG=$T/v52b_scripts/log_${LABEL}.txt
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$WT/src:$WT/tests/cpu:$WT/scripts" \
  $PY -B -m pytest -q -p no:cacheprovider -o tmp_path_retention_policy=none --basetemp="$BT" -x --tb=line "$@" > "$LOG" 2>&1
rc=$?
end=$(date +%s)
summary=$(grep -E "passed|failed|error" "$LOG" | tail -1)
echo -e "$MUT\t$LABEL\trc=$rc\t$((end-start))s\t$summary" >> $OUT
rm -rf "$BT"
git checkout -q -- . && git clean -fdq
