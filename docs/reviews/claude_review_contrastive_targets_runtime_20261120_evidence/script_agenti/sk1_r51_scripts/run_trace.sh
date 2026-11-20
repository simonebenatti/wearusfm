#!/bin/bash
# Skeptic R5-1: full receipt set on the instrumented clone, sentinel log in trace_full.log
set -u
T=/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp
C=$T/sk1_r51_a
PY=/Users/simonebenatti/.venvs/wearusfm-tok/bin/python
cd "$C" || exit 2
FILES=()
while IFS= read -r line; do FILES+=("$line"); done < <("$PY" -c "import json;print('\n'.join(x['path'] for x in json.load(open('results/infra/contrastive_targets_20261118/amendment_20261119/validation_final/local_validation.json'))['test_files']))")
echo "files=${#FILES[@]}"
rm -f "$T/sk1_r51_scripts/trace_full.log"
START=$(date +%s)
SK1_TRACE="$T/sk1_r51_scripts/trace_full.log" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$C/src:$C/tests/cpu:$C/scripts" "$PY" -B -m pytest -q -p no:cacheprovider -o tmp_path_retention_policy=none -rfE \
  --basetemp="$T/sk1_r51_bt_trace" "${FILES[@]}" > "$T/sk1_r51_scripts/trace_full.out" 2>&1
echo "rc=$? secs=$(( $(date +%s) - START ))"
tail -5 "$T/sk1_r51_scripts/trace_full.out"
rm -rf "$T/sk1_r51_bt_trace"
