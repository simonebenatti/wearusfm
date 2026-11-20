#!/bin/bash
# Skeptic R5-1: run the throwaway negative tests on the original code and under S01-S04
T=/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp
C=$T/sk1_r51_c
PY=/Users/simonebenatti/.venvs/wearusfm-tok/bin/python
cd "$C" || exit 2
for MID in NONE S01_status S02_one_block S03_cohort_plans S04_declared_plan_hashes; do
  if [ "$MID" != NONE ]; then
    "$PY" -B -c "
import sys
ns = {}
exec(open('$T/sk1_r51_scripts/mutants.py').read().split('def apply')[0], ns)
from pathlib import Path
p = Path('scripts/contrastive_release.py'); text = p.read_text()
for old, new in ns['M']['$MID']:
    assert text.count(old) == 1
    text = text.replace(old, new)
p.write_text(text)
" || exit 3
  fi
  echo "== $MID $(git diff --numstat | tr '\t' ' ')"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$C/src:$C/tests/cpu:$C/scripts" "$PY" -B -m pytest -q -p no:cacheprovider -o tmp_path_retention_policy=none \
    -rf --basetemp="$T/sk1_r51_bt_neg_$MID" "$T/sk1_r51_scripts/test_sk1_negative.py" 2>&1 | grep -E "passed|failed|^FAILED" | cut -c1-220
  git checkout -- scripts/contrastive_release.py
  rm -rf "$T/sk1_r51_bt_neg_$MID"
  echo "clean=$(git status --porcelain | wc -l | tr -d ' ')"
done
