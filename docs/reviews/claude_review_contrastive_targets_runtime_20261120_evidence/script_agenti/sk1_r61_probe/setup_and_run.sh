#!/bin/bash
# Recreate the temp clone, commit the synthetic review only there, run the probe, clean up.
set -euo pipefail
CL=/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_targets/clone
T=/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp
P=$T/sk1_r61_probe
C=$T/sk1_r61_clone_rerun
BT=$T/sk1_r61_bt_rerun
git clone -q --shared "$CL" "$C"
git -C "$C" -c advice.detachedHead=false checkout -q --detach 0efa4d45
mkdir -p "$C/docs/reviews"; cp "$P/sk1_probe_review.md" "$C/docs/reviews/sk1_probe_review.md"
git -C "$C" add docs/reviews/sk1_probe_review.md
git -C "$C" -c user.email=skeptic@example.invalid -c user.name=skeptic commit -q -m "probe review (temp clone only)"
cd "$C"
set +e
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$C/src:$C/tests/cpu:$C/scripts" /Users/simonebenatti/.venvs/wearusfm-tok/bin/python -B -m pytest -q -s \
  -p no:cacheprovider --rootdir="$P" --basetemp="$BT" "$P/test_sk1_r61_arm.py"
rc=$?
set -e
cd /
rm -rf /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp/sk1_r61_clone_rerun /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp/sk1_r61_bt_rerun
exit $rc
