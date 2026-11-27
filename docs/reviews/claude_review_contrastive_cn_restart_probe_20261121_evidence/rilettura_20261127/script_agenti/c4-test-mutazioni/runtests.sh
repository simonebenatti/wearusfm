#!/bin/bash
# usage: runtests.sh <outfile> <basetemp> [extra pythonpath prefix] -- files...
cd /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_probe2/clone
out=$1; bt=$2; pre=$3; shift 3
rm -rf "$bt"
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=${pre}src:scripts:scripts/diagnostics \
 /Users/simonebenatti/.venvs/wearusfm-tok/bin/python -B -m pytest -q -p no:cacheprovider -rA $XFLAG --basetemp="$bt" "$@" > "$out" 2>&1
echo "exit=$?" >> "$out"
