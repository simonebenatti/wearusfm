#!/bin/bash
# Operational census only; outside the reviewed source manifest. No signal data.
set -euo pipefail
: "${WORK:?}" "${CODE_DIR:?}" "${EXPECTED_SHA:?}" "${SLURM_CPUS_PER_TASK:?}"
[ "$SLURM_CPUS_PER_TASK" = 8 ]
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONPATH="$CODE_DIR/src:$CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 - <<'TCN_PY'
import json, os
# Same imports as the production runner before its numerical census.
from pose_temponet import runner, train, provenance as P
train.configure_numerics()
context = P.numeric_context("gpu")
expected = "12cb2a0b3207b66e260d3ca490db93ee0c029bb8f03d766df89a9edba6b7eb3f"
observed = P.digest(context)
node = os.environ.get("SLURMD_NODENAME", os.environ.get("SLURM_JOB_NODELIST", ""))
result = {"schema": "TCNH-second-node-numeric-inventory-v1",
          "slurm_job_id": os.environ["SLURM_JOB_ID"], "node": node,
          "code_dir": os.environ["CODE_DIR"], "expected_code_sha": os.environ["EXPECTED_SHA"],
          "source_sha256": P.source_identity(), "numeric_context": context,
          "numeric_sha256": observed, "expected_numeric_sha256": expected,
          "matches": observed == expected, "different_node": node != "lrdn0059" and bool(node),
          "signal_inputs_read": False}
print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
P.require(result["source_sha256"] == "9aaf1c8b6a050a524dfb60af65df9406313d4fd0e690dab1097bf11df2ec67ea", "inventory source mismatch")
P.require(result["different_node"], "inventory must use a second node")
P.require(result["matches"], "second-node complete numerical context differs from gpu8; stop")
TCN_PY
