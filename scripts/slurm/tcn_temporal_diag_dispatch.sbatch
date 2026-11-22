#!/bin/bash
# The stdin wrapper only verifies and dispatches the frozen diagnostic launcher.
set -euo pipefail
: "${WORK:?}" "${CODE_DIR:?}" "${EXPECTED_SHA:?}"
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || exit 2
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
exec bash "$CODE_DIR/scripts/slurm/tcn_temporal_diag.sbatch"
