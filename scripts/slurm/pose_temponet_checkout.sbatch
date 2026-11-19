#!/bin/bash
# One allocated preflight includes the frozen checkout and historical parity.
set -euo pipefail
: "${WORK:?}" "${CAMPAIGN:?}" "${CAMPAIGN_SHA:?}" "${CODE_DIR:?}" "${EXPECTED_SHA:?}"
[ "$CAMPAIGN_SHA" = "$EXPECTED_SHA" ] || { echo 'Checkout SHA mismatch' >&2; exit 2; }
[[ "$CAMPAIGN_SHA" =~ ^[0-9a-f]{40}$ ]] || exit 2
[ "$CODE_DIR" = "$WORK/wearusfm_campaigns/$CAMPAIGN" ] || exit 2
[ "${SLURM_RESTART_COUNT:-0}" = 0 ] || exit 2
TCN_CHECKOUT_SCRIPT=$(mktemp "${TMPDIR:-/tmp}/pose-temponet-checkout.XXXXXX")
trap 'rm -f "$TCN_CHECKOUT_SCRIPT"' EXIT
git -C "$WORK/wearusfm" show "$CAMPAIGN_SHA:scripts/slurm/campaign_checkout.sbatch" > "$TCN_CHECKOUT_SCRIPT"
bash "$TCN_CHECKOUT_SCRIPT"
bash "$CODE_DIR/scripts/slurm/pose_temponet.sbatch"
