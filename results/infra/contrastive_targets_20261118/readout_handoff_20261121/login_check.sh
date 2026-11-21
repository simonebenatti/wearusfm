#!/bin/bash
# Validate the prospective checkout or the seven-path CPU readout on the login node.
set -euo pipefail
phase=${1:?expected checkout or readout}
sha=${2:?full execution SHA}
[[ "$sha" =~ ^[0-9a-f]{40}$ ]]
campaign=contrastive-targets-readout-20261121
repo="${WORK:?}/wearusfm"
dest="$WORK/wearusfm_campaigns/$campaign"
pkg=results/infra/contrastive_targets_20261118
handoff="$pkg/readout_handoff_20261121"
binding="$pkg/releases/audit_20261120/review_binding.json"
audit="$pkg/audit_execution_20261120"
cal="$pkg/calibration_execution_20261121"
cuda="$pkg/cuda_execution_20261121"
full="$pkg/benchmark_execution_20261121"
reload="$pkg/reloads_execution_20261121"
run_root="$WORK/wearusfm_runs/results/contrastive_targets_20261118"
[[ "$(git -C "$repo" rev-parse "refs/heads/deploy/$campaign^{commit}")" == "$sha" ]]
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT

if [[ "$phase" == checkout && $# == 2 ]]; then
    [[ ! -e "$dest" && ! -L "$dest" ]]
    mkdir "$tmp/tree"
    git -C "$repo" archive "$sha" -- scripts src/wearusfm "$pkg/plan.json" "$pkg/preflight_plan.json" \
        "$handoff" "$pkg/review_configs/calibration.config.json" "$pkg/releases" \
        "$audit/audit_accounting.json" "$audit/audit.sacct.tsv" "$cal/allocation.sacct.tsv" "$cal/remote_sha256.txt" \
        "$cuda/allocation.sacct.tsv" "$cuda/remote_sha256.txt" \
        "$full/allocation.sacct.tsv" "$full/remote_sha256.txt" "$full/checkpoints_remote.txt" \
        "$cal/checkpoint_remote_receipt.txt" "$reload/allocation.sacct.tsv" "$reload/collection_readout.json" \
        "$reload/CN_A_remote_integrity_and_log.txt" "$reload/CN_B_remote_integrity_and_log.txt" \
        "$reload/CR_A_remote_integrity_and_log.txt" "$reload/CR_B_remote_integrity_and_log.txt" \
        results/infra/tokenizer_execution_20261103/19618/T100-G/config.json \
        results/infra/contrastive_comparison_20261116/review_response/coverage_manifest_reference.json \
        results/infra/contrastive_comparison_20261116/review_response/gradient_reference.json | tar -x -C "$tmp/tree"
    export GIT_DIR="$(git -C "$repo" rev-parse --absolute-git-dir)"
    cd "$tmp/tree"
elif [[ "$phase" == readout && $# == 3 && "$3" =~ ^[0-9]+$ ]]; then
    row=$(sacct -X -n -P -j "$3" --format=JobIDRaw,State,ExitCode,JobName%80,AllocCPUS,TimelimitRaw)
    printf '%s\n' "$row"
    [[ "$row" == "$3|COMPLETED|0:0|CONTRAST-CHECKOUT-readout-targets-v3-20261118|4|4" ]]
    export CODE_DIR="$dest" EXPECTED_SHA="$sha"
    source "$dest/scripts/slurm/lib_code_dir.sh"
    code_dir_setup
else
    echo 'Expected checkout, or readout and completed checkout ID' >&2
    exit 2
fi

# Reconcile immutable prerequisite partitions of the ledger with live SLURM history.
fields=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,TimelimitRaw
check_accounting() {
    local ids=$1 receipt=$2
    sacct -X -P -j "$ids" --format="$fields" | LC_ALL=C sort > "$tmp/current.tsv"
    cut -d '|' -f1-6 "$receipt" | LC_ALL=C sort > "$tmp/expected.tsv"
    cmp "$tmp/current.tsv" "$tmp/expected.tsv"
    cat "$tmp/current.tsv"
}
check_accounting 64937087 "$audit/audit.sacct.tsv"
check_accounting 65014846,65015116 "$cal/allocation.sacct.tsv"
check_accounting 65016695,65016794 "$cuda/allocation.sacct.tsv"
check_accounting 65017186,65017240,65017310,65020151 "$full/allocation.sacct.tsv"
check_accounting 65023425,65033016,65033756,65033804,65034107 "$reload/allocation.sacct.tsv"
row=$(sacct -X -n -P -j 64936161 --format=JobIDRaw,State,ExitCode,AllocCPUS,TimelimitRaw)
[[ "$row" == '64936161|COMPLETED|0:0|4|4' ]]
[[ "$(env -u GIT_DIR git -C "$WORK/wearusfm_campaigns/contrastive-targets-audit-20261119" rev-parse HEAD)" == 9abde2951cd9b2a25aa699faceb93132de6732f8 ]]
[[ "$(env -u GIT_DIR git -C "$WORK/wearusfm_campaigns/contrastive-targets-calibration-20261120" rev-parse HEAD)" == e05c05826daa9a6feda47f1f75742ea329c84f5f ]]
[[ "$(env -u GIT_DIR git -C "$WORK/wearusfm_campaigns/contrastive-targets-cuda-20261121" rev-parse HEAD)" == 4e880b9f6787d7294712370d00ed485bfbe45735 ]]
[[ "$(env -u GIT_DIR git -C "$WORK/wearusfm_campaigns/contrastive-targets-full-20261121" rev-parse HEAD)" == 746112dd711903e5c92ade2efd90834264b2676a ]]
[[ "$(env -u GIT_DIR git -C "$WORK/wearusfm_campaigns/contrastive-targets-reloads-20261121" rev-parse HEAD)" == 12e215a978b9d30ea6d670f58af1a0f97387d477 ]]
for dir in "$cal" "$cuda"; do
    checksum_list="$PWD/$dir/remote_sha256.txt"
    (cd "$run_root" && sha256sum --check --strict "$checksum_list")
done

# Summary, metric files, trace and per-rank diagnostics live relative to the benchmark folder.
full_checksums="$PWD/$full/remote_sha256.txt"
(cd "$run_root/benchmark" && sha256sum --check --strict "$full_checksums")

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/contrastive_release.py check-binding --binding "$binding"
python3 scripts/contrastive_release.py check-audit --audit "$run_root/audit.json" --accounting "$audit/audit_accounting.json"
python3 "$handoff/validate_readout.py"
