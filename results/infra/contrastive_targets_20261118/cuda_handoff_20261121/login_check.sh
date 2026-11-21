#!/bin/bash
# Run the frozen checks on the login node. The rank-exit probe and the GPU worker belong to the allocation.
set -euo pipefail
phase=${1:?checkout or cuda}
sha=${2:?full execution SHA}
[[ "$sha" =~ ^[0-9a-f]{40}$ ]]
campaign=contrastive-targets-cuda-20261121
repo="${WORK:?}/wearusfm"
dest="$WORK/wearusfm_campaigns/$campaign"
pkg=results/infra/contrastive_targets_20261118
release="$pkg/releases/cuda_20261121"
binding="$pkg/releases/audit_20261120/review_binding.json"
audit_account="$pkg/audit_execution_20261120"
cal_account="$pkg/calibration_execution_20261121"
run_root="$WORK/wearusfm_runs/results/contrastive_targets_20261118"
audit="$run_root/audit.json"
out="$run_root/cuda.json"
logical=CONTRAST-CUDA-targets-v3-20261118
[[ "$(git -C "$repo" rev-parse "refs/heads/deploy/$campaign^{commit}")" == "$sha" ]]
for path in "$out" "${out%.json}" "$out.rank_exit_probe"; do
    if [[ -e "$path" || -L "$path" ]]; then
        echo "CUDA output already exists: $path" >&2
        exit 2
    fi
done
[[ -f "$audit" && -f "$run_root/calibration/checkpoint.pt" ]]
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
audit_checkout=$(sacct -X -n -P -j 64936161 --format=JobIDRaw,State,ExitCode,AllocCPUS,TimelimitRaw)
[[ "$audit_checkout" == '64936161|COMPLETED|0:0|4|4' ]]
[[ "$(git -C "$WORK/wearusfm_campaigns/contrastive-targets-audit-20261119" rev-parse HEAD)" == 9abde2951cd9b2a25aa699faceb93132de6732f8 ]]
[[ "$(git -C "$WORK/wearusfm_campaigns/contrastive-targets-calibration-20261120" rev-parse HEAD)" == e05c05826daa9a6feda47f1f75742ea329c84f5f ]]
fields=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,TimelimitRaw
sacct -X -P -j 64937087 --format="$fields" > "$tmp/audit.sacct"
sacct -X -P -j 65014846,65015116 --format="$fields" > "$tmp/calibration.sacct"

if [[ "$phase" == checkout && $# == 2 ]]; then
    [[ ! -e "$dest" && ! -L "$dest" ]]
    mkdir "$tmp/tree"
    git -C "$repo" archive "$sha" -- scripts src/wearusfm "$pkg/plan.json" "$pkg/preflight_plan.json" \
        "$pkg/review_configs/calibration.config.json" "$pkg/releases" "$audit_account/audit_accounting.json" \
        "$audit_account/audit.sacct.tsv" "$cal_account/allocation.sacct.tsv" "$cal_account/remote_sha256.txt" \
        results/infra/tokenizer_execution_20261103/19618/T100-G/config.json \
        results/infra/contrastive_comparison_20261116/review_response/coverage_manifest_reference.json \
        results/infra/contrastive_comparison_20261116/review_response/gradient_reference.json | tar -x -C "$tmp/tree"
    export GIT_DIR="$(git -C "$repo" rev-parse --absolute-git-dir)"
    cd "$tmp/tree"
elif [[ "$phase" == cuda && $# == 3 && "$3" =~ ^[0-9]+$ ]]; then
    row=$(sacct -X -n -P -j "$3" --format=JobIDRaw,State,ExitCode,JobName%80,AllocCPUS,TimelimitRaw)
    printf '%s\n' "$row"
    [[ "$row" == "$3|COMPLETED|0:0|CONTRAST-CHECKOUT-cuda-targets-v3-20261118|4|4" ]]
    export CODE_DIR="$dest" EXPECTED_SHA="$sha"
    source "$dest/scripts/slurm/lib_code_dir.sh"
    code_dir_setup
else
    echo 'Invalid phase or missing completed checkout ID' >&2
    exit 2
fi

cmp "$tmp/audit.sacct" "$audit_account/audit.sacct.tsv"
# The accounting snapshot includes extended TRES columns. Match every stable field from both live allocations.
cut -d '|' -f1-6 "$cal_account/allocation.sacct.tsv" | LC_ALL=C sort > "$tmp/calibration.expected"
LC_ALL=C sort "$tmp/calibration.sacct" > "$tmp/calibration.current"
cmp "$tmp/calibration.expected" "$tmp/calibration.current"
cat "$tmp/audit.sacct" "$tmp/calibration.sacct"
sha_list="$PWD/$cal_account/remote_sha256.txt"
(cd "$run_root" && sha256sum --check --strict "$sha_list")
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/contrastive_release.py check-binding --binding "$binding"
python3 scripts/contrastive_release.py check-audit --audit "$audit" --accounting "$audit_account/audit_accounting.json"
python3 scripts/contrastive_release.py make-job --binding "$binding" --cell-id "$logical" \
    --evidence "$release/cuda_evidence.json" --result-out "$out" --out "$tmp/job.json"
cmp "$release/cuda.job.json" "$tmp/job.json"
python3 - "$release/cuda.job.json" "$logical" "$out" "$cal_account/allocation.sacct.tsv" <<'PY'
import json
import sys
from pathlib import Path

import torch
from contrastive_cuda_check import technical_config
from contrastive_job import build_command, rank_exit_report
from contrastive_prepare import _read, authenticate_calibration
from contrastive_release import checked_pin
from contrastive_run import validate_recipe
from wearusfm.training.pairing import state_sha256
from wearusfm.training.run import check_input_files

job = _read(sys.argv[1])
out = Path(sys.argv[3]).resolve()
run_dir = out.parent / "calibration"
command = build_command(job, "gpu", 1800)
report_path = checked_pin(job["calibration_report"])
summary_path = checked_pin(job["calibration_summary"])
if (report_path != out.parent / "calibration_report.json" or summary_path != run_dir / "summary.json"
        or checked_pin(job["calibration_accounting"]) != Path(sys.argv[4]).resolve()):
    raise ValueError("CUDA is bound to a different calibration")
measured = authenticate_calibration(_read(report_path), run_dir)
if (measured["run_code"]["sha"] != "e05c05826daa9a6feda47f1f75742ea329c84f5f"
        or measured["source_sha256"] != job["source_sha256"]
        or measured["input_sha256"] != {name: pin["sha256"] for name, pin in job["inputs"].items()}):
    raise ValueError("Prior calibration commit, source, or data pins differ")

# build_command checked the checkpoint bytes; repeat the worker's content checks on CPU, before any submission.
checkpoint_path = Path(job["calibration_checkpoint"]["path"]).resolve()
if checkpoint_path != run_dir / "checkpoint.pt":
    raise ValueError("CUDA requires the original calibration checkpoint")
checkpoint = torch.load(str(checkpoint_path), mmap=True, map_location="cpu", weights_only=False)
state_hash = measured["initial_state_sha256"]
if (checkpoint.get("step") != 512 or checkpoint["optimizer"]["state"]
        or checkpoint.get("contrastive_initial_sha256") != state_hash
        or state_sha256(checkpoint["student"]) != state_hash or state_sha256(checkpoint["teacher"]) != state_hash):
    raise ValueError("Checkpoint is not the authenticated calibration initial state")
del checkpoint

config_path = checked_pin(job["config"])
cfg = validate_recipe(_read(config_path), "calibration")
if (job["id"] != sys.argv[2] or job["out"] != str(out) or job["arm"] is not None or job["mode"] != "cuda"
        or job.get("repeat") is not None or "release" in job
        or cfg.contrastive["arm"] != "C0" or cfg.contrastive["calibration_only"] is not True
        or cfg.loader.rms_window_ms != 100 or cfg.loader.ms_fast_window_ms != 125):
    raise ValueError("Job, recipe or arm outside the technical CUDA stage")
arms = []
for name in ("CN", "CR"):
    tech = technical_config(cfg, name)
    if (tech.contrastive["arm"] != name or tech.contrastive["calibration_only"]
            or tech.max_steps != 2 or tech.batch_size != 8 or tech.datasets != ("emg2pose",)
            or tech.contrastive["nce_weight"] != 1 or tech.contrastive["rnc_weight"] != 1):
        raise ValueError("Technical override differs for " + name)
    arms.append({"arm": name, "steps": tech.max_steps, "global_batch": tech.batch_size,
                 "nce_weight": tech.contrastive["nce_weight"], "rnc_weight": tech.contrastive["rnc_weight"]})
input_pins = job["inputs"]
check_input_files(Path(input_pins["manifest"]["path"]), Path(input_pins["window_seconds"]["path"]),
                  cfg.nested, input_pins["manifest"]["sha256"])
prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
          str(Path("scripts/contrastive_cuda_check.py").resolve())]
if command[:len(prefix)] != prefix:
    raise ValueError("Unexpected CUDA worker")
argv = command[len(prefix):]
if len(argv) % 2:
    raise ValueError("Invalid CUDA worker argv")
flags = {}
for flag, value in zip(argv[::2], argv[1::2]):
    flags.setdefault(flag, []).append(value)
expected = {"--config": [str(config_path)], "--config-sha256": [job["config"]["sha256"]],
            "--expected-source-sha256": [job["source_sha256"]],
            "--calibration-checkpoint": [str(checkpoint_path)],
            "--calibration-checkpoint-sha256": [job["calibration_checkpoint"]["sha256"]],
            "--rank-exit-report": [str(rank_exit_report(out))], "--out": [str(out)],
            "--pose-stats-sha256": [input_pins["pose_stats"]["sha256"]],
            "--root": [str(Path(root).resolve(strict=True)) for root in job["roots"]]}
for key in ("manifest", "scales", "window_seconds"):
    flag = "--" + key.replace("_", "-")
    expected[flag] = [str(Path(input_pins[key]["path"]).resolve(strict=True))]
    expected[flag + "-sha256"] = [input_pins[key]["sha256"]]
if flags != expected:
    raise ValueError("Generated CUDA arguments differ from the published cell")
print(json.dumps({"job_id": job["id"], "job_arm": job["arm"], "stage": "cuda", "config_arm": cfg.contrastive["arm"],
                  "technical_arms": arms, "calibration_weights_authenticated": measured["weights"],
                  "checkpoint_state_sha256": state_hash, "execution_argv": command, "execution_started": False},
                 indent=2))
PY
