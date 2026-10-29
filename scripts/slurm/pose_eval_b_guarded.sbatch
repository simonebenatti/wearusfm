#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -J pose_eval_b_guarded
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/pose_eval_b_guarded_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/pose_eval_b_guarded_%j.out

# Operational wrapper sent with submit_job.py --script-stdin: no new deployment.
# Resource caps MUST be explicit on sbatch: extract 1 GPU/8 CPU/60G/40min;
# report 0 GPU/8 CPU/30G/20min. Scientific programs remain at frozen 24139ab.
# Usage: this_script extract CHECKPOINT VAL.npz | report VAL.npz REPORT.json
# Required environment: CODE_DIR, EXPECTED_SHA, WRAPPER_SHA256, WORK, SCRATCH.
# WRAPPER_SHA256 is the submitter's hash of this versioned source, not of a
# Slurm-spooled copy. It is provenance supplied at submission, not self-proof.
set -euo pipefail
if [ "$#" -eq 0 ]; then
  set -- "${B_EVAL_MODE:?}" "${B_EVAL_INPUT:?}" "${B_EVAL_OUTPUT:?}"
fi
[ "$#" -eq 3 ] || { echo 'expected MODE INPUT OUTPUT' >&2; exit 2; }
MODE="$1"; INPUT="$2"; OUTPUT="$3"
case "$MODE" in extract|report) ;; *) echo 'MODE must be extract or report' >&2; exit 2;; esac
: "${CODE_DIR:?}" "${EXPECTED_SHA:?}" "${WRAPPER_SHA256:?}" "${WORK:?}" "${SCRATCH:?}"
[ "$EXPECTED_SHA" = 24139ab901c052357a46a65bc20466f2eb1e8358 ] || { echo 'wrong frozen checkout' >&2; exit 2; }
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
python3 - "$MODE" "$INPUT" "$OUTPUT" <<'PY_B_GUARD'
import csv
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from wearusfm.data import pose_targets as PT
from wearusfm.harness import pose_eval as PE
from wearusfm.ingest.emg2pose import _SAFE_RE

CODE_SHA = "24139ab901c052357a46a65bc20466f2eb1e8358"
CONFIG_SHA = "f55fa6a5f117ad8898c20bbaf8b20f288c169a9a9a12e7bf3bfdad26fef1d157"
STATS_SHA = "757f2aeebf688063747d50e55d983dd6b3c4cea268ce43d2bae06f3e99524a7e"
MANIFEST_SHA = "6d6d8167b882eb4079fcc566748028dc6de81692859f220a0246291c198574ec"
CHECKPOINT_BYTES = 454215776
VAL_RECORDINGS, VAL_USERS = 1950, 15


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm_user(value):
    value = _SAFE_RE.sub("_", str(value))
    return value if value.startswith("u") else "u" + value


def paths():
    work, scratch = Path(os.environ["WORK"]), Path(os.environ["SCRATCH"])
    return {"csv": work / "data/raw/emg2pose_mini/emg2pose_metadata.csv",
            "processed": scratch / "data/processed/emg2pose",
            "pose": scratch / "data/processed/emg2pose_pose",
            "manifest": work / "wearusfm_runs/results/passo6/derived_v1_1/manifest_v1_1_nested_1.json.gz"}


def fresh_output(path, provenance):
    require(path.is_absolute(), "output must be absolute")
    resolved = path.resolve()
    forbidden = [Path(os.environ["CODE_DIR"]).resolve(), Path(os.environ["WORK"]) / "wearusfm",
                 Path(os.environ["WORK"]) / "wearusfm_campaigns"]
    require(not any(resolved == p.resolve() or p.resolve() in resolved.parents for p in forbidden), "output inside code")
    require(not path.exists() and not path.is_symlink(), "output already exists")
    require(not provenance.exists() and not provenance.is_symlink(), "provenance already exists")


def extract_preflight(source, output, loc):
    import torch

    require(source.stat().st_size == CHECKPOINT_BYTES, "checkpoint size differs from completed B")
    checkpoint_sha = sha(source)
    config_file, summary_file = source.parent / "config.json", source.parent / "summary.json"
    require(sha(config_file) == CONFIG_SHA, "B config file SHA mismatch")
    config, summary = json.loads(config_file.read_text()), json.loads(summary_file.read_text())
    require(summary.get("steps") == 5000 and summary.get("stopped") == "max_steps", "B summary not completed at 5000")
    require(summary.get("code", {}).get("sha") == CODE_SHA, "B training code SHA mismatch")
    require(summary.get("pose_stats_sha256") == STATS_SHA, "B summary pose stats mismatch")
    # Trusted checkpoint produced by our completed B job; never use this on an untrusted pickle.
    state = torch.load(source, map_location="cpu", weights_only=False)
    require(state.get("step") == 5000 and state.get("config") == config, "checkpoint step/config differs from completed B")
    require(state.get("pose_stats_sha256") == STATS_SHA, "checkpoint pose stats mismatch")
    require(config.get("model", {}).get("pose_head") is True, "checkpoint has no pose head")
    student = state.get("student", {})
    dim = config["model"]["dim"]
    for key, shape in {"pose_head.norm.weight": (dim,), "pose_head.norm.bias": (dim,),
                       "pose_head.out.weight": (20, dim), "pose_head.out.bias": (20,)}.items():
        value = student.get(key)
        require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape and bool(torch.isfinite(value).all()),
                "missing, malformed or nonfinite student head: " + key)
    require(bool(student["pose_head.out.weight"].count_nonzero()), "student head is still zero")
    del state, student
    require(sha(loc["pose"] / PT.STATS_FILE) == STATS_SHA, "current pose stats mismatch")
    require(sha(loc["manifest"]) == MANIFEST_SHA, "training manifest mismatch")
    with gzip.open(loc["manifest"], "rt") as stream:
        manifest = json.load(stream)
    train_users = {norm_user(r["subject"]) for row in manifest["rows"] for r in [dict(zip(manifest["columns"], row))]
                   if r["dataset"] == "emg2pose" and r["split"] == "pretraining"}
    require(bool(train_users), "empty emg2pose training user set")
    with loc["csv"].open(newline="") as stream:
        val_rows = [r for r in csv.DictReader(stream) if r["split"] == "val"]
    val_users = sorted({norm_user(r["user"]) for r in val_rows})
    require(len(val_rows) == VAL_RECORDINGS and len(val_users) == VAL_USERS, "CSV full val is not 1950 recordings / 15 users")
    require(not set(val_users) & train_users, "validation users overlap B training")
    records = PE.recordings_for(loc["csv"], loc["processed"], "val")
    require(len(records) == VAL_RECORDINGS, "processed val recordings missing")
    require(len({r["path"] for r in records}) == VAL_RECORDINGS, "duplicate validation recording")
    fingerprints = []
    for rec in records:
        processed = Path(rec["path"])
        pose = PT.pose_dir(loc["pose"], processed)
        required = [processed / "data_int16.npy", pose / PT.POSE_FILE, pose / PT.VALID_FILE]
        require(all(p.is_file() and p.stat().st_size > 0 for p in required), "missing processed/pose arrays: " + rec["path"])
        fingerprints.append({"recording": rec["stem"], "metadata_sha256": sha(processed / "metadata.json"),
                             "array_bytes": {str(p): p.stat().st_size for p in required}})
    return {"checkpoint": str(source), "checkpoint_bytes": CHECKPOINT_BYTES, "checkpoint_sha256": checkpoint_sha,
            "checkpoint_step": 5000, "config_file_sha256": CONFIG_SHA, "summary_sha256": sha(summary_file),
            "training_code_sha": CODE_SHA, "pose_stats_sha256": STATS_SHA, "train_manifest_sha256": MANIFEST_SHA,
            "csv": str(loc["csv"]), "csv_sha256": sha(loc["csv"]), "pose_root": str(loc["pose"]),
            "which": "student", "split": "val", "n_windows_per_recording": 10, "window_s": 4.0, "seed": PE.SEED_WINDOWS,
            "val_users": val_users, "n_train_users": len(train_users), "n_recordings": len(records), "records": records,
            "processed_metadata_and_array_sizes": fingerprints,
            "note": "Preflight only: frozen extraction follows; cache SHA is first recorded by report preflight. Array contents are not rehashed."}


def report_preflight(source, output, loc):
    provenance_file = Path(str(source) + ".extract_provenance.json")
    before = json.loads(provenance_file.read_text())
    require(before.get("mode") == "extract" and before.get("output") == str(source), "extraction provenance/cache path mismatch")
    require(before.get("wrapper_sha256") == os.environ["WRAPPER_SHA256"], "extraction wrapper SHA mismatch")
    require(before.get("training_code_sha") == CODE_SHA and before.get("checkpoint_step") == 5000, "extraction checkpoint provenance mismatch")
    require(before.get("config_file_sha256") == CONFIG_SHA and before.get("train_manifest_sha256") == MANIFEST_SHA,
            "extraction training config/manifest mismatch")
    require(before.get("which") == "student" and before.get("split") == "val" and before.get("n_recordings") == VAL_RECORDINGS,
            "extraction protocol mismatch")
    require(sha(loc["pose"] / PT.STATS_FILE) == STATS_SHA == before.get("pose_stats_sha256"), "current/extraction stats mismatch")
    require(sha(loc["csv"]) == before["csv_sha256"], "CSV changed since extraction preflight")
    checkpoint = Path(before["checkpoint"])
    # Require the original B file here: this campaign keeps it available; no silent loss of immutability verification.
    require(checkpoint.is_file() and sha(checkpoint) == before["checkpoint_sha256"], "checkpoint changed or inaccessible since extraction")
    cache_sha = sha(source)
    data = PE.load_npz(source)
    cfg = data["config"]
    require(data["has_head"] is True, "cache has no trained head predictions")
    for key, value in {"which": "student", "split": "val", "code_sha": CODE_SHA, "pose_stats_sha256": STATS_SHA,
                       "checkpoint_pose_stats_sha256": STATS_SHA, "checkpoint": before["checkpoint"],
                       "pose_root": before["pose_root"], "n_windows": 10, "window_s": 4.0, "seed": PE.SEED_WINDOWS}.items():
        require(cfg.get(key) == value, "cache provenance mismatch: " + key)
    require(data["n_recordings"] == VAL_RECORDINGS and len(before["records"]) == VAL_RECORDINGS, "cache recording count mismatch")
    meta, mask = data["meta"], data["mask"]
    n = len(meta)
    require(n > 0 and data["lat"].shape[0] == n and mask.ndim == 3 and mask.shape[0] == n and mask.shape[2] == 20,
            "cache window shapes/counts mismatch")
    require(data["pred"].shape == data["target"].shape == mask.shape and data["std"].shape == (n, 20), "cache pose shape mismatch")
    require(np.isfinite(data["pred"]).all() and np.isfinite(data["target"]).all() and np.isfinite(data["std"]).all(),
            "nonfinite cache prediction/target/std")
    valid = mask.any(axis=(1, 2))
    expected_users = set(before["val_users"])
    require(len(expected_users) == VAL_USERS and {norm_user(m["user"]) for m in meta} == expected_users,
            "cache must contain all 15 validation users")
    require({norm_user(m["user"]) for m, ok in zip(meta, valid) if ok} == expected_users,
            "cache has a validation user without valid pose support")
    identities, observed = set(), set()
    for m in meta:
        ri = m["recording"]
        require(type(ri) is int and 0 <= ri < VAL_RECORDINGS, "invalid cache recording index")
        rec = before["records"][ri]
        require(all(m[k] == rec[k] for k in ("user", "session", "stage", "side", "stem")), "cache recording metadata mismatch")
        identity = (ri, m["start"])
        require(identity not in identities, "duplicate cache window")
        identities.add(identity)
        observed.add(ri)
    return {"cache": str(source), "cache_sha256": cache_sha, "cache_bytes": source.stat().st_size,
            "extract_provenance": str(provenance_file), "extract_provenance_sha256": sha(provenance_file),
            "checkpoint": str(checkpoint), "checkpoint_sha256": before["checkpoint_sha256"], "checkpoint_step": 5000,
            "pose_stats_sha256": STATS_SHA, "which": "student", "split": "val", "n_users_with_valid_pose": len(expected_users),
            "n_windows": n, "n_windows_with_valid_pose": int(valid.sum()), "n_recordings_requested": VAL_RECORDINGS,
            "n_recordings_with_windows": len(observed), "n_recordings_without_windows": VAL_RECORDINGS - len(observed),
            "note": "Preflight only: frozen E1/E2 report follows, without --train; no scientific thresholds changed."}


def main(mode, source, output):
    wrapper_sha = os.environ["WRAPPER_SHA256"]
    require(len(wrapper_sha) == 64 and all(c in "0123456789abcdef" for c in wrapper_sha), "invalid WRAPPER_SHA256")
    require(os.environ.get("WEARUSFM_CODE_SHA") == CODE_SHA, "preflight is not using the frozen checkout")
    source, output = Path(source), Path(output)
    require(source.is_absolute() and source.is_file(), "input must be an existing absolute file")
    provenance = Path(str(output) + (".extract_provenance.json" if mode == "extract" else ".report_provenance.json"))
    fresh_output(output, provenance)
    result = (extract_preflight if mode == "extract" else report_preflight)(source, output, paths())
    result.update(mode=mode, output=str(output), wrapper_sha256=wrapper_sha, execution_code_sha=CODE_SHA,
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with provenance.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"preflight": "passed", "mode": mode, "provenance": str(provenance)}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
PY_B_GUARD
if [ "$MODE" = extract ]; then
  exec bash "$CODE_DIR/scripts/slurm/pose_eval_extract.sbatch" "$INPUT" "$OUTPUT" --which student --split val --seed 20261026 --n-windows 10 --window-s 4 --batch 32
else
  exec bash "$CODE_DIR/scripts/slurm/pose_eval_report.sbatch" "$INPUT" "$OUTPUT"
fi
