"""Counterexamples for Claude e198a87. Synthetic files only; no jobs or model training.

Run with one BLAS thread and pass the Claude checkout. Assertions describe the reviewed
defects, not desired regression behavior; change them when the implementation is repaired.
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile

import h5py
import numpy as np

repo = Path(sys.argv[1])
sys.path[:0] = [str(repo / "src"), str(repo / "scripts"), str(repo / "tests/cpu")]
import pose_audit as audit
import pose_head_synthetic as synthetic
from test_pose_audit import _setup
from wearusfm.ingest import emg2pose_pose as ingest
from wearusfm.training import run

# The true target at t is exactly the central feature at t; neighboring features
# make all tested lags representable. A nonzero argmin is not an index failure.
rng = np.random.default_rng(5)
windows = []
for index in range(12):
    raw = rng.normal(size=164)
    windows.append({"user": "synthetic", "order": index,
                    "feat": np.stack([raw[2 + k:162 + k] for k in range(-2, 3)], axis=1),
                    "target": np.repeat(raw[2:162, None], 20, axis=1),
                    "valid": np.ones(160, bool), "dyn": np.ones(160, bool)})
scan = audit.ridge_per_user(windows, lags=2, seed=20261027)["synthetic"]["lag_scan"]
assert scan["best_lag"] == 2
print("perfectly_aligned_contextual_features", scan)

# The independent oracle does not apply the ingestion rule for NaN/Inf.
angles = np.ones((100, 20), np.float32)
angles[0, 0] = np.nan
target, mask, counts = audit.oracle_patch_targets(angles, 0, 2, 50, 280, np.zeros(20), np.ones(20))
bad = ingest.invalid_rows(angles, 280)
blocks, valid = ingest.block_reduce(angles, ~bad)
assert bad[0] and not valid[0] and np.isfinite(blocks[valid]).all()
assert mask[0] and counts[0] == 5 and np.isnan(target[0, 0])
print("nonfinite_oracle", {"production_first_block_valid": bool(valid[0]), "oracle_valid_blocks_first_patch": int(counts[0]),
                            "oracle_target_is_nan": bool(np.isnan(target[0, 0]))})

# The synthetic schedule is one update ahead of the real decay.
cfg = run.with_wsd(run.sanity_config(), 1300, warmup_steps=1000)
from dataclasses import replace
cfg = replace(cfg, lr=1e-4)
real = np.array([run._lr_at(s - 1, cfg) for s in range(1, 1301)])
toy = np.array([synthetic.lr_at(s, "dbg") for s in range(1, 1301)])
assert not np.array_equal(real, toy)
assert np.isclose(real.sum() / 1e-4, 671) and np.isclose(toy.sum() / 1e-4, 670)
print("schedule_doses", {"real": float(real.sum() / 1e-4), "toy": float(toy.sum() / 1e-4)})

with tempfile.TemporaryDirectory(prefix="r3-audit-review-") as tmp:
    tmp = Path(tmp)
    root, poses, csv, archive = _setup(tmp)
    argv = ["--csv", str(csv), "--processed-root", str(root), "--pose-root", str(poses), "--tar", str(archive),
            "--n-users", "2", "--n-recordings", "4", "--n-windows", "5", "--context", "1", "--lags", "6"]
    out = tmp / "complete.json"
    with contextlib.redirect_stdout(io.StringIO()):
        audit.main([*argv, "--out", str(out)])
    report = json.loads(out.read_text())
    with tarfile.open(archive) as tf:
        member = next(x for x in tf if x.name.endswith(".hdf5"))
        with h5py.File(io.BytesIO(tf.extractfile(member).read()), "r") as h5:
            raw_emg = h5["emg2pose/timeseries"]["emg"][:100]
        stem = Path(member.name).stem
    directory = next(root.glob(f"*/{stem}"))
    emg_meta = json.loads((directory / "metadata.json").read_text())
    processed = np.load(directory / "data_int16.npy", mmap_mode="r")[:100].astype(float) / emg_meta["int16_scale"]
    max_error = float(np.abs(raw_emg - processed).max())
    assert max_error > 1
    keys = ("identita_ok_tutte", "ricostruzione_ok_tutte", "oracolo_ok_tutte")
    assert all(report["sintesi"][k] for k in keys)
    print("raw_emg_not_checked", {"max_raw_processed_error": max_error, **{k: report["sintesi"][k] for k in keys}})

    out = tmp / "missing.json"
    with contextlib.redirect_stdout(io.StringIO()):
        audit.main([*argv, "--tar-index-budget-s", "0", "--out", str(out)])
    missing = json.loads(out.read_text())
    s = missing["sintesi"]
    assert missing["tar"]["found"] == 0 and s["n_ricostruite"] == 0 and s["n_finestre_oracolo"] == 0
    assert s["ricostruzione_ok_tutte"] and s["oracolo_ok_tutte"]
    print("no_raw_but_true_flags", {"tar": missing["tar"], **{k: s[k] for k in (
        "n_ricostruite", "n_finestre_oracolo", "ricostruzione_ok_tutte", "oracolo_ok_tutte")}})
