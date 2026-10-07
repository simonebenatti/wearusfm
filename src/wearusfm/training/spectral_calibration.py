"""Train-only calibration provenance, shared by offline producer and trainer."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from wearusfm.model.spectral_step1 import READOUT_VERSION
from wearusfm.model.spectral_targets import TARGET_VERSION


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def calibration_signature(cfg, manifest, scales, window_s):
    return {"target_version": TARGET_VERSION, "readout_version": READOUT_VERSION,
            "manifest_sha256": sha256_file(manifest), "loader": json.loads(json.dumps(asdict(cfg.loader))),
            "datasets": list(cfg.datasets) if cfg.datasets is not None else None,
            "scales_sha256": json_sha256(scales or {}), "window_seconds_sha256": json_sha256(window_s)}


def read_calibration(path, signature):
    with np.load(path, allow_pickle=False) as data:
        y, valid = data["y"].copy(), data["valid"].copy()
        subject, split = data["subject"].copy(), data["split"].copy()
        meta = json.loads(str(data["metadata_json"].item()))
    if y.ndim != 2 or y.shape[1] != 32 or valid.shape != y.shape or valid.dtype != bool:
        raise ValueError("invalid calibration array shapes/dtype")
    if not np.isfinite(y[valid]).all() or not len(y):
        raise ValueError("empty/nonfinite calibration")
    if subject.shape != (len(y),) or split.shape != subject.shape or subject.dtype.kind not in "US" or split.dtype.kind not in "US":
        raise ValueError("calibration subject/split must be string vectors")
    if not np.all(split == "train") or any("/" not in s for s in subject.tolist()):
        raise ValueError("calibration must contain TRAIN-only qualified subjects")
    if sorted(set(subject.tolist())) != meta.get("subjects"):
        raise ValueError("calibration subject provenance mismatch")
    if any(meta.get(key) != value for key, value in signature.items()):
        raise ValueError("calibration target/manifest/preprocessing signature mismatch")
    if meta.get("role") != "train" or not meta.get("subjects") or not meta.get("split_sha256") or not meta.get("subset_sha256"):
        raise ValueError("calibration needs explicit TRAIN-only subject/subset provenance")
    return y, valid, meta
