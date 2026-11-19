"""Pinned historical cache inputs, without reinterpreting their producer identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import consolidation_long_pose as historical
import tokenizer_pose_readout as reference

CROP = 6
PRODUCER_FIELDS = ("which", "split", "checkpoint_step", "checkpoint_sha256", "code_sha",
                   "source_sha256", "recipe_sha256", "pose_stats_sha256", "checkpoint_pose_stats_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pin(pin):
    require(isinstance(pin, dict) and {"path", "sha256"} <= pin.keys(), "invalid file pin")
    return reference.verify_file(pin["path"], pin["sha256"], pin.get("bytes"))


def lengths_for(data):
    values = [m["n_patches"] for m in data["meta"]]
    require(all(type(n) is int and 2*CROP < n <= data["target"].shape[1] for n in values),
            "invalid real-patch lengths")
    return np.asarray(values, dtype=np.int64)


def central_support(mask, lengths, key_time_valid=None):
    mask, lengths = np.asarray(mask), np.asarray(lengths)
    require(mask.ndim == 3 and mask.dtype == bool and mask.shape[1] > 2*CROP,
            "invalid pose mask")
    require(lengths.shape == (len(mask),) and lengths.dtype.kind in "iu" and
            ((lengths > 2*CROP) & (lengths <= mask.shape[1])).all(), "invalid patch lengths")
    available = np.arange(mask.shape[1])[None, :] < lengths[:, None]
    require(not np.any(mask & ~available[..., None]), "target mask includes padded context")
    if key_time_valid is not None:
        valid = np.asarray(key_time_valid)
        require(valid.dtype == bool and valid.shape == available.shape and np.array_equal(valid, available),
                "key_time_valid differs from contiguous real context")
    return reference.T.central_mask(mask, lengths, offset=CROP)


def validate_producer(data, producer):
    """Compare immutable embedded provenance with a separately pinned producer.

    No current reader source hash is accepted as an implicit historical hash.
    The caller obtains producer from the pinned original extraction receipt.
    """
    require(isinstance(producer, dict) and set(PRODUCER_FIELDS) <= producer.keys(),
            "missing pinned historical producer fields")
    require(producer["which"] == "teacher" and producer["checkpoint_step"] == 49046 and
            producer["split"] in ("train", "val"), "wrong historical teacher/dose/split")
    for key in PRODUCER_FIELDS:
        value = producer[key]
        if key.endswith("sha256") or key == "code_sha":
            size = 40 if key == "code_sha" else 64
            require(isinstance(value, str) and len(value) == size and
                    all(c in "0123456789abcdef" for c in value), "invalid producer digest: " + key)
        require(data["config"].get(key) == value, "historical producer mismatch: " + key)
    require(producer["pose_stats_sha256"] == producer["checkpoint_pose_stats_sha256"],
            "historical target standardization mismatch")
    for key, value in {"n_windows": 10, "window_s": 4., "seed": 20261026, "per_side": False}.items():
        require(data["config"].get(key) == value, "historical window selection mismatch: " + key)
    return {key: producer[key] for key in PRODUCER_FIELDS}


def validate_cache(data, *, producer=None, target_reference=None, selection=None):
    result = historical.validate_cache(data, target_reference, selection)
    require(data["dim"] == 384 and data["target"].shape[1:] == (160, 20),
            "expected frozen 160-patch, 384-feature, 20-joint cache")
    central = central_support(data["mask"], lengths_for(data), data["key_time_valid"])
    result = {**result, "supported_windows": int(central.any(axis=(1, 2)).sum()),
              "central_mask_sha256": reference.arrays_hash(central)}
    if producer is not None:
        result["historical_producer"] = validate_producer(data, producer)
    return result


def read_cache_pin(pin, *, producer, reference_pin, depth=None):
    """Load one pinned split and verify its unchanged historical target support.

    Full validation precedes dropping an unused depth. No held-data decision is
    made here: the phase controller must keep validation out of selection.
    """
    require(depth in (None, "pre", "post"), "invalid feature depth")
    states = verify_pin(pin)
    states.update(verify_pin(reference_pin))
    target_reference = reference.read_targets(reference_pin["path"])
    reference.check_selection(target_reference, reference_pin)
    data = historical.read_cache(pin["path"], target_reference, reference_pin)
    details = validate_cache(data, producer=producer, target_reference=target_reference, selection=reference_pin)
    if "windows" in pin:
        require(len(data["meta"]) == pin["windows"], "cache window count differs from pin")
    if "users" in pin:
        require(len({m["user"] for m in data["meta"]}) == pin["users"], "cache user count differs from pin")
    require(states == {p: reference.T.file_state(p) for p in states}, "pinned input changed while reading")
    data["lengths"] = lengths_for(data)
    data["central_mask"] = central_support(data["mask"], data["lengths"], data["key_time_valid"])
    if depth is not None:
        data["features"] = data["pre" if depth == "pre" else "lat"]
        del data["pre"], data["lat"]
        data["feature_depth"] = depth
    data["input_verification"] = details
    return data


def view_inputs(data, depth="post", indices=None):
    """Return raw FP32 latents; learned LayerNorm belongs to the head."""
    require(depth in ("pre", "post"), "invalid depth")
    if "features" in data:
        require(data["feature_depth"] == depth, "requested view was not retained")
        features = data["features"]
    else:
        features = data["pre" if depth == "pre" else "lat"]
    lengths = lengths_for(data)
    if indices is not None:
        features, lengths = features[indices], lengths[indices]
    available = np.arange(features.shape[1])[None, :] < lengths[:, None]
    require(features.ndim == 3 and features.shape[-1] == 384 and
            np.isfinite(features[available]).all(), "invalid real latent context")
    return np.where(available[..., None], features, 0).astype(np.float32)


def alternating_user_folds(data, expected=None):
    users = sorted({m["user"] for m in data["meta"]})
    require(len(users) >= 2 and all(isinstance(u, str) and u for u in users), "invalid user list")
    folds = [users[::2], users[1::2]]
    require(expected is None or folds == expected, "user folds differ from frozen manifest")
    return folds


def subset(data, indices):
    indices = np.asarray(indices)
    require(indices.ndim == 1 and indices.dtype.kind in "iu" and
            ((indices >= 0) & (indices < len(data["meta"]))).all(), "invalid subset indices")
    require(len(set(indices.tolist())) == len(indices), "duplicate subset windows")
    n = len(data["meta"])
    out = {key: (value[indices] if isinstance(value, np.ndarray) and value.ndim and len(value) == n else value)
           for key, value in data.items()}
    out["meta"] = [data["meta"][i].copy() for i in indices]
    # This is a cache view, not a new historical recording selection.
    out["subset_indices"] = indices.copy()
    return out


def read_predictions(pin):
    states = verify_pin(pin)
    with np.load(pin["path"], allow_pickle=False) as archive:
        require(set(archive.files) == {"prediction"}, "unexpected historical prediction payload")
        result = archive["prediction"]
    require(result.ndim == 3 and result.dtype == np.float64,
            "historical crop predictions must be float64")
    require(states == {p: reference.T.file_state(p) for p in states}, "prediction file changed while reading")
    return result
