"""Read-only, pinned inputs for the PSD-3 frozen-feature reader comparison.

The stable entry schema requires ``sha256`` and ``bytes``. Optional scalar pins
have the cache's names (for example ``checkpoint_sha256`` and ``code_sha``);
``extraction_recipe`` is a decoded dictionary. Optional ``n_windows`` and
``n_subjects`` constrain row and subject counts. No model/dataset name is
inferred from the filename. Inventory reads NPY headers and small metadata,
never feature payloads; hashing the entire archive is explicitly opt-in.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from wearusfm.harness.downstream_ab import fingerprint


METADATA_KEYS = (
    "checkpoint", "checkpoint_sha256", "input_sha256", "splits_sha256",
    "extraction_recipe", "code_sha", "encoder_dim", "channels", "task",
    "protocol", "calibration_protocol", "feature_set", "montage_signature",
)
MAX_METADATA_BYTES = 1 << 20


def hash_file(path: str | Path) -> str:
    """SHA256 of file bytes, without loading the file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream)
    # These caches are written with scalar and homogeneous array dtypes. NPY
    # v3's Unicode field names are not part of this input contract.
    raise ValueError(f"unsupported NPY version: {version}")


def inspect_cache(path: str | Path, *, compute_sha: bool = False) -> dict:
    """Inspect an NPZ without materializing X, y or subject arrays.

    ``arrays`` maps member names to shape/dtype/order and uncompressed bytes.
    ``metadata`` contains only the whitelist of small scalar fields above.
    ``sha256`` is null unless ``compute_sha=True`` was explicitly requested.
    """
    path = Path(path)
    before = _signature(path)
    arrays, metadata = {}, {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            _require(member.filename.endswith(".npy") and "/" not in member.filename,
                     f"unexpected NPZ member: {member.filename}")
            key = member.filename[:-4]
            _require(key and key not in arrays, f"duplicate/empty NPZ key: {key}")
            with archive.open(member) as stream:
                shape, fortran_order, dtype = _header(stream)
                header_bytes = stream.tell()
            _require(not dtype.hasobject, f"object dtype is not allowed: {key}")
            elements = 1
            for length in shape:
                _require(isinstance(length, int) and length >= 0, f"invalid shape: {key}")
                elements *= length
            payload_bytes = elements * dtype.itemsize
            _require(member.file_size == header_bytes + payload_bytes,
                     f"NPY payload size does not match header: {key}")
            arrays[key] = {"shape": list(shape), "dtype": dtype.str,
                           "fortran_order": bool(fortran_order),
                           "uncompressed_bytes": payload_bytes}
            if key in METADATA_KEYS:
                _require(shape == () and payload_bytes <= MAX_METADATA_BYTES,
                         f"metadata must be a small scalar: {key}")
                with archive.open(member) as stream:
                    value = np.lib.format.read_array(stream, allow_pickle=False).item()
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                if key in ("extraction_recipe", "montage_signature"):
                    _require(isinstance(value, str), f"JSON metadata must be text: {key}")
                    value = json.loads(value)
                    if key == "extraction_recipe":
                        _require(isinstance(value, dict), "extraction_recipe must be a JSON object")
                metadata[key] = value
    digest = hash_file(path) if compute_sha else None
    _require(_signature(path) == before, "cache changed during inventory")
    return {"path": str(path), "bytes": before[2], "sha256": digest,
            "arrays": arrays, "metadata": metadata}


def _labels(value: np.ndarray, name: str) -> None:
    _require(value.ndim == 1 and len(value) > 0, f"{name} must be a nonempty vector")
    kind = value.dtype.kind
    if kind in "biuf":
        _require(bool(np.isfinite(value).all()), f"{name} contains nonfinite values")
    elif kind in "US":
        _require(bool((np.char.str_len(np.char.strip(value)) > 0).all()),
                 f"{name} contains empty/blank labels")
    else:
        raise ValueError(f"unsupported {name} dtype: {value.dtype}")


def load_cache_view(path: str | Path, entry: dict, view: str):
    """Return ``(X, y, subjects, provenance)`` for a pinned pool or dec view.

    The archive hash is checked before materializing only the selected feature
    array, y and s. Small metadata is read by ``inspect_cache``. The file's stat
    identity must remain unchanged throughout the operation. Full-array
    finiteness, metadata pins and row/subject coverage are checked before use.
    """
    _require(view in ("pool", "dec"), "view must be pool or dec")
    _require(isinstance(entry, dict), "entry must be a dictionary")
    digest = entry.get("sha256")
    _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
             "entry requires a lowercase SHA256")
    size = entry.get("bytes")
    _require(isinstance(size, int) and not isinstance(size, bool) and size > 0,
             "entry requires positive integer bytes")
    path = Path(path)
    before = _signature(path)
    _require(before[2] == size, "cache byte count differs from pin")
    _require(hash_file(path) == digest, "cache SHA256 differs from pin")
    info = inspect_cache(path)
    meta = info["metadata"]
    for key in METADATA_KEYS:
        if key in entry:
            _require(key in meta and meta[key] == entry[key], f"metadata pin differs: {key}")
    key = "x_" + view
    for name in (key, "y", "s"):
        _require(name in info["arrays"], f"cache is missing {name}")
    with np.load(path, allow_pickle=False) as cache:
        x, y, subjects = cache[key], cache["y"], cache["s"]
    _require(x.ndim == 2 and min(x.shape) > 0 and x.dtype.kind in "biuf",
             "X must be a nonempty real numeric matrix")
    _require(bool(np.isfinite(x).all()), "X contains nonfinite values")
    _labels(y, "y")
    _labels(subjects, "s")
    # JSON reports preserve textual identity as Unicode, including byte-string
    # NPZ labels. Do not stringify numeric labels or mixed object arrays.
    if y.dtype.kind == "S":
        y = np.char.decode(y, "utf-8")
    if subjects.dtype.kind == "S":
        subjects = np.char.decode(subjects, "utf-8")
    _require(len(x) == len(y) == len(subjects), "X/y/s row counts differ")
    if "encoder_dim" in meta:
        dim = meta["encoder_dim"]
        _require(isinstance(dim, int) and not isinstance(dim, bool) and dim > 0,
                 "invalid encoder_dim")
        if view == "pool":
            _require(x.shape[1] == 2 * dim, "pool width differs from encoder_dim")
        elif "channels" in meta:
            channels = meta["channels"]
            _require(isinstance(channels, int) and not isinstance(channels, bool) and channels > 0,
                     "invalid channels")
            _require(x.shape[1] == channels * dim, "dec width differs from channels/encoder_dim")
    n_subjects = len(np.unique(subjects))
    for name, actual in (("n_windows", len(y)), ("n_subjects", n_subjects)):
        if name in entry:
            _require(isinstance(entry[name], int) and not isinstance(entry[name], bool)
                     and entry[name] == actual, f"count pin differs: {name}")
    _require(_signature(path) == before, "cache changed during load")
    provenance = {"path": str(path), "sha256": digest, "bytes": size, "view": view,
                  "shape": list(x.shape), "dtype": x.dtype.str, "n_subjects": n_subjects,
                  "metadata": meta, "y_sha256": fingerprint(y),
                  "subjects_sha256": fingerprint(subjects),
                  "canonical_rows_sha256": fingerprint(list(range(len(y))), y.tolist(), subjects.tolist()),
                  "ordered_rows_sha256": fingerprint(np.arange(len(y), dtype=np.int64), y, subjects)}
    return x, y, subjects, provenance


def _string_ids(values, name: str) -> np.ndarray:
    """Check original scalars before NumPy can coerce mixed IDs into strings."""
    _require(isinstance(values, (list, tuple, np.ndarray)), f"{name} must be an ID vector")
    if isinstance(values, np.ndarray):
        _require(values.ndim == 1, f"{name} must be an ID vector")
    text = all(isinstance(value, (str, np.str_)) for value in values)
    binary = all(isinstance(value, (bytes, np.bytes_)) for value in values)
    _require(text or binary, f"{name} ID types must be uniformly text or bytes strings")
    result = np.asarray(values)
    _labels(result, name)
    _require(result.dtype.kind in "US", f"{name} must have string IDs")
    return result


def validate_roles(roles: list[dict], subjects) -> None:
    """Require disjoint, exhaustive folds and exactly one held fold per subject.

    IDs are nonempty Unicode/byte strings; ``subjects`` can contain repeated
    row-level IDs. Each role list itself must contain each subject at most once.
    No integer-to-string coercion or sorting changes the caller's row mapping.
    """
    subjects = _string_ids(subjects, "subjects")
    universe = set(subjects.tolist())
    _require(isinstance(roles, (list, tuple)) and len(roles) >= 2, "at least two folds required")
    held_counts = dict.fromkeys(universe, 0)
    for index, fold in enumerate(roles):
        _require(isinstance(fold, dict) and all(k in fold for k in ("train", "val", "held")),
                 f"fold {index} requires train/val/held")
        sets = {}
        for name in ("train", "val", "held"):
            values = _string_ids(fold[name], f"fold {index} {name}")
            _require(values.dtype.kind == subjects.dtype.kind, f"fold {index} ID types differ")
            sets[name] = set(values.tolist())
            _require(len(sets[name]) == len(values), f"fold {index} repeats {name} subjects")
        train, val, held = (sets[k] for k in ("train", "val", "held"))
        _require(not (train & val or train & held or val & held), f"fold {index} roles overlap")
        _require(train | val | held == universe, f"fold {index} does not partition all subjects")
        for subject in held:
            held_counts[subject] += 1
    _require(all(n == 1 for n in held_counts.values()), "held subjects must appear exactly once across folds")
