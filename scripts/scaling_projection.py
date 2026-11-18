"""Frozen, data-independent S5 projections for the 300M scaling comparison.

Generation accepts no data and no configurable seed. The same backbone matrix
serves pose and classification; the local matrix serves the second pool block.
Projection precedes pose LayerNorm or the classifier's train-fold scaler.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

PROTOCOL = "scaling-projection-20261104-v1"
SEEDS = {"backbone": 2026110401, "local": 2026110402}
SHAPE = (896, 384)
ALGORITHM = {"generator": "numpy.Generator(PCG64)", "distribution": "standard_normal_float64",
             "shape": list(SHAPE), "qr": "numpy.linalg.qr(mode=reduced)",
             "sign": "positive diagonal of R; matching column signs in Q",
             "storage": "C-contiguous little-endian float32", "rescaling": None,
             "data_independent": True, "pool_block_order": ["backbone", "local"]}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_state(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f4").tobytes()).hexdigest()


def recipe():
    return {"protocol": PROTOCOL, "algorithm": ALGORITHM, "seeds": SEEDS}


def generate():
    result = {}
    for name, seed in SEEDS.items():
        raw = np.random.Generator(np.random.PCG64(seed)).standard_normal(SHAPE, dtype=np.float64)
        q, r = np.linalg.qr(raw, mode="reduced")
        diagonal = np.diag(r)
        require(np.isfinite(diagonal).all() and (diagonal != 0).all(), "degenerate projection QR")
        q *= np.where(diagonal < 0, -1., 1.)[None, :]
        result[name] = np.asarray(q, dtype="<f4", order="C")
    return result


def _validate_matrix(matrix):
    require(isinstance(matrix, np.ndarray) and matrix.shape == SHAPE and matrix.dtype == np.dtype("<f4") and
            matrix.flags.c_contiguous and np.isfinite(matrix).all(), "invalid projection shape/precision/values")
    gram = matrix.astype(np.float64).T @ matrix.astype(np.float64)
    require(np.allclose(gram, np.eye(SHAPE[1]), atol=2e-6, rtol=0), "projection columns are not orthonormal")


def save(directory, matrices=None):
    """Write NPY files first and manifest last; return the immutable manifest pin.

    Optional precomputed matrices must exactly equal this implementation's fixed
    generation. They cannot supply another seed, fit a dataset, or alter scale.
    """
    expected = generate()
    matrices = expected if matrices is None else matrices
    require(set(matrices) == set(SEEDS), "need both projection blocks")
    for name in SEEDS:
        _validate_matrix(matrices[name])
        require(np.array_equal(matrices[name], expected[name]), "matrices differ from fixed generation")
    directory = Path(directory)
    require(directory.is_absolute() and not directory.exists() and not directory.is_symlink() and
            directory.resolve() == directory, "projection output must be a new absolute directory")
    directory.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, matrix in matrices.items():
        path = directory / (name + ".npy")
        with path.open("xb") as stream:
            np.save(stream, matrix, allow_pickle=False)
        files[name] = {"file": path.name, "sha256": sha(path), "bytes": path.stat().st_size,
                       "array_sha256": array_sha(matrix), "shape": list(SHAPE), "dtype": "<f4", "seed": SEEDS[name]}
    manifest = {**recipe(), "complete": True, "files": files,
                "versions": {"python": platform.python_version(), "numpy": np.__version__},
                "generator_source_sha256": sha(__file__)}
    path = directory / "manifest.json"
    with path.open("x") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate projection manifest key")
        result[key] = value
    return result


@dataclass(frozen=True)
class Projection:
    backbone: np.ndarray
    local: np.ndarray
    manifest: dict
    states: dict
    pin: dict


def load(manifest_pin):
    require(isinstance(manifest_pin, dict) and set(manifest_pin) <= {"path", "sha256", "bytes"} and
            {"path", "sha256"} <= set(manifest_pin), "invalid projection manifest pin")
    path = Path(manifest_pin["path"])
    require(path.is_absolute(), "projection manifest path must be absolute")
    states = {str(path): file_state(path)}
    require(sha(path) == manifest_pin["sha256"] and ("bytes" not in manifest_pin or
            path.stat().st_size == manifest_pin["bytes"]), "projection manifest SHA/size mismatch")
    manifest = json.loads(path.read_text(), object_pairs_hook=_unique)
    require(manifest.get("complete") is True and all(manifest.get(k) == v for k, v in recipe().items()),
            "projection algorithm/seed/protocol differs")
    require(manifest.get("generator_source_sha256") == sha(__file__), "projection generator source differs")
    versions = manifest.get("versions")
    require(isinstance(versions, dict) and set(versions) == {"python", "numpy"} and
            all(isinstance(v, str) and v for v in versions.values()), "missing projection generator versions")
    files = manifest.get("files", {})
    require(set(files) == set(SEEDS), "missing projection block")
    matrices = {}
    for name, seed in SEEDS.items():
        record = files[name]
        require(set(record) == {"file", "sha256", "bytes", "array_sha256", "shape", "dtype", "seed"} and
                record["file"] == name + ".npy" and record["shape"] == list(SHAPE) and record["dtype"] == "<f4" and
                record["seed"] == seed, "projection file recipe differs")
        array_path = path.parent / record["file"]
        states[str(array_path)] = file_state(array_path)
        require(sha(array_path) == record["sha256"] and array_path.stat().st_size == record["bytes"],
                "projection matrix SHA/size mismatch")
        matrix = np.load(array_path, allow_pickle=False)
        _validate_matrix(matrix)
        require(array_sha(matrix) == record["array_sha256"], "projection matrix byte hash differs")
        matrix.flags.writeable = False
        matrices[name] = matrix
    require(states == {p: file_state(p) for p in states}, "projection changed during load")
    return Projection(matrices["backbone"], matrices["local"], manifest, states, dict(manifest_pin))


def project_pose(lat, projection):
    require(isinstance(projection, Projection), "pinned projection required")
    require(isinstance(lat, np.ndarray) and lat.ndim == 3 and lat.shape[-1] == SHAPE[0] and
            lat.dtype in (np.dtype("float16"), np.dtype("float32")) and np.isfinite(lat).all(),
            "expected finite FP16/FP32 896-wide pose sequence; sanitize padding first")
    out = np.asarray(lat, dtype=np.float32) @ projection.backbone
    require(out.dtype == np.float32 and np.isfinite(out).all(), "nonfinite projected pose features")
    return out


def project_pool(x_pool, projection):
    require(isinstance(projection, Projection), "pinned projection required")
    require(isinstance(x_pool, np.ndarray) and x_pool.ndim == 2 and x_pool.shape[1] == 2 * SHAPE[0] and
            x_pool.dtype == np.float32 and np.isfinite(x_pool).all(), "expected finite FP32 pool1792")
    out = np.concatenate((x_pool[:, :SHAPE[0]] @ projection.backbone,
                          x_pool[:, SHAPE[0]:] @ projection.local), axis=1)
    require(out.dtype == np.float32 and np.isfinite(out).all(), "nonfinite projected pool features")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new external matrix directory; no data input")
    args = parser.parse_args(argv)
    print(json.dumps(save(args.out), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
