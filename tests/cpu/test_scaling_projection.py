"""Fixed-S5 matrix generation, provenance, block order and dtype oracles."""
import copy
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import scaling_projection as P


@pytest.fixture(scope="module")
def projection(tmp_path_factory):
    pin = P.save(tmp_path_factory.mktemp("s5") / "matrices")
    return P.load(pin)


def test_generation_is_data_independent_and_does_not_consume_global_rng(projection):
    assert not inspect.signature(P.generate).parameters
    before = np.random.get_state()
    actual = P.generate()
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    for name in P.SEEDS:
        np.testing.assert_array_equal(actual[name], getattr(projection, name))
        assert actual[name].shape == (896, 384) and actual[name].dtype == np.float32
    assert not np.array_equal(actual["backbone"], actual["local"])


def test_projection_matches_independent_positive_diagonal_qr_oracle(projection):
    for name, seed in P.SEEDS.items():
        gaussian = np.random.Generator(np.random.PCG64(seed)).standard_normal((896, 384), dtype=np.float64)
        q, r = np.linalg.qr(gaussian, mode="reduced")
        for column in range(384):
            if r[column, column] < 0:
                q[:, column] *= -1
        actual = getattr(projection, name)
        np.testing.assert_array_equal(actual, q.astype(np.float32))
        np.testing.assert_allclose(actual.astype(float).T @ actual, np.eye(384), atol=2e-7, rtol=0)
        assert np.all(np.diag(actual.astype(float).T @ gaussian) > 0)
        assert projection.manifest["files"][name]["array_sha256"] == P.array_sha(actual)


def test_pool_uses_distinct_matrices_in_backbone_then_local_order(projection):
    x = np.random.default_rng(7).normal(size=(3, 1792)).astype(np.float32)
    got = P.project_pool(x, projection)
    np.testing.assert_array_equal(got[:, :384], x[:, :896] @ projection.backbone)
    np.testing.assert_array_equal(got[:, 384:], x[:, 896:] @ projection.local)
    assert got.dtype == np.float32 and got.shape == (3, 768)
    assert not np.allclose(got[:, :384], x[:, :896] @ projection.local)
    assert not projection.backbone.flags.writeable and not projection.local.flags.writeable


def test_pose_keeps_all_patches_and_converts_fp16_before_projection(projection):
    x = np.random.default_rng(9).normal(size=(2, 17, 896)).astype(np.float16)
    actual = P.project_pose(x, projection)
    assert actual.shape == (2, 17, 384) and actual.dtype == np.float32
    np.testing.assert_array_equal(actual, x.astype(np.float32) @ projection.backbone)


@pytest.mark.parametrize("kind", ("pool_width", "pool_half", "pool_nan", "pose_width", "pose_double", "pose_nan"))
def test_projection_rejects_incompatible_or_nonfinite_inputs(projection, kind):
    if kind.startswith("pool"):
        x = np.zeros((2, 1792), np.float32)
        if kind == "pool_width": x = x[:, :-1]
        if kind == "pool_half": x = x.astype(np.float16)
        if kind == "pool_nan": x[0, 0] = np.nan
        fn = P.project_pool
    else:
        x = np.zeros((2, 17, 896), np.float16)
        if kind == "pose_width": x = x[..., :-1]
        if kind == "pose_double": x = x.astype(np.float64)
        if kind == "pose_nan": x[0, 0, 0] = np.nan
        fn = P.project_pose
    with pytest.raises(ValueError):
        fn(x, projection)


def test_save_will_not_attest_another_orthonormal_projection(tmp_path, projection):
    matrices = {name: getattr(projection, name).copy() for name in P.SEEDS}
    matrices["backbone"][:, 0] *= -1
    with pytest.raises(ValueError, match="fixed generation"):
        P.save(tmp_path / "changed", matrices)
    assert not (tmp_path / "changed").exists()


@pytest.mark.parametrize("mutation", ("seed", "algorithm", "source", "matrix_bytes", "matrix_array_hash", "matrix_scale", "swapped"))
def test_loading_requires_frozen_manifest_and_exact_matrix_bytes(tmp_path, projection, mutation):
    import shutil
    source = Path(projection.pin["path"]).parent
    destination = tmp_path / "bundle"
    shutil.copytree(source, destination)
    path = destination / "manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "seed": manifest["seeds"]["backbone"] += 1
    if mutation == "algorithm": manifest["algorithm"]["rescaling"] = "sqrt(896/384)"
    if mutation == "source": manifest["generator_source_sha256"] = "0" * 64
    if mutation == "matrix_bytes": (destination / "backbone.npy").write_bytes(b"corrupt")
    if mutation == "matrix_array_hash": manifest["files"]["backbone"]["array_sha256"] = "0" * 64
    if mutation == "swapped": manifest["files"]["backbone"]["file"] = "local.npy"
    if mutation == "matrix_scale":
        matrix = projection.backbone * 2
        target = destination / "backbone.npy"
        np.save(target, matrix, allow_pickle=False)
        manifest["files"]["backbone"].update(sha256=P.sha(target), bytes=target.stat().st_size, array_sha256=P.array_sha(matrix))
    path.write_text(json.dumps(manifest))
    pin = {"path": str(path), "sha256": P.sha(path), "bytes": path.stat().st_size}
    with pytest.raises(ValueError):
        P.load(pin)


def test_manifest_pin_itself_is_checked(projection):
    pin = {**projection.pin, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="manifest SHA"):
        P.load(pin)
