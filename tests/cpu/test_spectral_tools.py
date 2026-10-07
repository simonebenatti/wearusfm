"""Step 1 CLI on synthetic signals: real loader, small encoder, provenance guards."""
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from wearusfm.data import pretraining_loader as L
from wearusfm.model.fm import WearUsFM
from wearusfm.model.spectral_step1 import READOUT_VERSION
from wearusfm.model.spectral_targets import TARGET_VERSION
from wearusfm.training import run as R
from wearusfm.training.spectral_calibration import calibration_signature, read_calibration, sha256_file
from test_pretraining_loader import _tree


@pytest.fixture
def cli(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts" / "spectral_step1.py"
    spec = importlib.util.spec_from_file_location("spectral_step1_cli_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Capture the small config BEFORE replacing sanity_config (small_config uses it).
    small = R.small_config(datasets=None)
    monkeypatch.setattr(module.R, "sanity_config", lambda: small)
    monkeypatch.setattr(module, "source_commit", lambda: "a" * 40)
    return module, replace(R.with_window1_rules(small), datasets=None)

@pytest.fixture
def inputs(tmp_path, cli):
    module, cfg = cli
    root, manifest = _tree(tmp_path)
    index = R.load_index(manifest, [root], None)
    scales = {f"{r['dataset']}/{r['subject']}/{r['session']}": 2. for r in index.rows}
    loader = L.PretrainLoader(index, replace(cfg.loader, time_weighted=False), scales)
    window_s = {f"{r['dataset']}/{r['subject']}/{r['session']}": L.mean_window_s(loader.view(r), cfg.loader)
                for r in index.rows}
    scale_path, window_path = tmp_path / "scales.json", tmp_path / "window_seconds.json"
    scale_path.write_text(json.dumps({"scales": scales}))
    # This is the actual schema emitted by scripts/window_seconds.py.
    window_path.write_text(json.dumps({"manifest": str(manifest), "window_s": window_s}))
    split_path = tmp_path / "split.json"
    base = ["--manifest", str(manifest), "--root", str(root), "--datasets", "all", "--seed", "7"]
    assert module.main(["split", *base, "--out", str(split_path)]) == 0
    data_args = [*base, "--splits", str(split_path), "--scales", str(scale_path),
                 "--window-seconds", str(window_path), "--windows-per-subject", "2"]
    checkpoint = tmp_path / "small.pt"
    model = WearUsFM(cfg.model)
    torch.save({"config": R.config_to_dict(cfg), "student": model.state_dict(), "teacher": model.state_dict()}, checkpoint)
    return {"args": data_args, "base": base, "split": split_path, "checkpoint": checkpoint,
            "signature": calibration_signature(cfg, manifest, scales, window_s), "index": index}


def test_split_calibrate_extract_probe_roundtrip(tmp_path, cli, inputs):
    module, cfg = cli
    roles = json.loads(inputs["split"].read_text())
    assert set(roles) == {"train", "val", "test"}
    assert len({s for group in roles.values() for s in group}) == 3
    second_split = tmp_path / "split_again.json"
    module.main(["split", *inputs["base"], "--out", str(second_split)])
    assert second_split.read_bytes() == inputs["split"].read_bytes()
    calibration = tmp_path / "calibration.npz"
    assert module.main(["calibrate", *inputs["args"], "--out", str(calibration)]) == 0
    y, valid, meta = read_calibration(calibration, inputs["signature"])
    assert y.shape == valid.shape == (2, 32) and valid.any()
    assert meta["role"] == "train" and meta["subjects"] == roles["train"]
    assert meta["split_sha256"] == sha256_file(inputs["split"])
    with np.load(calibration, allow_pickle=False) as data:
        assert set(data["split"]) == {"train"}
        assert set(data["subject"]) == set(roles["train"])
    features = tmp_path / "features.npz"
    assert module.main(["extract", *inputs["args"], "--checkpoint", str(inputs["checkpoint"]),
                        "--device", "cpu", "--which", "teacher", "--out", str(features)]) == 0
    with np.load(features, allow_pickle=False) as data:
        feature_meta = json.loads(str(data["metadata_json"].item()))
        assert data["X"].shape == (6, 2 * cfg.model.dim)
        assert np.isfinite(data["X"]).all()
        assert feature_meta["checkpoint_sha256"] == sha256_file(inputs["checkpoint"])
        assert feature_meta["readout_version"] == READOUT_VERSION
        assert feature_meta["target_version"] == TARGET_VERSION
        assert feature_meta["which"] == "teacher"
        # Identical seed and train subjects give identical calibration targets.
        is_train = data["split"] == "train"
        np.testing.assert_array_equal(data["y"][is_train], y)
        np.testing.assert_array_equal(data["valid"][is_train], valid)
    report = tmp_path / "report.json"
    assert module.main(["probe", "--data", str(features), "--out", str(report),
                        "--expected-checkpoint-sha256", sha256_file(inputs["checkpoint"])]) == 0
    result = json.loads(report.read_text())
    assert set(result["views"]) == {"backbone", "local", "p1p2"}
    assert all(len(view["coordinates"]) == 32 for view in result["views"].values())
    with pytest.raises(ValueError, match="NEW"):
        module.main(["calibrate", *inputs["args"], "--out", str(calibration)])
    rejected = tmp_path / "wrong_hash_report.json"
    with pytest.raises(ValueError, match="provenance"):
        module.main(["probe", "--data", str(features), "--out", str(rejected),
                     "--expected-checkpoint-sha256", "0" * 64])
    assert not rejected.exists()


def test_calibration_rejects_false_train_only_provenance(tmp_path):
    signature = {"target_version": TARGET_VERSION, "readout_version": READOUT_VERSION, "manifest_sha256": "a" * 64}
    meta = dict(signature, role="train", subjects=["ds/one"], split_sha256="b" * 64,
                subset_sha256="c" * 64, source_commit="d" * 40)
    y, valid = np.arange(64).reshape(2, 32).astype(np.float32), np.ones((2, 32), bool)
    good_subject, good_split = np.array(["ds/one", "ds/one"]), np.array(["train", "train"])
    correct = tmp_path / "correct.npz"
    np.savez(correct, y=y, valid=valid, subject=good_subject, split=good_split, metadata_json=np.array(json.dumps(meta)))
    np.testing.assert_array_equal(read_calibration(correct, signature)[0], y)
    cases = [
        (good_subject, np.array(["train", "val"]), meta),
        (np.array(["ds/one", "ds/two"]), good_split, meta),
        (good_subject[:1], good_split, meta),
        (np.array(["one", "one"]), good_split, dict(meta, subjects=["one"])),
    ]
    for i, (subject, split, metadata) in enumerate(cases):
        bad = tmp_path / f"invalid_{i}.npz"
        np.savez(bad, y=y, valid=valid, subject=subject, split=split, metadata_json=np.array(json.dumps(metadata)))
        with pytest.raises(ValueError):
            read_calibration(bad, signature)


def test_extract_rejects_checkpoint_replaced_during_load(tmp_path, cli, inputs, monkeypatch):
    module, _ = cli
    original_load = module.load_model

    def load_and_replace(path, **kwargs):
        model = original_load(path, **kwargs)
        Path(path).write_bytes(b"replacement checkpoint after original weights were loaded")
        return model

    monkeypatch.setattr(module, "load_model", load_and_replace)
    out = tmp_path / "unstable_features.npz"
    with pytest.raises(ValueError, match="checkpoint|changed"):
        module.main(["extract", *inputs["args"], "--checkpoint", str(inputs["checkpoint"]), "--out", str(out)])
    assert not out.exists()


def test_roles_reject_nonpretraining_and_shared_subjects(tmp_path, cli, inputs):
    module, _ = cli
    roles = json.loads(inputs["split"].read_text())
    shared = dict(roles, val=roles["train"])
    path = tmp_path / "invalid_roles.json"
    path.write_text(json.dumps(shared))
    with pytest.raises(ValueError, match="leakage"):
        module.read_roles(path, inputs["index"])
    path.write_text(json.dumps(dict(roles, test=["external/held_out"])))
    with pytest.raises(ValueError, match="PRETRAINING"):
        module.read_roles(path, inputs["index"])
