"""Small end-to-end cache/subject-fold checks, without remote data or jobs."""
import importlib.util
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from wearusfm.harness.downstream_ab import fingerprint
from wearusfm.harness.psd3_inputs import hash_file
from wearusfm.harness.psd3_readers import FeatureConfig, LogisticConfig, MLPConfig

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("psd3_cli", ROOT / "scripts/psd3_readout.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.fixture
def entry(tmp_path):
    rng = np.random.default_rng(7)
    s = np.repeat([f"s{i}" for i in range(8)], 6)
    y = np.tile(["rest", "move"], 24)
    x = rng.normal(size=(48, 4)).astype(np.float32)
    x[:, 0] += (y == "move") * 3
    path = tmp_path / "cache.npz"
    np.savez(path, x_pool=x, x_dec=x.astype(np.float16), y=y, s=s)
    roles = []
    for i in range(4):
        held = [f"s{2*i}", f"s{2*i+1}"]
        rest = [f"s{j}" for j in range(8) if f"s{j}" not in held]
        roles.append({"train": rest[2:], "val": rest[:2], "held": held})
    return {"path": str(path), "pins": {"sha256": hash_file(path), "bytes": path.stat().st_size},
            "roles": roles, "subjects": sorted(set(s)), "roles_sha256": fingerprint(roles),
            "expected_oof_windows": 48, "expected_oof_rows_sha256": fingerprint(np.arange(48, dtype=np.int64)),
            "model": "B", "dataset": "uci_emg", "task": "p2"}


def test_cache_to_both_readers_exact_oof_support(entry, monkeypatch):
    # Identity fields need not be embedded in this artificial cache; attach only
    # after the loader wrapper verifies the exact bytes for the end-to-end test.
    original = cli.load_cache_view
    def load(path, pins, view):
        arrays = original(path, {k: pins[k] for k in ("sha256", "bytes")}, view)
        return arrays
    monkeypatch.setattr(cli, "load_cache_view", load)
    entry["pins"].update(input_sha256="a"*64, checkpoint_sha256="b"*64)
    monkeypatch.setattr(cli, "source_identity", lambda: {"source_sha256": "c"*64, "versions": {}})
    progress = []
    report = cli.run_shard(entry, "pool", feature_config=FeatureConfig(k=2),
        logistic_config=LogisticConfig(feature_dim=2, c_grid=(0.1, 1.0), max_iter=50),
        mlp_config=MLPConfig(feature_dim=2, hidden_dim=3, epochs=2, lambdas=(0.0, 0.01)),
        progress=progress.append)
    assert report["status"] == "completed"
    assert report["rows"] == list(range(48))
    assert set(report["predictions"]["mlp"]) == {"0", "1"}
    assert len(progress) == 8
    for fold in report["folds"]:
        assert fold["logistic"]["audit"][0]["train_fingerprint"] == fold["mlp"]["audit"][0]["train_fingerprint"]
        assert fold["preprocessing"]["fit_role"] == "train"
    assert all(len(p) == 48 for r in report["predictions"].values() for p in r.values())


def test_oof_guard_runs_before_any_fit(entry, monkeypatch):
    entry["expected_oof_rows_sha256"] = "0"*64
    monkeypatch.setattr(cli, "prepare_features", lambda *a, **k: pytest.fail("fit called before OOF guard"))
    with pytest.raises(ValueError, match="OOF row order/support"):
        cli.run_shard(entry, "pool")


def test_filter_class_support_is_same_for_all_readers():
    y = np.asarray(["a", "b", "a", "new", "b", "new"])
    s = np.asarray(["train", "train", "val", "val", "held", "held"])
    roles = [{"train": ["train"], "val": ["val"], "held": ["held"]}]
    train, val, held, dropped = cli.fold_indices(y, s, roles)[0]
    assert train.tolist() == [0, 1]
    assert val.tolist() == [2] and held.tolist() == [4]
    assert dropped == {"val": 1, "held": 1}


def test_plan_pins_match_archived_evidence():
    plan = cli.make_plan(ROOT / "results/infra/pilot_posa/b19618_20261029/final")
    assert len(plan["entries"]) == 10
    assert plan["logistic_fits"] + plan["mlp_fits"] == 1040
    assert sum(e["pins"]["bytes"] for e in plan["entries"]) == 5431115084
    plan["entries"][0]["roles"][0]["train"].append("unknown")
    with pytest.raises(ValueError):
        cli.validate_plan(plan)


def test_writer_does_not_overwrite_or_emit_nonfinite(tmp_path):
    path = tmp_path / "report.json"
    cli.write_new(path, {"ok": True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        cli.write_new(path, {"ok": False})
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        cli.write_new(tmp_path / "invalid.json", {"x": float("nan")})
    assert not (tmp_path / "invalid.json").exists()


def test_complete_default_report_validation_and_rejection(tmp_path, monkeypatch):
    rng = np.random.default_rng(17)
    s = np.repeat([f"s{i}" for i in range(8)], 40)
    y = np.tile(["rest", "move"], 160)
    x = rng.normal(size=(320, 130)).astype(np.float32)
    x[:, 0] += (y == "move") * 4
    metadata = {k: char*64 for k, char in (("checkpoint_sha256", "a"), ("input_sha256", "b"),
                ("splits_sha256", "c"), ("code_sha", "d"))}
    metadata["extraction_recipe"] = {"fixture": True}
    path = tmp_path / "full.npz"
    np.savez(path, x_pool=x, x_dec=x.astype(np.float16), y=y, s=s,
             **{k: np.asarray(json.dumps(v) if isinstance(v, dict) else v) for k, v in metadata.items()})
    roles = []
    for i in range(4):
        held = [f"s{2*i}", f"s{2*i+1}"]
        rest = [f"s{j}" for j in range(8) if f"s{j}" not in held]
        roles.append({"train": rest[2:], "val": rest[:2], "held": held})
    ent = {"path": str(path), "pins": {"sha256": hash_file(path), "bytes": path.stat().st_size, **metadata},
           "roles": roles, "subjects": sorted(set(s)), "roles_sha256": fingerprint(roles),
           "expected_oof_windows": 320, "expected_oof_rows_sha256": fingerprint(np.arange(320, dtype=np.int64)),
           "model": "B", "dataset": "uci_emg", "task": "p2"}
    identity = {"files": {"fixture": "e"*64}, "versions": {}}
    identity["source_sha256"] = fingerprint(identity["files"])
    monkeypatch.setattr(cli, "source_identity", lambda: copy.deepcopy(identity))
    report = json.loads(json.dumps(cli.run_shard(ent, "pool")))
    report["plan_sha256"] = "f"*64
    cli.validate_report(report, ent, "f"*64)
    alterations = [
        lambda r: r.update(status="failed"),
        lambda r: r.update(view="dec"),
        lambda r: r.update(subjects=["unknown"] * len(r["subjects"])),
        lambda r: r["y"].__setitem__(0, "move"),
        lambda r: r["runtime"].update(source_sha256="0"*64),
        lambda r: r["folds"].pop(),
        lambda r: r["folds"][0]["mlp"]["audit"].pop(),
        lambda r: r["folds"][0]["mlp"]["audit"][0]["epochs"].pop(),
        lambda r: r["folds"][0]["logistic"]["audit"][0].update(train_fingerprint="x"),
        lambda r: r["predictions"]["mlp"]["0"].__setitem__(0, "unknown_train_class"),
    ]
    for change in alterations:
        broken = copy.deepcopy(report)
        change(broken)
        with pytest.raises(ValueError):
            cli.validate_report(broken, ent, "f"*64)


def test_mutated_source_during_run_is_rejected(entry, monkeypatch):
    original = cli.load_cache_view
    def load(path, pins, view):
        return original(path, {k: pins[k] for k in ("sha256", "bytes")}, view)
    monkeypatch.setattr(cli, "load_cache_view", load)
    entry["pins"].update(input_sha256="a"*64, checkpoint_sha256="b"*64)
    identities = iter([{"source_sha256": "a"*64}, {"source_sha256": "b"*64}])
    monkeypatch.setattr(cli, "source_identity", lambda: next(identities))
    with pytest.raises(RuntimeError, match="changed during shard"):
        cli.run_shard(entry, "pool", feature_config=FeatureConfig(k=2),
            logistic_config=LogisticConfig(feature_dim=2, c_grid=(1.0,), max_iter=50),
            mlp_config=MLPConfig(feature_dim=2, hidden_dim=3, epochs=1, lambdas=(0.0,)))
