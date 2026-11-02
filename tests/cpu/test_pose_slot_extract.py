"""Extraction identity, streaming, original RNG indices and numerical parity."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import pose_slot_extract as X
import pose_slot_features as F


def slots(xs):
    values = torch.as_tensor([float(x[0, 0]) for x in xs])[:, None, None, None]
    return (values + torch.linspace(-1, 1, 384)[None, None, None, :]
            + torch.arange(64)[None, None, :, None] * .001
            + torch.arange(160)[None, :, None, None] * .0001)


@pytest.fixture
def scene(tmp_path, monkeypatch):
    pose_root, emg_root = tmp_path / "pose", tmp_path / "emg"
    records, reference = [], {k: [] for k in ("lat", "target", "mask", "std", "pose_var", "meta")}
    for ri in range(4):
        directory = emg_root / f"u{ri}" / f"session{ri}"
        pd = pose_root / directory.parent.name / directory.name
        directory.mkdir(parents=True)
        pd.mkdir(parents=True)
        (directory / "metadata.json").write_text("{}")
        np.save(directory / "data_int16.npy", np.zeros((5, 2), np.int16))
        (pd / X.PT.META_FILE).write_text("{}")
        np.save(pd / X.PT.POSE_FILE, np.zeros((5, 20), np.float16))
        np.save(pd / X.PT.VALID_FILE, np.ones(5, bool))
        records.append({"path": str(directory), "user": f"u{ri}", "session": f"s{ri}",
                        "stage": "train", "side": "left", "stem": directory.name})
    seen_scales = []

    def view(path, split_at_gaps):
        return SimpleNamespace(path=path, fs=2000., ri=int(path.parent.name[1:]))

    def windows(view, *args):
        return [SimpleNamespace(trial=None, start=8000 * j, stop=8000 * (j + 1),
                                span=(0, 16000), n_patches=160) for j in range(2)]

    def estimate(view, cfg, rng):
        value = float(rng.uniform(1, 2))
        seen_scales.append((view.ri, value))
        return value

    def read(view, trial, start, stop, span, cfg):
        return np.full((2, 3), view.ri + 1 + start / 80000, np.float32)

    def target(row, view, window, patch_ms):
        y = (np.arange(160)[:, None] * .001 + np.arange(20)[None, :] * .002 + view.ri * .1).astype(np.float32)
        mask = np.ones_like(y, bool)
        mask[::5] = False
        return y, mask, np.full(20, .5, np.float32)

    store = SimpleNamespace(root=pose_root, window=target)

    class Model:
        cfg = SimpleNamespace(dim=384, k_latents=64)
        calls = []

        def eval(self):
            return self

        def encode(self, inp, visible):
            assert visible is None and not torch.is_grad_enabled()
            self.calls.append([x.copy() for x in inp])
            return SimpleNamespace(z=slots(inp))

    ev = X.PE.EvalConfig(device="cpu")
    for ri, record in enumerate(records):
        v = view(Path(record["path"]), True)
        scale = np.random.default_rng([20261026, ri]).uniform(1, 2)
        ws = windows(v)
        xs = [read(v, w.trial, w.start, w.stop, w.span, ev.loader) / np.float32(scale) for w in ws]
        mean = slots(xs).mean(2).numpy().astype(np.float16)
        for j, w in enumerate(ws):
            y, mask, sd = target({}, v, w, 25.)
            reference["lat"].append(mean[j])
            reference["target"].append(y)
            reference["mask"].append(mask)
            reference["std"].append(sd)
            reference["pose_var"].append(np.float32(y[mask.any(1)].var(0).mean()))
            reference["meta"].append(X.window_metadata(record, ri, v, w))
    for key in reference.keys() - {"meta"}:
        reference[key] = np.asarray(reference[key])
    monkeypatch.setattr(X.L.SessionView, "open", view)
    monkeypatch.setattr(X.L, "estimate_session_scale", estimate)
    monkeypatch.setattr(X.L, "read_window", read)
    monkeypatch.setattr(X.PE, "fixed_windows", windows)
    monkeypatch.setattr(X, "model_inputs", lambda xs, view, ev: xs)
    return SimpleNamespace(model=Model(), records=records, store=store, ev=ev, reference=reference,
                           seen_scales=seen_scales, out=tmp_path / "output")


def extract(scene, **kwargs):
    defaults = dict(model_name="B", split="train", projection_record=F.recipe(), log=lambda *a, **k: None)
    defaults.update(kwargs)
    return X.extract_stream(scene.model, scene.records, scene.store, scene.ev, scene.reference, scene.out, **defaults)


def test_stream_preserves_original_record_indices_and_produces_only_compact_array(scene):
    result = extract(scene, selected_recording_indices=[1, 3])
    assert result["reference_row_indices"] == [2, 3, 6, 7]
    assert result["selected_original_recording_indices"] == [1, 3]
    assert [i for i, scale in scene.seen_scales] == [1, 3]
    for ri, scale in scene.seen_scales:
        assert scale == np.random.default_rng([20261026, ri]).uniform(1, 2)
    residual = np.load(scene.out / "residual.npy", mmap_mode="r", allow_pickle=False)
    assert residual.dtype == np.float32 and residual.shape == (4, 160, 336)
    expected = torch.cat([F.projected_residuals(slots(xs)) for xs in scene.model.calls]).numpy()
    np.testing.assert_array_equal(residual, expected)
    assert set(p.name for p in scene.out.iterdir()) == {"residual.npy", "receipt.json"}
    assert result["residual"]["sha256"] == X.T.D.file_sha256(scene.out / "residual.npy")
    assert result["mean_parity"]["passed"] and result["mean_parity"]["n_unequal_fp16"] == 0
    assert result["mean_parity_gate"] == X.MEAN_GATE
    assert len(result["raw_inputs_states"]) == 10


@pytest.mark.parametrize("field", ["target", "mask", "std", "pose_var", "meta", "lat"])
def test_changed_reference_identity_aborts_without_complete_receipt(scene, field):
    if field == "meta":
        scene.reference[field][0]["start"] += 1
    elif field == "mask":
        scene.reference[field][0, 0, 0] = ~scene.reference[field][0, 0, 0]
    elif field == "lat":
        scene.reference[field][0, 0, 0] += np.float16(.1)
    else:
        scene.reference[field].flat[0] += .1
    with pytest.raises(ValueError, match="historical"):
        extract(scene)
    assert not (scene.out / "receipt.json").exists()


def test_neighbor_target_availability_never_changes_features(scene):
    first = extract(scene)
    original = np.load(scene.out / "residual.npy")
    scene.out = scene.out.with_name("second")
    old_window = scene.store.window

    def empty_targets(*args):
        y, mask, sd = old_window(*args)
        return np.zeros_like(y), np.zeros_like(mask), sd

    scene.store.window = empty_targets
    scene.reference["target"][:] = 0
    scene.reference["mask"][:] = False
    scene.reference["pose_var"][:] = np.nan
    second = extract(scene)
    np.testing.assert_array_equal(np.load(scene.out / "residual.npy"), original)
    assert first["mean_parity"] == second["mean_parity"]


def test_recording_mutation_and_final_guard_cannot_leave_complete_receipt(scene, monkeypatch):
    original = scene.model.encode

    def changed(inp, visible):
        result = original(inp, visible)
        Path(scene.records[0]["path"], "metadata.json").write_text('{"changed":true}')
        return result

    monkeypatch.setattr(scene.model, "encode", changed)
    with pytest.raises(ValueError, match="recording input changed"):
        extract(scene)
    assert not (scene.out / "receipt.json").exists()


def test_final_source_guard_failure_does_not_publish_receipt(scene):
    def bad():
        raise ValueError("code changed")
    with pytest.raises(ValueError, match="code changed"):
        extract(scene, final_guard=bad)
    assert not (scene.out / "receipt.json").exists()
    with pytest.raises(ValueError, match="already exists"):
        extract(scene)


def test_nonfinite_slot_is_rejected_even_without_pose_labels(scene, monkeypatch):
    original = scene.model.encode
    scene.reference["mask"][:] = False

    def bad(inp, visible):
        result = original(inp, visible)
        result.z[0, 0, 63, 383] = torch.nan
        return result

    monkeypatch.setattr(scene.model, "encode", bad)
    with pytest.raises(ValueError, match="nonfinite backbone slot"):
        extract(scene)
    assert not (scene.out / "receipt.json").exists()


def test_mean_gate_catches_small_raw_change_amplified_by_layernorm():
    reference = np.zeros((2, 384), np.float16)
    actual = reference.astype(np.float32)
    actual[0, 0] = 1e-4
    assert np.allclose(actual, reference, rtol=1e-3, atol=1e-3)
    with pytest.raises(ValueError, match="historical mean parity failed"):
        X.compare_mean(actual, reference)


def test_mean_gate_checks_denominator_and_not_ulp_near_zero():
    reference = np.linspace(-1, 1, 384).astype(np.float16)[None, :]
    actual = reference.astype(np.float32) * 1.003
    with pytest.raises(ValueError, match="historical mean parity failed"):
        X.compare_mean(actual, reference)
    reference = np.zeros((1, 384), np.float16)
    result = X.compare_mean(np.full((1, 384), 1e-6, np.float32), reference)
    assert result["max_ulp_fp16_difference"] > 2 and result["n_unequal_fp16"] == 384
    assert result["normalized_max_abs"] == 0 and result["denominator_relative_max"] == 0


def test_preflight_rejects_val_and_does_not_renumber_record_selection(scene):
    with pytest.raises(ValueError, match="train only"):
        extract(scene, split="val", selected_recording_indices=[0])
    for bad in ([True], [3, 1], [1, 1], [4], []):
        with pytest.raises(ValueError, match="recording indices"):
            X.selected_reference_rows(scene.reference, scene.records, bad)


def test_preflight_cli_reads_train_only_and_finalizes_after_both_models(scene, monkeypatch, tmp_path):
    import pose_slot_plan as P
    from wearusfm.harness import fm_features
    calls = []
    plan = {"mean_parity_gate": X.MEAN_GATE, "output_root": str(tmp_path / "campaign"),
            "paths": {"pose_root": str(scene.store.root)}, "preflight_recording_indices": [0, 1, 2, 3],
            "models": {m: {"checkpoint": str(tmp_path / f"{m}.pt"), "checkpoint_sha256": m*64} for m in X.T.MODELS},
            "caches": {m: {s: {"path": f"{m}/{s}", "sha256": "a"*64, "n_windows": 8} for s in ("train", "val")}
                       for m in X.T.MODELS}}
    pp = tmp_path / "plan.json"
    pp.write_text(json.dumps(plan))
    monkeypatch.setattr(P, "validate_plan", lambda plan: None)
    monkeypatch.setattr(P, "runtime_identity", lambda sha: {"code_sha": sha})
    monkeypatch.setattr(P, "validate_checkpoint", lambda plan, model: {"model": model})

    def inspect(plan, model, split):
        assert split == "train"
        calls.append((model, split))
        return {"states": {}}

    monkeypatch.setattr(P, "validate_reference", inspect)
    monkeypatch.setattr(X, "global_input_states", lambda plan: {})
    monkeypatch.setattr(X, "load_records", lambda *args: scene.records)
    monkeypatch.setattr(X.T, "read_data", lambda pin: copy.deepcopy(scene.reference))
    monkeypatch.setattr(X.PT, "PoseStore", lambda path: scene.store)
    monkeypatch.setattr(fm_features, "load_model", lambda *a, **kw: scene.model)
    out = Path(plan["output_root"]) / "preflight"
    assert X.main(["preflight", "--plan", str(pp), "--expected-code-sha", "a"*40,
                   "--out", str(out), "--device", "cpu"]) == 0
    receipt = json.loads((out / "receipt.json").read_text())
    assert calls == [("B", "train"), ("R3c", "train")]
    assert set(receipt["outputs"]) == {"B/train", "R3c/train"}
    assert receipt["complete"] and set(receipt["full_extraction_window_scaled_seconds"]) == set(X.T.MODELS)
    assert not (out / "B/val").exists()
    attestation = P.validate_preflight(plan, "a"*40, X.T.D.file_sha256(pp))
    assert attestation["path"] == str(out / "receipt.json")
    assert attestation["sha256"] == X.T.D.file_sha256(out / "receipt.json")
    monkeypatch.setattr(P, "validate_reference", lambda plan, model, split: {"states": {}})
    full_out = Path(plan["output_root"]) / "B"
    assert X.main(["extract", "--model", "B", "--plan", str(pp), "--expected-code-sha", "a"*40,
                   "--out", str(full_out), "--device", "cpu"]) == 0
    complete = json.loads((full_out / "receipt.json").read_text())
    assert complete["complete"] and complete["preflight"] == attestation
    assert set(complete["outputs"]) == {"B/train", "B/val"}


def test_full_cli_without_preflight_aborts_before_output_and_model(tmp_path, monkeypatch):
    import pose_slot_plan as P
    from wearusfm.harness import fm_features
    root = tmp_path / "campaign"
    plan = {"output_root": str(root), "mean_parity_gate": X.MEAN_GATE}
    pp = tmp_path / "plan.json"
    pp.write_text(json.dumps(plan))
    monkeypatch.setattr(P, "validate_plan", lambda plan: None)
    monkeypatch.setattr(P, "runtime_identity", lambda sha: {"code_sha": sha})
    monkeypatch.setattr(fm_features, "load_model", lambda *a, **kw: pytest.fail("model loaded before preflight"))
    monkeypatch.setattr(X, "global_input_states", lambda *a: pytest.fail("input payload read before preflight"))
    with pytest.raises(FileNotFoundError):
        X.main(["extract", "--model", "B", "--plan", str(pp), "--expected-code-sha", "a"*40,
                "--out", str(root / "B"), "--device", "cpu"])
    assert not root.exists()


def completed_extraction(scene, tmp_path):
    root = tmp_path / "full"
    scene.out = root / "B" / "train"
    checkpoint_sha, reference_sha, code_sha, plan_sha = "a"*64, "b"*64, "c"*40, "d"*64
    extract(scene, identity={"checkpoint_sha256": checkpoint_sha, "reference_cache_sha256": reference_sha,
                             "code_sha": code_sha, "plan_sha256": plan_sha})
    cache = tmp_path / "reference.npz"
    np.savez(cache, meta_json=np.asarray(json.dumps(scene.reference["meta"])))
    plan = {"output_root": str(root), "models": {"B": {"checkpoint_sha256": checkpoint_sha}},
            "mean_parity_gate": X.MEAN_GATE,
            "caches": {"B": {"train": {"path": str(cache), "n_windows": 8, "sha256": reference_sha}}}}
    child_path = scene.out / "receipt.json"
    parent_path = scene.out.parent / "receipt.json"
    parent = {"protocol": X.PROTOCOL, "complete": True, "stage": "extract", "code_sha": code_sha,
              "plan_sha256": plan_sha,
              "outputs": {"B/train": {"receipt": str(child_path), "receipt_sha256": X.T.D.file_sha256(child_path),
                                      "n_windows": 8},
                          "B/val": {"receipt": str(root / "B/val/receipt.json"), "receipt_sha256": "f"*64,
                                    "n_windows": 18548}}}
    parent_path.write_text(json.dumps(parent))
    return plan, code_sha, plan_sha, child_path, parent_path


def test_full_receipt_is_accepted_by_readout_guard_and_corruption_rejected(scene, tmp_path):
    import pose_slot_plan as P
    plan, code_sha, plan_sha, child_path, parent_path = completed_extraction(scene, tmp_path)
    checked = P.validate_residual(plan, "B", "train", expected_code_sha=code_sha, expected_plan_sha256=plan_sha)
    assert checked["receipt"]["complete"] and checked["receipt"]["mean_parity"]["passed"]
    path = scene.out / "residual.npy"
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="SHA/size mismatch"):
        P.validate_residual(plan, "B", "train", expected_code_sha=code_sha, expected_plan_sha256=plan_sha)


PARITY_METRICS = ("normalized_patch_rms_max", "normalized_max_abs", "denominator_relative_max")


@pytest.mark.parametrize("keys,value,error", [
    (("mean_parity_gate", "eps"), 1e-2, "parity gate mismatch"),
    (("mean_parity", "passed"), False, "parity not passed"),
    *[(("mean_parity", key), X.MEAN_GATE[key] * 2, "parity outside gate") for key in PARITY_METRICS],
    *[(("mean_parity", key), float("nan"), "parity outside gate") for key in PARITY_METRICS],
    (("reference_row_indices",), list(reversed(range(8))), "rows incomplete or reordered"),
    (("checkpoint_step",), 5000, "checkpoint step mismatch"),
    (("mean_parity", "n_elements"), 8 * 160 * 384 - 1, "parity support incomplete"),
    (("code_sha",), "e"*40, "residual code mismatch"),
    (("plan_sha256",), "e"*64, "residual plan mismatch"),
])
def test_tampered_split_receipt_rejected_even_with_updated_parent_hash(scene, tmp_path, keys, value, error):
    import pose_slot_plan as P
    plan, code_sha, plan_sha, child_path, parent_path = completed_extraction(scene, tmp_path)
    child = json.loads(child_path.read_text())
    target = child
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    # JSON NaN is deliberately injected as malformed external evidence.
    child_path.write_text(json.dumps(child))
    parent = json.loads(parent_path.read_text())
    parent["outputs"]["B/train"]["receipt_sha256"] = X.T.D.file_sha256(child_path)
    for key in ("code_sha", "plan_sha256"):
        parent[key] = child[key]
    parent_path.write_text(json.dumps(parent))
    with pytest.raises(ValueError, match=error):
        P.validate_residual(plan, "B", "train", expected_code_sha=code_sha, expected_plan_sha256=plan_sha)


@pytest.mark.parametrize("case,error", [
    ("incomplete", "incomplete model extraction"),
    ("preflight", "incomplete model extraction"),
    ("protocol", "incomplete model extraction"),
    ("code", "extraction identity mismatch"),
    ("plan", "extraction identity mismatch"),
    ("missing_val", "missing full splits"),
    ("child_path", "extraction receipt mismatch"),
    ("child_sha", "extraction receipt mismatch"),
    ("count", "extraction receipt mismatch"),
])
def test_tampered_model_receipt_cannot_authorize_split(scene, tmp_path, case, error):
    import pose_slot_plan as P
    plan, code_sha, plan_sha, child_path, parent_path = completed_extraction(scene, tmp_path)
    parent = json.loads(parent_path.read_text())
    if case == "incomplete":
        parent["complete"] = False
    elif case == "preflight":
        parent["stage"] = "preflight"
    elif case == "protocol":
        parent["protocol"] = "other"
    elif case in ("code", "plan"):
        parent["code_sha" if case == "code" else "plan_sha256"] = "e"*64
    elif case == "missing_val":
        del parent["outputs"]["B/val"]
    elif case == "child_path":
        parent["outputs"]["B/train"]["receipt"] = str(child_path.with_name("other.json"))
    elif case == "child_sha":
        parent["outputs"]["B/train"]["receipt_sha256"] = "e"*64
    elif case == "count":
        parent["outputs"]["B/train"]["n_windows"] -= 1
    parent_path.write_text(json.dumps(parent))
    with pytest.raises(ValueError, match=error):
        P.validate_residual(plan, "B", "train", expected_code_sha=code_sha, expected_plan_sha256=plan_sha)
