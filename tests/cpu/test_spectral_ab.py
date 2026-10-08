import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from wearusfm.harness.spectral_ab import paired_spectral_comparison, validate_training_pair
from wearusfm.harness.spectral_probe import r2, spectral_probe
from wearusfm.training import run as R
from wearusfm.training.pairing import batch_sha256, state_sha256
from test_spectral_step1 import _calibration, _setup


def test_actual_paired_trainer_initialization_and_all_batches(tmp_path):
    root, manifest, cfg, batch, y, m = _setup(tmp_path)
    calibration = tmp_path/"cal.npz"
    _calibration(calibration, cfg, manifest, y, m)
    cfg = replace(cfg, max_steps=3, pairing_audit=True, keep_grad_every=1)
    runs = []
    for arm, weight in (("A", 0.), ("B", .05)):
        out = tmp_path/arm
        R.train(R.with_spectral_keep(cfg, calibration, weight), manifest, [root], out, log=lambda _: None)
        run = {k: json.loads((out/f"{k}.json").read_text()) for k in ("config", "summary", "pairing")}
        run["metrics"] = [json.loads(line) for line in (out/"metrics.jsonl").read_text().splitlines()]
        runs.append(run)
    assert validate_training_pair(*runs, expected_steps=3)["all_batches_identical"]
    # Auditing did not change the model/RNG path in arm B.
    plain = tmp_path/"plain"
    R.train(R.with_spectral_keep(replace(cfg, pairing_audit=False), calibration, .05),
            manifest, [root], plain, log=lambda _: None)
    audited, original = [torch.load(p/"checkpoint.pt", weights_only=False) for p in (tmp_path/"B", plain)]
    assert state_sha256(audited["student"]) == state_sha256(original["student"])
    assert torch.equal(audited["torch_rng"], original["torch_rng"])
    before = (tmp_path/"B"/"checkpoint.pt").read_bytes()
    with pytest.raises(ValueError, match="uninterrupted"):
        R.train(R.with_spectral_keep(cfg, calibration, .05), manifest, [root], tmp_path/"B")
    assert (tmp_path/"B"/"checkpoint.pt").read_bytes() == before
    broken = copy.deepcopy(runs[1])
    broken["metrics"][1]["batch_sha256"] = "f"*64
    with pytest.raises(ValueError, match="data/targets/masks differ"):
        validate_training_pair(runs[0], broken, expected_steps=3)
    broken = copy.deepcopy(runs[1]); broken["summary"]["stopped"] = "limite di tempo"
    with pytest.raises(ValueError, match="incomplete"):
        validate_training_pair(runs[0], broken, expected_steps=3)
    broken = copy.deepcopy(runs[1]); broken["config"]["lr"] *= 2
    with pytest.raises(ValueError, match="ONLY"):
        validate_training_pair(runs[0], broken, expected_steps=3)
    broken = copy.deepcopy(runs[1]); broken["pairing"]["student_initial_sha256"] = "f"*64
    with pytest.raises(ValueError, match="not paired"):
        validate_training_pair(runs[0], broken, expected_steps=3)


@pytest.mark.parametrize("field", ["signals", "visible", "kind", "qc_valid", "anchor_targets", "windows", "rows"])
def test_batch_fingerprint_detects_real_changes_and_preserves_rng(tmp_path, field):
    _, _, _, batch, _, _ = _setup(tmp_path)
    before = torch.get_rng_state().clone()
    a = batch_sha256(batch)
    assert torch.equal(before, torch.get_rng_state())
    changed = copy.deepcopy(batch)
    if field == "signals": changed.signals[0][0, 0] += 1
    elif field == "visible": changed.visible[0, 0] = ~changed.visible[0, 0]
    elif field == "kind": changed.kind[0, 0] += 1
    elif field == "qc_valid": changed.qc_valid[0] = ~changed.qc_valid[0]
    elif field == "anchor_targets": changed.anchor_targets[0].ms_fast[0, 0, 0] += 1
    elif field == "windows": changed.windows[0] = replace(changed.windows[0], start=changed.windows[0].start+1)
    else: changed.rows[0]["session"] = "other"
    assert batch_sha256(changed) != a
    assert batch_sha256(copy.deepcopy(batch)) == a


def test_state_hash_handles_scalar_buffer_bf16_and_layout():
    a = {"scalar": torch.tensor(True), "bf16": torch.arange(8).to(torch.bfloat16)}
    b = {"bf16": a["bf16"].clone(), "scalar": a["scalar"].clone()}
    assert state_sha256(a) == state_sha256(b)
    b["bf16"][0] += 1
    assert state_sha256(a) != state_sha256(b)


def synthetic_report(error_scale):
    # Each subject has a constant target locally, but pooled R2 is well defined.
    # The comparison MUST use pooled sufficient statistics, not discard subjects.
    subject = np.repeat(["ds/a", "ds/b", "ds/c"], [2, 4, 6])
    y = np.repeat([1., 3., 8.], [2, 4, 6])
    prediction = y + error_scale
    rows = []
    for j in range(32):
        stats = {}
        for s in sorted(set(subject)):
            target, pred = y[subject == s], prediction[subject == s]
            stats[s] = {"n": len(target), "r2": None, "target_mean": float(target.mean()),
                        "target_m2": float(np.square(target-target.mean()).sum()),
                        "sse": float(np.square(target-pred).sum())}
        rows.append({"coordinate": j, "status": "ok", "r2": r2(y, prediction), "per_subject": stats})
    # Add nonzero within-subject variance so bootstrap draws of one subject are assessable.
    for row in rows:
        for stat in row["per_subject"].values(): stat["target_m2"] = stat["n"]*.25
        row["r2"] = 1-len(y)*error_scale**2/(np.square(y-y.mean()).sum()+len(y)*.25)
    return {"coordinates": rows}


def test_paired_subject_bootstrap_reconstructs_pooled_r2_and_is_deterministic():
    a, b = synthetic_report(1.), synthetic_report(.1)
    result = paired_spectral_comparison([(a, b), (copy.deepcopy(a), copy.deepcopy(b))], draws=200)
    subject = np.repeat(["ds/a", "ds/b", "ds/c"], [2, 4, 6])
    y = np.repeat([1., 3., 8.], [2, 4, 6])
    denominator = np.square(y-y.mean()).sum() + len(y)*.25
    assert result["primary"]["delta"] == pytest.approx(len(subject)*(.99)/denominator)
    assert result == paired_spectral_comparison([(a, b), (a, b)], draws=200)
    assert result["decision"] == "promising_run_fresh_P1_P2"
    assert result["bootstrap"]["shared_across_arms_and_model_seeds"]


def test_identical_arms_zero_ci_and_negative_r2_not_truncated():
    bad = synthetic_report(100.)
    result = paired_spectral_comparison([(bad, bad), (bad, bad)], draws=100)
    assert result["families"]["fast_shape"]["a_r2"] < 0
    assert result["primary"]["delta"] == 0
    assert result["primary"]["paired_subject_ci95"] == [0., 0.]
    assert result["decision"] == "not_promising_at_this_weight_and_endpoint"


def test_bootstrap_rejects_target_mismatch_missing_coordinate_and_one_seed():
    a, b = synthetic_report(1.), synthetic_report(.1)
    bad = copy.deepcopy(b); bad["coordinates"][0]["per_subject"]["ds/a"]["target_mean"] += 1
    with pytest.raises(ValueError, match="different test targets"):
        paired_spectral_comparison([(a, bad), (a, b)], draws=20)
    bad = copy.deepcopy(b); bad["coordinates"][0]["status"] = "constant_test"
    with pytest.raises(ValueError, match="no omitted"):
        paired_spectral_comparison([(a, bad), (a, b)], draws=20)
    with pytest.raises(ValueError, match="two paired"):
        paired_spectral_comparison([(a, b)], draws=20)


def test_ridge_sufficient_statistics_match_actual_pooled_r2():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(180, 4)); y = np.tile(x[:, 0, None], (1, 32))
    split = np.repeat(["train", "val", "test"], 60)
    subject = np.repeat(["ds/a", "ds/b", "ds/c", "ds/d", "ds/e", "ds/f"], 30)
    report = spectral_probe(x, y, np.ones_like(y, bool), subject, split)
    row = report["coordinates"][0]
    stats = list(row["per_subject"].values())
    n = np.array([s["n"] for s in stats]); means = np.array([s["target_mean"] for s in stats])
    grand = (n*means).sum()/n.sum()
    denom = sum(s["target_m2"]+s["n"]*(s["target_mean"]-grand)**2 for s in stats)
    assert 1-sum(s["sse"] for s in stats)/denom == pytest.approx(row["r2"], abs=1e-12)


def test_pilot_builder_fixed_identity_and_unsafe_final_checkpoint(tmp_path):
    script = Path(__file__).resolve().parents[2]/"scripts"/"step1_ab.py"
    import sys
    sys.path.insert(0, str(script.parent))
    try:
        spec = importlib.util.spec_from_file_location("step1_ab", script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        cal = tmp_path/"calibration.npz"; cal.write_bytes(b"builder-only")
        ca = R.config_to_dict(module.build_config("A", 1, cal))
        cb = R.config_to_dict(module.build_config("B", 1, cal))
        assert ca["jepa"].pop("keep_weight") == 0 and cb["jepa"].pop("keep_weight") == .05
        assert ca == cb and ca["max_steps"] == 4000 and ca["warmup_steps"] == 1000
        with pytest.raises(ValueError, match="nonfinite final"):
            module._finite_state({"optimizer": {"moment": torch.tensor(float("nan"))}})
    finally:
        sys.path.pop(0)


def test_compare_cache_pipeline_and_mismatched_preprocessing(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    script = Path(__file__).resolve().parents[2]/"scripts"/"step1_ab.py"
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("step1_ab_pipeline", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    commit = "c"*40
    monkeypatch.setattr(module, "checked_commit", lambda expected: commit)
    cfg = {"jepa": {"keep_weight": 0.}, "max_steps": 4000, "pairing_audit": True, "seed": 0,
           "target_version": "target", "readout_version": "readout"}
    run = {"config": cfg, "summary": {"steps": 4000, "stopped": "max_steps"},
           "pairing": {"student_initial_sha256": "a"*64, "teacher_initial_sha256": "a"*64,
                       "seed": 0, "num_workers": 6, "resume": False},
           "metrics": [{"step": i, "batch_sha256": "b"*64} for i in range(1, 4001)],
           "provenance": {"arm": "A", "checkpoint_sha256": "a"*64, "extraction_signature": {"loader": "signed"}}}

    def fake_run(path, _):
        arm, seed = path.name.split("_seed")
        result = copy.deepcopy(run)
        result["config"]["jepa"]["keep_weight"] = module.WEIGHTS[arm]
        result["config"]["seed"] = result["pairing"]["seed"] = int(seed)
        result["provenance"]["arm"] = arm
        return result

    monkeypatch.setattr(module, "read_run", fake_run)
    subject = np.repeat([f"emg2qwerty/s{i:03}" for i in range(100)], 100)
    split = np.repeat(["train", "val", "test"], [8000, 1000, 1000])
    y = np.tile(np.linspace(-1, 1, 10000)[:, None], (1, 32)).astype(np.float32)
    meta = {"source_commit": commit, "checkpoint_sha256": "a"*64, "loader": "signed",
            "subset_sha256": module.SUBSET_SHA256, "split_sha256": module.INPUT_SHA256["splits"],
            "manifest_sha256": module.INPUT_SHA256["manifest"], "which": "teacher",
            "seed": 0, "windows_per_subject": 100, "target_version": "target", "readout_version": "readout"}
    for model_seed in (0, 1):
        for arm in ("A", "B"):
            name = f"{arm}_seed{model_seed}"
            np.savez_compressed(tmp_path/f"{name}.npz", X=np.zeros((10000, 768), np.float32), y=y,
                                valid=np.ones_like(y, bool), subject=subject, split=split,
                                metadata_json=np.array(json.dumps(meta)))
            target = y[split == "test", 0].astype(np.float64)
            names = subject[split == "test"]
            error = .02 if arm == "A" else .005
            stats = {}
            for s in sorted(set(names)):
                t = target[names == s]
                stats[s] = {"n": len(t), "target_mean": float(t.mean()),
                            "target_m2": float(np.square(t-t.mean()).sum()), "sse": len(t)*error**2}
            report = {"coordinates": [{"coordinate": j, "status": "ok", "r2": r2(target, target+error),
                                       "per_subject": stats} for j in range(32)]}
            (tmp_path/f"{name}.json").write_text(json.dumps({"metadata": meta,
                "features_sha256": module.sha256_file(tmp_path/f"{name}.npz"),
                "views": {v: report for v in ("backbone", "local", "p1p2")}}))
    args = SimpleNamespace(expected_commit=commit, out=tmp_path/"comparison.json",
                           runs_root=tmp_path, eval_root=tmp_path)
    assert module.compare(args) == 0
    result = json.loads(args.out.read_text())
    assert result["primary_view"] == "p1p2" and result["decision"] == "promising_run_fresh_P1_P2"
    assert len(result["artifacts"]) == 4
    # Metadata drift must fail before reporting another apparent success.
    bad = json.loads((tmp_path/"B_seed1.json").read_text()); bad["metadata"]["loader"] = "other"
    (tmp_path/"B_seed1.json").write_text(json.dumps(bad))
    args.out = tmp_path/"rejected.json"
    with pytest.raises(ValueError, match="identity mismatch"):
        module.compare(args)
    assert not args.out.exists()
