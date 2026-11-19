"""Operational regressions from Claude's review, exercising real CPU paths."""
from contextlib import redirect_stdout
from datetime import datetime
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from pose_temponet import benchmark as B, controls as C, execution as E, heads, linear
from pose_temponet import provenance as P, runner as R, train as T, workflow as W
from test_pose_temponet_benchmark import fixture as cache_fixture
from test_pose_temponet_controls import _builder_fixture
from test_pose_temponet_provenance import prepared
from test_pose_temponet_runner import runtime, initial_chain
from test_pose_temponet_train import dataset, factory, evaluator_for


def test_progress_records_unfinished_curve_and_initial_hashes(capsys):
    fitting = [dataset(7, seed=10), dataset(7, seed=11)]
    recipe = T._TestRecipe(initial_epochs=2, maximum_epochs=3)
    trajectories = [T.Trajectory(factory, d, lr=.001, seed=0, fold=f, _recipe=recipe)
                    for f, d in enumerate(fitting)]
    pair = T.PairedCandidate(trajectories, evaluator_for(fitting))
    pair.advance(2, on_checkpoint=R._progress)
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["point"]["exposure"] == 0
    assert records[0]["initial_sha256_per_fold"] == {str(t.fold): t.initial_sha256 for t in trajectories}
    assert records[-1]["point"]["exposure"] == 2
    assert any(0 < r["point"]["exposure"] < 1 for r in records)


def test_refits_print_initial_hash_before_any_update(monkeypatch):
    fitting = [dataset(7, seed=10), dataset(7, seed=11)]
    recipe = T._TestRecipe(initial_epochs=1, maximum_epochs=2)
    selected = T._run_search(factory, fitting, evaluator_for(fitting), device="cpu", recipe=recipe)["selection"]
    output = io.StringIO()
    original = T.Trajectory.train_to_step
    observed = []
    def checked(self, target):
        latest = json.loads(output.getvalue().splitlines()[-1])
        assert latest == {"event": "refit_initialization", "seed": self.seed, "fold": 2,
                          "initial_sha256": self.initial_sha256, "updates": 0}
        observed.append(self.seed)
        return original(self, target)
    monkeypatch.setattr(T.Trajectory, "train_to_step", checked)
    with redirect_stdout(output):
        T._run_refits(factory, dataset(14), selected, device="cpu", recipe=recipe)
    assert observed == [0, 1, 2]


def test_nondiagnostic_outer_gate_is_saved_before_neural_fit(runtime, monkeypatch):
    repo, plan, opened, _ = initial_chain(runtime)
    for fold in (0, 1):
        R.run_stage(plan, E.job_id("wls_calibration", fold=fold), source_root=repo)
    gate = {"status": "control_non_diagnostic", "linear_gate": {"status": "diagnostic"},
            "temporal_gate": {"status": "control_non_diagnostic"}}
    monkeypatch.setattr(C, "combine_reference_folds", lambda rows: gate)
    monkeypatch.setattr(W, "run_selection", lambda *a, **k: pytest.fail("neural selection must not start"))
    identifier = E.job_id("controls", head="L0", world="linear")
    with pytest.raises(R.GateNotPassed) as raised:
        R.run_stage(plan, identifier, source_root=repo)
    assert raised.value.status == "control_non_diagnostic"
    receipt = P.read_json_pin(P.file_pin(P.receipt_path(plan, identifier)))
    assert receipt["status"] == "gate_failed" and not receipt["complete"]
    assert receipt["details"]["gate_status"] == "control_non_diagnostic"
    assert P.read_json_pin(receipt["artifacts"]["combined_gate"]) == gate
    assert P.read_json_pin(receipt["artifacts"]["failed_gate"]) == gate
    assert datetime.fromisoformat(receipt["details"]["finished_at"]).utcoffset().total_seconds() == 0


@pytest.mark.parametrize("kind", ["control_non_diagnostic", "degenerate"])
def test_calibration_gate_status_survives_failure_receipt(runtime, monkeypatch, kind):
    repo, plan, _, _ = initial_chain(runtime)
    def calibrate(*args, **kwargs):
        if kind == "degenerate":
            C.normalize_component(np.zeros((2, 16, 2)), np.ones((2, 4, 2), bool), [0])
        return {"outer_fold": 0, "components": {}, "calibration": {"status": kind}}
    monkeypatch.setattr(C, "build_positive_control_fold", calibrate)
    identifier = E.job_id("wls_calibration", fold=0)
    with pytest.raises(C.GateNotPassed):
        R.run_stage(plan, identifier, source_root=repo)
    receipt = P.read_json_pin(P.file_pin(P.receipt_path(plan, identifier)))
    expected = "control_non_calibrated" if kind == "degenerate" else kind
    assert receipt["status"] == "gate_failed" and receipt["details"]["gate_status"] == expected
    assert P.read_json_pin(receipt["artifacts"]["failed_gate"])["status"] == expected


def test_temporal_margin_fits_clean_target_and_scores_noisy_target():
    data, components = _builder_fixture()
    fit, held = np.arange(0, 8, 2), np.arange(1, 8, 2)
    selected = {"linear": {"a": 0., "sigma": 1., "alpha": .1},
                "temporal": {"a": .4, "sigma": 1., "alpha": .1}}
    values = C.evaluate_outer_references(data, components, fit, held, selected, block_windows=2)
    diagnostics = values["linear_diagnostics"]
    assert abs(diagnostics["L0"]["temporal_margin_recovery_fitting"]) < 1e-12
    for name in ("L0", "L13"):
        row = diagnostics[name]
        assert row["temporal_margin_fit_target"] == "clean_y_star"
        assert row["temporal_margin_score_target"] == "observed_y"
        # Fitting y instead would absorb realized noise, changing the diagnostic.
        assert row["temporal_margin_clean_fit_statistics"] != row["worlds"]["temporal"]["fitting"]["statistics"]


def test_first_load_measurement_reused_without_second_loader():
    data = cache_fixture()
    loaded, evidence = B._load_measurement(lambda: data, {"source": "pin"})
    actual = B._input_io(loaded, lambda: pytest.fail("must not reread"), {"source": "pin"}, evidence)
    assert actual["load_scope"] == "first_application_load" and not actual["os_page_cache_controlled"]
    bad = dict(evidence, population_sha256="0"*64)
    with pytest.raises(ValueError, match="first-load"):
        B._input_io(data, None, {"source": "pin"}, bad)


def test_benchmark_score_runs_production_cast_mask_finite_and_metric_path(monkeypatch):
    data = cache_fixture(n=80, time=160, features=384, joints=20)
    data["meta"][0]["n_patches"] = 150
    data["lat"][0, 150:] = np.nan
    data["mask"][0, 150:] = False
    folds = (np.arange(40), np.arange(40, 80))
    bundle = B._scoring_bundle(data, folds)
    model = heads.make_head("L0", seed=0)
    original = W._score_model
    calls = []
    def spy(*args, **kwargs):
        calls.append((args[2], tuple(args[3]), args[4]))
        return original(*args, **kwargs)
    monkeypatch.setattr(W, "_score_model", spy)
    actual = B._production_score(model, bundle, 0, folds[0], "post", "cpu")
    expected = original(model, bundle, 0, folds[0], "post", "cpu")
    assert actual == expected and calls == [(0, tuple(folds[0]), "post")]
    data["lat"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="latent context"):
        B._production_score(model, bundle, 0, folds[0], "post", "cpu")


def test_benchmark_measures_real_artifacts_separately_from_memory_snapshots(monkeypatch, tmp_path):
    data = cache_fixture(n=80, time=160, features=384, joints=20)
    identity = {"source": "pin"}
    _, measured = B._load_measurement(lambda: data, identity)
    original_trajectory, original_score, original_batch = T.Trajectory, B._production_score, B._prediction_batch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(T, "configure_numerics", lambda: {"test_only": True})
    monkeypatch.setattr(B, "_hardware", lambda: {"test_only": True})
    monkeypatch.setattr(B, "_validation_surrogate", lambda d: (0, 0))
    monkeypatch.setattr(heads, "HEADS", ("L0",))
    monkeypatch.setattr(T, "Trajectory", lambda *a, **k: original_trajectory(*a, **{**k, "device": "cpu"}))
    scores = []
    def score(*args, **kwargs):
        scores.append(kwargs["include_clean"])
        return original_score(*args, **{**kwargs, "device": "cpu"})
    monkeypatch.setattr(B, "_production_score", score)
    monkeypatch.setattr(B, "_prediction_batch", lambda *a, **k: original_batch(*a, **{**k, "device": "cpu"}))
    monkeypatch.setattr(T, "save_checkpoint", lambda *a, **k: pytest.fail("not the production artifact writer"))
    proxy = tmp_path/"source.npz"; proxy.write_bytes(b"cost proxy")
    result = B.benchmark_gpu(data, input_loader=lambda: pytest.fail("first load already measured"),
                             input_measurement=measured, output_dir=tmp_path/"cost", reference_byte_count=10,
                             reference_proxy_path=proxy, allowed_train_users={m["user"] for m in data["meta"]},
                             input_identity=identity)
    assert len(scores) == 33 and all(scores)
    for cell in result["cells"]["L0"].values():
        assert cell["evaluation_path"] == "workflow._score_model" and cell["dataset_identity_warmed"]
        assert cell["save_scope"] == "in_memory_state_capture"
        assert cell["load_state_scope"] == "in_memory_restore"
        assert cell["artifact_write_bytes"] > 0
        for key in ("save", "load_state", "artifact_write", "artifact_read"):
            assert len(cell[key+"_seconds"]) == 10 and min(cell[key+"_seconds"]) > 0
    assert len(list((tmp_path/"cost").glob("*.pt"))) == 33
