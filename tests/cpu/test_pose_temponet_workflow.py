"""End-to-end train-only selection adapters and the complete-matrix held barrier."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from pose_temponet import controls as C, data as D, metrics as M, train as T, workflow as W


class SmallHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(20))
        self.bias = nn.Parameter(torch.zeros(20))

    def forward(self, x):
        return x[..., :20] * self.scale + self.bias


def factory(seed, fold):
    return SmallHead()


def cache(users=("u3", "u0", "u2", "u1"), windows=2):
    rng = np.random.default_rng(41)
    n = len(users) * windows
    x = rng.normal(size=(n, 160, 384)).astype(np.float16)
    target = x[..., :20].astype(np.float32) * np.linspace(.6, 1.5, n)[:, None, None]
    mask = np.ones(target.shape, dtype=bool)
    mask[0, 70:90, 4:] = False
    return {"lat": x, "pre": x * np.float16(.7), "target": target.astype(np.float32),
            "mask": mask, "std": np.ones((n, 20), np.float32),
            # Deliberately false: internal adapters must recompute from real/y* targets.
            "pose_var": np.full(n, 999., dtype=np.float32),
            "key_time_valid": np.ones((n, 160), bool),
            "config": {"split": "train", "which": "teacher", "checkpoint_step": 49046},
            "meta": [{"user": user, "recording": i, "start": 0, "n_patches": 160}
                     for i, user in enumerate(np.repeat(users, windows))]}


def bundle(c=None):
    c = cache() if c is None else c
    return W._prepare_real_folds(c, allowed_train_users=sorted({m["user"] for m in c["meta"]}),
                                 source_identity={"cache_sha256": "a" * 64, "protocol_sha256": "b" * 64},
                                 expected_user_count=4)


def recipe():
    return T._TestRecipe(initial_epochs=1, maximum_epochs=2, last_epochs=1, batch_size=4)


def pass_gate(b):
    # Fixtures isolate workflow, not the separately tested calibration mathematics.
    return W._lock({"status": "passed", "complete": True, "recipe": W._json(recipe().__dict__),
                    **{k: b.manifest[k] for k in ("population_sha256", "source_identity", "user_folds")}}, "gate_sha256")


def relock(report, **changes):
    report = copy.deepcopy(report)
    report.pop("selection_sha256")
    report.update(changes)
    return W._lock(report, "selection_sha256")


def stats(mse, edyn, count=4, ss=40.):
    return {"mse_sum": mse*count, "window_count": count, "dynamic_sse": edyn*ss, "dynamic_target_ss": ss}


def artifacts(b):
    output = []
    for fold in (0, 1):
        y = b.cache["target"].astype(np.float64)
        components = {"L": y * .5, "T": np.roll(y, 1, axis=1) * .25,
                      "epsilon": np.random.default_rng(fold).normal(size=y.shape) * 4}
        thresholds = {}
        for world, a in (("linear", 0.), ("temporal", .3)):
            clean, _ = C.synthetic_target(components, a, .8)
            thresholds[world + "_threshold"] = M.training_threshold(clean[b.fitting[fold]], b.cache["mask"][b.fitting[fold]])
        output.append({"outer_fold": fold,
                       "partition": C.validate_outer_partition(b.cache["meta"], b.fitting[fold], b.held[fold], b.manifest["users"]),
                       "components": components,
                       "calibration": {"status": "calibrated", "selected": {
                           "linear": {"a": 0., "sigma": .8}, "temporal": {"a": .3, "sigma": .8}}},
                       "outer_scored_after_inner_selection": True,
                       "outer_reference_statistics": {**thresholds,
                           "linear": {"constant": stats(1., 1.), "ridge": stats(.2, .2),
                                      "oracle": stats(.18, .18), "clean_ls": stats(.18, .18)},
                           "temporal": {"constant": stats(2., 2.), "clean_ls": stats(1., 1.), "oracle": stats(.92, .92)}}})
    return output


def control_bundle(b, aa, world):
    return W._prepare_control_folds(b.cache, aa, world, allowed_train_users=b.manifest["users"],
                                    source_identity=b.manifest["source_identity"], expected_user_count=4)


def test_partition_uses_users_and_real_fit_only_thresholds_and_rejects_held():
    b = bundle()
    assert b.manifest["user_folds"] == [["u0", "u2"], ["u1", "u3"]]
    for fold in (0, 1):
        fit, held = b.fitting[fold], b.held[fold]
        assert not set(fit) & set(held)
        assert b.thresholds[fold] == M.training_threshold(b.cache["target"][fit], b.cache["mask"][fit])
        assert not np.any(b.pose_variances[fold] == 999.)
    changed = copy.deepcopy(b.cache)
    changed["target"][b.held[0]] *= 20
    b2 = bundle(changed)
    assert b2.thresholds[0] == b.thresholds[0]
    assert b2.thresholds[1] != b.thresholds[1]
    changed["config"]["split"] = "val"
    with pytest.raises(ValueError, match="train cache"):
        bundle(changed)
    with pytest.raises(ValueError, match="allowed users"):
        W.prepare_real_folds(b.cache, allowed_train_users=b.manifest["users"], source_identity={"pin": "x"})


def test_synthetic_control_scores_noisy_y_but_selects_dynamics_from_clean_y_star():
    b = bundle()
    aa = artifacts(b)
    c = control_bundle(b, aa, "temporal")
    assert c.reference_gate["status"] == "diagnostic"
    fit, held = c.fitting[0], c.held[0]
    clean, noisy = C.synthetic_target(aa[0]["components"], .3, .8)
    np.testing.assert_array_equal(c.targets[0], noisy)
    np.testing.assert_array_equal(c.clean_targets[0], clean)
    np.testing.assert_array_equal(c.pose_variances[0], M.dynamic_variance(clean, b.cache["mask"]))
    panel = W._metric_panel(c, 0, held)
    selected = panel["pose_var"] >= c.thresholds[0]
    assert not np.array_equal(selected, M.dynamic_variance(noisy[held], panel["mask"]) >= c.thresholds[0])
    prediction = clean[held, 6:-6]
    observed = M.selection_statistics(prediction, panel, c.thresholds[0])
    clean_score = M.selection_statistics(prediction, W._metric_panel(c, 0, held, clean=True), c.thresholds[0])
    assert observed["dynamic_sse"] > 0 and clean_score["dynamic_sse"] == 0
    aa[0]["partition"]["held_users"] = ["intruder"]
    with pytest.raises(ValueError, match="another fold"):
        control_bundle(b, aa, "linear")


def test_real_fit_cannot_start_before_global_control_gate_or_after_input_mutation():
    b = bundle()
    with pytest.raises(ValueError, match="positive controls"):
        W._run_selection(b, "L0", recipe=recipe(), factory=factory)
    gate = pass_gate(b)
    b.cache["target"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="changed after preparation"):
        W._run_selection(b, "L0", control_gate=gate, recipe=recipe(), factory=factory)


@pytest.fixture(scope="module")
def real_result():
    torch.set_num_threads(1)
    b = bundle()
    gate = pass_gate(b)
    result = W._run_selection(b, "L0", control_gate=gate, recipe=recipe(), factory=factory)
    return b, gate, result


def test_search_selected_outer_statistics_are_actual_paired_mse_checkpoint(real_result):
    b, gate, result = real_result
    report = result["selection"]
    W._check_selection(json.loads(json.dumps(report)))
    selected = report["selected"]
    pair = result["states"]["candidate_states"][selected["lr"]]
    for fold in (0, 1):
        model = factory(0, fold)
        model.load_state_dict(pair["best_states"][fold]["model"])
        with torch.no_grad():
            prediction = model(torch.from_numpy(D.view_inputs(b.cache, "post", b.held[fold]))).numpy()[:, 6:-6]
        expected = M.selection_statistics(prediction, W._metric_panel(b, fold, b.held[fold]), b.thresholds[fold])
        assert report["selected_outer_statistics"][f"fold{fold}"] == expected
    assert report["selected_outer_values"]["pooled"]["MSE"] == selected["mse"]
    assert report["selected_outer_values"]["pooled"]["E_dyn"] == selected["edyn"]
    assert report["positive_gate_sha256"] == gate["gate_sha256"]
    assert len(report["curves"]) >= 4 and all(c["points"][0]["exposure"] == 0 for c in report["curves"])


def control_selections(real_result):
    b, _, result = real_result
    aa = artifacts(b)
    reports = {}
    for world in W.WORLDS:
        cb = control_bundle(b, aa, world)
        reports[world] = {}
        for head in W.CONTROL_HEADS:
            value = .22 if world == "linear" else .94
            panel = {"fold0": stats(value, value), "fold1": stats(value, value)}
            panel["pooled"] = C.pool_statistics(list(panel.values()))
            reports[world][head] = relock(result["selection"], head=head, world=world,
                                          bundle=cb.manifest, selected_outer_statistics=panel,
                                          optimizer_alarms=[], optimization_unresolved=False)
    return aa, reports


def test_controls_require_all_heads_both_folds_and_pool_but_s2_and_temporal_mlp_do_not_veto(real_result):
    aa, reports = control_selections(real_result)
    reports["temporal"]["MLP"] = relock(reports["temporal"]["MLP"], optimizer_alarms=["upper_edge"],
                                          objective_tradeoff={"flag": True})
    gate = W.qualify_controls(aa, reports)
    assert gate["status"] == "passed"
    assert gate["temporal_L0_MLP_descriptive_only"]
    changed = copy.deepcopy(reports)
    del changed["linear"]["MLP"]
    with pytest.raises(ValueError, match="six positive"):
        W.qualify_controls(aa, changed)
    bad = copy.deepcopy(reports["linear"]["L0"]["selected_outer_statistics"])
    bad["fold0"] = stats(.6, .6)
    bad["pooled"] = C.pool_statistics([bad["fold0"], bad["fold1"]])
    reports["linear"]["L0"] = relock(reports["linear"]["L0"], selected_outer_statistics=bad,
                                      optimizer_alarms=["late_minimum"])
    assert W.qualify_controls(aa, reports)["status"] == "optimization_unresolved"
    reports["linear"]["L0"] = relock(reports["linear"]["L0"], optimizer_alarms=[])
    assert W.qualify_controls(aa, reports)["status"] == "recipe_not_admitted"


def test_json_selection_to_refits_and_lock_all_fifteen_before_batched_inference(real_result, monkeypatch):
    b, gate, result = real_result
    selections, refits = {}, {}
    for head in W.HEADS:
        selected = relock(result["selection"], head=head,
                          feature_array_sha256=W._array_hash(W._features(b.cache, W._feature_depth(head))))
        selected = json.loads(json.dumps(selected))
        selections[head] = selected
        refits[head] = W._run_refits(b, selected, gate, recipe=recipe(), factory=factory)
        assert set(refits[head]["states"]["refit_states"]) == {0, 1, 2}
        assert set(refits[head]["manifest"]["fitting_diagnostics"]) == {"0", "1", "2"}
    dev_pin = {"cache_sha256": "c" * 64}
    lock = W._lock_inference_manifest(selections, refits, development_identity=dev_pin, production=False)
    assert len(lock["cells"]) == 15
    with pytest.raises(ValueError, match="test recipe"):
        W.lock_inference_manifest(selections, refits, development_identity=dev_pin)
    incomplete = dict(refits)
    incomplete.pop("TCN_pre")
    with pytest.raises(ValueError, match="all five"):
        W._lock_inference_manifest(selections, incomplete, development_identity=dev_pin, production=False)
    broken = copy.deepcopy(refits)
    broken["L0"]["states"]["refit_states"][2]["model"]["scale"][0] += 1
    with pytest.raises(ValueError, match="payload"):
        W._lock_inference_manifest(selections, broken, development_identity=dev_pin, production=False)
    dev = cache(users=("v0", "v1", "v2"), windows=13)  # 39 rows: one full batch and seven.
    dev["config"]["split"] = "val"
    state = refits["L0"]["states"]["refit_states"][0]
    calls = []
    original = D.view_inputs
    def observed(data, depth="post", indices=None):
        assert indices is not None and len(indices) <= 32
        calls.append(len(indices))
        return original(data, depth, indices)
    monkeypatch.setattr(D, "view_inputs", observed)
    got = W._predict_locked(dev, "L0", 0, state, lock, development_identity=dev_pin,
                            production=False, factory=factory)
    assert calls == [32, 7]
    assert got["prediction"].shape == (39, 148, 20) and got["prediction"].dtype == np.float32
    assert got["receipt"]["complete"]
    bad_lock = copy.deepcopy(lock)
    bad_lock.pop("inference_lock_sha256")
    bad_lock["cells"].pop("TCN_pre/2")
    bad_lock = W._lock(bad_lock, "inference_lock_sha256")
    calls.clear()
    with pytest.raises(ValueError, match="all fifteen"):
        W._predict_locked(dev, "L0", 0, state, bad_lock, development_identity=dev_pin,
                           production=False, factory=factory)
    assert calls == []
    dev["meta"][0]["user"] = b.manifest["users"][0]
    with pytest.raises(ValueError, match="overlaps training"):
        W._predict_locked(dev, "L0", 0, state, lock, development_identity=dev_pin,
                           production=False, factory=factory)


def test_refit_rejects_changed_features_even_with_same_targets(real_result):
    b, gate, result = real_result
    changed = copy.deepcopy(b.cache)
    changed["lat"][0, 0, 0] += 1
    b2 = bundle(changed)
    with pytest.raises(ValueError, match="features differ"):
        W._run_refits(b2, result["selection"], gate, recipe=recipe(), factory=factory)


def test_batch_statistics_match_one_shot_with_masked_windows_and_unequal_user_support():
    c = cache(windows=20)
    c["mask"][0] = False
    b = bundle(c)
    model = SmallHead()
    with torch.no_grad():
        model.scale.copy_(torch.linspace(.1, 1., 20))
    fold, indices = 1, b.held[1]
    assert len(indices) == 40
    got = W._score_model(model, b, fold, indices, "post", "cpu")
    with torch.no_grad():
        prediction = model(torch.from_numpy(D.view_inputs(c, "post", indices))).numpy()[:, 6:-6]
    expected = M.selection_statistics(prediction, W._metric_panel(b, fold, indices), b.thresholds[fold])
    assert got["window_count"] == expected["window_count"]
    for key in ("mse_sum", "dynamic_sse", "dynamic_target_ss"):
        assert got[key] == pytest.approx(expected[key], rel=1e-14)


def test_public_preparation_has_fixed_forty_users_and_detects_wrong_checkpoint():
    c = cache(users=tuple(f"u{i:02}" for i in range(40)), windows=1)
    got = W.prepare_real_folds(c, allowed_train_users=sorted({m["user"] for m in c["meta"]}),
                               source_identity={"cache_sha256": "d" * 64})
    assert [len(x) for x in got.manifest["user_folds"]] == [20, 20]
    c["config"]["checkpoint_step"] = 19618
    with pytest.raises(ValueError, match="49046"):
        W.prepare_real_folds(c, allowed_train_users=got.manifest["users"], source_identity={"pin": "wrong"})


def test_fp16_host_storage_preserves_fp32_trajectory_predictions_and_gradients():
    c = cache()
    c["meta"][0]["n_patches"] = 100
    c["key_time_valid"][0, 100:] = False
    c["mask"][0, 100:] = False
    c["lat"][0, 100:] = np.nan
    b = bundle(c)
    indices = np.arange(len(c["meta"]))
    compact = W._training_set(b, indices, "post", 0)
    assert compact.features.dtype == torch.float16
    assert torch.isfinite(compact.features).all()
    assert torch.count_nonzero(compact.features[0, 100:]) == 0
    assert np.isnan(c["lat"][0, 100:]).all()  # Padding cleanup must not mutate the verified cache.
    previous = T.TensorTrainingSet(torch.from_numpy(D.view_inputs(c, "post", indices)),
                                    compact.target, compact.support, compact.window_ids)
    assert previous.features.dtype == torch.float32
    actual_factory = lambda seed, fold: W.heads.make_head("TCN_post", seed=seed)
    trajectories = [T.Trajectory(actual_factory, dataset, lr=.003, seed=0, fold=0,
                                  _recipe=recipe()) for dataset in (compact, previous)]
    for trajectory in trajectories:
        trajectory.train_to_step(3)
    left, right = trajectories
    assert T.payload_sha256(left.model.state_dict()) == T.payload_sha256(right.model.state_dict())
    assert T.payload_sha256(left.optimizer.state_dict()) == T.payload_sha256(right.optimizer.state_dict())
    assert left.epoch_losses == right.epoch_losses
    for a, z in zip(left.model.parameters(), right.model.parameters()):
        assert torch.equal(a.grad, z.grad)
    batch, _, _ = compact.batch(torch.arange(compact.size), torch.device("cpu"))
    assert batch.dtype == torch.float32
    left.model.eval(); right.model.eval()
    with torch.no_grad():
        assert torch.equal(left.model(batch), right.model(batch))
