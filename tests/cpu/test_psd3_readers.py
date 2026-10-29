"""Isolation, objective and reproducibility of the new PSD3 reader experiment."""

from dataclasses import FrozenInstanceError, replace
import inspect
import json
import warnings

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import functional as F

from wearusfm.harness import psd3_readers as readers


def _data():
    rng = np.random.default_rng(21)
    train = rng.normal(size=(24, 3)).astype(np.float32)
    val = rng.normal(size=(12, 3)).astype(np.float32)
    held = rng.normal(size=(9, 3)).astype(np.float32)
    labels = lambda x: np.where(x[:, 0] > 0, "open", "closed")
    return train, labels(train), val, labels(val), held


def _tiny_mlp(**changes):
    return replace(readers.MLPConfig(feature_dim=3, hidden_dim=4, epochs=2, batch_size=7), **changes)


def test_preprocessing_is_train_only_and_matches_the_frozen_transform():
    rng = np.random.default_rng(71)
    xt, xv, xh = (rng.normal(size=(n, 7)) for n in (20, 8, 9))
    config = readers.FeatureConfig(k=3)
    result = readers.prepare_features(xt, xv, xh, config=config)
    changed = readers.prepare_features(xt, xv * 100 + 500, xh * -1000, config=config)
    assert np.array_equal(result.train, changed.train)
    assert result.metadata["transform_fingerprint"] == changed.metadata["transform_fingerprint"]
    assert not np.array_equal(result.val, changed.val)
    scaler = StandardScaler().fit(xt)
    pca = PCA(n_components=3, svd_solver="randomized", whiten=False, iterated_power=7,
              n_oversamples=10, random_state=20261029).fit(scaler.transform(xt))
    for actual, original in zip((result.train, result.val, result.held), (xt, xv, xh)):
        assert np.array_equal(actual, pca.transform(scaler.transform(original)).astype(np.float32))
        assert actual.dtype == np.float32 and actual.shape == (len(original), 3)
    assert result.metadata["variance_explained"] == pytest.approx(pca.explained_variance_ratio_.sum())
    assert result.metadata["second_scaler"] is False
    json.dumps(result.metadata, allow_nan=False)


@pytest.mark.parametrize("n,width,k", [(4, 8, 4), (3, 8, 4), (8, 3, 4)])
def test_preprocessing_refuses_to_cap_requested_dimension(n, width, k):
    with pytest.raises(ValueError, match="no automatic cap"):
        readers.prepare_features(np.ones((n, width)), np.ones((3, width)), np.ones((2, width)),
                                 config=readers.FeatureConfig(k=k))


@pytest.mark.parametrize("config", [
    lambda: readers.FeatureConfig(k=0), lambda: readers.FeatureConfig(seed=-1),
    lambda: readers.LogisticConfig(c_grid=(1., 1.)), lambda: readers.LogisticConfig(tol=np.nan),
    lambda: readers.LogisticConfig(max_iter=True), lambda: readers.MLPConfig(epochs=0),
    lambda: readers.MLPConfig(lambdas=(-.1,)), lambda: readers.MLPConfig(seeds=(0, 0)),
    lambda: readers.MLPConfig(lr=float("inf")), lambda: readers.MLPConfig(seeds=[0, 1]),
])
def test_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        config()


def test_configuration_is_immutable():
    with pytest.raises(FrozenInstanceError):
        readers.MLPConfig().epochs = 2


def test_numpy_configuration_scalars_remain_json_serializable():
    config = _tiny_mlp(epochs=np.int64(1), lr=np.float32(.001),
                       lambdas=(np.float32(0),), seeds=(np.int64(0),))
    result = readers.fit_mlp(*_data(), config=config)
    json.dumps(result, allow_nan=False)


def test_logistic_seven_train_only_fits_tie_break_and_fit_warning_attribution(monkeypatch):
    xt, yt, xv, yv, xh = _data()
    calls = []

    class FakeLogistic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_params(self):
            return self.kwargs

        def fit(self, x, y):
            calls.append((x.copy(), y.copy(), self.kwargs))
            self.coef_, self.intercept_ = np.zeros((1, x.shape[1])), np.zeros(1)
            self.n_iter_ = np.array([self.kwargs["max_iter"]])
            warnings.warn("iteration budget exhausted", ConvergenceWarning)
            return self

        def predict(self, x):
            return np.zeros(len(x), dtype=int)

    monkeypatch.setattr(readers, "LogisticRegression", FakeLogistic)
    config = readers.LogisticConfig(feature_dim=3)
    result = readers.fit_logistic(xt, yt, xv, yv, xh, config=config)
    assert len(calls) == len(result["audit"]) == result["expected_fits"] == 7
    assert [a["phase"] for a in result["audit"]] == ["candidate"] * 6 + ["refit"]
    assert result["selected_hyperparameter"] == 1e-3
    for (x, y, kwargs), audit in zip(calls, result["audit"]):
        assert np.array_equal(x, xt) and len(y) == len(yt)
        assert kwargs["max_iter"] == 12000 and kwargs["tol"] == 1e-4
        assert kwargs["class_weight"] == "balanced" and kwargs["warm_start"] is False
        assert kwargs["solver"] == "lbfgs" and kwargs["penalty"] == "l2"
        assert audit["warnings"] == [{"category": "ConvergenceWarning", "message": "iteration budget exhausted"}]
        assert audit["reached_iteration_limit"] is True
    assert set(result["predictions"]["0"]) == {"closed"}
    assert not result["metric_warnings"]
    json.dumps(result, allow_nan=False)


def test_logistic_exception_is_not_a_completed_report(monkeypatch):
    def fail_fit(self, x, y):
        warnings.warn("before failure", ConvergenceWarning)
        raise RuntimeError("synthetic fit failure")

    monkeypatch.setattr(readers.LogisticRegression, "fit", fail_fit)
    with pytest.raises(RuntimeError, match="synthetic fit failure") as caught:
        readers.fit_logistic(*_data(), config=readers.LogisticConfig(feature_dim=3))
    audit = caught.value.psd3_audit
    assert len(audit) == 1 and audit[0]["status"] == "failed"
    assert audit[0]["warnings"][0]["category"] == "ConvergenceWarning"
    assert "synthetic fit failure" in audit[0]["exception"]


@pytest.mark.parametrize("kind", ["unicode", "bytes", "numeric"])
def test_real_logistic_supports_labels_and_held_features_do_not_select_c(kind):
    xt, yt, xv, yv, xh = _data()
    if kind == "bytes":
        yt, yv = yt.astype("S"), yv.astype("S")
    elif kind == "numeric":
        yt, yv = (yt == "open").astype(int), (yv == "open").astype(int)
    config = readers.LogisticConfig(feature_dim=3, c_grid=(.01, 1.0), max_iter=500)
    original = readers.fit_logistic(xt, yt, xv, yv, xh, config=config)
    changed = readers.fit_logistic(xt, yt, xv, yv, -xh * 1000, config=config)
    assert original["selected_hyperparameter"] == changed["selected_hyperparameter"]
    assert original["candidates"] == changed["candidates"]
    assert original["scores"] == changed["scores"]
    assert "y_held" not in inspect.signature(readers.fit_logistic).parameters
    assert set(original["predictions"]["0"]) <= set(original["classes"])
    json.dumps(original, allow_nan=False)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_features_rejected_for_every_role(bad):
    data = list(_data())
    for index in (0, 2, 4):
        changed = list(data)
        changed[index] = data[index].copy()
        changed[index][0, 0] = bad
        with pytest.raises(ValueError, match="nonfinite"):
            readers.fit_mlp(*changed, config=_tiny_mlp())
        with pytest.raises(ValueError, match="nonfinite"):
            readers.fit_logistic(*changed, config=readers.LogisticConfig(feature_dim=3))


@pytest.mark.parametrize("bad", [np.array([" "] * 24), np.zeros(24, dtype=complex),
                                  np.full(24, np.nan), np.array(["a"] * 24, dtype=object)])
def test_invalid_labels_are_rejected(bad):
    xt, _, xv, yv, xh = _data()
    with pytest.raises(ValueError):
        readers.fit_logistic(xt, bad, xv, yv, xh, config=readers.LogisticConfig(feature_dim=3))


def test_mlp_objective_uses_train_class_weights_batch_mean_and_no_bias_penalty():
    labels = np.array([0, 0, 0, 1])
    weights = readers._class_weights(labels, 2)
    assert weights.tolist() == pytest.approx([2 / 3, 2.0])
    model = readers._MLP(3, 4, 2, 0)
    with torch.no_grad():
        model.weight1.fill_(1)
        model.weight2.fill_(2)
        model.bias1.fill_(100)
        model.bias2.fill_(-200)
    logits = torch.tensor([[1., -1.], [-1., 1.], [0., 0.]])
    target = torch.tensor([0, 1, 1])
    tw = torch.tensor(weights, dtype=torch.float32)
    objective, ce, penalty = readers._objective(model, logits, target, tw, .1)
    manual_ce = (-F.log_softmax(logits, dim=1)[torch.arange(3), target] * tw[target]).sum() / 3
    assert ce.item() == pytest.approx(manual_ce.item())
    assert penalty.item() == pytest.approx(.5 * .1 * (12 + 8 * 4))
    assert objective.item() == pytest.approx(ce.item() + penalty.item())
    assert ce.item() != pytest.approx(F.cross_entropy(logits, target, weight=tw).item())


def test_mlp_tie_prefers_strongest_lambda_and_only_selected_models_predict_held(monkeypatch):
    xt, yt, xv, yv, xh = _data()
    config = _tiny_mlp()
    created, predictions = [], []
    original_init, original_predict = readers._MLP.__init__, readers._predict_mlp

    def spy_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    def spy_predict(model, x, batch_size):
        predictions.append((created.index(model), len(x), len(created)))
        return original_predict(model, x, batch_size)

    monkeypatch.setattr(readers._MLP, "__init__", spy_init)
    monkeypatch.setattr(readers, "_predict_mlp", spy_predict)
    monkeypatch.setattr(readers, "_score", lambda *args: .5)
    result = readers.fit_mlp(xt, yt, xv, yv, xh, config=config)
    assert result["selected_hyperparameter"] == .01
    assert len(created) == len(result["audit"]) == result["expected_fits"] == 6
    assert [p for p in predictions if p[1] == len(xh)] == [(4, len(xh), 6), (5, len(xh), 6)]
    assert result["prediction_policy"] == "separate_seeds_no_ensemble_no_refit"
    assert set(result["predictions"]) == {"0", "1"}
    for seed in config.seeds:
        rows = [a for a in result["audit"] if a["seed"] == seed]
        assert len({r["initialization_fingerprint"] for r in rows}) == 1
        assert len({r["permutation_fingerprint"] for r in rows}) == 1
        assert all(r["parameter_count"] == 3 * 4 + 4 + 4 * 2 + 2 for r in rows)
        assert all(len(r["epochs"]) == config.epochs for r in rows)
        assert all(r["stopping_reason"] == "fixed_epoch_budget" for r in rows)
    assert "y_held" not in inspect.signature(readers.fit_mlp).parameters
    json.dumps(result, allow_nan=False)


def test_mlp_determinism_held_isolation_and_no_global_rng_mutation():
    data = _data()
    numpy_before, torch_before = np.random.get_state(), torch.random.get_rng_state().clone()
    config = _tiny_mlp(lambdas=(0., .01))
    readers.prepare_features(data[0], data[2], data[4], config=readers.FeatureConfig(k=2))
    readers.fit_logistic(*data, config=readers.LogisticConfig(feature_dim=3, c_grid=(.1,)))
    original = readers.fit_mlp(*data, config=config)
    repeated = readers.fit_mlp(*data, config=config)
    changed = readers.fit_mlp(*data[:4], data[4] * -100, config=config)
    assert original["predictions"] == repeated["predictions"]
    for field in ("candidates", "scores", "selected_hyperparameter"):
        assert original[field] == repeated[field] == changed[field]
    for a, b in zip(original["audit"], repeated["audit"]):
        assert [e["train_loss"] for e in a["epochs"]] == [e["train_loss"] for e in b["epochs"]]
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.random.get_rng_state())


def test_mlp_nonfinite_training_raises_with_partial_diagnostics(monkeypatch):
    def broken_objective(model, logits, targets, weights, regularization):
        value = logits.sum() * float("nan")
        return value, value, value

    monkeypatch.setattr(readers, "_objective", broken_objective)
    with pytest.raises(FloatingPointError, match="training loss") as caught:
        readers.fit_mlp(*_data(), config=_tiny_mlp())
    assert len(caught.value.psd3_audit) == 1
    assert caught.value.psd3_audit[0]["status"] == "failed"
    assert caught.value.psd3_audit[0]["epochs"] == []


def test_mlp_learns_a_small_nonlinear_xor_without_held_labels():
    corners = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], dtype=np.float32)
    labels = np.array(["same", "different", "different", "same"])
    xt, yt = np.tile(corners, (8, 1)), np.tile(labels, 8)
    config = readers.MLPConfig(feature_dim=2, hidden_dim=8, epochs=80, batch_size=32,
                               lr=.03, lambdas=(0.,), seeds=(0,))
    result = readers.fit_mlp(xt, yt, corners, labels, corners, config=config)
    assert result["scores"]["0"]["train"] == result["scores"]["0"]["val"] == 1.0
    assert result["predictions"]["0"] == labels.tolist()
