"""Claude review: matched head search, numerical outcomes and frozen parity."""
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from wearusfm.harness import finetuning as F, fm_features as FF
from wearusfm.model.fm import FMConfig, WearUsFM


def checkpoint():
    cfg = FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=1,
                   decoder_layers=1, muscle_dropout=.4, window_ms=100., tokenizer_activation="gelu")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = WearUsFM(cfg)
    return {"config": {"model": asdict(cfg)}, "teacher": model.state_dict()}


@pytest.mark.parametrize("candidate", [0, 1])
def test_head_grid_matches_by_candidate_and_encoder_grid_is_unchanged(candidate):
    state = checkpoint()
    before = {k: v.clone() for k, v in state["teacher"].items()}
    assert F.LR_GRID == {"frozen": (1e-3, 1e-2), "finetune": (1e-5, 1e-4), "scratch": (1e-4, 1e-3)}
    for regime in F.REGIMES:
        classifier = F.build_classifier(state, regime, 2)
        lr = F.LR_GRID[regime][candidate]
        optimizer = F.optimizer_for(classifier, regime, lr)
        assert optimizer.param_groups[0]["lr"] == pytest.approx((1e-3, 1e-2)[candidate])
        assert optimizer.param_groups[0]["lr"] == F.HEAD_LR_MULTIPLIER[regime] * lr
        if regime == "frozen":
            assert len(optimizer.param_groups) == 1
        else:
            assert optimizer.param_groups[1]["lr"] == lr
        # A real optimization step must not mutate the source checkpoint.
        sum(p.square().sum() for p in classifier.head.parameters()).backward()
        optimizer.step()
    for key in before:
        torch.testing.assert_close(state["teacher"][key], before[key], atol=0, rtol=0)


def test_frozen_logits_match_historical_features_with_two_crops():
    state = checkpoint()
    before = {k: v.clone() for k, v in state["teacher"].items()}
    windows = np.random.default_rng(81).normal(size=(2, 900, 8)).astype(np.float32)
    saved_windows = windows.copy()
    subjects = np.array(["a", "b"])
    montage = FF.myo8_montage("epn612", 200.)
    prepared = F.PreparedWindows(windows, subjects, 200., montage)
    assert len(prepared.crops) == 2
    classifier = F.build_classifier(state, "frozen", 3)
    classifier.train()  # The trainable head must still leave the encoder in eval.
    historical = FF.extract_features(classifier.encoder, windows, 200., montage, subjects, batch=2)
    observed_head_inputs = []
    hook = classifier.head.register_forward_pre_hook(lambda _m, args: observed_head_inputs.append(args[0].detach().clone()))
    with torch.no_grad():
        actual = classifier(prepared.batch([0, 1], "cpu"))
    hook.remove()
    with torch.no_grad():
        expected = classifier.head(torch.from_numpy(historical))
    torch.testing.assert_close(observed_head_inputs[0], torch.from_numpy(historical), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert not classifier.encoder.training
    np.testing.assert_array_equal(windows, saved_windows)
    for key in before:
        torch.testing.assert_close(state["teacher"][key], before[key], atol=0, rtol=0)


class BadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return torch.full_like(grad, float("inf"))


class ToyClassifier(torch.nn.Module):
    def __init__(self, fault=None):
        super().__init__()
        self.encoder = torch.nn.Identity()
        self.head = torch.nn.Linear(2, 2)
        self.fault = fault

    def forward(self, x):
        if self.fault == "implementation":
            raise RuntimeError("implementation bug")
        if self.fault == "oom":
            raise torch.OutOfMemoryError("synthetic out of memory")
        logits = self.head(x)
        if self.fault == "prediction":
            return logits * float("nan")
        if self.fault == "gradient":
            return BadGradient.apply(logits)
        return logits


class ToyData:
    def __init__(self):
        self.x = torch.tensor([[1., -1.], [-1., 1.], [1., 0.], [0., 1.], [2., 0.], [0., 2.]])

    def batch(self, indices, device):
        assert not set(indices) & {4, 5}, "held must not enter candidate fit"
        return self.x[indices].to(device)


def fit_toy(model, **kwargs):
    return F.fit_candidate(model, ToyData(), np.array([0, 1, 0, 1, 0, 1]),
                           {"train": np.array([0, 1]), "val": np.array([2, 3]), "held": np.array([4, 5])},
                           kwargs.pop("weights", torch.ones(2)), "frozen", kwargs.pop("lr", 1e-3), "cpu",
                           batch=2, epochs=kwargs.pop("epochs", 2), **kwargs)


@pytest.mark.parametrize("kind", ["prediction", "loss", "gradient"])
def test_numerical_divergence_is_specific_and_reports_progress(kind):
    model = ToyClassifier(fault=None if kind == "loss" else kind)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(F.NumericalDivergence) as caught:
        fit_toy(model, weights=torch.full((2,), float("nan")) if kind == "loss" else torch.ones(2))
    failure = caught.value
    assert failure.kind == kind
    assert failure.step == 0 and failure.epoch == 1 and failure.history == []
    assert failure.context == {"phase": "train", "attempted_update": 1}
    assert json.loads(json.dumps(failure.as_dict(), allow_nan=False))["kind"] == kind
    for key in before:
        torch.testing.assert_close(model.state_dict()[key], before[key], atol=0, rtol=0)


def test_prediction_divergence_preserves_history_snapshot():
    history = [{"epoch": 1, "step": 7, "val_bacc": .5}]
    with pytest.raises(F.NumericalDivergence) as caught:
        F.predict(ToyClassifier("prediction"), ToyData(), np.array([2, 3]), "cpu",
                  step=8, epoch=2, history=history)
    failure = caught.value
    history[0]["val_bacc"] = .9
    assert failure.history[0]["val_bacc"] == .5
    assert failure.step == 8 and failure.epoch == 2
    assert failure.context["phase"] == "evaluation"
    exported = failure.as_dict()
    exported["history"].clear()
    assert len(failure.history) == 1


@pytest.mark.parametrize("fault, expected", [("implementation", RuntimeError), ("oom", torch.OutOfMemoryError)])
def test_runtime_errors_are_not_reclassified_as_numerical(fault, expected):
    with pytest.raises(expected) as caught:
        fit_toy(ToyClassifier(fault))
    assert type(caught.value) is expected


def test_configuration_errors_remain_configuration_errors():
    with pytest.raises(ValueError, match="unregistered learning rate"):
        fit_toy(ToyClassifier(), lr=123.)
    with pytest.raises(ValueError, match="positive training budget"):
        fit_toy(ToyClassifier(), epochs=0)


def test_best_epoch_is_explicit_and_uses_first_tie(monkeypatch):
    monkeypatch.setattr(F, "predict", lambda *args, **kwargs: np.array([0, 1]))
    report = fit_toy(ToyClassifier(), epochs=3)
    assert report["best_epoch"] == 1
    assert report["val_bacc"] == 1.
    assert report["updates"] == 3 and len(report["history"]) == 3
    assert report["convergence_claim"] is False
