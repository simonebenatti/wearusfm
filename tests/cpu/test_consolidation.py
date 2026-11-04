"""Regression checks for continuation identity, depth parity and downstream isolation."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import consolidation_continue as C
import consolidation_finetune as S
from wearusfm.harness import finetuning as F, fm_features as FF
from wearusfm.harness.backbone_depth import depth_readouts, extract_windows
from wearusfm.model.fm import FMConfig, WearUsFM
from wearusfm.training import run as R


def small():
    cfg = FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=1,
                   decoder_layers=1, muscle_dropout=.4, window_ms=100., tokenizer_activation="gelu")
    torch.manual_seed(42)
    model = WearUsFM(cfg)
    from dataclasses import asdict
    return model, {"config": {"model": asdict(cfg)}, "teacher": model.state_dict()}


def data():
    w = np.random.default_rng(1).normal(size=(8, 60, 8)).astype(np.float32)
    s = np.array(["a"] * 4 + ["b"] * 2 + ["held"] * 2)
    return F.PreparedWindows(w, s, 200., FF.myo8_montage("epn612", 200.)), s


@pytest.mark.parametrize("step", C.PHASES)
def test_continuation_is_exact_recipe_and_global_lr(step):
    receipt = json.loads(C.SOURCE.read_text())
    cmd, cfg = C.command_for(receipt, step, "/tmp/new-run")
    old = json.loads((C.BASE / "19618/T100-G/config.json").read_text())
    assert {**R.config_to_dict(cfg), "max_steps": 19618} == old
    assert R._lr_at(19618, cfg) == 1e-4
    assert R._lr_at(39237, cfg) == 1e-4
    assert R._lr_at(49045, cfg) < 1e-7
    assert ("--branch-from" in cmd) == (step == 30000)
    assert receipt["command"][receipt["command"].index("--max-steps") + 1] == "19618"


def test_no_unregistered_dose_or_changed_source():
    receipt = json.loads(C.SOURCE.read_text())
    with pytest.raises(ValueError):
        C.command_for(receipt, 60000, "/tmp/new")
    receipt["command"][receipt["command"].index("--lr") + 1] = ".001"
    with pytest.raises(ValueError, match="recipe"):
        C.command_for(receipt, 30000, "/tmp/new")


def test_depth_observes_real_forward_and_cleans_hook():
    model, _ = small()
    prepared, _ = data()
    inp = prepared.batch([0, 1], "cpu")[0]
    model.eval()
    ref = model.encode(inp, None)
    result = depth_readouts(model, inp)
    torch.testing.assert_close(result["sequences"]["post"], ref.z.float().mean(2))
    torch.testing.assert_close(result["vectors"]["post"], result["pool"][:, :16])
    assert result["sequences"]["pre"].shape == result["sequences"]["post"].shape
    assert not model.pool._forward_hooks
    assert not torch.allclose(result["vectors"]["pre"], result["vectors"]["post"])
    model.train()
    with pytest.raises(ValueError, match="eval"):
        depth_readouts(model, inp)


def test_depth_extraction_matches_historical_crops_and_batching():
    model, _ = small()
    w = np.random.default_rng(9).normal(size=(2, 900, 8)).astype(np.float32)
    s = np.array(["a", "b"])
    montage = FF.myo8_montage("epn612", 200.)
    actual = extract_windows(model, w, s, 200., montage, batch=2, retain_sequences=True)
    expected = FF.extract_features(model, w, 200., montage, s, batch=2)
    np.testing.assert_allclose(actual["pool"], expected, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(actual["post"], expected[:, :16], atol=1e-6, rtol=1e-5)
    again = extract_windows(model, w, s, 200., montage, batch=1)
    np.testing.assert_allclose(actual["pre"], again["pre"], atol=1e-6, rtol=1e-5)
    assert actual["temporal"]["pre"].shape == (2, 2, 160, 16)
    assert actual["crop_bounds"] == [(0, 800), (100, 900)]


@pytest.mark.parametrize("dim", [384, 896])
def test_depth_width_is_not_hardcoded(dim):
    from types import SimpleNamespace
    from wearusfm.model.fm import Encoded
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.Identity()
            self.cfg = SimpleNamespace(dim=dim)
        def encode(self, inp, mask):
            lat = self.pool(torch.ones(1, 4, 2, dim))
            valid = torch.ones(1, 4, dtype=torch.bool)
            vis = torch.ones(2, 4, dtype=torch.bool)
            return Encoded(torch.ones(2, 4, dim), lat * 2, valid, torch.zeros(2, dim), vis, vis)
    inp = SimpleNamespace(counts=[2], qc_valid=torch.ones(2, dtype=torch.bool))
    model = Fake().eval()
    result = depth_readouts(model, inp)
    assert result["vectors"]["pre"].shape == result["vectors"]["post"].shape == (1, dim)
    assert result["pool"].shape == (1, 2 * dim)


def test_three_regimes_share_head_but_only_adaptation_updates_encoder():
    _, state = small()
    prepared, _ = data()
    before = {k: v.clone() for k, v in state["teacher"].items()}
    heads = []
    for regime in F.REGIMES:
        classifier = F.build_classifier(state, regime, 2)
        heads.append({k: v.clone() for k, v in classifier.head.state_dict().items()})
        if regime == "scratch":
            assert not torch.equal(classifier.encoder.tokenizer.proj.weight, before["tokenizer.proj.weight"])
        classifier.train()
        initial = classifier.encoder.tokenizer.proj.weight.detach().clone()
        opt = F.optimizer_for(classifier, regime, F.LR_GRID[regime][0])
        loss = torch.nn.functional.cross_entropy(classifier(prepared.batch([0, 1], "cpu")), torch.tensor([0, 1]))
        loss.backward()
        opt.step()
        assert torch.equal(initial, classifier.encoder.tokenizer.proj.weight) == (regime == "frozen")
        assert classifier.encoder.training == (regime != "frozen")
        assert all(p.grad is None for p in classifier.encoder.decoder.parameters())
    for h in heads[1:]:
        for k in h:
            torch.testing.assert_close(h[k], heads[0][k], rtol=0, atol=0)
    for k, value in state["teacher"].items():
        torch.testing.assert_close(value, before[k], rtol=0, atol=0)


def test_labels_and_subject_isolation():
    y = np.array(["rest", "move", "rest", "move", "rest", "move", "rest", "move"])
    _, s = data()
    roles = F.role_indices(s, {"train": ["a"], "val": ["b"], "held": ["held"]})
    labels, classes, weights = F.encode_labels(y, roles["train"])
    assert classes[labels].tolist() == y.tolist()
    assert weights.tolist() == [1., 1.]
    with pytest.raises(ValueError, match="overlapping"):
        F.role_indices(s, {"train": ["a"], "val": ["a"], "held": ["held"]})
    y[-1] = "new"
    with pytest.raises(ValueError, match="absent"):
        F.encode_labels(y, roles["train"])


def test_candidate_never_reads_held_and_restores_validation_choice():
    _, state = small()
    prepared, s = data()
    roles = F.role_indices(s, {"train": ["a"], "val": ["b"], "held": ["held"]})
    seen = []
    original = prepared.batch
    def guarded(indices, device):
        seen.extend(indices)
        assert not set(indices) & set(roles["held"])
        return original(indices, device)
    prepared.batch = guarded
    classifier = F.build_classifier(state, "finetune", 2)
    report = F.fit_candidate(classifier, prepared, np.arange(8) % 2, roles, torch.ones(2),
                             "finetune", 1e-4, "cpu", batch=2, epochs=2)
    assert report["updates"] == 4
    assert report["val_bacc"] == max(x["val_bacc"] for x in report["history"])
    assert set(seen) == set(range(6))


def test_recipe_hash_cannot_be_changed_by_adding_a_benchmark_receipt():
    plan = json.loads((ROOT / "results/infra/consolidation_20261104/finetune_plan.json").read_text())
    assert S.recipe_identity(plan) == plan["recipe_sha256"]
    plan["benchmark_receipt"] = {"path": "new", "sha256": "abc"}
    assert S.recipe_identity(plan) == plan["recipe_sha256"]
    plan["epochs"] += 1
    assert S.recipe_identity(plan) != plan["recipe_sha256"]


def test_draft_plans_cannot_execute_before_claude_review(monkeypatch, tmp_path):
    from types import SimpleNamespace
    monkeypatch.setattr(C.T.P, "code_identity", lambda code: code)
    directory = ROOT / "results/infra/consolidation_20261104"
    for name in ("continuation_plan.json", "finetune_plan.json"):
        plan = json.loads((directory / name).read_text())
        plan["status"] = "awaiting_claude_review"
        (tmp_path / name).write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Claude review"):
        C.execute(tmp_path / "continuation_plan.json", 30000, "a" * 40)
    with pytest.raises(ValueError, match="Claude review"):
        S.execute(SimpleNamespace(plan=tmp_path / "finetune_plan.json", expected_code="a" * 40))
