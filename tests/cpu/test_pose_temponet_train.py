from dataclasses import asdict
from pathlib import Path
import copy
import json
import math
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from pose_temponet import train as T
from pose_temponet import selection as S


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.hidden = nn.Linear(4, 8)
        self.dropout = nn.Dropout(.25)
        self.out = nn.Linear(8, 2)
        self.affine = nn.Linear(4, 2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x):
        z = self.norm(x)
        return self.affine(z) + self.out(self.dropout(torch.relu(self.hidden(z))))


def factory(seed, fold):
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(100000 + seed).get_state())
        return TinyHead()


def dataset(n=9, seed=18, masked_empty=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 16, 4, generator=g)
    y = torch.stack((x[..., 0] + .4 * x[..., 1], x[..., 2] - .7 * x[..., 3]), -1)
    support = torch.ones(n, 4, 2, dtype=torch.bool)
    if masked_empty:
        support[-1] = False
        y[-1] = float("nan")
    return T.TensorTrainingSet(x.half(), y, support, [f"window-{n-i:03d}" for i in range(n)])


def evaluator_for(data):
    def evaluate(model, fold):
        d = data[fold]
        x, y, mask = d.batch(torch.arange(d.size), next(model.parameters()).device)
        p = model(x)[:, 6:-6]
        count = mask.sum((1, 2))
        error = torch.where(mask, p-y, 0)
        mse_sum = (error.square().sum((1, 2))/count).sum()
        # Fixture supports all central times; independent centered oracle.
        pc = p-p.mean(1, keepdim=True)
        yc = y-y.mean(1, keepdim=True)
        return S.FoldScore(float(mse_sum), len(y), float((pc-yc).square().sum()), float(yc.square().sum()))
    return evaluate


def trajectory(data=None, **kwargs):
    return T.Trajectory(factory, data or dataset(), lr=kwargs.pop("lr", .003), seed=kwargs.pop("seed", 0),
                        fold=kwargs.pop("fold", 0), _recipe=kwargs.pop("_recipe", T._TestRecipe()), **kwargs)


def test_partial_epoch_resume_preserves_adam_dropout_and_shuffle_exactly(tmp_path):
    data = dataset()
    original = trajectory(data)
    original.train_to_step(4)  # Three steps per epoch; checkpoint inside epoch two.
    state = original.state()
    path = tmp_path / "state.pt"
    digest = T.save_checkpoint(path, state)
    original.train_to_step(15)
    expected = original.state()
    torch.rand(257)  # External random work must not alter the trajectory.
    resumed = trajectory(data)
    resumed.restore(T.load_checkpoint(path, expected_sha256=digest))
    resumed.train_to_step(15)
    actual = resumed.state()
    for key in ("model", "optimizer", "dropout_cpu", "shuffle_rng", "epoch_losses", "epoch_order_sha256"):
        assert T.payload_sha256(expected[key]) == T.payload_sha256(actual[key]), key
    assert actual["updates"] == 15


def test_training_does_not_consume_global_rng_and_keeps_last_small_batch():
    before = torch.get_rng_state().clone()
    data = dataset(n=8, masked_empty=True)
    t = trajectory(data)
    assert torch.equal(before, torch.get_rng_state())
    assert data.size == 7 and data.excluded_windows == 1 and t.steps_per_epoch == 2
    t.train_to_step(2)
    assert torch.equal(before, torch.get_rng_state())
    assert t.epoch_losses[0]["window_count"] == 7
    assert t.position == 0 and t.permutation is None
    assert all(torch.isfinite(p).all() for p in t.model.parameters())


def test_same_shuffle_and_initial_state_across_learning_rates_and_target_worlds():
    a, b = dataset(), dataset()
    b.target = b.target * 2 + .3
    left, right = trajectory(a, lr=.001), trajectory(b, lr=.01)
    assert left.initial_sha256 == right.initial_sha256
    left.train_to_step(9)
    right.train_to_step(9)
    assert left.epoch_order_sha256 == right.epoch_order_sha256
    assert T.payload_sha256(left.model.state_dict()) != T.payload_sha256(right.model.state_dict())


def test_checkpoint_cannot_change_dataset_or_partial_epoch_position():
    original = trajectory()
    original.train_to_step(4)
    state = original.state()
    with pytest.raises(ValueError, match="provenance"):
        trajectory(dataset(seed=19)).restore(state)
    broken = copy.deepcopy(state)
    broken["position"] += 1
    broken.pop("payload_sha256")
    broken["payload_sha256"] = T.payload_sha256(broken)
    with pytest.raises(ValueError, match="shuffle"):
        trajectory().restore(broken)
    broken = copy.deepcopy(state)
    broken["updates"] += 1
    with pytest.raises(ValueError, match="checksum"):
        trajectory().restore(broken)


def test_checkpoint_cannot_silently_change_weight_decay_or_loss_accounting():
    original = trajectory()
    original.train_to_step(4)
    for field in ("optimizer", "loss_windows"):
        state = original.state()
        if field == "optimizer":
            state["optimizer"]["param_groups"][0]["weight_decay"] = 0
        else:
            state["loss_windows"] += 1
        state.pop("payload_sha256")
        state["payload_sha256"] = T.payload_sha256(state)
        with pytest.raises(ValueError, match="recipe|history"):
            trajectory().restore(state)


def test_checkpoint_file_hash_is_verified_before_deserialization(tmp_path):
    path = tmp_path / "state.pt"
    digest = T.save_checkpoint(path, trajectory().state())
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="file hash"):
        T.load_checkpoint(path, expected_sha256=digest)


def test_paired_checkpoint_chooses_one_shared_exposure_not_independent_fold_minima():
    fitting = [dataset(n=7, seed=10), dataset(n=7, seed=11)]
    counts = [0, 0]
    # Exposures 0, .5, 1, 2, 3. Independent fold minima would be (2, 1).
    scores = [[5, 4, 1, .1, 3], [20, 15, 9, 12, 14]]
    def evaluator(model, fold):
        i = counts[fold]
        counts[fold] += 1
        return S.FoldScore(scores[fold][i], 1, 1 if i == 2 else 2, 1)
    pair = T.PairedCandidate([trajectory(d, fold=f) for f, d in enumerate(fitting)], evaluator)
    pair.advance(3)
    assert pair.best_exposure == 1
    assert [s["updates"] for s in pair.best_states] == [2, 2]
    assert pair.curve().best().mse == 5
    assert [t.updates for t in pair.trajectories] == [6, 6]


def test_paired_midrun_restore_reproduces_curve_and_both_selected_states():
    fitting = [dataset(n=7, seed=10), dataset(n=9, seed=11)]
    evaluator = evaluator_for(fitting)
    def fresh():
        return T.PairedCandidate([trajectory(d, fold=f) for f, d in enumerate(fitting)], evaluator)
    full = fresh()
    full.advance(3)
    interrupted = fresh()
    saved = []
    class Interrupted(Exception):
        pass
    def checkpoint(pair):
        if pair.points[-1].exposure == .5:
            saved.append(pair.state())
            raise Interrupted
    with pytest.raises(Interrupted):
        interrupted.advance(3, checkpoint)
    resumed = fresh()
    resumed.restore(saved[0])
    resumed.advance(3)
    assert resumed.curve() == full.curve()
    assert T.payload_sha256(resumed.best_states) == T.payload_sha256(full.best_states)
    assert T.payload_sha256(resumed.state()) == T.payload_sha256(full.state())


def test_inner_evaluation_identity_is_required_and_preserved_on_resume():
    fitting = [dataset(), dataset(seed=22)]
    evaluator = evaluator_for(fitting)
    pair = T.PairedCandidate([trajectory(d, fold=f) for f, d in enumerate(fitting)], evaluator, "a" * 64)
    pair.advance(3)
    other = T.PairedCandidate([trajectory(d, fold=f) for f, d in enumerate(fitting)], evaluator, "b" * 64)
    with pytest.raises(ValueError, match="provenance"):
        other.restore(pair.state())
    with pytest.raises(ValueError, match="pinned inner"):
        T.run_search(factory, [], evaluator, device="cpu")


def test_untriggered_continuation_and_mixed_pair_are_rejected():
    fitting = [dataset(n=7), dataset(n=7, seed=20)]
    calls = [0, 0]
    def evaluator(model, fold):
        k = calls[fold]
        calls[fold] += 1
        value = 1 if k == 2 else 2  # Epoch one, not the final epoch.
        return S.FoldScore(value, 1, value, 1)
    pair = T.PairedCandidate([trajectory(d, fold=f) for f, d in enumerate(fitting)], evaluator)
    pair.advance(3)
    with pytest.raises(ValueError, match="not triggered"):
        pair.advance(6)
    state = pair.state()
    state["current_states"][0] = copy.deepcopy(state["best_states"][0])
    state.pop("payload_sha256")
    state["payload_sha256"] = T.payload_sha256(state)
    with pytest.raises(ValueError, match="current pair exposure"):
        pair.restore(state)


def test_small_complete_search_locks_mse_selection_and_refits_three_fresh_seeds():
    fitting = [dataset(n=7, seed=10), dataset(n=7, seed=11)]
    recipe = T._TestRecipe(initial_epochs=2, maximum_epochs=3)
    fitting_calls = []
    evaluator = evaluator_for(fitting)
    def fitting_evaluator(model, fold):
        fitting_calls.append(fold)
        return evaluator(model, fold)
    result = T._run_search(factory, fitting, evaluator, device="cpu", recipe=recipe,
                           fitting_evaluator=fitting_evaluator)
    report = result["selection"]
    assert report["selection_locked"] and report["recipe"]["test_only"]
    assert len(report["curves"]) in (4, 5)
    assert fitting_calls == [0, 1]  # Only the selected pair, not every LR.
    assert report["selected"]["exposure"] > 0
    full = dataset(n=14)
    # The production handoff persists the selection as JSON between jobs.
    persisted_report = json.loads(json.dumps(report))
    refits = T._run_refits(factory, full, persisted_report, device="cpu", recipe=recipe)
    assert refits["refits_locked"] and set(refits["refit_states"]) == {0, 1, 2}
    expected_steps = math.ceil(report["selected"]["exposure"] * math.ceil(full.size / recipe.batch_size))
    for seed, state in refits["refit_states"].items():
        assert state["seed"] == seed and state["fold"] == 2 and state["updates"] == expected_steps
        assert state["lr"] == report["selected"]["lr"]
    changed = copy.deepcopy(report)
    changed["selected"]["lr"] *= 2
    with pytest.raises(ValueError, match="locked, intact"):
        T._run_refits(factory, full, changed, device="cpu", recipe=recipe)


def test_production_entry_points_refuse_test_recipe_before_any_training():
    with pytest.raises(ValueError, match="exact frozen recipe"):
        T.run_search(factory, [], lambda *_: None, device="cpu", recipe=T._TestRecipe())
    with pytest.raises(ValueError, match="exact frozen recipe"):
        T.run_refits(factory, dataset(), {}, device="cpu", recipe=T._TestRecipe())
    with pytest.raises(ValueError, match="production readout"):
        T.Trajectory(factory, dataset(), lr=.001, seed=0, fold=0)
