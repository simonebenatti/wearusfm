import copy
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from wearusfm.training.contrastive_memory import (
    CandidateQueue, WindowIdentity, admissible_mask, gather_objects,
    gather_variable, identities_from_batch,
)


def ident(i=0, *, dataset="A", subject="u", session="s", trial=None, fs=200.0):
    return WindowIdentity(dataset, subject, session, trial, i * 2000, i * 2000 + 800, fs)


def test_batch_identity_ignores_montage_and_requires_explicit_coordinates():
    b = SimpleNamespace(rows=[{"dataset": "A", "subject": "u", "session": "s"}],
                        windows=[SimpleNamespace(trial=None, start=0, stop=800)], fs=[200.], presented=["full"])
    first = identities_from_batch(b)
    b.presented = ["virtual"]
    assert identities_from_batch(b) == first == (ident(),)
    del b.windows[0].trial
    with pytest.raises(ValueError, match="explicit"):
        identities_from_batch(b)
    with pytest.raises(ValueError):
        WindowIdentity.from_dict({"dataset": "A"})


@pytest.mark.parametrize("change", [dict(start=-1), dict(stop=0), dict(fs=0), dict(fs=float("nan")),
                                  dict(trial=True), dict(dataset=""), dict(subject=None), dict(start=0.1)])
def test_bad_identity_rejected(change):
    with pytest.raises(ValueError):
        replace(ident(), **change)


def test_negative_filters_same_subject_and_boundary_in_seconds():
    q = ident()
    candidates = [q, replace(q, start=799, stop=1000), replace(q, start=1599, stop=2000),
                  replace(q, start=1600, stop=2000), replace(q, trial=1),
                  replace(q, subject="another"), replace(q, session="another"), replace(q, dataset="B"),
                  replace(q, fs=400., start=3200, stop=4000)]
    assert admissible_mask([q], candidates).tolist() == [[False, False, False, True, False, True, True, False, True]]
    assert admissible_mask([], candidates).shape == (0, 9)
    assert admissible_mask([q], []).shape == (1, 0)
    assert not admissible_mask([q], [replace(q, start=400, stop=1200)], gap_seconds=0).item()


def test_snapshot_readonly_current_preferred_and_lowest_index_tie():
    q = CandidateQueue(3, 2)
    keys = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    q.commit(keys, [ident(0), ident(1)], step=0)
    before = copy.deepcopy(q.state_dict())
    current = torch.tensor([[5., 6.], [7., 8.], [9., 10.]], requires_grad=True)
    snap = q.snapshot(1, current, [ident(0), ident(0), ident(2)])
    assert snap.identities == (ident(1), ident(0), ident(2))
    assert snap.keys.tolist() == [[3., 4.], [5., 6.], [9., 10.]]
    assert snap.from_current == (False, True, True)
    assert snap.steps == (0, 1, 1)
    assert snap.global_indices == (1, 0, 2)
    assert not snap.keys.requires_grad
    assert q.last_step == before["last_step"] and len(q) == 2
    assert all(torch.equal(a["key"], b["key"]) for a, b in zip(q.state_dict()["entries"], before["entries"]))


def test_capacity_is_per_dataset_and_invalid_copy_does_not_erase_valid_key():
    q = CandidateQueue(2, 3)
    q.commit(torch.arange(8.).view(4, 2), [ident(0), ident(1), ident(2), ident(0, dataset="B")], step=0)
    assert len(q) == 3
    cur = torch.tensor([[float("nan"), 0.], [11., 12.]])
    snap = q.snapshot(1, cur, [ident(1), ident(3)], torch.tensor([False, True]))
    assert ident(1) in snap.identities
    q.commit(cur, [ident(1), ident(3)], torch.tensor([False, True]), step=1)
    assert [e["identity"]["dataset"] for e in q.state_dict()["entries"]].count("A") == 2
    assert len(q) == 3
    with pytest.raises(ValueError, match="expects update"):
        q.commit(cur, [ident(1), ident(3)], torch.tensor([False, True]), step=1)
    with pytest.raises(ValueError, match="expects update"):
        q.snapshot(3, cur, [ident(1), ident(3)])


def test_age_resume_and_state_not_aliased():
    q = CandidateQueue(3, 2)
    key = torch.tensor([[1., 2.]])
    q.commit(key, [ident()], step=0)
    empty = torch.empty(0, 2)
    q.commit(empty, [], step=1)
    snap = q.snapshot(2, empty, [])
    assert len(snap.identities) == 1  # age exactly max_age is admissible
    q.commit(empty, [], step=2)
    saved = q.state_dict()
    restored = CandidateQueue(3, 2)
    restored.load_state_dict(saved)
    saved["entries"][0]["key"].zero_()
    assert restored.state_dict()["entries"][0]["key"].tolist() == [1., 2.]
    assert not restored.snapshot(3, empty, []).identities
    for model in (q, restored):
        model.commit(torch.tensor([[9., 8.]]), [ident(1)], step=3)
    assert torch.equal(q.snapshot(4, empty, []).keys, restored.snapshot(4, empty, []).keys)


@pytest.mark.parametrize("mutation", ["policy", "duplicate", "future", "old", "nonfinite", "shape", "extra", "order", "version_type", "policy_type"])
def test_corrupt_queue_checkpoint_rejected_atomically(mutation):
    q = CandidateQueue(3, 2)
    q.commit(torch.ones(2, 2), [ident(0), ident(1)], step=0)
    state = copy.deepcopy(q.state_dict())
    if mutation == "policy": state["capacity_per_dataset"] += 1
    elif mutation == "duplicate": state["entries"][1]["identity"] = state["entries"][0]["identity"]
    elif mutation == "future": state["entries"][1]["step"] = 1
    elif mutation == "old": state["last_step"] = 3
    elif mutation == "nonfinite": state["entries"][1]["key"][0] = float("nan")
    elif mutation == "shape": state["entries"][1]["key"] = torch.ones(3)
    elif mutation == "extra": state["entries"][1]["other"] = 7
    elif mutation == "order": state["entries"].reverse()
    elif mutation == "version_type": state["version"] = True
    elif mutation == "policy_type": state["max_age"] = 2.0
    restored = CandidateQueue(3, 2)
    with pytest.raises(ValueError): restored.load_state_dict(state)
    assert restored.last_step == -1 and len(restored) == 0


def test_single_process_gather_preserves_grad_only_when_requested():
    x = torch.randn(3, 4, requires_grad=True)
    assert gather_variable(x, with_grad=True) is x
    assert not gather_variable(x).requires_grad
    assert gather_objects([ident(0), ident(1)]) == [ident(0), ident(1)]


def _gather_worker(rank, world, store, output, all_empty):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=45))
    try:
        # Uneven partitions with an empty rank, then equal FULL loss on each rank.
        sizes = [0] * world if all_empty else list(range(world))
        n = sum(sizes)
        x = (torch.arange(n * 3, dtype=torch.float64).view(n, 3) + 1.) / 20.
        start = sum(sizes[:rank])
        model = torch.nn.Linear(3, 2, bias=False).double()
        with torch.no_grad(): model.weight.copy_(torch.tensor([[.2, -.3, .4], [.5, .1, -.2]], dtype=torch.float64))
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        local = ddp(x[start:start + sizes[rank]])
        gathered = gather_variable(local, with_grad=True)
        # Every candidate participates on both sides; this exercises all-gather's
        # summed backward, followed by DDP mean, not a detached-key approximation.
        loss = (gathered @ gathered.T).square().sum() / max(n * n, 1)
        loss.backward()
        ref = torch.nn.Linear(3, 2, bias=False).double()
        with torch.no_grad(): ref.weight.copy_(model.weight)
        full = ref(x)
        ((full @ full.T).square().sum() / max(n * n, 1)).backward()
        assert torch.allclose(model.weight.grad, ref.weight.grad, atol=1e-12, rtol=1e-12)
        detached = gather_variable(local)
        assert not detached.requires_grad and torch.equal(detached, gathered.detach())
        objects = gather_objects([(rank, i) for i in range(sizes[rank])])
        assert objects == [(r, i) for r in range(world) for i in range(sizes[r])]
        torch.save({"grad": model.weight.grad, "n": n, "loss": loss.detach()}, f"{output}.{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world,all_empty", [(2, False), (4, False), (2, True)])
def test_distributed_variable_gather_matches_full_gradient(tmp_path, world, all_empty):
    mp.spawn(_gather_worker, args=(world, str(tmp_path / "store"), str(tmp_path / "out"), all_empty),
             nprocs=world, join=True)
    states = [torch.load(tmp_path / f"out.{r}.pt", weights_only=False) for r in range(world)]
    assert all(torch.equal(states[0]["grad"], s["grad"]) for s in states[1:])
