"""Numerical and provenance checks for the single-update communication observer."""
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "diagnostics"))
from contrastive_cn_restart_observe import ReductionObservation, load_observation


class Project(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.scale = torch.nn.Parameter(torch.tensor(0.75, dtype=torch.float64))

    def forward(self, x):
        return (self.linear(x).double() * self.scale).square().sum()


def _worker(rank, world, directory, address):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=address, rank=rank, world_size=world,
                            timeout=timedelta(seconds=45))
    try:
        results = []
        for observed in (False, True):
            torch.manual_seed(87)
            model = Project()
            ddp = DistributedDataParallel(model)
            if observed:
                obs = ReductionObservation(ddp, model, expected_parameters=3)
            opt = torch.optim.SGD(model.parameters(), lr=.001)
            for update in range(3):
                x = torch.tensor([rank + 1., 2 * rank - 1., update + .5])
                # Expected rank-local unscaled gradient independently of DDP.
                local_grads = torch.autograd.grad(model(x), tuple(model.parameters()))
                opt.zero_grad(set_to_none=True)
                if observed and update == 1:
                    obs.start()
                ddp(x).backward()
                if observed and update == 1:
                    obs.finish(directory, rank)
                    receipt, vectors = load_observation(directory, rank, expected_parameters=3)
                    assert len(receipt["buckets"]) == 2  # mixed dtype
                    for i, p in enumerate(model.parameters()):
                        assert torch.equal(vectors["local"][i], local_grads[i])
                        assert torch.equal(vectors["reduced"][i], p.grad)
                    with pytest.raises(ValueError, match="only be observed once"):
                        obs.start()
                opt.step()
            results.append([p.detach().clone() for p in model.parameters()])
        for plain, with_hook in zip(*results):
            assert torch.equal(plain, with_hook)
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def observation(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cn_ddp")
    mp.spawn(_worker, args=(4, str(folder), (folder / "store").as_uri()), nprocs=4, join=True)
    return folder


def test_true_local_and_reduced_gradients_and_independent_rank_files(observation):
    left_report, left = load_observation(observation, 0, expected_parameters=3)
    right_report, right = load_observation(observation, 1, expected_parameters=3)
    assert [p["name"] for p in left_report["parameters"]] == ["scale", "linear.weight", "linear.bias"]
    assert left_report["buckets"] == right_report["buckets"]
    other_ranks = [load_observation(observation, rank, expected_parameters=3)[1] for rank in (2, 3)]
    for pid, (a, b, final_a, final_b) in enumerate(zip(left["local"], right["local"], left["reduced"], right["reduced"])):
        assert not torch.equal(a, b)
        assert torch.equal(final_a, final_b)
        all_local = [a, b, *(v["local"][pid] for v in other_ranks)]
        torch.testing.assert_close(final_a, torch.stack(all_local).mean(0))
        assert all(torch.equal(final_a, v["reduced"][pid]) for v in other_ranks)



@pytest.fixture
def capture(monkeypatch):
    """Small CPU bucket whose gradient shares the same storage as in GradBucket."""
    model = torch.nn.Linear(2, 1)
    ddp = SimpleNamespace(process_group=None, register_comm_hook=lambda state, hook: None)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 4)
    observer = ReductionObservation(ddp, model, expected_parameters=2)
    observer.start()
    flat = torch.tensor([.25, -.5, .75])
    members = list(model.parameters())
    views = [flat[:2].view_as(model.weight), flat[2:].view_as(model.bias)]
    bucket = SimpleNamespace(buffer=lambda: flat, parameters=lambda: members,
                             gradients=lambda: views, index=lambda: 0)
    return observer, model, bucket


@pytest.fixture
def observation_files(capture, tmp_path, monkeypatch):
    observer, model, bucket = capture
    observer._snapshot(bucket)
    bucket.buffer().mul_(2)
    for param, grad in zip(model.parameters(), bucket.gradients(), strict=True):
        param.grad = grad.clone()
    observer.finish(tmp_path, 0)
    native_load = torch.load

    def torch22_load(filename, **kwargs):
        if kwargs.get("mmap") and not isinstance(filename, str):
            raise ValueError("mmap requires a string filename in Torch 2.2.0a0")
        return native_load(filename, **kwargs)
    monkeypatch.setattr(torch, "load", torch22_load)
    return tmp_path


def test_torch22_mmap_keeps_offsets_and_uses_bytes_for_vector_identity(observation_files):
    path = observation_files / "observation.rank0.json"
    receipt = json.loads(path.read_text())
    # CPU summation order may vary across environments. Scientific distances
    # are computed from the authenticated vectors, not these summaries.
    receipt["parameters"][1]["local_norm"] *= 1 + 1e-9
    path.write_text(json.dumps(receipt))
    _, values = load_observation(observation_files, 0, expected_parameters=2)
    assert values["local"][1].storage_offset() == 2
    assert values["local"][1].item() == .75
    assert values["reduced"][1].item() == 1.5


@pytest.mark.parametrize(("field", "replacement", "reason"), [
    (("schema",), "other", "invalid observation receipt or vector pin"),
    (("update",), 544, "invalid observation receipt or vector pin"),
    (("rank",), 1, "invalid observation receipt or vector pin"),
    (("world_size",), 0, "invalid observation receipt or vector pin"),
    (("vectors", "file"), "other.pt", "invalid observation receipt or vector pin"),
    (("vectors", "bytes"), 0, "invalid observation receipt or vector pin"),
    (("vectors", "sha256"), "0" * 64, "invalid observation receipt or vector pin"),
    (("parameters", 1, "name"), "weight", "invalid parameter or bucket inventory"),
    (("buckets", 0, "index"), 5, "invalid parameter or bucket inventory"),
    (("buckets", 0, "parameter_ids"), [0, 0], "invalid parameter or bucket inventory"),
    (("parameters", 0, "bucket_position"), 99, "bucket/parameter mapping differs"),
    (("parameters", 0, "bucket_index"), 1, "bucket/parameter mapping differs"),
    (("parameters", 0, "id"), 1, "bucket/parameter mapping differs"),
    (("buckets", 0, "dtype"), "torch.float64", "bucket/parameter mapping differs"),
    (("parameters", 0, "shape"), [99], "observed gradient differs from its receipt"),
    (("parameters", 0, "local_sha256"), "0" * 64, "observed gradient differs from its receipt"),
    (("parameters", 0, "reduced_sha256"), "0" * 64, "observed gradient differs from its receipt"),
    (("parameters", 1, "bucket_element_offset"), 1, "observed gradient differs from its receipt"),
    (("buckets", 0, "nbytes"), 8, "parameter span outside its bucket or overlapped"),
    (("parameters", 0, "local_norm"), -1.0, "invalid descriptive gradient norm"),
    (("parameters", 0, "reduced_norm"), None, "invalid descriptive gradient norm"),
])
def test_reader_rejects_the_specific_corrupted_field(observation_files, field, replacement, reason):
    path = observation_files / "observation.rank0.json"
    record = json.loads(path.read_text())
    owner = record
    for key in field[:-1]:
        owner = owner[key]
    owner[field[-1]] = replacement
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=reason):
        load_observation(observation_files, 0, expected_parameters=2)


def test_wrong_expected_parameter_count_rejected_at_read_and_capture(observation_files, capture):
    with pytest.raises(ValueError, match="invalid parameter or bucket inventory"):
        load_observation(observation_files, 0, expected_parameters=3)
    _, model, _ = capture
    with pytest.raises(ValueError, match="unexpected number of trainable parameters"):
        ReductionObservation(SimpleNamespace(process_group=None), model, expected_parameters=3)


def test_vector_file_symlink_is_rejected(observation_files):
    path = observation_files / "vectors.rank0.pt"
    moved = path.with_suffix(".real")
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(ValueError, match="invalid observation receipt or vector pin"):
        load_observation(observation_files, 0, expected_parameters=2)


def test_snapshot_requires_gradient_views_of_the_communicated_buffer(capture):
    obs, _, bucket = capture
    bucket.gradients()[0] = bucket.gradients()[0].clone()
    with pytest.raises(ValueError, match="DDP bucket gradient view"):
        obs._snapshot(bucket)


@pytest.mark.parametrize(("fault", "reason"), [
    ("unrecorded", "incomplete step-545 bucket coverage"),
    ("missing_gradient", "duplicated bucket member or missing reduced gradient"),
    ("different_preclip", "reduced bucket differs from the actual preclip parameter gradient"),
    ("wrong_rank", "missing observation or incorrect process-group rank"),
])
def test_finish_rejects_incomplete_or_wrong_gradient_evidence(fault, reason, capture, tmp_path):
    obs, model, bucket = capture
    if fault != "unrecorded":
        obs._snapshot(bucket)
    for p, v in zip(model.parameters(), bucket.gradients(), strict=True):
        p.grad = v.clone()
    if fault == "missing_gradient":
        model.bias.grad = None
    if fault == "different_preclip":
        model.bias.grad.add_(1)
    with pytest.raises(ValueError, match=reason):
        obs.finish(tmp_path, int(fault == "wrong_rank"))
