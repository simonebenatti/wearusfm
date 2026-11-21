"""Numerical and provenance checks for the single-update communication observer."""
from datetime import timedelta
import json
from pathlib import Path
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


@pytest.mark.parametrize("field", ["pin", "name_collision", "rank", "shape", "gradient_hash", "slot", "bucket_members"])
def test_rank_receipt_must_bind_original_vectors_and_bucket_members(observation, tmp_path, field):
    source = observation / "observation.rank0.json"
    record = json.loads(source.read_text())
    (tmp_path / "vectors.rank0.pt").hardlink_to(observation / "vectors.rank0.pt")
    if field == "pin":
        record["vectors"]["sha256"] = "0" * 64
    elif field == "name_collision":
        record["parameters"][1]["name"] = record["parameters"][0]["name"]
    elif field == "rank":
        record["rank"] = 1
    elif field == "shape":
        record["parameters"][0]["shape"] = [99]
    elif field == "gradient_hash":
        record["parameters"][0]["local_sha256"] = "0" * 64
    elif field == "slot":
        record["parameters"][0]["bucket_position"] = 99
    else:
        record["buckets"][0]["parameter_ids"] = []
    (tmp_path / source.name).write_text(json.dumps(record))
    with pytest.raises(ValueError) as info:
        load_observation(tmp_path, 0, expected_parameters=3)
    print("MSG", field, str(info.value))
    assert "mmap" not in str(info.value)
