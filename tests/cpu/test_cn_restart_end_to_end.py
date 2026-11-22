"""Read a complete on-disk intervention through the production comparison CLI on CPU."""
import copy
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
import contrastive_cn_restart_compare as reader
from contrastive_cn_restart_observe import ReductionObservation
from contrastive_target_resume import WearUsFM, attach_contrastive_head, config_from_dict


def dump(path, data):
    path.write_text(json.dumps(data))


def json_lines(path, data):
    path.write_text("".join(json.dumps(row) + "\n" for row in data))


def endpoint(cfg, ordinary):
    # get_rng_state_all() on rank 0 includes three dormant generators in addition
    # to its active GPU. rank_rng is gathered independently from all four ranks.
    ranks = [{"torch_rng": torch.Generator().manual_seed(8 + rank).get_state(),
              "cuda_rng": torch.full((16,), 64 + rank, dtype=torch.uint8)} for rank in range(4)]
    return {"step": 576, "world_size": 4, "config": cfg,
            "student": {"weight": torch.tensor([1., 2.])}, "teacher": {"weight": torch.tensor([1., 2.])},
            "optimizer": {"state": {0: {"step": torch.tensor(576.), "exp_avg": torch.tensor([.5, .6])}}},
            "contrastive_state": {"last_step": 575, "keys": torch.ones((1, 2))},
            "rank_rng": ranks, "torch_rng": ranks[0]["torch_rng"].clone(),
            "cuda_rng": [ranks[0]["cuda_rng"].clone()] +
                        [torch.full((16,), 22 if ordinary else 33, dtype=torch.uint8) for _ in range(3)]}


def rank_evidence(folder, variant, rank, names, groups, excess):
    # Use the actual observation writer on synthetic communication buffers. The
    # local/reduced views on disk share their bucket storage, including offsets.
    capture = object.__new__(ReductionObservation)
    capture.capturing = True
    capture.rank, capture.world_size, capture.parameters, capture.buckets = rank, 4, names, []
    for group_index, ids in enumerate(groups):
        sizes = [names[i][1].numel() for i in ids]
        first = names[ids[0]][1]
        local = torch.full((sum(sizes),), rank + 1., dtype=first.dtype)
        reduced = torch.full_like(local, 2.5)
        members, offset = [], 0
        for i, size in zip(ids, sizes, strict=True):
            name, p = names[i]
            assert first.dtype == p.dtype
            if name == "anchor_heads.rms.bias":
                # Different valid fp32 addition orders can produce 0 or +/- 2^-26
                # from these cancelling rank contributions, after division by four.
                local[offset] = [1., 2.**-24, -1., -2.**-24][rank]
                reduced[offset] = (1e-3 if excess else 2.**-26) if variant == "warm" else 0.
            p.grad = reduced.narrow(0, offset, size).view_as(p).clone()
            members.append((i, tuple(p.shape), p.stride(), offset))
            offset += size
        capture.buckets.append((group_index, members, local, reduced))
    capture.finish(folder, rank)

    trace = reader.rows(folder / f"trace-{rank}.jsonl", 544, 576)[0]
    h = "b" * 64
    before = {k: h for k in ("optimizer", "prepared_runtime_sha256", "anchor_targets_sha256",
                               "numpy_rng", "python_rng", "student_buffers_sha256", "teacher_buffers_sha256")}
    before.update(grad_none=True, input_sha256=trace["input_sha256"],
                  rng_torch_cpu=trace["rng_cpu_before_sha256"], rng_torch_device=trace["rng_cuda_before_sha256"])
    before.update({k: trace["boundary"][k] for k in ("student", "teacher", "contrastive_state")})
    dump(folder / f"boundary.rank{rank}.json",
         {"schema": "cn-ddp-boundary-v1", "rank": rank, "world_size": 4, "update": 545,
          "variant": variant, "null_warmups": int(variant == "warm"), "host": "fixture",
          "device_name": "fixture-gpu", "device_index": rank, "gpu_uuid": f"fixture-{rank}",
          "before": before, "after_warmup": before})
    post = {k: h for k in ("post_clip_gradient_sha256", "exp_avg_sha256", "exp_avg_sq_sha256",
                            "step_sha256", "parameter_sha256")}
    dump(folder / f"post_update.rank{rank}.json",
         {"schema": "cn-ddp-optimizer-update-v1", "rank": rank, "update": 545,
          "rows": [dict(id=i, name=n, **post) for i, (n, _) in enumerate(names)]})


def make_case(path, monkeypatch, change):
    original = reader.ARCHIVE
    o, w, a, c = [path / name for name in ("ordinary", "warm", "historical_ordinary", "historical_continuous")]
    for folder in (o, w, a, c):
        folder.mkdir()
    # Preserve the identities of all 270 trainable tensors and all 7 warm groups.
    cfg = reader.read(original["ordinary"] / "config.json")
    cfg["model"].update(dim=16, n_heads=4, k_latents=4)
    with torch.random.fork_rng(devices=[]):
        names = list(attach_contrastive_head(WearUsFM(config_from_dict(cfg).model)).named_parameters())
    assert len(names) == 270 and names[253][0] == "anchor_heads.rms.bias"

    torch.save({"step": 544, "config": cfg}, c / "checkpoint_step0000544.pt")
    fork_sha = reader.file_sha(c / "checkpoint_step0000544.pt")
    start = {"phase": "reload_a", "copied_checkpoint_sha256": fork_sha,
             "source": {"sha256": fork_sha, "payload_sha256": "a" * 64}}
    for folder in (a, o, w):
        dump(folder / "config.json", cfg)
        dump(folder / "start.json", start)
        dump(folder / "summary.json", {"steps": 576, "stopped": "max_steps"})

    ordinary, continuous = endpoint(cfg, True), endpoint(cfg, False)
    ordinary["optimizer"]["state"][0]["exp_avg"].add_(.02)
    chosen = copy.deepcopy(continuous)
    for idx in (1, 2, 3):
        chosen["cuda_rng"][idx] = ordinary["cuda_rng"][idx].clone()
    if change == "different_weight":
        chosen["student"]["weight"][0] += .125
    elif change == "dormant_not_ordinary":
        chosen["cuda_rng"][3] = continuous["cuda_rng"][3].clone()
    elif change == "wrong_active":
        chosen["cuda_rng"][0][0] += 1
    elif change == "wrong_rank_rng":
        chosen["rank_rng"][3]["cuda_rng"][0] += 1
    elif change == "wrong_torch_rng":
        chosen["torch_rng"][0] += 1
    elif change == "wrong_counter":
        chosen["contrastive_state"]["last_step"] -= 1
    torch.save(ordinary, a / "checkpoint.pt")
    shutil.copyfile(a / "checkpoint.pt", o / "checkpoint.pt")  # preserve raw bytes for fidelity check
    if change == "ordinary_dormant":
        damaged = copy.deepcopy(ordinary)
        damaged["cuda_rng"][3][0] += 1
        torch.save(damaged, o / "checkpoint.pt")
    torch.save(continuous, c / "checkpoint.pt")
    torch.save(chosen, w / "checkpoint.pt")

    for variant, history, folder in (("ordinary", a, o), ("warm", c, w)):
        archive = original[variant]
        metrics = reader.rows(archive / "metrics.jsonl", 544, 576, allow_prefix=variant == "warm")
        json_lines(history / "metrics.jsonl", metrics)
        json_lines(folder / "metrics.jsonl", metrics)
        for rank in range(4):
            profile = f"numeric_profile.rank{rank}.json"
            trace = f"trace-{rank}.jsonl"
            shutil.copyfile(archive / profile, history / profile)
            shutil.copyfile(archive / profile, folder / profile)
            shutil.copyfile(archive / trace, history / trace)
            shutil.copyfile(original["ordinary"] / trace, folder / trace)
            ddp_path = f"trace-{rank}.ddp.jsonl"
            ddp = reader.rows(original["warm"] / ddp_path, 544, 576)
            hot = [list(map(int, chunk.split())) for chunk in ddp[0]["rebuilt_per_bucket_param_indices"].split(",")]
            cold = [[i for i, (_, p) in enumerate(names) if p.dtype == dtype] for dtype in (torch.float32, torch.float64)]
            def byte_sizes(groups):
                return ", ".join(str(sum(names[i][1].numel() * names[i][1].element_size() for i in g)) for g in groups)
            if variant == "warm":
                for row in ddp:
                    row.update(bucket_sizes=byte_sizes(cold), rebuilt_bucket_sizes=byte_sizes(hot))
                json_lines(history / ddp_path, ddp)
            rank_evidence(folder, variant, rank, names, hot if variant == "warm" else cold, change == "excess")
    monkeypatch.setattr(reader, "FORK_SHA", fork_sha)
    monkeypatch.setattr(reader, "ENDPOINT", {"ordinary": reader.file_sha(a / "checkpoint.pt"),
                                            "warm": reader.file_sha(c / "checkpoint.pt")})
    monkeypatch.setattr(reader, "ARCHIVE", {"ordinary": a, "warm": c})
    return o, w, a, c


@pytest.mark.parametrize("change, expected", [
    ("ordinary_rounding", "SUPPORTED"), ("different_weight", "PARTIAL_SUPPORT"),
    ("excess", "HYPOTHESIS_UNSUPPORTED"), ("dormant_not_ordinary", "INCONCLUSIVE"),
    ("wrong_active", "INCONCLUSIVE"), ("wrong_rank_rng", "INCONCLUSIVE"),
    ("wrong_torch_rng", "INCONCLUSIVE"), ("wrong_counter", "INCONCLUSIVE"), ("ordinary_dormant", "INCONCLUSIVE")
])
def test_complete_restart_readout_from_four_folders(tmp_path, monkeypatch, change, expected):
    o, w, a, c = make_case(tmp_path, monkeypatch, change)
    real_load = torch.load
    mmap_reads = []

    def load_as_cluster_22(filename, *args, **kwargs):
        if kwargs.get("mmap"):
            if not isinstance(filename, str):
                raise ValueError("f must be a string filename in order to use mmap argument")
            mmap_reads.append(Path(filename))
        return real_load(filename, *args, **kwargs)

    monkeypatch.setattr(torch, "load", load_as_cluster_22)
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["probe", "--ordinary", str(o), "--warm", str(w),
                                      "--historical-ordinary", str(a), "--historical-continuous", str(c),
                                      "--report", str(report)])
    reader.main()
    result = reader.read(report)
    assert result["status"] == expected, result["reason"]
    assert result["training_approved"] is False and result["historical_status"] == "FAILED_REQUIRES_REVIEW"
    if expected != "INCONCLUSIVE":
        assert {path for path in mmap_reads if path.name.startswith("vectors.")} == {
            folder / f"vectors.rank{rank}.pt" for folder in (o, w) for rank in range(4)}
        assert len(result["per_parameter_and_rank"]) == 4 * 270
        assert result["endpoint"]["warm"]["exact_payload"] is False
    if change in ("ordinary_rounding", "excess"):
        scalar = [row["reduced"] for row in result["per_parameter_and_rank"] if row["name"] == "anchor_heads.rms.bias"]
        assert len(scalar) == 4
        for row in scalar:
            assert row["relative_l2"] == 1.0
            assert row["screen_denominator"] == (1 + 2.**-24) / 2
            assert (row["screen_relative_l2"] > 1e-6) == (change == "excess")
    if expected == "PARTIAL_SUPPORT":
        assert result["endpoint"]["warm"]["floating_tensors_different"] == ["student.weight"]
    if expected == "INCONCLUSIVE":
        reason = {"dormant_not_ordinary": "cuda_rng.3", "wrong_active": "cuda_rng.0",
                  "wrong_rank_rng": "rank_rng.3.cuda_rng", "wrong_torch_rng": "torch_rng",
                  "wrong_counter": "contrastive_state.last_step", "ordinary_dormant": "did not reproduce historical ordinary"}
        assert reason[change] in result["reason"]
        if change != "ordinary_dormant":
            assert result["endpoint"]["warm"]["scientific_metadata_or_schema_different"] == [reason[change]]
    if expected == "SUPPORTED":
        assert result["endpoint"]["warm"]["exact_scientific_payload"]
        assert {r["path"] for r in result["endpoint"]["warm"]["process_rng"] if r["matches_ordinary"]
                and not r["matches_continuous"]} == {"cuda_rng.1", "cuda_rng.2", "cuda_rng.3"}
