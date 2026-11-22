"""Prospective, strictly bounded readout for the common-fork intervention."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
import contrastive_cn_restart_compare as compare


@pytest.mark.parametrize("local,reduced,excess,end,expected", [
    (False, True, False, True, "SUPPORTED"), (False, True, False, False, "PARTIAL_SUPPORT"),
    (False, False, False, True, "HYPOTHESIS_UNSUPPORTED"), (True, True, False, True, "HYPOTHESIS_UNSUPPORTED"),
    (False, True, True, True, "HYPOTHESIS_UNSUPPORTED"),
])
def test_criteria_fixed_before_measurement(local, reduced, excess, end, expected):
    assert compare.classify(local, reduced, excess, end, True)[0] == expected


def test_warm_first_update_mismatch_prevents_full_support():
    assert compare.classify(False, True, False, True, False)[0] == "PARTIAL_SUPPORT"


def test_per_tensor_relative_metric_near_zero_and_bitwise_zero():
    metric = compare.compare_vector(torch.tensor([-0., 1.]), torch.tensor([0., 2.]))
    assert metric["different_elements"] == 2 and metric["difference_l2"] == 1.
    assert metric["relative_l2"] == 0.5 and metric["max_abs"] == 1.
    metric = compare.compare_vector(torch.tensor([0.]), torch.tensor([1e-20], dtype=torch.float32))
    assert metric["relative_l2"] == pytest.approx(1e-8)
    assert not compare.identical(torch.tensor([-0.]), torch.tensor([0.]))
    with pytest.raises(ValueError, match="invalid gradient"):
        compare.compare_vector(torch.tensor([float("nan")]), torch.tensor([0.]))


def test_all_checkpoint_leaves_checked_and_classified():
    original = {"student": {"layer": torch.tensor([0.])}, "optimizer": {"state": {0: {"step": torch.tensor(576.)}}},
                "rank_rng": [torch.tensor([1], dtype=torch.uint8)], "config": {"seed": 0}}
    new = copy.deepcopy(original)
    assert compare.checkpoint_differences(original, new)["exact_payload"]
    new["student"]["layer"].neg_()
    new["rank_rng"][0].add_(1)
    new["optimizer"]["state"][0]["step"] += 1
    new["config"]["seed"] = 1
    result = compare.checkpoint_differences(original, new)
    assert result["floating_tensors_different"] == ["student.layer"]
    assert result["metadata_or_schema_different"] == ["optimizer.state.0.step", "rank_rng.0", "config.seed"]


def layout(groups, sizes, dtypes):
    return {"buckets": [{"nbytes": size, "parameter_ids": ids} for ids, size in zip(groups, sizes)],
            "parameters": [{"id": i, "dtype": d} for i, d in enumerate(dtypes)]}


def test_layout_historical_cold_dtype_order_and_warm_actual_sequence():
    cold = layout([[1], [0, 2]], [8, 32], ["float", "double", "float"])
    warm = layout([[2], [1], [0]], [16, 8, 16], ["float", "double", "float"])
    arch = {"bucket_sizes": "32, 8", "rebuilt_bucket_sizes": "16, 8, 16",
            "rebuilt_per_bucket_param_indices": "2, 1, 0"}
    assert compare.layouts(cold, warm, arch)["ordinary"] == cold["buckets"]
    warm["buckets"].reverse()
    with pytest.raises(ValueError, match="warm actual bucket"):
        compare.layouts(cold, warm, arch)
    cold["buckets"][1]["parameter_ids"].reverse()
    with pytest.raises(ValueError, match="ordinary initial"):
        compare.layouts(cold, warm, arch)


def test_no_measurement_cannot_promote_training(tmp_path, monkeypatch):
    args = ["check", "--warm", str(tmp_path / "w"), "--ordinary", str(tmp_path / "o"),
            "--historical-continuous", str(tmp_path / "h"), "--historical-ordinary", str(tmp_path / "i"),
            "--report", str(tmp_path / "report.json")]
    monkeypatch.setattr(sys, "argv", args)
    compare.main()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "INCONCLUSIVE" and report["training_approved"] is False
    assert report["historical_status"] == "FAILED_REQUIRES_REVIEW"


def test_trace_joins_and_gradient_fields_on_real_archived_legs():
    # CN_B is a separate archived replica of CN_A. Both preserve the continuous suffix inputs.
    a = compare.ARCHIVE["ordinary"]
    b = a.with_name("CN_B")
    c = compare.ARCHIVE["warm"]
    reference = compare.metrics_evidence(b, a, "ordinary")
    assert reference["differences"] == []
    warm_reference = compare.metrics_evidence(b, c, "warm")
    assert any(d["field"] == "grad_norm" and d["step"] == 545 for d in warm_reference["differences"])
    # The new probe uses the historical resume_a scheduler stage for both variants.
    for rank in range(4):
        assert (compare.branch_traces(a, a, "ordinary", rank)["boundary"]
                == compare.branch_traces(a, c, "warm", rank)["boundary"])


def test_mutated_warmup_state_fails_before_verdict(tmp_path):
    h = "c" * 64
    keys = ("student", "teacher", "optimizer", "contrastive_state", "prepared_runtime_sha256", "anchor_targets_sha256", "input_sha256",
            "rng_torch_cpu", "rng_torch_device", "numpy_rng", "python_rng", "student_buffers_sha256",
            "teacher_buffers_sha256")
    before = {**dict.fromkeys(keys, h), "grad_none": True}
    trace = {"boundary": dict.fromkeys(("student", "teacher", "contrastive_state"), h), "input_sha256": h,
             "rng_cpu_before_sha256": h, "rng_cuda_before_sha256": h}
    record = {"schema": "cn-ddp-boundary-v1", "rank": 0, "world_size": 4, "variant": "warm", "update": 545,
              "null_warmups": 1, "before": before, "after_warmup": before.copy()}
    path = tmp_path / "boundary.rank0.json"
    path.write_text(json.dumps(record))
    assert compare.check_boundary(tmp_path, "warm", 0, trace)["before"] == before
    record["after_warmup"]["optimizer"] = "0" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="warmup changed state"):
        compare.check_boundary(tmp_path, "warm", 0, trace)


def test_post_update_trace_includes_each_optimizer_tensor_in_order(tmp_path):
    key_names = ("post_clip_gradient_sha256", "exp_avg_sha256", "exp_avg_sq_sha256", "step_sha256", "parameter_sha256")
    row = {"id": 0, "name": "p", **dict.fromkeys(key_names, "a" * 64)}
    data = {"schema": "cn-ddp-optimizer-update-v1", "update": 545, "rank": 1, "rows": [row]}
    path = tmp_path / "post_update.rank1.json"
    path.write_text(json.dumps(data))
    identity = [(0, "p", (1,), "float32")]
    assert compare.post_update_hashes(tmp_path, 1, identity) == [dict.fromkeys(key_names, "a" * 64)]
    row["name"] = "q"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="parameter ordering"):
        compare.post_update_hashes(tmp_path, 1, identity)


def saved_endpoint(seed):
    def rng(value):
        return torch.tensor([value, 5, 9], dtype=torch.uint8)
    return {"step": 576, "torch_rng": rng(0), "cuda_rng": [rng(0), rng(seed), rng(seed), rng(seed)],
            "rank_rng": [{"torch_rng": rng(rank + 1), "cuda_rng": rng(rank)} for rank in range(4)],
            "student": {"weight": torch.tensor([2.])}, "queue_counter": 32}


def test_process_rng_is_not_the_active_rng_on_any_rank():
    continuous = saved_endpoint(24)
    ordinary = saved_endpoint(12)
    warm = copy.deepcopy(ordinary)
    result = compare.warm_endpoint_differences(continuous, warm, ordinary["cuda_rng"])
    inactive = ["cuda_rng.1", "cuda_rng.2", "cuda_rng.3"]
    assert result["metadata_or_schema_different"] == inactive
    assert not result["exact_payload"] and result["exact_scientific_payload"]
    assert [entry["path"] for entry in result["process_rng"]] == inactive
    assert all(entry["matches_ordinary"] and not entry["matches_continuous"] for entry in result["process_rng"])

    warm["cuda_rng"][2] = continuous["cuda_rng"][2].clone()
    warm["cuda_rng"][0][0] += 1
    warm["torch_rng"][0] += 1
    warm["rank_rng"][2]["cuda_rng"][0] += 1
    warm["queue_counter"] += 1
    result = compare.warm_endpoint_differences(continuous, warm, ordinary["cuda_rng"])
    assert not result["exact_scientific_payload"]
    assert set(result["scientific_metadata_or_schema_different"]) == {
        "cuda_rng.0", "cuda_rng.2", "torch_rng", "rank_rng.2.cuda_rng", "queue_counter"}
    assert "cuda_rng.2" not in result["metadata_or_schema_different"]


def test_only_specific_unused_rng_values_can_be_excluded():
    continuous = saved_endpoint(24)
    ordinary = saved_endpoint(12)
    warm = copy.deepcopy(ordinary)
    warm["cuda_rng"][1] = ordinary["cuda_rng"][1] = torch.tensor([5, 9], dtype=torch.uint8)
    result = compare.warm_endpoint_differences(continuous, warm, ordinary["cuda_rng"])
    assert not result["exact_scientific_payload"]
    assert result["scientific_metadata_or_schema_different"] == ["cuda_rng.1"]
    assert not result["process_rng"][0]["same_shape"]
    warm["cuda_rng"][1] = warm["cuda_rng"][1].float()
    with pytest.raises(ValueError, match=r"invalid RNG state at cuda_rng\.1"):
        compare.warm_endpoint_differences(continuous, warm, ordinary["cuda_rng"])
    del warm["cuda_rng"][1]
    with pytest.raises(ValueError, match=r"expected 4 entries at cuda_rng"):
        compare.warm_endpoint_differences(continuous, warm, ordinary["cuda_rng"])


@pytest.mark.parametrize("pid,name,expected", [(253, "anchor_heads.rms.bias", False),
                                               (257, "anchor_heads.env.bias", False),
                                               (252, "anchor_heads.rms.weight", True)])
def test_scalar_cancellation_keeps_the_raw_ratio_and_a_falsifiable_bound(pid, name, expected):
    def row(delta):
        return {"id": pid, "name": name, "reduced": compare.compare_vector(torch.tensor([0.]), torch.tensor([delta]))}
    small = row(1e-7)
    assert compare.screen_reduced_gradients([small], {253: 1., 257: 1.}) is expected
    assert small["reduced"]["relative_l2"] == 1.
    assert small["reduced"]["screen_relative_l2"] == pytest.approx(1. if expected else 1e-7)
    assert compare.screen_reduced_gradients([row(.01)], {253: 1., 257: 1.})
    if pid in compare.SCALAR_ADDEND_PARAMETERS:
        small["name"] = "misidentified"
        with pytest.raises(ValueError, match="named scalar"):
            compare.screen_reduced_gradients([small], {253: 1., 257: 1.})
