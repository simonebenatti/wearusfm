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
