"""CPU contract tests; no CUDA reproducibility or timing is claimed."""
import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import contrastive_numeric_profile as P


def test_profile_selects_only_math_disables_tf32_and_restores_on_exception():
    before = P.numeric_flags()
    with pytest.raises(RuntimeError, match="test abort"):
        with P.numeric_profile(require_cuda=False) as flags:
            P.assert_numeric_flags(flags)
            assert flags["cudnn_tf32"] is False
            assert flags["matmul_tf32"] is False
            assert flags["math_sdp_enabled"] is True
            assert flags["flash_sdp_enabled"] is flags["mem_efficient_sdp_enabled"] is False
            assert not flags["deterministic_algorithms"]
            raise RuntimeError("test abort")
    assert P.numeric_flags() == before


@pytest.mark.parametrize("field,value", [
    ("matmul_tf32", True), ("cudnn_tf32", True), ("math_sdp_enabled", False),
    ("flash_sdp_enabled", True), ("mem_efficient_sdp_enabled", True),
    ("cudnn_sdp_enabled", True), ("deterministic_algorithms", True),
    ("float32_matmul_precision", "high"),
])
def test_changed_profile_flag_fails_closed(field, value):
    with P.numeric_profile(require_cuda=False) as flags:
        with pytest.raises(P.NumericProfileUnavailable):
            P.assert_numeric_flags({**flags, field: value})


def test_actual_autocast_is_required_inside_backward_not_inferred_from_plan():
    with P.numeric_profile(require_cuda=False) as flags:
        with pytest.raises(P.NumericProfileUnavailable, match="actual bf16"):
            P.assert_numeric_flags(flags, require_autocast=True)
        P.assert_numeric_flags({**flags, "autocast_cuda_enabled": True,
                               "autocast_cuda_dtype": "torch.bfloat16"}, require_autocast=True)


def test_missing_cuda_never_qualifies_profile(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    before = P.numeric_flags()
    with pytest.raises(P.NumericProfileUnavailable, match="unavailable"):
        with P.numeric_profile():
            pytest.fail("CPU must not pass a CUDA release gate")
    assert P.numeric_flags() == before


@pytest.mark.parametrize("active", [False, True])
def test_torch22_cuda_autocast_apis_record_actual_state(monkeypatch, active):
    calls = []

    def enabled():  # PyTorch 2.2 rejects the modern device argument.
        calls.append("cuda_enabled")
        return active

    def dtype():
        calls.append("cuda_dtype")
        return torch.bfloat16

    monkeypatch.setattr(torch, "is_autocast_enabled", enabled)
    monkeypatch.delattr(torch, "get_autocast_dtype", raising=False)
    monkeypatch.setattr(torch, "get_autocast_gpu_dtype", dtype, raising=False)
    with P.numeric_profile(require_cuda=False) as flags:
        assert flags["autocast_cuda_enabled"] is active
        assert flags["autocast_cuda_dtype"] == "torch.bfloat16"
        if active:
            P.assert_numeric_flags(flags, require_autocast=True)
        else:
            with pytest.raises(P.NumericProfileUnavailable, match="actual bf16"):
                P.assert_numeric_flags(flags, require_autocast=True)
    assert calls == ["cuda_enabled", "cuda_dtype"] * 2


@pytest.mark.parametrize("missing", ["enabled", "dtype", "required_flag"])
def test_unavailable_numeric_api_is_inconclusive_not_attribute_error(monkeypatch, missing):
    if missing == "enabled":
        monkeypatch.delattr(torch, "is_autocast_enabled")
    elif missing == "dtype":
        monkeypatch.delattr(torch, "get_autocast_dtype", raising=False)
        monkeypatch.delattr(torch, "get_autocast_gpu_dtype", raising=False)
    else:
        monkeypatch.delattr(torch, "get_float32_matmul_precision")
    with pytest.raises(P.NumericProfileUnavailable, match="API unavailable"):
        with P.numeric_profile(require_cuda=False):
            pytest.fail("missing API must not yield a valid profile")
