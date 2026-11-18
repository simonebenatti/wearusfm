"""CPU checks of the CUDA harness' construction and fixed acceptance rule."""
import importlib.util
from pathlib import Path
from dataclasses import asdict, replace

import pytest
import torch

from test_contrastive_runtime import _config
from test_pose_training import _pose_setup
from wearusfm.training import run as R
from wearusfm.training.contrastive_objectives import ContrastiveConfig, attach_contrastive_head
from wearusfm.model.fm import WearUsFM

SPEC = importlib.util.spec_from_file_location("contrastive_cuda_check_test", Path(__file__).resolve().parents[2] / "scripts/contrastive_cuda_check.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def test_technical_overrides_keep_active_calibrated_weights_and_real_fixtures(tmp_path):
    root, manifest, pose_root = _pose_setup(tmp_path)
    base = _config(pose_root, "C0", calibration=True)
    cfg = C.technical_config(base, "CN")
    cc = ContrastiveConfig(**cfg.contrastive)
    assert cfg.batch_size == 8 and cfg.max_steps == 2 and cfg.jepa == base.jepa
    assert cc.weights(0) == (1., 0.) and not cc.calibration_only
    assert not cc.coverage_exempt_datasets and cc.coverage_quotas is None
    batches = C.build_batches(cfg, manifest, [root], {}, {})
    assert len(batches) == 2
    for batch in batches:
        assert batch.pose[:2] == [None, None]
        assert all(p is not None for p in batch.pose[2:])
        assert len({signal.shape[-1] for signal in batch.signals}) > 1
    torch.manual_seed(cfg.seed)
    model = attach_contrastive_head(WearUsFM(cfg.model))
    reference = C.trajectory(cfg, model.state_dict(), batches, R.DistContext(), "cpu", "fp32")
    result = C.compare_trajectory(reference, reference, reference, C._flat(model.parameters()), "fp32")
    assert result["passed"] and len(result["updates"]) == 2


def _row(shift=0.):
    keys = ("total", "pose", "contrast_nce", "contrast_rnc", "contrast_global_grad_base",
            "contrast_global_grad_nce", "contrast_global_grad_rnc")
    return {"metrics": {**{k: 1. + shift for k in keys}, "contrast_rnc_windows": 3, "contrast_nce_eligible": 6},
            "gradient": torch.tensor([1. + shift], dtype=torch.float64),
            "student": torch.tensor([1. + shift], dtype=torch.float64),
            "teacher": torch.tensor([1. + shift], dtype=torch.float64),
            "projector_grad_norm": 1., "queue_identity_sha256": "same"}


def test_fp32_and_bf16_criteria_reject_excess_error_and_empty_support():
    initial = torch.zeros(1, dtype=torch.float64)
    fp32, bf16 = [_row()], [_row(.001)]
    assert C.compare_trajectory([_row(1e-6)], fp32, fp32, initial, "fp32")["passed"]
    assert not C.compare_trajectory([_row(1e-4)], fp32, fp32, initial, "fp32")["passed"]
    assert C.compare_trajectory([_row(.0019)], bf16, fp32, initial, "bf16")["passed"]
    assert not C.compare_trajectory([_row(.003)], bf16, fp32, initial, "bf16")["passed"]
    zero = _row()
    zero["metrics"]["contrast_nce_eligible"] = 0
    assert not C.compare_trajectory([zero], fp32, fp32, initial, "fp32")["passed"]


def test_reference_rejects_live_process_group_before_any_collective(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="must precede"):
        C.trajectory(None, None, None, R.DistContext(), "cpu", "fp32")


def test_bf16_diagnostic_floor_accepts_observed_rounding_and_keeps_every_other_limit():
    """Regression for CUDA 64730941 (18/11/2026, CR bf16, update 2, contrast_global_grad_nce).

    The one-process bf16 reference was 1.06e-4 from fp32, the four-rank bf16 value 1.39e-3
    from the reference: both are bf16 rounding (unit roundoff about 3.9e-3). The amended
    rule (post hoc, Claude) never holds a bf16 diagnostic norm tighter than 1e-2."""
    assert C.CRITERIA["bf16_diagnostic_relative_floor"] == .01 == C.CRITERIA["bf16_loss_relative"]
    initial = torch.zeros(1, dtype=torch.float64)
    fp32, reference, actual = _row(), _row(), _row()
    reference["metrics"]["contrast_global_grad_nce"] = 1. + 1.06e-4
    actual["metrics"]["contrast_global_grad_nce"] = (1. + 1.06e-4) * (1. + 1.39e-3)
    result = C.compare_trajectory([actual], [reference], [fp32], initial, "bf16")
    row = result["updates"][0]
    assert result["passed"] and row["limits"]["contrast_global_grad_nce"] == .01
    old_limit = 1.06e-4 / 1. + C.CRITERIA["bf16_noise_floor"]
    assert row["errors"]["contrast_global_grad_nce"] > old_limit  # the frozen rule rejected exactly this case
    assert all(row["limits"][k] == .01 for k in ("contrast_global_grad_base", "contrast_global_grad_nce", "contrast_global_grad_rnc"))
    high, above = _row(), _row()
    high["metrics"]["contrast_global_grad_rnc"] = 1.02
    above["metrics"]["contrast_global_grad_rnc"] = 1.02 * 1.015
    kept = C.compare_trajectory([above], [high], [fp32], initial, "bf16")  # never tighter than the previous rule
    assert kept["passed"] and kept["updates"][0]["limits"]["contrast_global_grad_rnc"] == pytest.approx(.02 + 1e-5, rel=1e-12)
    assert C.compare_trajectory([fp32], [fp32], [fp32], initial, "fp32")["updates"][0]["limits"] == {
        **dict.fromkeys(("total", "pose", "contrast_nce", "contrast_rnc", "contrast_global_grad_base",
                         "contrast_global_grad_nce", "contrast_global_grad_rnc", "gradient"), 1e-5), "student": .01, "teacher": .01}
    assert all(row["limits"][k] == .01 for k in ("total", "pose", "contrast_nce", "contrast_rnc"))
    # A diagnostic error above the floor still fails, in each of the three norms.
    for key in ("contrast_global_grad_base", "contrast_global_grad_nce", "contrast_global_grad_rnc"):
        bad = _row()
        bad["metrics"][key] = reference["metrics"][key] * 1.011
        assert not C.compare_trajectory([bad], [reference], [fp32], initial, "bf16")["passed"]
    # The floor never loosens FP32 nor the bf16 gradient and parameter limits.
    assert not C.compare_trajectory([actual], [fp32], [fp32], initial, "fp32")["passed"]
    for key in ("gradient", "student", "teacher"):
        shifted = _row()
        shifted[key] = torch.tensor([1.002], dtype=torch.float64)
        assert not C.compare_trajectory([shifted], [_row()], [_row()], initial, "bf16")["passed"]
