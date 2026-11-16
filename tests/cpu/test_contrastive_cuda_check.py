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
