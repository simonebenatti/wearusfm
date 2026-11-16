import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from wearusfm.training.contrastive_objectives import ContrastiveConfig

spec = importlib.util.spec_from_file_location("contrastive_audit", Path(__file__).resolve().parents[2]/"scripts/contrastive_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def batch(step, pose=True, hidden=False):
    return SimpleNamespace(index=step, rows=[dict(dataset="A", subject=str(s), session="one") for s in range(4)],
        windows=[SimpleNamespace(trial=None, start=step*2000, stop=step*2000+200) for s in range(4)],
        fs=[200.]*4, counts=[1]*4, n_patches=[8]*4, qc_valid=np.ones(4,dtype=bool),
        visible=np.full((4,8),not hidden,dtype=bool),
        pose=[(np.zeros((8,20)),np.ones((8,20),dtype=bool)) for s in range(4)] if pose else None)


def config():
    return ContrastiveConfig(min_negatives=1,capacity_per_dataset=8,fill_steps=0,rnc_stride=2,enforce_coverage=False)


def test_audit_uses_real_masks_and_reports_support_not_embedding_success():
    good=audit.audit_batches(([batch(i)] for i in range(8)),config(),["A"])
    assert good["passed"] and good["coverage_partial_block"]["nce_fraction"]==1
    assert all(r["rnc_windows"]==4 and r["pose_subjects"]==4 for r in good["records"])
    hidden=audit.audit_batches(([batch(i,hidden=True)] for i in range(8)),config(),["A"])
    assert not hidden["passed"] and hidden["coverage_partial_block"]["nce_fraction"]==0
    assert hidden["coverage_partial_block"]["rnc_active_fraction"]==1
    assert hidden["origin_support_sha256"]!=good["origin_support_sha256"]


def test_audit_does_not_confuse_many_unlabelled_windows_with_rnc_support():
    result=audit.audit_batches(([batch(i,pose=False)] for i in range(8)),config(),["A"])
    assert not result["passed"] and result["coverage_partial_block"]["rnc_active_fraction"]==0
    assert result["coverage_partial_block"]["nce_fraction"]==1


def test_missing_dataset_and_wrong_stream_order_fail():
    result=audit.audit_batches(([batch(i)] for i in range(8)),config(),["A","absent"])
    assert not result["passed"] and result["coverage_partial_block"]["datasets"]["absent"]["queries"]==0
    with pytest.raises(ValueError,match="stream index"):
        audit.audit_batches([[batch(1)]],config(),["A"])
