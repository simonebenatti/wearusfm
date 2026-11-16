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
    return ContrastiveConfig(nce_weight=.1,rnc_weight=.1,min_negatives=1,capacity_per_dataset=8,fill_steps=0,rnc_stride=2,enforce_coverage=False)


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


def test_audit_reports_exact_calibration_panel_before_larger_audit():
    from dataclasses import replace
    result = audit.audit_batches(([batch(i)] for i in range(1024)), replace(config(), fill_steps=256), ["A"])
    panel = result["calibration_panel"]
    assert panel["updates"] == 256 and panel["through_step"] == 511
    assert panel["datasets"]["A"]["queries"] == 256 * 4
    assert result["coverage_partial_block"]["updates"] == 768
    row = result["records"][-1]
    assert row["contrast_nce_negative_count__A"] > 0
    assert row["contrast_rnc_max_instant_multiplicity"] == 1
    assert row["contrast_rnc_anchors"] == row["rnc_anchors"]


def test_descriptor_budget_checks_compute_limit_before_opening_stream(monkeypatch):
    from dataclasses import replace
    cfg = audit.R.sanity_config()
    cfg = replace(cfg, loader=replace(cfg.loader, open_sessions=64, block_pool=16, pose_root="pose"))
    monkeypatch.setattr(audit.resource, "getrlimit", lambda _: (1024, 1024))
    budget = audit.descriptor_budget(cfg)
    assert budget["estimated_open_descriptor_bound"] == 800
    assert budget["emg_cache_per_loader"] == budget["pose_cache_per_loader"] == 4
    monkeypatch.setattr(audit.resource, "getrlimit", lambda _: (256, 256))
    with pytest.raises(ValueError, match="no stream opened"):
        next(audit.replay(cfg, None, {}, {}, 1))


def test_audit_cache_cap_preserves_actual_loader_origins_masks_and_signal(tmp_path, monkeypatch):
    from dataclasses import replace
    from test_pretraining_loader import _tree, _cfg
    root, manifest = _tree(tmp_path)
    index = audit.R.L.ManifestIndex.load(manifest, [root])
    cfg = audit.R.sanity_config()
    cfg = replace(cfg, batch_size=4, loader=_cfg(open_sessions=64, block_pool=2, time_weighted=False, virtual=None))
    monkeypatch.setattr(audit.resource, "getrlimit", lambda _: (65536, 65536))
    streams = [audit.replay(cfg, index, {}, None, 16, world_size=1, workers=2, cache_cap=cap) for cap in (1, 64)]
    for left, right in zip(*streams):
        a, b = left[0], right[0]
        assert audit.identities_from_batch(a) == audit.identities_from_batch(b)
        assert np.array_equal(a.visible, b.visible)
        assert all(np.array_equal(x, y) for x, y in zip(a.signals, b.signals))
