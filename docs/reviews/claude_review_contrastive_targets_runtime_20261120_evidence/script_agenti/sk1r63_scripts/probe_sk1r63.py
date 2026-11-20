"""Independent probes (sk1r63) for read_preflight branches the repository tests do not exercise.

They reuse the repository's own readout_fixture and only change the stubbed outcome.
"""
import pytest

import contrastive_benchmark_readout as B
import contrastive_target_resume
from test_contrastive_targets_readout import readout_fixture


def _with_compare(monkeypatch, outcome):
    monkeypatch.setattr(contrastive_target_resume, "compare_three_paths", lambda *a, **k: dict(outcome))


def test_failed_common_checkpoint_comparison_blocks(tmp_path, monkeypatch):
    kwargs, *_ = readout_fixture(tmp_path, monkeypatch)
    _with_compare(monkeypatch, {"status": "FAILED_REQUIRES_REVIEW", "passed": False})
    result = B.read_preflight(**kwargs)
    assert result["status"] == "FAILED_REQUIRES_REVIEW"
    assert any("common-checkpoint" in r for r in result["failure_reasons"])


def test_inconclusive_common_checkpoint_comparison_blocks(tmp_path, monkeypatch):
    kwargs, *_ = readout_fixture(tmp_path, monkeypatch)
    _with_compare(monkeypatch, {"status": "INCONCLUSIVE", "passed": False})
    result = B.read_preflight(**kwargs)
    assert result["status"] == "INCONCLUSIVE_REQUIRES_REVIEW"
    assert result["failure_reasons"] == []


def test_reload_evidence_is_read_with_its_own_resume_phase(tmp_path, monkeypatch):
    kwargs, _, _, calls, _ = readout_fixture(tmp_path, monkeypatch)
    assert B.read_preflight(**kwargs)["status"] == "PASSED_FOR_REVIEW"
    phases = {name: kw.get("expected_phase", "full") for name, _, kw in calls}
    assert phases == {"C0": "full", "CN": "full", "CR": "full", "CN_A": "resume_a",
                      "CN_B": "resume_b", "CR_A": "resume_a", "CR_B": "resume_b"}


def test_trajectory_drift_remains_descriptive_inside_read_preflight(tmp_path, monkeypatch):
    kwargs, *_ = readout_fixture(tmp_path, monkeypatch)
    original = B._run_evidence

    def drifted(folder, arm, *args, **kw):
        summary, cfg, records, pins = original(folder, arm, *args, **kw)
        if arm == "CN" and kw.get("first_update", args[3] if len(args) > 3 else 1) == 1:
            records = [dict(r, total=r["total"] * 1.01) for r in records]
        return summary, cfg, records, pins

    monkeypatch.setattr(B, "_run_evidence", drifted)
    result = B.read_preflight(**kwargs)
    assert result["trajectory_drift_descriptive"]["CN"]["fields"]["total"]["outside_count"] == 256
    assert result["status"] == "PASSED_FOR_REVIEW" and result["failure_reasons"] == []


def test_reload_origin_suffix_mismatch_is_rejected(tmp_path, monkeypatch):
    kwargs, *_ = readout_fixture(tmp_path, monkeypatch)
    original = B._run_evidence

    def shifted(folder, arm, *args, **kw):
        summary, cfg, records, pins = original(folder, arm, *args, **kw)
        if kw.get("first_update") == 545:
            records = [dict(r, origin=[-1]) for r in records]
        return summary, cfg, records, pins

    monkeypatch.setattr(B, "_run_evidence", shifted)
    with pytest.raises(ValueError, match="continuous suffix"):
        B.read_preflight(**kwargs)
