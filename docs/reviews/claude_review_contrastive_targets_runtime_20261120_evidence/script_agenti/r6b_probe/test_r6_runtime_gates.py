"""r6 probe: runtime release gates are present (repository tests stub them out)."""
import pytest
import contrastive_release as REL
import contrastive_run as runner
import test_contrastive_run as TR

REAL_BINDING, REAL_ACCOUNTING = REL.validate_review_binding, REL.validate_audit_accounting


def test_runtime_rejects_unbound_review_and_missing_accounting(tmp_path, monkeypatch):
    release, cfg, inputs, audit = TR.release_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(REL, "validate_review_binding", REAL_BINDING)
    with pytest.raises(ValueError, match="authenticated package review binding"):
        runner.validate_release(release, "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)
    monkeypatch.setattr(REL, "validate_review_binding", lambda value: None)
    bad = dict(release, review_pin="another-commit")
    with pytest.raises(ValueError, match="review commit differs"):
        runner.validate_release(bad, "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)
    monkeypatch.setattr(REL, "validate_audit_accounting", REAL_ACCOUNTING)
    with pytest.raises(ValueError, match="complete artifact path"):
        runner.validate_release(release, "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)
    with pytest.raises(ValueError, match="completed new audit accounting"):
        runner.validate_release(dict(release, audit_sacct=None), "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)
