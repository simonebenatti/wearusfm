"""r6 probe: read_preflight maps compare_three_paths outcomes; evidence pins of the comparison."""
import pytest
import contrastive_benchmark_readout as B
import contrastive_target_resume
import test_contrastive_targets_readout as T


@pytest.mark.parametrize("returned,expected", [
    ({"status": "FAILED_REQUIRES_REVIEW", "passed": False}, "FAILED_REQUIRES_REVIEW"),
    ({"status": "INCONCLUSIVE", "passed": False}, "INCONCLUSIVE_REQUIRES_REVIEW"),
    ({"status": "PASSED", "passed": True}, "PASSED_FOR_REVIEW"),
])
def test_resume_outcome_mapping(tmp_path, monkeypatch, returned, expected):
    kwargs, *_ = T.readout_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(contrastive_target_resume, "compare_three_paths", lambda *a, **k: dict(returned))
    result = B.read_preflight(**kwargs)
    print("\n", returned["status"], "->", result["status"], result["failure_reasons"], result["inconclusive_reasons"])
    assert result["status"] == expected and result["training_approved"] is False


def test_comparison_evidence_pins_are_replaced(tmp_path, monkeypatch):
    kwargs, *_ = T.readout_fixture(tmp_path, monkeypatch)
    pins = {phase: {"trace-0.jsonl": "T" * 64, "start.json": "S" * 64, "checkpoint.pt": "C" * 64}
            for phase in ("continuous", "reload_a", "reload_b")}
    monkeypatch.setattr(contrastive_target_resume, "compare_three_paths",
                        lambda *a, **k: {"status": "PASSED", "passed": True, "evidence_sha256": pins})
    result = B.read_preflight(**kwargs)
    kept = result["arms"]["CN"]["resume"]["evidence_sha256"]
    print("\nkeys kept:", sorted(kept), {k: sorted(v) for k, v in kept.items()})
    assert "continuous" not in kept and not any("trace-0.jsonl" in v or "start.json" in v for v in kept.values())
