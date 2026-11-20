"""r6 probe: current behaviour of read_preflight branches that no repository test exercises."""
import copy, json
from pathlib import Path
import pytest
import contrastive_benchmark_readout as B
import contrastive_numeric_profile as NP
import test_contrastive_targets_readout as T
import test_contrastive_benchmark_readout as TB


def test_B04_reload_origin_suffix_mismatch_is_refused(tmp_path, monkeypatch):
    kwargs, *_ = T.readout_fixture(tmp_path, monkeypatch)
    original = B._run_evidence
    def evidence(folder, arm, cohort, cfg, provenance, first_update=1, **kw):
        s, c, records, pins = original(folder, arm, cohort, cfg, provenance, first_update, **kw)
        if Path(folder).name == "CR_B":
            records = copy.deepcopy(records); records[3]["origin"] = ["other"]
        return s, c, records, pins
    monkeypatch.setattr(B, "_run_evidence", evidence)
    with pytest.raises(ValueError, match="continuous suffix"):
        B.read_preflight(**kwargs)


def test_B16_old_trajectory_tolerance_violation_stays_descriptive(tmp_path, monkeypatch):
    kwargs, *_ = T.readout_fixture(tmp_path, monkeypatch)
    original = B._run_evidence
    def evidence(folder, arm, cohort, cfg, provenance, first_update=1, **kw):
        s, c, records, pins = original(folder, arm, cohort, cfg, provenance, first_update, **kw)
        if Path(folder).name == "CN":
            records = copy.deepcopy(records); records[9]["total"] = 2.; records[200]["grad_norm"] = 5.
        return s, c, records, pins
    monkeypatch.setattr(B, "_run_evidence", evidence)
    result = B.read_preflight(**kwargs)
    drift = result["trajectory_drift_descriptive"]["CN"]["fields"]
    print("\nstatus", result["status"], "outside", drift["total"]["outside_count"], drift["grad_norm"]["outside_count"])
    assert result["status"] == "PASSED_FOR_REVIEW" and drift["total"]["outside_count"] == 1


def test_B14_uniform_wrong_profile_is_inconclusive_not_failed(tmp_path, monkeypatch):
    kwargs, *_ = T.readout_fixture(tmp_path, monkeypatch)
    for folder in [*kwargs["full_dirs"].values(), *kwargs["resumed_dirs"].values()]:
        for rank in range(4):
            path = Path(folder) / f"numeric_profile.rank{rank}.json"
            row = json.loads(path.read_text()); row["flags"]["cudnn_tf32"] = True
            path.write_text(json.dumps(row))
    result = B.read_preflight(**kwargs)
    ctx = result["numeric_context_across_runs"]
    print("\nstatus", result["status"], "missing", len(ctx["missing"]), ctx["missing"][0])
    assert result["status"] == "INCONCLUSIVE_REQUIRES_REVIEW" and len(ctx["missing"]) == 28 and result["failure_reasons"] == []


def test_B11_cuda_receipt_without_rank_profile_is_refused(tmp_path, monkeypatch):
    captured = {}
    real = TB.B._cuda_evidence
    def spy(path, *args):
        captured["report"] = json.loads(Path(path).read_text()); captured["args"] = args
        return real(path, *args)
    monkeypatch.setattr(TB.B, "_cuda_evidence", spy)
    TB.test_cuda_receipt_binds_calibration_config_and_complete_numeric_checks(tmp_path)
    report = captured["report"]
    for row in report["ranks"]:
        row.pop("numeric_profile")
    path = tmp_path / "cuda_noprofile.json"; path.write_text(json.dumps(report))
    with pytest.raises(NP.NumericProfileUnavailable) as err:
        real(path, *captured["args"])
    print("\nrefused:", type(err.value).__name__, str(err.value)[:90])
