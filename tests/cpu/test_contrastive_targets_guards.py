"""Independent negative barriers for the prospective target cohort (M-1/N-4)."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import contrastive_prepare as P
import contrastive_run as R
import contrastive_benchmark_readout as B
from test_contrastive_prepare import fixture_run


def write(path, value):
    path.write_text(json.dumps(value))
    return path


def audit_payload(cfg, summary, inputs):
    return {"source_sha256": P.source_digest(), "inputs": inputs,
            "steps": 1024, "world_size": 4, "workers_per_rank": 6, "passed": True,
            "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "expected_datasets": summary["contrastive"]["expected_datasets"],
            "calibration_panel": summary["contrastive"]["coverage_partial_block"]}


def test_validate_audit_config_digest_barrier_is_reached_with_current_source(tmp_path):
    cfg, summary, _ = fixture_run(tmp_path / "calibration")
    paths = {name: write(tmp_path / (name + ".json"), {}) for name in ("manifest", "scales", "window_seconds", "pose_stats")}
    data = audit_payload(cfg, summary, {name: R.sha(path) for name, path in paths.items()})
    path = write(tmp_path / "audit.json", data)
    R.validate_audit(path, paths, cfg)
    # Relabelled historical config: every outer source/input/coverage guard passes.
    historical_cfg = copy.deepcopy(cfg)
    historical_cfg["loader"].update(rms_window_ms=25., ms_fast_window_ms=50.)
    data["config_sha256"] = hashlib.sha256(json.dumps(historical_cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write(path, data)
    with pytest.raises(ValueError, match="audit stream/config differs"):
        R.validate_audit(path, paths, cfg)


def test_read_preflight_config_digest_barrier_is_reached_with_current_source(tmp_path, monkeypatch):
    folder = tmp_path / "calibration"
    cfg, summary, _ = fixture_run(folder)
    receipt = P.calibration_report(folder)
    receipt_path = write(tmp_path / "calibration_receipt.json", receipt)
    cohort = tmp_path / "cohort"
    monkeypatch.setattr(sys, "argv", ["contrastive_prepare", "calibrated-configs", "--receipt", str(receipt_path),
                                     "--run-dir", str(folder), "--out", str(cohort)])
    P.main()
    audit = audit_payload(cfg, summary, receipt["input_sha256"])
    audit["config_sha256"] = "0" * 64
    audit_path = write(tmp_path / "audit.json", audit)
    with pytest.raises(ValueError, match="different inputs/config/source"):
        B.read_preflight(cohort / "cohort.json", {arm: tmp_path / arm for arm in ("C0", "CN", "CR")},
                        {arm: tmp_path / arm for arm in ("CN_A", "CN_B", "CR_A", "CR_B")}, tmp_path / "unread_cuda.json",
                        cpu_audit_path=audit_path, calibration_path=receipt_path, calibration_run_dir=folder,
                        cpu_audit_sacct_path=tmp_path / "unread_accounting.json")


def test_current_calibration_receipt_guard_is_reached_after_recipe_validation(tmp_path):
    folder = tmp_path / "calibration"
    fixture_run(folder)
    receipt = P.calibration_report(folder)
    P.authenticate_calibration(receipt, folder)
    assert receipt["source_sha256"] == P.source_digest()
    altered = copy.deepcopy(receipt)
    altered["weights"]["nce"] *= 1.001
    with pytest.raises(ValueError, match="calibration receipt differs from authenticated recalculation"):
        P.authenticate_calibration(altered, folder)


def test_plan_pointer_and_target_fields_agree_with_actual_review_config_builder():
    assert P.PLAN == ROOT / "results/infra/contrastive_targets_20261118/plan.json"
    plan = P._read(P.PLAN)
    configs = P.build_review_configs(P._read(P.BASELINE))
    for arm in ("C0", "CN", "CR"):
        assert plan["arms"][arm]["rms_window_ms"] == configs[arm]["loader"]["rms_window_ms"] == 100.
        assert plan["arms"][arm]["ms_fast_window_ms"] == configs[arm]["loader"]["ms_fast_window_ms"] == 125.
        assert plan["arms"][arm]["target_cli_args"] == ["--rms-window-ms", "100", "--ms-fast-window-ms", "125"]
        assert configs[arm]["target_version"] == "step1_psd_power_shape_energy_fast125_v3"
    assert configs["calibration"]["loader"]["rms_window_ms"] == 100.
    assert configs["calibration"]["loader"]["ms_fast_window_ms"] == 125.


def test_readout_text_is_descriptive_about_both_changed_targets(tmp_path):
    reference = P._read(B.HISTORICAL_INITIAL_REFERENCE)
    rows = [{"step": i + 1, **{key: series[i] for key, series in reference["historical_first8"].items()}} for i in range(8)]
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = B.initial_calibration_description(tmp_path)
    assert result["descriptive_only"] is True
    assert "target supports changed from RMS25/fast50 to RMS100/fast125" in result["interpretation"]
    assert "even update 1 need not match" in result["interpretation"]
