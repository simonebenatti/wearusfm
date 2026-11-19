"""Legacy evidence guards must remain on every consumer's execution path."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest
from frozen_campaign_support import run_frozen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import contrastive_benchmark_readout as B
import contrastive_prepare as P
import contrastive_run as R
import contrastive_source_compat as S

CALIBRATION = ROOT / "results/infra/contrastive_comparison_20261116/calibration_completed_20261117"
CURRENT_SOURCE = "c" * 64


class AfterAuditGate(Exception):
    """Stop the callsite test before any CUDA/checkpoint evidence is read."""


@pytest.fixture
def audit_evidence(tmp_path):
    config = {"contrastive": {}}
    input_path = tmp_path / "manifest.json"
    input_path.write_text("{}\n")
    inputs = {"manifest": input_path}
    report = {
        "source_sha256": S.HISTORICAL_SOURCE_SHA256,
        "inputs": {"manifest": hashlib.sha256(input_path.read_bytes()).hexdigest()},
        "steps": 1024,
        "world_size": 4,
        "workers_per_rank": 6,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "expected_datasets": ["fixture"],
        "calibration_panel": {},
        "passed": True,
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report))
    return path, report, inputs, config


def _legacy_guard(monkeypatch, *, reject):
    calls = []

    def guard(kind, payload):
        calls.append((kind, payload))
        if reject:
            raise ValueError("legacy artifact pin rejected")
        return {"kind": kind}

    monkeypatch.setattr(S, "validate_historical_evidence", guard)
    return calls


@pytest.mark.parametrize("reject", [False, True])
def test_validate_audit_invokes_and_obeys_historical_guard(audit_evidence, monkeypatch, reject):
    path, report, inputs, config = audit_evidence
    monkeypatch.setattr(R, "source_digest", lambda: CURRENT_SOURCE)
    monkeypatch.setattr(R, "_coverage", lambda _: {"passed": True})
    calls = _legacy_guard(monkeypatch, reject=reject)

    if reject:
        with pytest.raises(ValueError, match="legacy artifact pin rejected"):
            R.validate_audit(path, inputs, config)
    else:
        assert R.validate_audit(path, inputs, config) == report
    assert calls == [("audit", path)]


@pytest.mark.parametrize("reject", [False, True])
def test_read_preflight_invokes_and_obeys_historical_audit_guard(audit_evidence, monkeypatch, reject):
    path, report, _, config = audit_evidence
    cohort_path = path.parent / "cohort.json"
    receipt_path = path.parent / "receipt.json"
    cohort_path.write_text(json.dumps({
        "endpoint": 576,
        "source_sha256": CURRENT_SOURCE,
        "configs": {arm: {"path": arm + ".json"} for arm in B.ARMS},
    }))
    receipt_path.write_text(json.dumps({"input_sha256": report["inputs"]}))
    # Other gates have their own tests. Reach the actual audit branch with only
    # three tiny JSON files, then stop before any GPU evidence/model loading.
    monkeypatch.setattr(B, "authenticate_calibration", lambda *args: None)
    monkeypatch.setattr(B, "calibrated_configs", lambda *args: {})
    monkeypatch.setattr(R, "validate_cohort", lambda *args: None)
    monkeypatch.setattr(P, "BASELINE", cohort_path)
    monkeypatch.setattr(P, "build_review_configs", lambda _: {"calibration": config})
    monkeypatch.setattr(P, "_coverage", lambda _: {"passed": True})

    def after_audit(*args):
        raise AfterAuditGate

    monkeypatch.setattr(B, "_cuda_evidence", after_audit)
    # The amended audit accounting gate follows the historical-source guard.
    # Isolate it here; its State/ExitCode/Elapsed checks have separate tests.
    import contrastive_release
    accounting_calls = []
    accounting_path = path.parent / "audit-sacct.json"

    def accounting(audit_path, sacct_path, *, expected_source):
        accounting_calls.append((audit_path, sacct_path, expected_source))
        return {"passed": True}

    monkeypatch.setattr(contrastive_release, "validate_audit_accounting", accounting)
    calls = _legacy_guard(monkeypatch, reject=reject)
    expected_error = ValueError if reject else AfterAuditGate
    with pytest.raises(expected_error, match="legacy artifact pin rejected" if reject else None):
        B.read_preflight(
            cohort_path,
            {arm: path.parent / arm for arm in B.ARMS},
            {arm + "_" + replica: path.parent / (arm + "-resumed-" + replica)
             for arm in ("CN", "CR") for replica in ("A", "B")},
            path.parent / "cuda.json",
            cpu_audit_path=path,
            calibration_path=receipt_path,
            calibration_run_dir=path.parent / "calibration-run",
            cpu_audit_sacct_path=accounting_path,
        )
    assert calls == [("audit", path)]
    assert accounting_calls == ([] if reject else [(path, accounting_path, CURRENT_SOURCE)])


def test_pinned_historical_receipt_still_requires_authenticated_recalculation(tmp_path):
    # Exercise the real historical exception on the exact reader release which
    # owns it. Current 100/125 sources must not inherit that exception.
    result = run_frozen(
        tmp_path, "8905e346b03e5d173cccc124aa40032bf588a191",
        ("src", "scripts", "results/infra/contrastive_comparison_20261116",
         "results/infra/tokenizer_execution_20261103/19618/T100-G/config.json"), """
import json,shutil
from unittest.mock import patch
import contrastive_prepare as P
import contrastive_source_compat as S
calibration=root/'results/infra/contrastive_comparison_20261116/calibration_completed_20261117'
receipt=P._read(calibration/'calibration_report.json')
metadata=S.validate_historical_evidence('calibration_receipt',receipt)
assert metadata['historical_source_sha256']==receipt['source_sha256']
current=P.source_digest()
assert current!=receipt['source_sha256']
folder=root/'mutated-run'
shutil.copytree(calibration/'run',folder)
summary_path=folder/'summary.json'
summary=P._read(summary_path)
summary['contrastive']['provenance']['source_sha256']=current
summary_path.write_text(json.dumps(summary))
guard_calls,recalculations=[],[]
original_guard,original_report=S.validate_historical_evidence,P.calibration_report
def guard(kind,payload):
    guard_calls.append(kind)
    return original_guard(kind,payload)
def recalculate(run_dir):
    measured=original_report(run_dir)
    recalculations.append(measured)
    return measured
with patch.object(S,'validate_historical_evidence',guard),patch.object(P,'calibration_report',recalculate):
    try:
        P.authenticate_calibration(receipt,folder)
    except ValueError as exc:
        assert 'differs from authenticated recalculation' in str(exc)
    else:
        raise AssertionError('pinned receipt bypassed authenticated recalculation')
assert guard_calls==['calibration_receipt']
assert len(recalculations)==1
assert recalculations[0]['source_sha256']==current
assert recalculations[0]['weights']==receipt['weights']
assert recalculations[0]['inputs']['summary.json']!=receipt['inputs']['summary.json']
print(json.dumps({'guard_calls':guard_calls,'recalculations':len(recalculations),'rejected':True}))
""")
    assert result == {"guard_calls": ["calibration_receipt"], "recalculations": 1, "rejected": True}
