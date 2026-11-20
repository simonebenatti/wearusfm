"""Effect probe (sk1r63): what a FAILED/INCONCLUSIVE common-checkpoint comparison becomes at read_preflight and
authenticate_preflight. Prints one JSON line per case; never asserts the expected value (it is run on original and mutated code)."""
import json

import pytest

import contrastive_benchmark_readout as B
import contrastive_target_resume
from test_contrastive_targets_readout import readout_fixture


@pytest.mark.parametrize("outcome", [{"status": "FAILED_REQUIRES_REVIEW", "passed": False},
                                     {"status": "INCONCLUSIVE", "passed": False}])
def test_effect(tmp_path, monkeypatch, outcome):
    kwargs, *_ = readout_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(contrastive_target_resume, "compare_three_paths", lambda *a, **k: dict(outcome))
    result = B.read_preflight(**kwargs)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(result))
    try:
        B.authenticate_preflight(receipt)
        accepted = True
    except ValueError:
        accepted = False
    print("EFFECT", json.dumps({"compare_status": outcome["status"], "readout_status": result["status"],
                               "failure_reasons": result["failure_reasons"], "inconclusive_reasons": result["inconclusive_reasons"],
                               "stored_resume_CN": result["arms"]["CN"]["resume"]["status"],
                               "authenticate_preflight_accepts": accepted}))
