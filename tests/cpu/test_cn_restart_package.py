"""The new diagnostic must not inherit permission from the November 20 release."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cn_restart_gate", ROOT / "scripts/diagnostics/contrastive_cn_restart.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_history_review_and_unbound_package_rejected(monkeypatch, tmp_path):
    plan, pins = gate.package()
    assert plan["legs"] == ["ordinary", "warm"]
    assert (plan["fork_step"], plan["trace_update"], plan["final_step"]) == (544, 545, 576)
    old = ROOT / "results/infra/contrastive_targets_20261118/releases/audit_20261120/review_binding.json"
    with pytest.raises(ValueError, match="precisely one"):
        review = json.loads(old.read_text())
        gate.review_fields((ROOT / review["review"]["path"]).read_text(), pins)
    monkeypatch.setattr(gate, "BINDING", tmp_path / "not_released.json")
    with pytest.raises(ValueError, match="New Claude review"):
        gate.require_binding()


@pytest.mark.parametrize("field", ["plan_sha256", "package_sha256", "status", "reviewed_commit"])
def test_changed_review_fields_rejected(field):
    pins = {"package_sha256": "a" * 64, "plan_sha256": "b" * 64}
    data = dict(schema="cn-restart-review-v1", status="APPROVED_FOR_DIAGNOSTIC",
                reviewed_commit="7" * 40, **pins)
    data[field] = "wrong"
    with pytest.raises(ValueError):
        gate.review_fields("```cn-restart-review\n" + json.dumps(data) + "\n```", pins)


def test_new_binding_checks_reviewed_git_bytes(tmp_path, monkeypatch):
    files = [tmp_path / "probe.py", tmp_path / "plan.json"]
    for f in files:
        f.write_text(f.name)
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "FILES", files[:1])
    monkeypatch.setattr(gate, "PLAN", files[1])
    sha = "7" * 40
    monkeypatch.setattr(gate, "git", lambda cmd, expr: expr.split(":")[1].encode() if expr.startswith(sha + ":") else b"")
    pins = {"package_sha256": "a" * 64, "plan_sha256": "b" * 64}
    data = dict(schema="cn-restart-review-v1", status="APPROVED_FOR_DIAGNOSTIC", reviewed_commit=sha, **pins)
    body = "```cn-restart-review\n" + json.dumps(data) + "\n```"
    assert gate.review_fields(body, pins) == data
    files[0].write_text("altered")
    with pytest.raises(ValueError, match="changed since Claude review"):
        gate.review_fields(body, pins)


def test_layout_cost_and_immutable_endpoints():
    plan, pins = gate.package()
    historical = json.loads(gate.checked_pin(plan["historical_job"], repository=True).read_text())
    assert plan["fork"] == historical["fork_checkpoint"]
    assert len(pins["plan_sha256"]) == len(pins["package_sha256"]) == 64
    assert plan["ordinary_reference"]["sha256"].startswith("bb39f339")
    assert plan["continuous_reference"]["sha256"].startswith("4743abaf")
    a = plan["allocation"]
    assert a["gpus"] * a["seconds"] / 3600 == 1
    assert a["cpus"] * a["seconds"] / 3600 == 8
    assert a["checkout_seconds"] * a["checkout_cpus"] == 960
    assert (plan["n_parameters"], plan["n_ranks"], plan["relative_gradient_roundoff_limit"],
            plan["gradient_norm_floor"]) == (270, 4, 1e-6, 1e-12)
    subprocess.run(["bash", "-n", gate.LAUNCHER], check=True)
