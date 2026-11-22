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


def test_plan_matches_worker_and_readout(monkeypatch):
    monkeypatch.syspath_prepend(str(gate.HERE))
    import contrastive_cn_restart_compare as reader
    import contrastive_cn_restart_leg as leg
    plan, _ = gate.package()
    assert plan["fork_step"] == leg.FORK_STEP == reader.FORK
    assert plan["trace_update"] == reader.FORK + 1
    assert plan["final_step"] == reader.END
    assert plan["n_parameters"] == leg.PARAMETERS == reader.PARAMETERS
    assert plan["n_ranks"] == plan["allocation"]["gpus"] == reader.RANKS
    assert plan["relative_gradient_roundoff_limit"] == reader.LIMIT
    assert plan["gradient_norm_floor"] == reader.DENOM_FLOOR
    rules = plan["readout_amendment"]
    assert rules["scalar_addend_parameters"] == {str(i): n for i, n in reader.SCALAR_ADDEND_PARAMETERS.items()}
    assert rules["warm_process_only_rng"] == list(reader.PROCESS_ONLY_CUDA_RNG)
    assert rules["require_process_only_rng_equal_to_ordinary"] is True
    assert rules["all_other_nonfloating_differences"] == "INCONCLUSIVE"
    assert (plan["fork"]["sha256"], plan["ordinary_reference"]["sha256"], plan["continuous_reference"]["sha256"]) == (
        reader.FORK_SHA, reader.ENDPOINT["ordinary"], reader.ENDPOINT["warm"])
    assert '--gpu-ore 1.0' in gate.PROTOCOL.read_text()
    assert 'scripts/diagnostics/contrastive_cn_restart.py" --validate-only' in gate.PROTOCOL.read_text()


@pytest.mark.parametrize("timeout_phase", [None, "ordinary", "warm", "compare"])
def test_phase_receipts_survive_subprocess_timeout(timeout_phase, tmp_path, monkeypatch):
    import contrastive_job

    root = tmp_path / "probe"
    plan = {"legs": ["ordinary", "warm"]}
    phases = [*plan["legs"], "compare"]
    monkeypatch.setattr(gate, "require_binding", lambda: plan)
    monkeypatch.setattr(gate, "commands", lambda p: (root, [["ordinary"], ["warm"]], ["compare"], []))
    monkeypatch.setattr(contrastive_job, "validate_allocation", lambda kind: None)
    monkeypatch.setattr(gate, "file_sha", lambda path: "3" * 64)
    monkeypatch.setattr(gate.subprocess, "check_output", lambda cmd, text: "0, GPU-0, bus, A100")
    for key, value in {"TORCH_DIST_INIT_BARRIER": "1", "SLURM_JOB_ID": "fixture",
                       "WEARUSFM_CODE_SHA": "6" * 40, "SLURM_JOB_END_TIME": "3000000000"}.items():
        monkeypatch.setenv(key, value)

    def execute(cmd, deadline):
        phase, = cmd
        if phase == timeout_phase:
            raise subprocess.TimeoutExpired(cmd, 1)
        if phase == "compare":
            (root / "report.json").write_text("{}")
    monkeypatch.setattr(gate, "checked_command", execute)
    if timeout_phase:
        with pytest.raises(subprocess.TimeoutExpired):
            gate.run(False)
        expected = phases[:phases.index(timeout_phase) + 1]
        receipt = json.loads((root / "execution_failure.json").read_text())
        assert receipt["phase"] == timeout_phase and receipt["completed_phases"] == expected[:-1]
        assert receipt["status"] == "INCONCLUSIVE" and receipt["training_approved"] is False
    else:
        gate.run(False)
        assert not (root / "execution_failure.json").exists()
        expected = phases
    records = [json.loads(row) for row in (root / "phase_timings.jsonl").read_text().splitlines()]
    assert [record["phase"] for record in records] == [phase for phase in expected for _ in range(2)]
    for phase, start, end in zip(expected, records[::2], records[1::2], strict=True):
        assert start["event"] == "start" and start["unix_s"] > 0
        assert end["event"] == ("failed" if phase == timeout_phase else "completed")
        assert end["elapsed_s"] >= 0
