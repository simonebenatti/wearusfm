"""Operational review regressions, without a cluster or any submission."""
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from pose_temponet import __main__ as CLI, execution as E, provenance as P, resources as R
from test_pose_temponet_execution import measured_pack


@pytest.mark.parametrize("seconds", [0.1, 277.5, 2787.5, 2787.51, 2800.001])
def test_measured_walltime_rounds_before_packing_commands_and_allocation(seconds, monkeypatch):
    from pose_temponet import runner
    row = R._job("fixture", "neural", seconds, gpus=1)
    expected = 60 * math.ceil(math.ceil(2 * seconds) / 60)
    assert row["requested_seconds"] == expected
    assert row["allocation_seconds"] == expected + 180
    assert row["gpu_hours"] == pytest.approx((expected + 180) / 3600)
    graph = E.job_graph()
    pack = measured_pack(graph)
    pack[0]["requested_seconds"] = expected
    ready = E.apply_training_pack(graph, pack)
    job = next(j for j in ready if j["stage"] == "controls")
    monkeypatch.setattr(E, "verify_closed_form_release", lambda *a: None)
    command = E.submit_command({"submission_released": True}, job,
                               checkout_sha="a" * 40, plan_path="/work/plan.json", plan_sha256="b" * 64)
    wall = "%02d:%02d:00" % (expected // 3600, expected // 60 % 60)
    assert "--time=" + wall in command
    for key, value in {"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "8", "SLURM_GPUS_ON_NODE": "1",
                       "SLURM_NNODES": "1", "SLURM_MEM_PER_NODE": "32768", "SLURM_RESTART_COUNT": "0"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw:
                        f"TimeLimit={wall} Partition=boost_usr_prod NumCPUs=8 NumNodes=1 "
                        "AllocTRES=cpu=8,mem=32G,node=1,gres/gpu=1")
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    assert runner._allocation(job)["gpus"] == 1


@pytest.mark.parametrize("invalid", [0, -60, True, 120.0, 119, 121, None])
def test_graph_and_pack_reject_non_minute_training_allocations(invalid):
    graph = E.job_graph()
    pack = measured_pack(graph)
    pack[0]["requested_seconds"] = invalid
    with pytest.raises(ValueError, match="whole-minute"):
        E.apply_training_pack(graph, pack)
    if invalid is not None:
        graph[0]["resources"]["requested_seconds"] = invalid
        with pytest.raises(ValueError, match="inclusive lane"):
            E.validate_graph(graph)


def test_submitter_agent_is_accepted_by_the_actual_parser():
    tree = ast.parse((ROOT / "scripts/submit_job.py").read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and any(isinstance(a, ast.Constant) and a.value == "--agente" for a in n.args))
    supported = ast.literal_eval(next(k.value for k in call.keywords if k.arg == "choices"))
    job = E.job_graph()[0]
    command = E.submit_command({"submission_released": True}, job,
                               checkout_sha="a" * 40, plan_path="/work/p.json", plan_sha256="b" * 64)
    assert command[command.index("--agente") + 1] == "gpt"
    assert command[command.index("--agente") + 1] in supported


def test_launcher_preserves_module_pythonpath_and_dispatches_frozen_code(tmp_path):
    text = (ROOT / "scripts/slurm/pose_temponet.sbatch").read_text()
    line = next(line for line in text.splitlines() if line.startswith("export PYTHONPATH="))
    env = {**os.environ, "CODE_DIR": "/frozen", "PYTHONPATH": "/module/numpy:/module/torch"}
    result = subprocess.run(["bash", "-c", line+'\nprintf "%s" "$PYTHONPATH"'],
                            env=env, capture_output=True, text=True, check=True)
    assert result.stdout == "/frozen/src:/frozen/scripts:/module/numpy:/module/torch"
    frozen = tmp_path / "frozen"
    scripts = frozen / "scripts/slurm"
    scripts.mkdir(parents=True)
    marker = tmp_path / "dispatched"
    (scripts / "lib_code_dir.sh").write_text('code_dir_setup() { [ "$EXPECTED_SHA" = "' + "a"*40 + '" ]; }\n')
    (scripts / "pose_temponet.sbatch").write_text('printf frozen > "' + str(marker) + '"\n')
    env.update(CODE_DIR=str(frozen), WORK=str(tmp_path), EXPECTED_SHA="a"*40)
    subprocess.run(["bash", str(ROOT / "scripts/slurm/pose_temponet_dispatch.sbatch")], env=env, check=True)
    assert marker.read_text() == "frozen"
    job = E.job_graph()[1]
    command = E.submit_command({"submission_released": True}, job,
                               checkout_sha="a" * 40, plan_path="/work/p.json", plan_sha256="b" * 64)
    assert "scripts/slurm/pose_temponet_dispatch.sbatch" in command


@pytest.mark.parametrize("commit", ["HEAD", "a" * 7, "a" * 39, "a" * 41, "g" * 40])
def test_commands_reject_non_full_plan_commit_before_any_git_or_plan_read(commit, monkeypatch):
    monkeypatch.setattr(P, "load_plan", lambda *a, **k: pytest.fail("read before commit guard"))
    with pytest.raises(ValueError, match="full plan commit"):
        CLI.main(["commands", "--plan", "absent", "--checkout-sha", "b"*40, "--plan-commit", commit,
                  "--plan-repository-path", "plan.json", "--stage", "preflight"])


@pytest.mark.parametrize("damage", ["short_sha", "absolute", "parent", "digest", "missing", None])
def test_checkout_checks_plan_before_any_campaign_mutation(tmp_path, damage):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "checkout_was_started"
    fake_git = fake_bin / "git"
    fake_git.write_text(f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
                        "arg=sys.argv[-1]\n"
                        "if arg.endswith(':plan.json'): print('{}',end='')\n"
                        "elif arg.endswith(':scripts/slurm/campaign_checkout.sbatch'):\n"
                        f" print('touch \\\"{marker}\\\"; exit 19')\n"
                        "else: sys.exit(1)\n")
    fake_git.chmod(0o755)
    fake_hash = fake_bin / "sha256sum"
    fake_hash.write_text(f"#!{sys.executable}\nimport hashlib,sys\nprint(hashlib.sha256(sys.stdin.buffer.read()).hexdigest()+'  -')\n")
    fake_hash.chmod(0o755)
    env = {**os.environ, "PATH": str(fake_bin)+os.pathsep+os.environ["PATH"], "WORK": str(tmp_path),
           "CAMPAIGN": "example", "CAMPAIGN_SHA": "a"*40, "EXPECTED_SHA": "a"*40,
           "CODE_DIR": str(tmp_path / "wearusfm_campaigns/example"), "TCN_PLAN_COMMIT": "b"*40,
           "TCN_PLAN_REPO_PATH": "plan.json", "TCN_PLAN_SHA256": hashlib.sha256(b"{}").hexdigest()}
    if damage == "short_sha": env["TCN_PLAN_COMMIT"] = "b"*7
    if damage == "absolute": env["TCN_PLAN_REPO_PATH"] = "/plan.json"
    if damage == "parent": env["TCN_PLAN_REPO_PATH"] = "x/../plan.json"
    if damage == "digest": env["TCN_PLAN_SHA256"] = "0"*64
    if damage == "missing": env["TCN_PLAN_REPO_PATH"] = "missing.json"
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm/pose_temponet_checkout.sbatch")],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert marker.exists() is (damage is None), result.stderr


def operator_file(tmp_path, observed="2026-11-19T12:00:00+00:00"):
    path = tmp_path / "checks.json"
    value = {"work_free_bytes": 14*1024**3, "pending_write_bytes": 1024**3,
             "no_duplicate_jobs": True, "serial_lane_checked": True, "reservations_checked": True,
             "balance_checked": True, "observed_at": observed}
    path.write_text(json.dumps(value))
    return path, value


@pytest.mark.parametrize("finished", ["2026-11-19T12:00:00Z", "2026-11-19T12:00:01Z", None,
                                     "2026-11-19T11:00:00", "garbage"])
def test_resource_plan_requires_new_checks_after_both_authenticated_benchmarks(tmp_path, monkeypatch, finished):
    path, _ = operator_file(tmp_path)
    observed_stages = []
    def verified(pin, plan, **kwargs):
        observed_stages.append(kwargs["stage"])
        assert kwargs["verify_artifacts"] is False
        return {"details": {"finished_at": finished if kwargs["stage"] == "gpu_benchmark" else "2026-11-19T10:00:00Z"},
                "receipt_sha256": "a"*64}
    monkeypatch.setattr(P, "verified_receipt", verified)
    with pytest.raises(ValueError, match="timestamp|newer"):
        CLI.refresh_operator_checks({}, path, {"path": "cpu.json"}, {"path": "gpu.json"})
    assert observed_stages == ["cpu_benchmark", "gpu_benchmark"]


def test_resource_plan_replaces_stale_quota_and_binds_new_file(tmp_path, monkeypatch):
    path, checks = operator_file(tmp_path)
    monkeypatch.setattr(P, "verified_receipt", lambda *a, **kw:
                        {"details": {"finished_at": "2026-11-19T11:59:59Z"}, "receipt_sha256": "a"*64})
    original = {"operator_checks": {"work_free_bytes": 999}, "jobs": []}
    result = CLI.refresh_operator_checks(original, path, {"path": "cpu.json"}, {"path": "gpu.json"})
    assert result["operator_checks"] == checks
    assert result["operator_evidence"] == P.file_pin(path)
    assert original["operator_checks"]["work_free_bytes"] == 999
    assert set(result["resource_operator_freshness"]["benchmarks"]) == {"cpu_benchmark", "gpu_benchmark"}
    with pytest.raises(SystemExit):
        CLI.parser().parse_args(["resource-plan", "--plan", "p", "--cpu-receipt", "c", "--gpu-receipt", "g", "--output", "o"])


def test_review_must_name_actual_code_or_source_before_materializing_it(monkeypatch):
    monkeypatch.setattr(CLI.subprocess, "check_output", lambda *a, **k: b"APPROVATO historical review")
    with pytest.raises(ValueError, match="review must name"):
        CLI.release_plan({"reader": {"source_sha256": "b"*64}}, review_commit="c"*40,
                         review_path="docs/review.md", reviewed_code_sha="a"*40, operator_checks={})


@pytest.mark.parametrize("status", ["control_non_calibrated", "nondiagnostic", "diagnostic"])
def test_no_neural_command_before_combined_closed_form_gate(tmp_path, monkeypatch, status):
    from pose_temponet import controls
    jobs = E.apply_training_pack(E.job_graph(), measured_pack(E.job_graph()))
    plan = {"jobs": jobs, "submission_released": True}
    calibrations = [j for j in jobs if j["stage"] == "wls_calibration"]
    proofs, receipts = {}, {}
    for fold, job in enumerate(calibrations):
        path = tmp_path / f"calibration-{fold}.json"
        path.write_text(json.dumps({"outer_fold": fold, "status": status}))
        pin = P.file_pin(path)
        # Local copies preserve original hashes, while remote paths remain in
        # the authenticated receipt. No remote access is needed for commands.
        receipts[job["id"]] = {"artifacts": {"calibration": {**pin, "path": f"/remote/{fold}.json"}}}
        proofs[job["id"]] = {"receipt": {"path": job["id"]}, "calibration": pin}
    monkeypatch.setattr(E, "verify_completed_predecessors", lambda p, ps: set(ps))
    monkeypatch.setattr(P, "verified_receipt", lambda pin, *a, **kw: receipts[pin["path"]])
    seen = []
    def combine(rows):
        seen.extend(rows)
        return {"status": status}
    monkeypatch.setattr(controls, "combine_reference_folds", combine)
    job = next(j for j in jobs if j["stage"] == "controls")
    kwargs = dict(checkout_sha="a"*40, plan_path="/work/p.json", plan_sha256="b"*64, completed_proofs=proofs)
    if status == "diagnostic":
        assert E.submit_command(plan, job, **kwargs)
    else:
        with pytest.raises(controls.GateNotPassed) as caught:
            E.submit_command(plan, job, **kwargs)
        assert caught.value.status == status
        assert caught.value.gate == {"status": status}
    assert sorted(x["outer_fold"] for x in seen) == [0, 1]
    proof = proofs[calibrations[0]["id"]]
    proof["calibration"] = {**proof["calibration"], "sha256": "f"*64}
    with pytest.raises(ValueError, match="mirrored calibration differs"):
        E.submit_command(plan, job, **kwargs)
    with pytest.raises(ValueError, match="both closed-form controls"):
        E.submit_command(plan, job, **{**kwargs, "completed_proofs": {}})


def test_resource_formula_counts_memory_snapshots_separately_from_artifact_io(monkeypatch):
    from test_pose_temponet_resources import receipts, gate
    observed = {}
    original = R._job
    def capture(name, phase, raw_seconds, **kwargs):
        observed[name] = raw_seconds
        return original(name, phase, raw_seconds, **kwargs)
    monkeypatch.setattr(R, "_job", capture)
    cpu, gpu = receipts()
    gate(cpu, gpu)
    before = observed.copy()
    # Increase filesystem write time only. Exactly 24 state-equivalents per
    # search and three per refit must be added before the safety factor.
    for cells in gpu["cells"].values():
        for cell in cells.values():
            cell["artifact_write_seconds"] = [1.002]*10
    gate(cpu, gpu)
    assert observed["select-TCN_post-real"] - before["select-TCN_post-real"] == pytest.approx(24)
    assert observed["refit-TCN_post"] - before["refit-TCN_post"] == pytest.approx(3)
    assert observed["final-inference"] == before["final-inference"]
    before = observed.copy()
    gpu["cells"]["TCN_post"]["full"]["artifact_read_seconds"] = [1.002]*10
    gpu["cells"]["TCN_post"]["full"]["save_seconds"] = [1.001]*10
    gate(cpu, gpu)
    assert observed["final-inference"] - before["final-inference"] == pytest.approx(3)
    assert observed["refit-TCN_post"] - before["refit-TCN_post"] == pytest.approx(3)


@pytest.mark.parametrize("status", ["diagnostic", "control_non_calibrated", "control_non_diagnostic"])
def test_commands_persist_closed_form_gate_before_emitting_any_command(tmp_path, monkeypatch, capsys, status):
    from pose_temponet import controls
    plan = {"jobs": [E.job_graph()[0], next(j for j in E.job_graph() if j["stage"] == "controls")]}
    plan_path, gate_path = tmp_path / "plan.json", tmp_path / "gate.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(P, "load_plan", lambda *a, **kw: plan)
    monkeypatch.setattr(CLI.subprocess, "check_output", lambda *a, **kw: plan_path.read_bytes())
    monkeypatch.setattr(E, "verify_completed_predecessors", lambda *a: {"fold0", "fold1"})
    gate = {"status": status, "linear_gate": {"passed": status == "diagnostic"}}
    def qualify(*args):
        if status != "diagnostic":
            raise controls.GateNotPassed("measured gate did not qualify", status=status, gate=gate)
        return gate
    monkeypatch.setattr(E, "verify_closed_form_release", qualify)
    calls = []
    def submit(*args, **kwargs):
        assert json.loads(gate_path.read_text())["gate"] == gate
        calls.append(args[1]["stage"])
        return ["python3", "scripts/submit_job.py", "--export=ALL"]
    monkeypatch.setattr(E, "submit_command", submit)
    argv = ["commands", "--plan", str(plan_path), "--checkout-sha", "a"*40, "--plan-commit", "b"*40,
            "--plan-repository-path", "plan.json", "--stage", "preflight", "--stage", "controls",
            "--closed-reference-gate-output", str(gate_path)]
    if status == "diagnostic":
        assert CLI.main(argv) == 0
        assert calls == ["preflight", "controls"]
        assert capsys.readouterr().out.count("scripts/submit_job.py") == 2
    else:
        with pytest.raises(controls.GateNotPassed):
            CLI.main(argv)
        assert calls == []
        assert capsys.readouterr().out == ""
    evidence = json.loads(gate_path.read_text())
    assert evidence["status"] == status and evidence["gate"] == gate
    assert evidence["phase_plan"] == P.file_pin(plan_path)
    original = gate_path.read_bytes()
    with pytest.raises(FileExistsError):
        CLI.main(argv)
    assert gate_path.read_bytes() == original


def test_controls_commands_require_an_explicit_new_gate_artifact_path(monkeypatch):
    monkeypatch.setattr(P, "load_plan", lambda *a, **kw: pytest.fail("read before output-path guard"))
    with pytest.raises(ValueError, match="closed-reference-gate-output"):
        CLI.main(["commands", "--plan", "absent", "--checkout-sha", "a"*40, "--plan-commit", "b"*40,
                  "--plan-repository-path", "plan.json", "--stage", "controls"])
