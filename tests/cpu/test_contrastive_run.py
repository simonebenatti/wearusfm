import copy
from pathlib import Path
import json
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import contrastive_prepare as P
import contrastive_run as runner
from test_contrastive_prepare import fixture_run


def recipes():
    return P.build_review_configs(P._read(P.BASELINE))


def measured(tmp_path):
    folder = tmp_path / "calibration"
    fixture_run(folder)
    return folder, P.calibration_report(folder)


def test_review_configs_cannot_accidentally_be_training_recipes():
    cfg = recipes()
    assert runner.validate_recipe(cfg["calibration"], "calibration").contrastive["calibration_only"]
    for arm in ("C0", "CN", "CR"):
        with pytest.raises(ValueError, match="calibrated"):
            runner.validate_recipe(cfg[arm], "training")
    with pytest.raises(ValueError, match="execution release"):
        runner.validate_release({"status": "REVIEW_REQUIRED"}, "calibration", Path("not_read"), {})


@pytest.mark.parametrize("arm", ["C0", "CN", "CR"])
def test_paired_recipes_only_use_authenticated_shared_weights(tmp_path, arm):
    folder, report = measured(tmp_path)
    cfg = P.calibrated_configs(report, folder, 19618)[arm]
    assert runner.validate_recipe(cfg, "training", report, folder).contrastive["arm"] == arm
    for field, value in (("rnc_stride", 4), ("fill_steps", 0), ("enforce_coverage", False), ("rnc_weight", .45)):
        bad = copy.deepcopy(cfg)
        bad["contrastive"][field] = value
        with pytest.raises(ValueError): runner.validate_recipe(bad, "training", report, folder)
    for endpoint in (544, 576):
        cfg["max_steps"] = endpoint
        assert runner.validate_recipe(cfg, "benchmark", report, folder).max_steps == endpoint
    cfg["max_steps"] = 64
    with pytest.raises(ValueError, match="frozen split"):
        runner.validate_recipe(cfg, "benchmark", report, folder)
    cfg["max_steps"] = 19618
    with pytest.raises(ValueError, match="evidence"):
        runner.validate_recipe(cfg, "training", {"status": "PASSED_FOR_REVIEW", "weights": {"nce": .1, "rnc": .1}})


def test_calibration_cannot_update_or_start_from_new_schedule():
    for key, value in (("arm", "CN"), ("calibration_only", False), ("fill_steps", 128)):
        cfg = recipes()["calibration"]
        cfg["contrastive"][key] = value
        with pytest.raises(ValueError): runner.validate_recipe(cfg, "calibration")


def release_fixture(tmp_path, monkeypatch):
    folder = tmp_path / "calibration"
    cfg, summary, _ = fixture_run(folder)
    inputs = {}
    for name in ("manifest", "scales", "window_seconds", "pose_stats"):
        inputs[name] = tmp_path / (name + ".json")
        inputs[name].write_text("{}")
    pins = {name: runner.sha(path) for name, path in inputs.items()}
    # This fixture validates release mechanics on tiny files; the real scientific
    # input pins have their own negative regression below.
    monkeypatch.setattr(runner, "validate_input_bindings", lambda value: None)
    audit = {"steps": 1024, "world_size": 4, "workers_per_rank": 6, "passed": True,
             "source_sha256": runner.source_digest(), "inputs": pins,
             "expected_datasets": summary["contrastive"]["expected_datasets"],
             "config_sha256": runner.hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
             "calibration_panel": summary["contrastive"]["coverage_partial_block"]}
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    config_path = folder / "config.json"
    release = {"status": "APPROVED_FOR_PREPARATION", "stage": "calibration",
               "source_sha256": runner.source_digest(), "config_sha256": runner.sha(config_path),
               "world_size": 4, "workers_per_rank": 6, "inputs": pins,
               "review_pin": "reviewed-commit", "time_limit_s": 900., "audit_sha256": runner.sha(audit_path)}
    return release, config_path, inputs, audit_path


def test_execution_release_rejects_changed_inputs_code_time_or_audit(tmp_path, monkeypatch):
    release, cfg, inputs, audit = release_fixture(tmp_path, monkeypatch)
    runner.validate_release(release, "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)
    for key, value in (("source_sha256", "changed"), ("config_sha256", "changed"), ("audit_sha256", "changed"),
                       ("world_size", 2), ("workers_per_rank", 6.), ("review_pin", None),
                       ("time_limit_s", float("nan")), ("time_limit_s", float("inf"))):
        bad = copy.deepcopy(release)
        bad[key] = value
        with pytest.raises(ValueError):
            runner.validate_release(bad, "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)
    inputs["manifest"].write_text("changed")
    with pytest.raises(ValueError, match="data pins"):
        runner.validate_release(release, "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)


def test_calibration_release_requires_passing_exact_audit_panel(tmp_path, monkeypatch):
    release, cfg, inputs, audit = release_fixture(tmp_path, monkeypatch)
    bad = P._read(audit)
    bad["calibration_panel"]["through_step"] = 510
    audit.write_text(json.dumps(bad))
    release["audit_sha256"] = runner.sha(audit)
    with pytest.raises(ValueError, match="frozen panel"):
        runner.validate_release(release, "calibration", cfg, inputs, time_limit_s=900., audit_path=audit)


def test_validate_only_reads_supplied_release_even_if_truncated(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(recipes()["calibration"]))
    release = tmp_path / "release.json"
    release.write_text('{"truncated":')
    monkeypatch.setattr(sys, "argv", ["contrastive_run", "--stage", "calibration", "--config", str(cfg),
        "--validate-only", "--release", str(release), "--manifest", "absent", "--root", "absent",
        "--scales", "absent", "--window-seconds", "absent", "--out-dir", str(tmp_path / "out"), "--time-limit-s", "900"])
    with pytest.raises(ValueError): runner.main()


def test_recipe_only_is_explicitly_not_release_validation(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(recipes()["calibration"]))
    monkeypatch.setattr(sys, "argv", ["contrastive_run", "--stage", "calibration", "--config", str(cfg), "--recipe-only"])
    runner.main()
    output = json.loads(capsys.readouterr().out)
    assert output["recipe_valid"] is True and output["release_valid"] is False


def test_generator_cohort_binds_all_arms_and_every_endpoint(tmp_path, monkeypatch):
    folder, receipt = measured(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    out = tmp_path / "cohort"
    monkeypatch.setattr(sys, "argv", ["contrastive_prepare", "calibrated-configs", "--receipt", str(receipt_path),
                                    "--run-dir", str(folder), "--out", str(out)])
    P.main()
    for arm in ("C0", "CN", "CR"):
        for suffix in ("", ".split", ".5000", ".19618"):
            runner.validate_cohort(out / "cohort.json", receipt_path, folder, out / (arm + suffix + ".config.json"))
    changed = P._read(out / "CR.config.json")
    changed["contrastive"]["rnc_weight"] *= 2
    (out / "CR.config.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="candidate pin"):
        runner.validate_cohort(out / "cohort.json", receipt_path, folder, out / "CN.config.json")


def test_resume_requires_exact_successful_checkpoint_and_summary(tmp_path):
    cohort = {"initial_state_sha256": "a" * 64, "calibration_sha256": "b" * 64}
    summary = {"steps": 544, "stopped": "max_steps", "contrastive": {"initial_sha256": "a" * 64,
               "config": {"arm": "CN"}, "provenance": {"calibration_sha256": "b" * 64}}}
    runner.torch.save({"step": 544, "world_size": 4, "contrastive_initial_sha256": "a" * 64,
                       "contrastive_provenance": summary["contrastive"]["provenance"],
                       "contrastive_state": {"queue": {"last_step": 543}}, "rank_rng": [{}] * 4}, tmp_path / "checkpoint.pt")
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    release = {"stage": "benchmark", "arm": "CN", "resume_checkpoint_sha256": runner.sha(tmp_path / "checkpoint.pt"),
               "resume_summary_sha256": runner.sha(path)}
    runner.validate_resume(release, tmp_path, cohort)
    (tmp_path / "STOP").write_text("failed")
    with pytest.raises(ValueError, match="successful matching"):
        runner.validate_resume(release, tmp_path, cohort)


def test_training_resource_gate_cannot_be_bypassed_without_measurements():
    with pytest.raises(ValueError, match="measured three-arm benchmark"):
        runner.validate_measured_budget(None, {}, {}, None)


def test_split_controller_derives_only_reviewed_bounded_continuation(tmp_path):
    import contrastive_split_benchmark as split
    before = {"max_steps": 544, "recipe": "unchanged"}
    a, b = tmp_path / "split.json", tmp_path / "full.json"
    a.write_text(json.dumps(before))
    b.write_text(json.dumps({**before, "max_steps": 576}))
    (tmp_path / "checkpoint.pt").write_bytes(b"exact544")
    (tmp_path / "summary.json").write_text("{}")
    initial = {"stage": "benchmark", "phase": "split", "arm": "CN", "config_sha256": runner.sha(a),
               "source_sha256": runner.source_digest(), "paired_resume": {"config_sha256": runner.sha(b),
               "total_time_limit_s": 1800, "maximum_resume_time_limit_s": 1800}}
    resumed = split.continuation_release(initial, a, b, tmp_path, 247.)
    assert resumed["phase"] == "resume" and resumed["time_limit_s"] == 247.
    assert resumed["resume_checkpoint_sha256"] == runner.sha(tmp_path / "checkpoint.pt")
    for remaining in (-1, 1801, float("nan")):
        with pytest.raises(ValueError): split.continuation_release(initial, a, b, tmp_path, remaining)
    b.write_text(json.dumps({"max_steps": 576, "recipe": "changed"}))
    initial["paired_resume"]["config_sha256"] = runner.sha(b)
    with pytest.raises(ValueError, match="more than"):
        split.continuation_release(initial, a, b, tmp_path, 247.)


def test_split_controller_does_not_retry_nonzero_phase(monkeypatch):
    import contrastive_split_benchmark as split
    calls = []
    class Child:
        pid = 12345
        def wait(self, timeout): return 3
        def poll(self): return 3
    monkeypatch.setattr(split.subprocess, "Popen", lambda *a, **k: calls.append(a) or Child())
    monkeypatch.setattr(split.os, "killpg", lambda *a: None)
    with pytest.raises(RuntimeError, match="no retry"):
        split.checked_command(["reviewed-phase"], split.time.monotonic() + 10)
    assert len(calls) == 1
