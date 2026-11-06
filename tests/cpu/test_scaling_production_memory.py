"""Small real evidence files for S3; no cluster/GPU/train or fabricated readiness."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from wearusfm.training import run as R

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("test_s3_guard", ROOT / "scripts/scaling_production_memory.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
CONTRACT = ROOT / "results/infra/scaling_300m_20261104/training_source_contract.json"
REFERENCE = ROOT / "results/infra/tokenizer_execution_20261103/5000/T100-G/summary.json"
CODE = "a" * 40


def write(path, value, lines=False):
    path.write_text("\n".join(json.dumps(row) for row in value) + "\n" if lines else json.dumps(value))
    return M.pin(path)


def production_memory_fixture(tmp_path, cfg, contract_sha):
    """500 tiny log rows exercise real pins/coverage; checkpoint bytes are only an I/O fixture."""
    run, out = tmp_path / "technical_run", tmp_path / "technical_receipts"
    run.mkdir(); out.mkdir()
    source_pin = M.pin(CONTRACT)
    assert source_pin["sha256"] == contract_sha
    stats = json.loads(CONTRACT.read_text())["inputs"]["pose_stats"]["sha256"]
    environment = json.loads(REFERENCE.read_text())["environment"]
    runtime = {**{k: environment[k] for k in M.RUNTIME_FIELDS}, "cudnn": 8907}
    config = R.config_to_dict(replace(cfg, max_steps=500))
    plan = {"protocol": M.PLAN_PROTOCOL, "arm": M.ARM, "steps": 500, "technical_only": True,
            "run_dir": str(run), "receipt_dir": str(out), "source_contract": source_pin,
            "reference_runtime_summary": M.pin(REFERENCE),
            "execution_source_sha256": {p: R.sha256_file(ROOT / p) for p in M.EXECUTION_SOURCES}}
    plan_pin = write(tmp_path / "plan.json", plan)
    windows = [[f"session-{i // 4}", None, (i % 4) * 8000, (i % 4 + 1) * 8000] for i in range(256)]
    users = [f"u{i // 8}" for i in range(256)]
    pose_set = {"users": sorted(set(users)), "windows": windows, "window_users": users,
                "recordings": list(range(64)), "batch_sha256": ["b" * 64] * 64, "pose_stats_sha256": stats}
    panel = [{"step": step, "panel_sha256": M.canonical_hash(pose_set), "n_windows": 256,
              "n_windows_valid": 256, "n_users": 32, "elapsed_s": 4., "mse": 2.,
              "per_window": [{"user": user, "window": w[1:]} for user, w in zip(users, windows)]} for step in (0, 500)]
    metrics = [{"step": i, "windows": 32, "total": 2., "pose": 1., "grad_norm": 1., "lr": R._lr_at(i - 1, cfg),
                "t_step_s": 2.8} for i in range(1, 501)]
    metrics[-1].update({k: 1. for k in ("shared_grad_pose", "shared_grad_anchors", "shared_grad_cos",
        "student_erank", "teacher_erank", "student_collapse_ratio", "teacher_collapse_ratio")})
    data = {"config": config,
        "summary": {"stopped": "max_steps", "steps": 500, "world_size": 4, "batch_per_process": 8,
            "parameters": 292054847, "prefetch": True, "accum_steps": 1, "micro_batch": 8,
            "environment": environment,
            "code": {"sha": CODE}, "elapsed_s": 1900., "gpu_max_memory_gib": 45., "pose_stats_sha256": stats},
        "metrics": metrics, "fixed_eval": [{"step": i, "batches": 8, "total": 2.} for i in (0, 500)],
        "fixed_set": {"seed": [cfg.seed, R.FIXED_EVAL_SEED], "windows": windows,
                      "batch_sha256": ["c" * 64] * 8},
        "pose_panel": panel, "pose_set": pose_set,
        "technical_marker": {"protocol": M.PROTOCOL, "technical_only": True, "fresh": True, "start_step": 0,
            "steps": 500, "training_source_contract_sha256": contract_sha, "code_sha": CODE, "slurm_job_id": "123456",
            "plan_sha256": M.canonical_hash(plan)}}
    artifacts = {key: write(run / M.ARTIFACT_NAMES[key], value, key in ("metrics", "fixed_eval", "pose_panel"))
                 for key, value in data.items()}
    (run / "checkpoint.pt").write_bytes(b"technical checkpoint I/O fixture; no scientific weights")
    artifacts["technical_checkpoint"] = M.pin(run / "checkpoint.pt")
    ranks = []
    for rank in range(4):
        ranks.append(write(out / f"rank_{rank}.json", {"protocol": M.RANK_PROTOCOL, "rank": rank,
            "local_rank": rank, "world_size": 4, "worker_returned": True, "returncode": 0,
            "code_sha": CODE, "slurm_job_id": "123456", "training_source_contract_sha256": contract_sha,
            "run_dir": str(run), "plan_sha256": M.canonical_hash(plan),
            "runtime": runtime,
            "peak_allocated_bytes": (42 + rank) * 2**30, "device_memory_bytes": 64 * 2**30,
            "peak_reserved_bytes": 46 * 2**30}))
    evidence = M.inspect_evidence(plan, artifacts, ranks, R.config_to_dict(cfg), contract_sha, CODE, "123456")
    return {"protocol": M.PROTOCOL, "complete": True, "arm": M.ARM, "technical_only": True,
            "scientific_checkpoints_written": False, "training_source_contract_sha256": contract_sha,
            "code_sha": CODE, "slurm_job_id": "123456", "plan": plan_pin, "artifacts": artifacts, "ranks": ranks, **evidence}


@pytest.fixture
def context():
    return M.module("scaling_train").check_source_contract(CONTRACT)


@pytest.fixture
def scene(context, tmp_path):
    return production_memory_fixture(tmp_path, context["config_300m"], context["identity"]["sha256"]), context


def check(scene):
    report, context = scene
    return M.validate_report(report, R.config_to_dict(context["config_300m"]), context["identity"]["sha256"])


def mutate_artifact(report, key, change):
    path = Path(report["artifacts"][key]["path"])
    lines = key in ("metrics", "fixed_eval", "pose_panel")
    value = M.read_json(path, lines)
    change(value)
    report["artifacts"][key] = write(path, value, lines)


def test_real_pins_and_complete_periodic_coverage_pass(scene):
    assert check(scene) == 45 * 2**30
    assert scene[0]["coverage_steps"] == {"fixed_eval": [0, 500], "pose_panel": [0, 500], "gradient_diagnostics": [500]}
    assert scene[0]["runtime_comparison"]["matched_fields"] == ["python", "torch", "cuda"]
    assert scene[0]["runtime_comparison"]["unavailable_in_reference"] == ["cudnn"]


def test_fixed_set_real_producer_flat_schema_is_accepted(scene, monkeypatch):
    from wearusfm.training import pairing
    cfg = scene[1]["config_300m"]
    class Loader:
        def __init__(self, *args):
            self.calls = 0
        def batch(self, size, rng):
            self.calls += 1
            return SimpleNamespace(
                rows=[{"dataset": "emg2pose", "subject": "u0", "session": "s0"}] * size,
                windows=[SimpleNamespace(trial=None, start=i * 8000, stop=(i + 1) * 8000) for i in range(size)])
    monkeypatch.setattr(R.L, "PretrainLoader", Loader)
    monkeypatch.setattr(pairing, "batch_sha256", lambda batch: "b" * 64)
    batches, actual = R._fixed_eval_set(None, cfg, {}, None)
    assert len(batches) == 8 and len(actual["windows"]) == 256 and len(actual["batch_sha256"]) == 8
    mutate_artifact(scene[0], "fixed_set", lambda x: (x.clear(), x.update(actual)))
    assert check(scene) == 45 * 2**30


def test_old_nested_fixed_set_schema_is_rejected(scene):
    mutate_artifact(scene[0], "fixed_set", lambda x: x.update(windows=[list(range(32))] * 8))
    with pytest.raises(ValueError, match="support"):
        check(scene)


@pytest.mark.parametrize("field", ["python", "torch", "cuda", "cudnn"])
def test_rank_runtime_difference_is_rejected(scene, field):
    report, _ = scene
    path = Path(report["ranks"][1]["path"])
    value = M.read_json(path)
    value["runtime"][field] = 1234 if field == "cudnn" else "different"
    report["ranks"][1] = write(path, value)
    with pytest.raises(ValueError, match="runtime"):
        check(scene)


def test_summary_runtime_difference_is_rejected(scene):
    mutate_artifact(scene[0], "summary", lambda x: x["environment"].update(cuda="different"))
    with pytest.raises(ValueError, match="runtime"):
        check(scene)


@pytest.mark.parametrize("field", ["python", "torch", "cuda"])
def test_prelaunch_runtime_gate_rejects_changed_historical_fields(field):
    reference = M.read_json(REFERENCE)
    runtime = {**{k: reference["environment"][k] for k in M.RUNTIME_FIELDS}, "cudnn": 8907}
    runtime[field] = "different"
    with pytest.raises(ValueError, match="runtime differs"):
        M.runtime_comparison(runtime, reference)


@pytest.mark.parametrize("bad", ["64", "499", "fixed_only0", "panel_only0", "grad", "diagnostics", "prefetch",
                                "resume", "metrics_gap", "metric_nan", "panel_support", "panel_identity", "summary", "stop", "named"])
def test_flags_cannot_replace_actual_production_coverage(scene, bad):
    report, _ = scene
    if bad in ("64", "499"):
        mutate_artifact(report, "config", lambda x: x.update(max_steps=int(bad)))
    if bad == "fixed_only0": mutate_artifact(report, "fixed_eval", lambda x: x.pop())
    if bad == "panel_only0": mutate_artifact(report, "pose_panel", lambda x: x.pop())
    if bad == "grad": mutate_artifact(report, "metrics", lambda x: x[-1].pop("shared_grad_pose"))
    if bad == "diagnostics": mutate_artifact(report, "metrics", lambda x: x[-1].pop("teacher_erank"))
    if bad == "prefetch": mutate_artifact(report, "summary", lambda x: x.update(prefetch=False))
    if bad == "resume": mutate_artifact(report, "technical_marker", lambda x: x.update(start_step=64))
    if bad == "metrics_gap": mutate_artifact(report, "metrics", lambda x: x.pop(20))
    if bad == "metric_nan": mutate_artifact(report, "metrics", lambda x: x[-1].update(pose=float("nan")))
    if bad == "panel_support": mutate_artifact(report, "pose_panel", lambda x: x[-1].update(n_windows=255))
    if bad == "panel_identity": mutate_artifact(report, "pose_panel", lambda x: x[-1].update(panel_sha256="f" * 64))
    if bad == "summary": mutate_artifact(report, "summary", lambda x: x.update(stopped="limite di tempo"))
    root = Path(report["artifacts"]["config"]["path"]).parent
    if bad == "stop": (root / "STOP").write_text("failed")
    if bad == "named": (root / "checkpoint_step0005000.pt").write_bytes(b"wrong scientific endpoint")
    with pytest.raises(ValueError): check(scene)


@pytest.mark.parametrize("bad", ["missing", "duplicate", "rank", "job", "code", "failed", "peak", "total", "summary_peak", "claim", "corrupt"])
def test_each_peak_needs_its_own_matching_rank_and_global_evidence(scene, bad):
    report, _ = scene
    if bad == "missing": report["ranks"].pop()
    elif bad == "duplicate": report["ranks"][3] = report["ranks"][2]
    elif bad == "claim": report["peak_allocated_bytes_by_rank"] = [1] * 4
    elif bad == "summary_peak": mutate_artifact(report, "summary", lambda x: x.update(gpu_max_memory_gib=41.))
    elif bad == "corrupt": Path(report["ranks"][0]["path"]).write_text("{}")
    else:
        path = Path(report["ranks"][0]["path"])
        row = M.read_json(path)
        if bad == "rank": row["rank"] = 3
        if bad == "job": row["slurm_job_id"] = "1"
        if bad == "code": row["code_sha"] = "b" * 40
        if bad == "failed": row["worker_returned"] = False
        if bad == "peak": row["peak_allocated_bytes"] = 60 * 2**30
        if bad == "total": row["device_memory_bytes"] = 10 * 2**30
        report["ranks"][0] = write(path, row)
    with pytest.raises(ValueError): check(scene)


def test_technical_recipe_changes_only_dose_output_and_operational_timeout(context, tmp_path):
    cfg, argv = M.recipe(context, tmp_path / "run", 3600)
    assert cfg.max_steps == 500 and cfg.schedule.total_steps == 49046 and cfg.warmup_steps == 1962
    assert cfg.pose_panel_every == cfg.pose_grad_every == cfg.fixed_eval_every == 500
    assert cfg.prefetch and cfg.batch_size == 32 and cfg.accum_steps == 1
    assert argv[-2:] == ["--time-limit-s", "3300"]
    assert "out_dir" not in R.config_to_dict(cfg) and "time_limit_s" not in R.config_to_dict(cfg)
    assert R.config_to_dict(replace(cfg, max_steps=5000)) == R.config_to_dict(context["config_300m"])


def test_original_64_step_claim_is_rejected_without_any_evidence(context):
    cfg = context["config_300m"]
    old = {"protocol": M.PROTOCOL, "complete": True, "arm": M.ARM,
           "training_source_contract_sha256": context["identity"]["sha256"],
           "config": R.config_to_dict(replace(cfg, max_steps=64)), "prefetch_exercised": True,
           "fixed_eval_exercised": True, "pose_panel_exercised": True}
    with pytest.raises(ValueError):
        M.module("scaling_train").check_production_memory(old, cfg, context["identity"]["sha256"])


def test_budget_uses_slower_full_recipe_proxy_then_margin_and_reserve(context):
    from test_scaling_train import preflight
    report = preflight(context)
    report["arms"]["T100-G-30M"]["normal_update_rank_max_seconds"] = [2.76668 / 3.90856] * 128
    report["arms"][M.ARM]["normal_update_rank_max_seconds"] = [2.76668] * 128
    estimate = M.resource_estimate(report, context)
    assert estimate["seconds_per_step"] > 2.76668
    assert estimate["walltime_seconds"] == 3600 and estimate["gpu_hours_cap"] == 4.
    assert estimate["cpu_core_hours_cap"] == 32. and not estimate["checkout_included"]
    report["arms"][M.ARM]["normal_update_rank_max_seconds"] = [12.] * 128
    with pytest.raises(ValueError, match="ceiling"):
        M.resource_estimate(report, context)


def test_producer_identity_is_part_of_the_memory_evidence(scene):
    report, _ = scene
    plan_path = Path(report["plan"]["path"])
    plan = M.read_json(plan_path)
    plan["execution_source_sha256"]["scripts/scaling_production_memory.py"] = "f" * 64
    report["plan"] = write(plan_path, plan)
    with pytest.raises(ValueError, match="SHA"):
        check(scene)
