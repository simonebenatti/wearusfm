"""Offline scientific-wrapper contracts: no cluster, GPU, training or submission."""
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import pytest

from wearusfm.training import run as R

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("scaling_train_test", ROOT / "scripts/scaling_train.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
CONTRACT = ROOT / "results/infra/scaling_300m_20261104/training_source_contract.json"


def pin(path):
    return {"path": str(path), "sha256": R.sha256_file(path)}


def write_json(path, value):
    path.write_text(json.dumps(value))
    return pin(path)


@pytest.fixture
def context():
    return M.check_source_contract(CONTRACT)


def plan_for(context, tmp_path):
    return {"protocol": M.PROTOCOL, "arm": M.ARM, "checkpoint_step": 5000,
            "source_contract": pin(CONTRACT),
            "scientific_source_sha256": context["contract"]["scientific_source_sha256"],
            "execution_source_sha256": {p: R.sha256_file(ROOT / p) for p in M.EXECUTION_SOURCES},
            "allocation": {"nodes": 1, "gpus": 4, "cpus": 32, "memory": "480G", "walltime_seconds": 21600},
            "run_dir": str(tmp_path / "run"), "receipt_dir": str(tmp_path / "receipt"),
            "gates": {name: {"path": str(tmp_path / (name + ".json")), "sha256": "a" * 64} for name in M.GATES},
            "control": {"checkpoint": {"path": "/leonardo_scratch/control5000.pt", "sha256": context["contract"]["reference_checkpoint_sha256"]},
                        "receipt": context["contract"]["reference_training_receipt"],
                        "retention_verified_until_utc": "2026-12-13T00:00:00Z", "required_until_utc": "2026-11-11T00:00:00Z"}}


def test_complete_source_contract_and_cli_exactly_match_preflight_without_remote_reads(context, tmp_path):
    assert len(context["contract"]["scientific_source_sha256"]) == 103
    assert context["contract"]["reference_code_sha"] == M.BASE_CODE_SHA
    cfg, command = M.validate_plan(plan_for(context, tmp_path), context)
    assert cfg.model.dim == 896 and cfg.model.n_heads == 14 and cfg.batch_size == 32
    assert cfg.max_steps == 5000 and cfg.warmup_steps == 1962 and cfg.schedule.total_steps == 49046
    assert R.config_to_dict(cfg) == R.config_to_dict(context["config_300m"])
    assert command[:3] == ["torchrun", "--standalone", "--nproc_per_node=4"]
    assert command[-2:] == ["--time-limit-s", "21300"]
    recovered = command[3:-2]
    original = context["comparison"]["arm_configurations"]["T100-G"]["validated_trainer_argv"]
    for flag in ("--preset", "--out-dir", "--max-steps"):
        recovered[recovered.index(flag) + 1] = original[original.index(flag) + 1]
    assert recovered == original


@pytest.mark.parametrize("change", ["missing", "changed", "extra", "reference", "dose", "schedule"])
def test_contract_rejects_incomplete_sources_and_scientific_changes(context, change):
    contract = copy.deepcopy(context["contract"])
    key = next(iter(contract["scientific_source_sha256"]))
    if change == "missing": del contract["scientific_source_sha256"][key]
    if change == "changed": contract["scientific_source_sha256"][key] = "a" * 64
    if change == "extra": contract["scientific_source_sha256"]["src/unreviewed.py"] = "a" * 64
    if change == "reference": contract["reference_code_sha"] = "a" * 40
    if change == "dose": contract["checkpoint_step"] = 19618
    if change == "schedule": contract["schedule"]["warmup_steps"] = 64
    with pytest.raises(ValueError):
        M.check_source_contract(contract)


@pytest.mark.parametrize("change", ["19618", "globalgpu", "no_gates", "placeholder", "wrong_control", "no_retention", "short_time", "no_exec_sources"])
def test_plan_does_not_accept_unreviewed_expansion_or_placeholder_gates(context, tmp_path, change):
    plan = plan_for(context, tmp_path)
    if change == "19618": plan["checkpoint_step"] = 19618
    if change == "globalgpu": plan["allocation"]["gpus"] = 8
    if change == "no_gates": plan["gates"].pop("evaluation_readiness")
    if change == "placeholder": plan["gates"]["preflight"]["sha256"] = "0" * 64
    if change == "wrong_control": plan["control"]["checkpoint"]["sha256"] = "a" * 64
    if change == "no_retention": del plan["control"]["retention_verified_until_utc"]
    if change == "short_time": plan["allocation"]["walltime_seconds"] = 300
    if change == "no_exec_sources": del plan["execution_source_sha256"]
    with pytest.raises(ValueError):
        M.validate_plan(plan, context)


def preflight(context):
    return {"protocol": M.S.PROTOCOL, "complete": True, "code_sha": M.PREFLIGHT_CODE_SHA,
            "paired_data_masks_queries_targets": True, "one_model_resident": True, "scientific_checkpoints_written": False,
            "batches_per_arm": 128, "native_warmup_batches": 64, "peak_stress_batches": 64,
            "world_size": 4, "windows_per_arm": 4096, "inputs": context["contract"]["inputs"],
            "arms": {name: {"normal_update_rank_max_seconds": [1. if name != M.ARM else 4.] * 128,
                            "memory_gate_passed": True, "peak_allocated_bytes": (16 if name != M.ARM else 40) * 2**30,
                            "device_memory_bytes": 64 * 2**30,
                            "initial": {"parameters": 28362013 if name != M.ARM else 292054847}}
                     for name in M.S.ARMS}}


def memory(context, tmp_path):
    from test_scaling_production_memory import production_memory_fixture
    return production_memory_fixture(tmp_path, context["config_300m"], context["identity"]["sha256"])


def test_resource_formula_uses_all112_measured_updates_not_a_favorable_phase(context):
    report = preflight(context)
    report["arms"][M.ARM]["normal_update_rank_max_seconds"] = [100.] * 8 + [3.] * 56 + [100.] * 8 + [5.] * 56
    estimate = M.check_preflight(report, context)
    assert estimate["measured_steps"] == 112
    assert estimate["mean_update_seconds"][M.ARM] == 4.
    assert estimate["ratio300_to30"] == 4.
    assert estimate["minimum_walltime_seconds"] == 25200  # exceeds first-tranche cap; run must stop


@pytest.mark.parametrize("change", ["partial", "wrong_inputs", "short_times", "nonfinite", "memory", "wrong_parameters"])
def test_preflight_gate_rejects_incomplete_or_wrong_measurements(context, change):
    report = copy.deepcopy(preflight(context))
    if change == "partial": report["batches_per_arm"] = 127
    if change == "wrong_inputs": report["inputs"]["manifest"]["sha256"] = "a" * 64
    if change == "short_times": report["arms"][M.ARM]["normal_update_rank_max_seconds"].pop()
    if change == "nonfinite": report["arms"][M.ARM]["normal_update_rank_max_seconds"][30] = float("nan")
    if change == "memory": report["arms"][M.ARM]["peak_allocated_bytes"] = 60 * 2**30
    if change == "wrong_parameters": report["arms"][M.ARM]["initial"]["parameters"] = 28362013
    with pytest.raises(ValueError):
        M.check_preflight(report, context)


@pytest.mark.parametrize("change", ["prefetch", "panel", "peak", "microbatch", "config", "job", "contract"])
def test_production_memory_requires_real_path_and_same_scientific_recipe(context, tmp_path, change):
    report = memory(context, tmp_path)
    if change == "prefetch": report["prefetch_exercised"] = False
    if change == "panel": report["pose_panel_exercised"] = False
    if change == "peak": report["peak_allocated_bytes_by_rank"][2] = 60 * 2**30
    if change == "microbatch": report["accum_steps"] = 2
    if change == "config": report["config"]["lr"] = 3e-4
    if change == "job": report["slurm_job_id"] = "pending"
    if change == "contract": report["training_source_contract_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        M.check_production_memory(report, context["config_300m"], context["identity"]["sha256"])


def test_valid_production_memory_measurement(context, tmp_path):
    assert M.check_production_memory(memory(context, tmp_path), context["config_300m"], context["identity"]["sha256"]) == 45 * 2**30


def test_extraction_alone_cannot_assert_readiness_even_with_dimension_checks(context):
    report = {"protocol": "scaling-evaluation-readiness-20261104-v1", "complete": True, "checkpoint_step": 5000,
              "training_source_contract_sha256": context["identity"]["sha256"],
              "dimensions": {"T100-G-30M": 384, M.ARM: 896}, "pool_widths": {"T100-G-30M": 768, M.ARM: 1792},
              "source_files": {"scripts/scaling_evaluation.py": "a" * 64}}
    with pytest.raises(ValueError, match="evaluators missing"):
        M.check_evaluation_readiness(report, context["identity"]["sha256"], {})


@pytest.fixture
def runtime(context, tmp_path, monkeypatch):
    """Real gate files and hashes, synthetic contents explicitly confined to tests."""
    plan = plan_for(context, tmp_path)
    pre = preflight(context)
    pre["arms"][M.ARM]["normal_update_rank_max_seconds"] = [3.] * 128
    plan["gates"]["preflight"] = write_json(tmp_path / "preflight.json", pre)
    plan["gates"]["production_memory"] = write_json(tmp_path / "production_memory.json", memory(context, tmp_path))
    # Readers are not implemented in the checkout. Isolate their separate unit
    # tests; the actual controller never stubs this readiness function.
    plan["gates"]["evaluation_readiness"] = write_json(tmp_path / "readiness.json", {
        "fixture": True, "control_required_until_utc": plan["control"]["required_until_utc"]})
    monkeypatch.setattr(M, "check_evaluation_readiness", lambda report, sha, states: None)
    review_text = tmp_path / "review.md"
    review_text.write_text("APPROVATO, synthetic fixture only")
    review = {"protocol": "scaling-scientific-review-20261104-v1", "complete": True, "verdict": "APPROVATO",
              "scope": "fresh300M5000", "training_source_contract_sha256": context["identity"]["sha256"],
              "execution_source_sha256": plan["execution_source_sha256"], "allocation": plan["allocation"],
              "gates_sha256": {k: plan["gates"][k]["sha256"] for k in M.GATES - {"scientific_review"}}, "review": pin(review_text)}
    plan["gates"]["scientific_review"] = write_json(tmp_path / "review.json", review)
    control = tmp_path / "control.pt"
    control.write_bytes(b"synthetic control only")
    context["contract"]["reference_checkpoint_sha256"] = R.sha256_file(control)
    plan["control"]["checkpoint"] = pin(control)
    return plan, context


def test_all_runtime_gate_files_are_read_and_control_is_hashed_before_launch(runtime):
    plan, context = runtime
    states = {}
    estimate = M.check_runtime_gates(plan, context, context["config_300m"], states, datetime(2026, 11, 4, tzinfo=timezone.utc))
    assert set(M.GATES) <= set(states) and "control_checkpoint" in states and "control_receipt" in states
    assert estimate["minimum_walltime_seconds"] <= plan["allocation"]["walltime_seconds"]


@pytest.mark.parametrize("change", ["missing_file", "hash", "review", "expired", "control_corrupt", "runtime_cap"])
def test_runtime_gate_failure_prevents_scientific_execution(runtime, change):
    plan, context = runtime
    if change == "missing_file": Path(plan["gates"]["production_memory"]["path"]).unlink()
    if change == "hash": plan["gates"]["preflight"]["sha256"] = "b" * 64
    if change == "review":
        path = Path(plan["gates"]["scientific_review"]["path"])
        report = json.loads(path.read_text()); report["verdict"] = "DA CORREGGERE"
        plan["gates"]["scientific_review"] = write_json(path, report)
    if change == "expired": plan["control"]["retention_verified_until_utc"] = "2026-11-04T01:00:00Z"
    if change == "control_corrupt": Path(plan["control"]["checkpoint"]["path"]).write_bytes(b"wrong control")
    if change == "runtime_cap": plan["allocation"]["walltime_seconds"] = 5400
    with pytest.raises((ValueError, FileNotFoundError)):
        M.check_runtime_gates(plan, context, context["config_300m"], {}, datetime(2026, 11, 4, tzinfo=timezone.utc))


def test_scratch_deadline_cannot_be_extended_by_a_plan(runtime):
    plan, context = runtime
    plan["control"]["checkpoint"]["path"] = "/leonardo_scratch/control.pt"
    plan["control"]["retention_verified_until_utc"] = "2027-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="scratch retention"):
        M.check_runtime_gates(plan, context, context["config_300m"], {}, datetime(2026, 11, 4, tzinfo=timezone.utc))


@pytest.mark.parametrize("change", [None, "gpu", "memory", "time", "node"])
def test_actual_allocation_must_match_reviewed_resources(change):
    fields = {"NumNodes": "1", "NumCPUs": "32", "TimeLimit": "06:00:00", "AllocTRES": "cpu=32,mem=480G,node=1,gres/gpu=4"}
    expected = {"walltime_seconds": 21600}
    if change == "gpu": fields["AllocTRES"] = "cpu=32,mem=480G,node=1,gres/gpu=8"
    if change == "memory": fields["AllocTRES"] = "cpu=32,mem=500G,node=1,gres/gpu=4"
    if change == "time": fields["TimeLimit"] = "08:00:00"
    if change == "node": fields["NumNodes"] = "2"
    if change:
        with pytest.raises(ValueError): M.check_allocation_fields(fields, expected, 4)
    else:
        M.check_allocation_fields(fields, expected, 4)


@pytest.fixture
def outputs(context, tmp_path):
    from test_tokenizer_train import state, SIGNATURE
    import torch
    cfg = context["config_300m"]
    run = tmp_path / "output"
    run.mkdir()
    summary = {"stopped": "max_steps", "steps": 5000, "world_size": 4, "parameters": 292054847, "accum_steps": 1}
    write_json(run / "summary.json", summary)
    write_json(run / "config.json", R.config_to_dict(cfg))
    # Tiny checkpoint tensors exercise the existing complete-state verifier;
    # no300M model construction or scientific computation is needed here.
    payload = state(cfg)
    torch.save(payload, run / "checkpoint_step0005000.pt")
    torch.save(payload, run / "checkpoint.pt")
    return run, cfg, SIGNATURE


def test_complete_phase_uses_existing_optimizer_rng_checkpoint_verifier(outputs):
    run, cfg, signature = outputs
    summary, named, final = M.verify_phase_outputs(run, cfg, signature)
    assert summary["steps"] == 5000 and named.name == "checkpoint_step0005000.pt" and final.name == "checkpoint.pt"


@pytest.mark.parametrize("change", ["incomplete", "stop", "configuration", "different_teacher", "optimizer", "rng", "checkpoint_dose", "missing_checkpoint"])
def test_no_complete_scientific_receipt_for_partial_or_corrupt_outputs(outputs, change):
    import torch
    run, cfg, signature = outputs
    if change == "incomplete":
        path = run / "summary.json"; report = json.loads(path.read_text()); report["stopped"] = "time_limit"; write_json(path, report)
    elif change == "stop":
        (run / "STOP").write_text("numerical failure")
    elif change == "configuration":
        path = run / "config.json"; report = json.loads(path.read_text()); report["lr"] = 3e-4; write_json(path, report)
    elif change == "missing_checkpoint":
        (run / "checkpoint_step0005000.pt").unlink()
    else:
        path = run / "checkpoint.pt"
        payload = torch.load(path, weights_only=False)
        if change == "different_teacher": payload["teacher"]["weight"][0] = 1
        if change == "optimizer": payload["optimizer"]["state"][0]["exp_avg"][0] = float("nan")
        if change == "rng": payload["rank_rng"] = []
        if change == "checkpoint_dose": payload["step"] = 4999
        torch.save(payload, path)
    with pytest.raises((ValueError, FileNotFoundError)):
        M.verify_phase_outputs(run, cfg, signature)
