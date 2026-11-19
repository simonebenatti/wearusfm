"""Adversarial evidence checks for the amended full/suffix readout.

Raw file formatting, missing numerical evidence and resume paths are independent
of training performance. No CUDA measurement is claimed by these fixtures.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import contrastive_benchmark_readout as B
import contrastive_prepare as P
from contrastive_numeric_profile import numeric_profile
from test_contrastive_target_parity import point
from test_contrastive_benchmark_readout import _run_folder


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value))
    return Path(path)


def test_scalar_drift_preserves_old_thresholds_but_is_only_descriptive():
    rows = [{"step": step, "total": 1., "grad_norm": 2.} for step in range(1, 257)]
    other = copy.deepcopy(rows)
    other[0]["total"] += .001
    other[255]["grad_norm"] += .1
    report = B.trajectory_drift(rows, other)
    assert report["descriptive_only"] and "passed" not in report and "status" not in report
    assert report["rel_tol"] == 1e-4 and report["abs_tol"] == 1e-6
    assert report["fields"]["total"]["outside_count"] == report["fields"]["grad_norm"]["outside_count"] == 1
    for key in ("total", "grad_norm"):
        assert len(report["fields"][key]["updates"]) == 256
    with pytest.raises(ValueError, match="exactly updates"):
        B.trajectory_drift(rows[1:], other)
    other[100]["total"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        B.trajectory_drift(rows, other)


def test_named_fork_checkpoint_is_a_technical_save_not_scientific_dose_change(monkeypatch):
    monkeypatch.setattr(P, "authenticate_calibration", lambda *args: {"weights": {"nce": .2, "rnc": .3}})
    technical = P.calibrated_configs({}, None, 576)
    for arm, cfg in technical.items():
        assert cfg["schedule"]["keep_checkpoints"] == [544, 5000, 19618]
        assert cfg["loader"]["rms_window_ms"] == 100 and cfg["loader"]["ms_fast_window_ms"] == 125
        assert cfg["max_steps"] == 576
        assert cfg["schedule"]["total_steps"] == 49046
    for endpoint in (5000, 19618):
        scientific = P.calibrated_configs({}, None, endpoint)
        for cfg in scientific.values():
            assert cfg["schedule"]["keep_checkpoints"] == [5000, 19618]
    assert all(c["schedule"]["keep_checkpoints"] == [544, 5000, 19618] for c in technical.values())


def readout_fixture(tmp_path, monkeypatch):
    import contrastive_run
    import contrastive_release
    import contrastive_rank_exit
    import contrastive_target_resume

    source, initial = "s" * 64, "i" * 64
    inputs = {name: name + "-hash" for name in ("manifest", "scales", "window_seconds", "pose_stats")}
    full = {arm: tmp_path / arm for arm in B.ARMS}
    reloads = {arm + "_" + letter: tmp_path / (arm + "_" + letter) for arm in ("CN", "CR") for letter in ("A", "B")}
    expected = {arm: {"contrastive": {"arm": arm}} for arm in B.ARMS}
    for arm, path in full.items():
        path.mkdir()
        write(path / "config.json", expected[arm])
        write(path / "summary.json", {"real": arm})
        (path / "checkpoint_step0000544.pt").write_bytes(arm.encode())
    for path in reloads.values():
        path.mkdir()
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "checkpoint.pt").write_bytes(b"fixed initial checkpoint")
    cal_path = write(tmp_path / "calibration.json", {"input_sha256": inputs})
    cohort = {"endpoint": 576, "source_sha256": source, "initial_state_sha256": initial,
              "calibration_sha256": B.sha(cal_path), "configs": {arm: {"path": arm + ".json"} for arm in B.ARMS}}
    cohort_path = write(tmp_path / "cohort.json", cohort)
    cal_cfg = {"contrastive": {"nce_weight": 1., "rnc_weight": 1.}}
    audit = write(tmp_path / "audit.json", {"passed": True, "source_sha256": source,
        "inputs": inputs, "steps": 1024, "world_size": 4, "workers_per_rank": 6,
        "config_sha256": canonical(cal_cfg)})
    accounting = write(tmp_path / "accounting.json", {"valid": True})
    rank_exit = write(tmp_path / "rank_exit.json", {"valid": True})
    cuda = write(tmp_path / "cuda.json", {"inputs": {"calibration_checkpoint": B.sha(calibration / "checkpoint.pt")},
        "rank_exit_report_path": str(rank_exit), "rank_exit_report_sha256": B.sha(rank_exit)})
    provenance = {"source_sha256": source, "calibration_sha256": B.sha(cal_path),
                  "cohort_sha256": B.sha(cohort_path), "input_sha256": inputs}
    with numeric_profile(require_cuda=False) as flags:
        for name, path in {**full, **reloads}.items():
            arm = name[:2]
            phase = "full" if name in full else "resume_" + name[-1].lower()
            for rank in range(4):
                write(path / f"numeric_profile.rank{rank}.json", {
                    "rank": rank, "world_size": 4, "source_sha256": source, "phase": phase,
                    "config_canonical_sha256": canonical(expected[arm]), "flags": flags})
    monkeypatch.setattr(B, "authenticate_calibration", lambda *args: None)
    monkeypatch.setattr(B, "calibrated_configs", lambda *args: expected)
    monkeypatch.setattr(P, "build_review_configs", lambda *args: {"calibration": cal_cfg})
    monkeypatch.setattr(P, "_coverage", lambda *args: {"passed": True})
    monkeypatch.setattr(contrastive_run, "validate_cohort", lambda *args: None)
    monkeypatch.setattr(contrastive_release, "validate_audit_accounting", lambda *args, **kwargs: {"valid": True})
    monkeypatch.setattr(B, "_cuda_evidence", lambda *args: json.loads(cuda.read_text()))
    monkeypatch.setattr(contrastive_rank_exit, "validate_receipt", lambda *args, **kwargs: None)
    monkeypatch.setattr(B, "_initial_from_calibration", lambda *args: {})
    monkeypatch.setattr(B, "config_from_dict", lambda cfg: cfg)
    monkeypatch.setattr(B, "initial_calibration_description", lambda *args: {})
    monkeypatch.setattr(B, "component_ratios", lambda *args: {name: {"median": .1, "within_screening_bounds": True} for name in ("nce", "rnc")})
    monkeypatch.setattr(B, "measured_allocation", lambda *args: {"allocated_gpu_hours": 2., "phase_allocation_seconds": [900, 900]})
    calls = []
    def evidence(folder, arm, cohort, cfg, provenance, first_update=1, **kwargs):
        calls.append((Path(folder).name, first_update, kwargs))
        records = [{"step": step, "total": 1., "grad_norm": 1., "origin": [step]}
                   for step in range(first_update, 577)]
        return {"gpu_max_memory_gib": 5.}, cfg, records, {"origin_sessions.json": "same", "summary.json": B.sha(full[arm] / "summary.json")}
    monkeypatch.setattr(B, "_run_evidence", evidence)
    pairs = []
    def compare(paths, *args, **kwargs):
        pairs.append(paths)
        return {"status": "PASSED", "passed": True}
    monkeypatch.setattr(contrastive_target_resume, "compare_three_paths", compare)
    parity = {"schema": 1, "profile": "bf16_math_tf32_off", "points": [point(128), point(256)],
              "source_sha256": source, "config_sha256": canonical(expected["C0"]),
              "summary_sha256": B.sha(full["C0"] / "summary.json"), "initial_state_sha256": initial,
              "provenance": provenance}
    parity_path = write(full["C0"] / "zero_weight_parity.json", parity)
    kwargs = dict(cohort_path=cohort_path, full_dirs=full, resumed_dirs=reloads, cuda_path=cuda,
                  cpu_audit_path=audit, calibration_path=cal_path, calibration_run_dir=calibration,
                  cpu_audit_sacct_path=accounting)
    return kwargs, parity_path, parity, calls, pairs


def test_readout_requires_both_reloads_and_only_suffix_545_576(tmp_path, monkeypatch):
    kwargs, parity_path, parity, calls, pairs = readout_fixture(tmp_path, monkeypatch)
    result = B.read_preflight(**kwargs)
    assert result["status"] == "PASSED_FOR_REVIEW"
    assert sorted((name, start) for name, start, _ in calls) == [
        ("C0", 1), ("CN", 1), ("CN_A", 545), ("CN_B", 545), ("CR", 1), ("CR_A", 545), ("CR_B", 545)]
    assert len(pairs) == 2
    assert all(set(p) == {"continuous", "reload_a", "reload_b"} for p in pairs)
    del kwargs["resumed_dirs"]["CR_B"]
    with pytest.raises(ValueError, match="two fresh reloads"):
        B.read_preflight(**kwargs)


@pytest.mark.parametrize("mutation", ["missing", "missing_point", "missing_support", "gradient"])
def test_missing_or_failed_parity_cannot_be_promoted_by_stored_pass(tmp_path, monkeypatch, mutation):
    kwargs, path, parity, _, _ = readout_fixture(tmp_path, monkeypatch)
    if mutation == "missing":
        path.unlink()
    else:
        parity["status"] = "PASSED"
        if mutation == "missing_point": parity["points"].pop()
        elif mutation == "missing_support": parity["points"][0]["ranks"][0]["cases"]["CNb"]["nce_eligible"] = 0.
        else: parity["points"][0]["ranks"][0]["cases"]["CNb"]["gradient_sha256"] = "f" * 64
        write(path, parity)
    result = B.read_preflight(**kwargs)
    assert result["status"] == ("FAILED_REQUIRES_REVIEW" if mutation == "gradient" else "INCONCLUSIVE_REQUIRES_REVIEW")
    assert result["training_approved"] is False


@pytest.mark.parametrize("field", ["source_sha256", "config_sha256", "summary_sha256", "initial_state_sha256", "provenance"])
def test_parity_receipt_is_bound_to_this_control_not_just_same_protocol(tmp_path, monkeypatch, field):
    kwargs, path, parity, _, _ = readout_fixture(tmp_path, monkeypatch)
    parity[field] = "wrong"
    write(path, parity)
    with pytest.raises(ValueError, match="bound to the full control run"):
        B.read_preflight(**kwargs)


def numeric_folder(tmp_path, phase="full", first=1):
    cfg, provenance, cohort = _run_folder(tmp_path)
    cohort["source_sha256"] = provenance["source_sha256"]
    # Production trainer and cohort serializer intentionally use different formatting.
    (tmp_path / "config.json").write_text(json.dumps(cfg, indent=1))
    source = tmp_path / "input.config.json"
    source.write_text(json.dumps(cfg, indent=2) + "\n")
    assert B.sha(source) != B.sha(tmp_path / "config.json")
    summary = json.loads((tmp_path / "summary.json").read_text())
    summary["gpu_max_memory_gib"] = 5.
    write(tmp_path / "summary.json", summary)
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records if row["step"] >= first))
    with numeric_profile(require_cuda=False) as flags:
        for rank in range(4):
            write(tmp_path / f"numeric_profile.rank{rank}.json", {
                "rank": rank, "world_size": 4, "source_sha256": cohort["source_sha256"], "phase": phase,
                "config_sha256": B.sha(source), "config_canonical_sha256": canonical(cfg), "flags": flags})
    return cfg, provenance, cohort


def test_numeric_profile_uses_semantic_config_identity_across_actual_serializer_formats(tmp_path):
    cfg, provenance, cohort = numeric_folder(tmp_path)
    summary, _, records, pins = B._run_evidence(tmp_path, "CN", cohort, cfg, provenance)
    assert len(records) == 576 and summary["gpu_max_memory_gib"] == 5.
    assert all(f"numeric_profile.rank{rank}.json" in pins for rank in range(4))


@pytest.mark.parametrize("field,value", [
    ("source_sha256", "wrong"), ("config_canonical_sha256", "wrong"),
    ("phase", "resume_a"), ("rank", 3), ("world_size", 1),
])
def test_numeric_receipt_cannot_be_reused_for_another_source_config_rank_or_phase(tmp_path, field, value):
    cfg, provenance, cohort = numeric_folder(tmp_path)
    path = tmp_path / "numeric_profile.rank0.json"
    row = json.loads(path.read_text())
    row[field] = value
    write(path, row)
    with pytest.raises(ValueError, match="numeric profile"):
        B._run_evidence(tmp_path, "CN", cohort, cfg, provenance)


def test_reload_evidence_requires_32_update_suffix_and_matching_profile_phase(tmp_path):
    cfg, provenance, cohort = numeric_folder(tmp_path, phase="resume_a", first=545)
    _, _, records, _ = B._run_evidence(tmp_path, "CN", cohort, cfg, provenance, first_update=545, expected_phase="resume_a")
    assert [r["step"] for r in records] == list(range(545, 577))
    with pytest.raises(ValueError, match="numeric profile"):
        B._run_evidence(tmp_path, "CN", cohort, cfg, provenance, first_update=545, expected_phase="resume_b")
    with pytest.raises(ValueError, match="incomplete"):
        B._run_evidence(tmp_path, "CN", cohort, cfg, provenance)


def test_missing_rank_numeric_flags_or_full_batch_memory_remains_blocking(tmp_path):
    cfg, provenance, cohort = numeric_folder(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    summary["gpu_max_memory_gib"] = None
    write(tmp_path / "summary.json", summary)
    with pytest.raises(ValueError, match="memory"):
        B._run_evidence(tmp_path, "CN", cohort, cfg, provenance)
    summary["gpu_max_memory_gib"] = 5.
    write(tmp_path / "summary.json", summary)
    (tmp_path / "numeric_profile.rank3.json").unlink()
    with pytest.raises(FileNotFoundError):
        B._run_evidence(tmp_path, "CN", cohort, cfg, provenance)


def test_numerical_context_must_match_across_separate_allocations(tmp_path, monkeypatch):
    kwargs, _, _, calls, _ = readout_fixture(tmp_path, monkeypatch)
    # Every rank in this allocation agrees and still satisfies bf16-math/TF32-off,
    # but its reduction setting differs from the other six allocations.
    folder = kwargs["resumed_dirs"]["CN_B"]
    for rank in range(4):
        path = folder / f"numeric_profile.rank{rank}.json"
        row = json.loads(path.read_text())
        row["flags"]["bf16_reduced_precision_reduction"] = not row["flags"]["bf16_reduced_precision_reduction"]
        write(path, row)
    report = B.read_preflight(**kwargs)
    assert report["status"] == "FAILED_REQUIRES_REVIEW"
    assert report["numeric_context_across_runs"]["status"] == "FAILED"
    assert len(report["numeric_context_across_runs"]["differences"]) == 4
    assert all(d["fields"] == ["bf16_reduced_precision_reduction"]
               for d in report["numeric_context_across_runs"]["differences"])
    assert calls == [] and report["arms"] == {} and report["training_allocations_gpu_hours"] == {}


def test_missing_profile_does_not_produce_a_scientific_negative(tmp_path, monkeypatch):
    kwargs, _, _, calls, _ = readout_fixture(tmp_path, monkeypatch)
    (kwargs["full_dirs"]["CR"] / "numeric_profile.rank2.json").unlink()
    report = B.read_preflight(**kwargs)
    assert report["status"] == "INCONCLUSIVE_REQUIRES_REVIEW"
    assert report["failure_reasons"] == []
    assert report["numeric_context_across_runs"]["missing"] == ["CR/rank2: missing profile receipt"]
    assert calls == [] and report["training_approved"] is False


def test_complete_consistent_profile_is_pinned_for_all_28_rank_receipts(tmp_path, monkeypatch):
    kwargs, _, _, _, _ = readout_fixture(tmp_path, monkeypatch)
    report = B.read_preflight(**kwargs)
    context = report["numeric_context_across_runs"]
    assert report["status"] == "PASSED_FOR_REVIEW" and context["status"] == "PASSED"
    assert context["differences"] == context["missing"] == []
    assert len(context["profile_receipts_sha256"]) == 7
    assert sum(len(rows) for rows in context["profile_receipts_sha256"].values()) == 28
