import copy
import importlib.util
import json
from pathlib import Path

import pytest
from wearusfm.training.contrastive_objectives import ContrastiveRuntime

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("contrastive_prepare", ROOT / "scripts/contrastive_prepare.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def baseline():
    return json.loads(P.BASELINE.read_text())


def fixture_run(path):
    path.mkdir()
    cfg = P.build_review_configs(baseline())["calibration"]
    reference = P.policy_reference("coverage", "manifest_reference")
    input_bindings = P._read(P.PLAN).get("input_sha256", {
        "manifest": reference["source_manifest"]["sha256"], "window_seconds": reference["source_window_seconds"]["sha256"],
        "scales": "b" * 64, "pose_stats": "b" * 64})
    datasets = list(cfg["contrastive"]["coverage_quotas"])
    counts = {d: 1 for d in datasets}
    counts[datasets[0]] += 32 - len(datasets)
    runtime = ContrastiveRuntime(cfg["contrastive"])
    runtime.expected_datasets = tuple(datasets)
    runtime.step = 511
    runtime.queue.last_step = 511
    runtime.coverage = {"updates": 256, "rnc_active": 256,
                        "datasets": {d: [n * 256, n * 256] for d, n in counts.items()}}
    coverage = runtime.coverage_report()
    summary = {"steps": 512, "stopped": "max_steps", "world_size": 4, "batch_per_process": 8, "workers_per_rank": 6,
               "code": {"sha": "f" * 40}, "contrastive": {
                   "config": cfg["contrastive"], "optimizer_updates": 0, "queue_last_step": 511,
                   "initial_sha256": "a" * 64, "final_student_sha256": "a" * 64, "final_teacher_sha256": "a" * 64,
                   "provenance": {"source_sha256": P.source_digest(),
                                  "input_sha256": input_bindings},
                   "expected_datasets": datasets, "coverage_completed_blocks": [], "coverage_partial_block": coverage}}
    records = []
    for step in range(1, 513):
        row = {"step": step, "contrastive_calibration_only": True,
               "contrast_nce_eligible": 32, "contrast_rnc_windows": 4}
        for d, n in counts.items():
            row["contrast_nce_queries__" + d] = n
            row["contrast_nce_eligible__" + d] = n
        if step > 256:
            row.update(contrast_global_grad_base=1., contrast_global_grad_nce=.5, contrast_global_grad_rnc=.25,
                       contrast_global_cos_nce_base=.1, contrast_global_cos_rnc_base=-.2)
        records.append(row)
    write_run(path, cfg, summary, records)
    return cfg, summary, records


def write_run(path, cfg, summary, records):
    (path / "config.json").write_text(json.dumps(cfg))
    (path / "summary.json").write_text(json.dumps(summary))
    (path / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def test_config_roundtrip_restores_nested_types_and_preserves_baseline():
    before = baseline()
    cfg = P.config_from_dict(before)
    assert isinstance(cfg.loader.mask, P.MaskSpec)
    assert isinstance(cfg.loader.virtual, P.VirtualSpec)
    assert isinstance(cfg.schedule, P.R.Schedule)
    assert isinstance(cfg.alarm, P.R.AlarmRule)
    assert isinstance(cfg.loader.mask.long, tuple)
    assert isinstance(cfg.loader.virtual.shapes[0], tuple)
    assert P.R.config_to_dict(P.config_from_dict(P.R.config_to_dict(cfg))) == P.R.config_to_dict(cfg)
    assert before == baseline()
    bad = copy.deepcopy(before)
    bad["unregistered"] = True
    with pytest.raises(TypeError): P.config_from_dict(bad)


def test_review_configs_share_recipe_and_training_weights_are_not_executable():
    configs = P.build_review_configs(baseline())
    shared = []
    for name in ("C0", "CN", "CR"):
        cfg = copy.deepcopy(configs[name])
        cc = cfg.pop("contrastive")
        assert cc["arm"] == name and cc["nce_weight"] is None and cc["rnc_weight"] is None
        with pytest.raises(ValueError): P.ContrastiveConfig(**cc)
        assert "contrastive_projection" not in cfg["model"]
        shared.append(cfg)
    assert shared[0] == shared[1] == shared[2]
    cal = configs["calibration"]
    assert cal["max_steps"] == 512 and cal["warmup_steps"] == 1962
    assert cal["schedule"]["total_steps"] == 49046
    cc = P.ContrastiveConfig(**cal["contrastive"])
    assert cc.calibration_only and cc.weights(511) == (0., 0.)


@pytest.mark.parametrize("field,value", [("batch_size", 64), ("lr", 3e-4), ("max_steps", 5000), ("seed", 3)])
def test_preparation_rejects_shifted_recipe(field, value):
    b = baseline()
    b[field] = value
    with pytest.raises(ValueError): P.build_review_configs(b)


def test_calibration_report_selects_same_first32_informative_batches(tmp_path):
    folder = tmp_path / "run"
    cfg, summary, records = fixture_run(folder)
    records[256]["contrast_global_grad_rnc"] = 0.
    records[257]["contrast_global_grad_nce"] = 0.
    write_run(folder, cfg, summary, records)
    report = P.calibration_report(folder)
    assert report["status"] == "PASSED_FOR_REVIEW" and report["training_approved"] is False
    assert report["selected_steps"] == list(range(259, 291))
    assert report["weights"] == {"nce": .2 * P.historical_gradient_multiplier(), "rnc": .4 * P.historical_gradient_multiplier()}
    assert len(report["batch_decisions"]) == 256
    assert report["batch_decisions"][0]["reasons"] == ["invalid_gradient_rnc"]
    assert all(len(p["sha256"]) == 64 for p in report["inputs"].values())


@pytest.mark.parametrize("kind", ["step", "optimizer", "student", "teacher", "workers", "world", "dataset", "metric", "config", "nan", "code_missing", "code_short", "code_nonhex", "float_step", "metric_coverage", "source", "float_summary", "input_binding"])
def test_calibration_rejects_inconsistent_or_incomplete_measurement(tmp_path, kind):
    folder = tmp_path / "run"
    cfg, summary, records = fixture_run(folder)
    if kind == "step": summary["steps"] = 511
    elif kind == "optimizer": summary["contrastive"]["optimizer_updates"] = 1
    elif kind == "student": summary["contrastive"]["final_student_sha256"] = "b" * 64
    elif kind == "teacher": summary["contrastive"]["final_teacher_sha256"] = "b" * 64
    elif kind == "workers": summary["workers_per_rank"] = 4
    elif kind == "world": summary["world_size"] = 2
    elif kind == "dataset": summary["contrastive"]["expected_datasets"].append("C")
    elif kind == "metric": records[257]["step"] = 257
    elif kind == "config": cfg["lr"] *= 2
    elif kind == "nan": records[257]["contrast_global_grad_nce"] = float("nan")
    elif kind == "code_missing": summary.pop("code")
    elif kind == "code_short": summary["code"]["sha"] = "f" * 39
    elif kind == "code_nonhex": summary["code"]["sha"] = "x" * 40
    elif kind == "float_step": records[257]["step"] = 258.
    elif kind == "metric_coverage": records[257]["contrast_nce_eligible"] -= 1
    elif kind == "source": summary["contrastive"]["provenance"]["source_sha256"] = "0" * 64
    elif kind == "float_summary": summary["steps"] = 512.
    elif kind == "input_binding": summary["contrastive"]["provenance"]["input_sha256"].pop("manifest")
    write_run(folder, cfg, summary, records)
    with pytest.raises(ValueError): P.calibration_report(folder)


@pytest.mark.parametrize("kind", ["weight", "support", "coverage"])
def test_calibration_reports_scientific_gate_failure_without_clamp_or_extension(tmp_path, kind):
    folder = tmp_path / "run"
    cfg, summary, records = fixture_run(folder)
    if kind == "weight":
        for r in records[256:]: r["contrast_global_grad_nce"] = 1e-3
    elif kind == "support":
        for r in records[256:]: r["contrast_rnc_windows"] = 0
        summary["contrastive"]["coverage_partial_block"].update(rnc_active_fraction=0., passed=False)
    else:
        block = summary["contrastive"]["coverage_partial_block"]
        for r in records[256:282]: r["contrast_rnc_windows"] = 0
        block["rnc_active_fraction"] = 230 / 256
        block["passed"] = False
    write_run(folder, cfg, summary, records)
    report = P.calibration_report(folder)
    assert report["status"] == "FAILED_REQUIRES_REVIEW" and report["training_approved"] is False
    assert report["failure_reasons"]
    if kind == "weight": assert report["weights"]["nce"] == 100. * P.historical_gradient_multiplier()
    if kind == "support": assert report["selected_steps"] == [] and report["weights"] == {}


@pytest.mark.parametrize("line", ['{"step":1,"step":2}', '{"step":1,"x":NaN}', '{"step":1,"x":1e999}'])
def test_metric_json_rejects_duplicate_and_nonfinite_numbers(line):
    with pytest.raises(ValueError): P._read_json_line(line)


def test_receipt_is_recomputed_not_trusted_and_all_arms_share_it(tmp_path):
    run = tmp_path / "run"
    fixture_run(run)
    receipt = P.calibration_report(run)
    configs = P.calibrated_configs(receipt, run)
    assert set(configs) == {"C0", "CN", "CR"}
    assert len({(c["contrastive"]["nce_weight"], c["contrastive"]["rnc_weight"]) for c in configs.values()}) == 1
    for mutate in (lambda r: r["weights"].update(nce=.9), lambda r: r["failure_reasons"].append("bad"),
                   lambda r: r.update(initial_state_sha256="b" * 64)):
        bad = copy.deepcopy(receipt)
        mutate(bad)
        with pytest.raises(ValueError): P.authenticate_calibration(bad, run)
    with (run / "metrics.jsonl").open("a") as out: out.write("{}\n")
    with pytest.raises(ValueError): P.authenticate_calibration(receipt, run)


def test_quota_exemption_keeps_counts_in_global_gate(tmp_path):
    run = tmp_path / "run"
    cfg, summary, records = fixture_run(run)
    block = summary["contrastive"]["coverage_partial_block"]
    for row in records[256:]:
        row["contrast_nce_eligible"] -= row["contrast_nce_eligible__capgmyo"]
        row["contrast_nce_eligible__capgmyo"] = 0
    block["datasets"]["capgmyo"].update(eligible=0, fraction=0.)
    block["nce_fraction"] = (8192 - block["datasets"]["capgmyo"]["queries"]) / 8192
    write_run(run, cfg, summary, records)
    report = P.calibration_report(run)
    assert report["status"] == "PASSED_FOR_REVIEW"
    assert report["coverage"]["datasets"]["capgmyo"]["exempt"]
    assert sum(v["queries"] for v in report["coverage"]["datasets"].values()) == 8192
    block["datasets"].pop("capgmyo")
    write_run(run, cfg, summary, records)
    with pytest.raises(ValueError): P.calibration_report(run)


def test_extreme_finite_norms_produce_failed_json_receipt_without_infinity(tmp_path):
    run = tmp_path / "run"
    cfg, summary, records = fixture_run(run)
    for r in records[256:]: r.update(contrast_global_grad_base=1e308, contrast_global_grad_nce=1e-308)
    write_run(run, cfg, summary, records)
    receipt = P.calibration_report(run)
    assert receipt["status"] == "FAILED_REQUIRES_REVIEW" and receipt["weights"]["nce"] is None
    json.dumps(receipt, allow_nan=False)


def test_repository_pins_are_portable():
    assert P._pin(P.BASELINE)["path"] == str(P.BASELINE.relative_to(ROOT))


def test_coverage_policy_rejects_consistently_repinned_alternate_data(tmp_path):
    folder = tmp_path / "run"
    _, summary, _ = fixture_run(folder)
    pins = summary["contrastive"]["provenance"]["input_sha256"]
    P.validate_input_bindings(pins)
    for name in ("manifest", "window_seconds"):
        changed = {**pins, name: "0" * 64}
        with pytest.raises(ValueError, match="coverage-policy source"):
            P.validate_input_bindings(changed)


def test_calibration_receipt_is_identical_after_evidence_directory_relocation(tmp_path):
    import shutil
    original = tmp_path / "cluster_checkout_output"
    fixture_run(original)
    receipt = P.calibration_report(original)
    copied = tmp_path / "mac_copy" / "calibration"
    shutil.copytree(original, copied)
    assert P.calibration_report(copied) == receipt
    assert P.authenticate_calibration(receipt, copied) == receipt
    assert all(pin["path"] == name for name, pin in receipt["inputs"].items())
