import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("contrastive_prepare", ROOT / "scripts/contrastive_prepare.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def baseline():
    return json.loads(P.BASELINE.read_text())


def fixture_run(path):
    path.mkdir()
    cfg = P.build_review_configs(baseline())["calibration"]
    coverage = {"through_step": 511, "updates": 256, "nce_fraction": 1., "rnc_active_fraction": 1.,
                "datasets": {d: {"queries": 4096, "eligible": 4096, "fraction": 1.} for d in ("A", "B")}, "passed": True}
    summary = {"steps": 512, "stopped": "max_steps", "world_size": 4, "batch_per_process": 8, "workers_per_rank": 6,
               "code": {"sha": "f" * 40}, "contrastive": {
                   "config": cfg["contrastive"], "optimizer_updates": 0, "queue_last_step": 511,
                   "initial_sha256": "a" * 64, "final_student_sha256": "a" * 64, "final_teacher_sha256": "a" * 64,
                   "expected_datasets": ["A", "B"], "coverage_completed_blocks": [], "coverage_partial_block": coverage}}
    records = []
    for step in range(1, 513):
        row = {"step": step, "contrastive_calibration_only": True,
               "contrast_nce_eligible": 32, "contrast_rnc_windows": 4}
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
        assert cfg["model"]["contrastive_projection"] is True
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
    records[257]["contrast_nce_eligible"] = 0
    write_run(folder, cfg, summary, records)
    report = P.calibration_report(folder)
    assert report["status"] == "PASSED_FOR_REVIEW" and report["training_approved"] is False
    assert report["selected_steps"] == list(range(259, 291))
    assert report["weights"] == {"nce": .2, "rnc": .4}
    assert len(report["batch_decisions"]) == 256
    assert report["batch_decisions"][0]["reasons"] == ["invalid_gradient_rnc"]
    assert all(len(p["sha256"]) == 64 for p in report["inputs"].values())


@pytest.mark.parametrize("kind", ["step", "optimizer", "student", "teacher", "workers", "world", "dataset", "metric", "config", "nan", "code_missing", "code_short", "code_nonhex"])
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
    else:
        block = summary["contrastive"]["coverage_partial_block"]
        block["rnc_active_fraction"] = 230 / 256
        block["passed"] = False
    write_run(folder, cfg, summary, records)
    report = P.calibration_report(folder)
    assert report["status"] == "FAILED_REQUIRES_REVIEW" and report["training_approved"] is False
    assert report["failure_reasons"]
    if kind == "weight": assert report["weights"]["nce"] == 100.
    if kind == "support": assert report["selected_steps"] == [] and report["weights"] == {}
