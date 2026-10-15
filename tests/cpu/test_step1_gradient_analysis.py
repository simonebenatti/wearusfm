import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_step1_gradients.py"
spec = importlib.util.spec_from_file_location("step1_gradient_analysis", SCRIPT)
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


def _config(name):
    return {"seed": int(name[-1]), "max_steps": 4000, "batch_size": 32, "keep_grad_every": 500,
            "grad_clip": 1.0, "pairing_audit": True, "datasets": ["emg2qwerty"],
            "keep_calibration_sha256": G.INPUT_SHA256["calibration"],
            "model": {"keep_readout": True, "multiscale_anchor": True, "rvq_codes": None},
            "loader": {"multiscale_anchor": True},
            "jepa": {"target": "b", "anchor_weight": .2, "keep_weight": 0.0 if name[0] == "A" else .05,
                     "loss_per_window": True, "ema_momentum": .996, "rvq_targets": "slab"}}


def _rows():
    return [{"step": step, "shared_grad_jepa": 2.0, "shared_grad_anchors": 1.0,
             "shared_grad_keep": .5, "grad_norm": 2.0 if i == 0 else .5}
            for i, step in enumerate(G.EXPECTED_LOG_STEPS)]


def _export(tmp_path):
    manifest = {"runs": {}}
    for name in G.RUNS:
        raw = ("" if name[0] == "A" else "".join(json.dumps(row) + "\n" for row in _rows())).encode()
        (tmp_path / f"{name}.jsonl").write_bytes(raw)
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(_config(name)))
        provenance = {"source_commit": G.TRAIN_COMMIT, "arm": name[0], "model_seed": int(name[-1]),
                      "input_sha256": G.INPUT_SHA256, "checkpoint_sha256": G.CHECKPOINTS[name]}
        (directory / "run_provenance.json").write_text(json.dumps(provenance))
        manifest["runs"][name] = {"metrics_sha256": G.sha256(f"full log {name}".encode()), "metrics_lines": 4000,
                                   "export_sha256": G.sha256(raw), "export_lines": 0 if name[0] == "A" else 8}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_weighted_ratios_are_not_weighted_twice_and_indexing_is_explicit(tmp_path):
    _export(tmp_path)
    result = G.analyze_export(tmp_path)
    assert result["training_source"]["sha256"] == G.TRAINER_SHA256
    for name in ("A_seed0", "A_seed1"):
        assert result["runs"][name]["status"] == "excluded_by_design"
        assert "norms_and_ratios" not in result["runs"][name]
    b = result["runs"]["B_seed0"]
    assert b["coverage"]["sampled_update_indices"] == list(range(0, 4000, 500))
    assert b["coverage"]["fraction"] == .002
    assert b["samples"][1]["log_step"] == 501 and b["samples"][1]["update_index"] == 500
    assert b["norms_and_ratios"]["anchors_over_jepa"]["median"] == .5
    assert b["norms_and_ratios"]["keep_over_jepa"]["median"] == .25
    assert b["clipping_at_diagnostic_samples"] == {"threshold": 1., "count": 1, "fraction": .125, "log_steps": [1]}
    assert result["pooled_B_descriptive"]["norms_and_ratios"]["keep_over_jepa"]["n"] == 16


@pytest.mark.parametrize("mutation", ["missing", "off_by_one", "duplicate", "unexpected_A"])
def test_sparse_coverage_must_match_the_historical_diagnostic(mutation):
    rows = _rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "off_by_one":
        rows[1]["step"] = 500
    elif mutation == "duplicate":
        rows[1]["step"] = 1
    if mutation == "unexpected_A":
        with pytest.raises(ValueError, match="expected diagnostic"):
            G.analyze_rows("A_seed0", rows, _config("A_seed0"))
    else:
        with pytest.raises(ValueError, match="expected diagnostic"):
            G.analyze_rows("B_seed0", rows, _config("B_seed0"))


@pytest.mark.parametrize("key,value", [("shared_grad_keep", float("nan")), ("shared_grad_anchors", float("inf")),
                                       ("shared_grad_jepa", 0.0), ("grad_norm", -1.0)])
def test_nonfinite_negative_and_undefined_ratios_are_rejected(key, value):
    rows = _rows()
    rows[3][key] = value
    with pytest.raises(ValueError, match="step 1501"):
        G.analyze_rows("B_seed0", rows, _config("B_seed0"))


@pytest.mark.parametrize("field", ["source_commit", "checkpoint_sha256", "model_seed", "input_sha256"])
def test_wrong_frozen_provenance_is_rejected(tmp_path, field):
    _export(tmp_path)
    path = tmp_path / "B_seed0" / "run_provenance.json"
    provenance = json.loads(path.read_text())
    provenance[field] = "changed"
    path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="provenance"):
        G.analyze_export(tmp_path)


def test_export_hash_and_unexpected_pair_configuration_are_rejected(tmp_path):
    manifest = _export(tmp_path)
    path = tmp_path / "B_seed0.jsonl"
    raw = path.read_bytes()
    path.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        G.analyze_export(tmp_path)
    path.write_bytes(raw)
    config_path = tmp_path / "B_seed0" / "config.json"
    cfg = json.loads(config_path.read_text())
    cfg["loader"]["mask_changed"] = True
    config_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="beyond model seed"):
        G.analyze_export(tmp_path)
    manifest["runs"]["A_seed0"]["metrics_lines"] = 3999
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="endpoint"):
        G.analyze_export(tmp_path)


def test_cli_creates_a_finite_report_and_refuses_overwrite(tmp_path):
    _export(tmp_path)
    out = tmp_path / "out.json"
    assert G.main(["--export-dir", str(tmp_path), "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["analysis"] == "step1_weighted_shared_gradients_descriptive_v1"
    with pytest.raises(SystemExit):
        G.main(["--export-dir", str(tmp_path), "--out", str(out)])
