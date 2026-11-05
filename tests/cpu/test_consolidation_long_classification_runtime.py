"""CPU execution gates for the long classification controller; no jobs or data."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import consolidation_long_classification_runtime as R
import consolidation_long_classification as M
import test_consolidation_long_classification as F


def measurements():
    """Complete timing evidence with simple explicit costs for projection tests."""
    result = {}
    for ds in M.DATASETS:
        result[ds] = {}
        for view in M.VIEWS:
            bench = {"protocol": M.PROTOCOL, "kind": "benchmark", "dataset": ds, "view": view,
                "step": 19618, "cache_sha256": hashlib.sha256(ds.encode()).hexdigest(),
                "timing_only": True, "validation_or_held_scored": False, "full_fold0_train": True,
                "max_iter": 12000, "sample_rows": 100, "full_train_rows": 100,
                "sample_classes": 2, "full_train_classes": 2, "encoder_dim": 384,
                "width": 768 if view == "pool" else 384,
                "fold_train_rows": [100, 100, 120, 80], "fold_train_classes": [2]*4,
                "load_s": 1., "scale_s": 1., "input_sha256": "1"*64, "splits_sha256": "2"*64,
                "roles_sha256": "3"*64, "ordered_rows_sha256": "4"*64, "checkpoint_sha256": "5"*64,
                "numeric_context": {"threads": 1}, "subjects": ["a", "b"], "fold0_train_subjects": ["a", "b"],
                "fits": [{"C": c, "fit_phase": "benchmark_train_only", "exception": None,
                    "max_iter": 12000, "n_iter": 25, "n_train": 100, "elapsed_s": 1.,
                    "convergence_warnings": 0, "converged_by_criterion": True} for c in M.C_GRID]}
            old = {"dataset": ds, "folds": [{"fold": f, "fit_seconds": 20., "train_rows": n, "classes": 2}
                    for f, n in enumerate(bench["fold_train_rows"])]}
            result[ds][view] = {"benchmark": bench, "historical_timing": old, "peak_rss_bytes": 1024**3}
    return result


def context(tmp_path):
    return {"code_sha": "c"*40, "source_sha256": "f"*64, "plan_pin": {"sha256": "e"*64},
        "plan": {"output_root": str(tmp_path), "recipe_sha256": "d"*64,
            "classification_benchmark_policy": copy.deepcopy(R.BENCHMARK_POLICY),
            "extraction_batch": {"classification": 64},
            "endpoints": {str(s): {"checkpoint": {"sha256": "a"*64}} for s in M.DOSES}}}


def finish(ctx, out, report):
    report = {**report, "complete": True, "checkpoint_unchanged": report["kind"] == "extract-classification",
              "frozen_teacher_unchanged": report["kind"] == "extract-classification"}
    R.C.write_json(Path(out)/"report.json", report)
    return report


def test_global_gate_covers40_logical_cells_and48_fixed_jobs_with_historical_cost_floor():
    gate = R.resource_gate(measurements())
    assert gate["new_fit_cells"] == 40 and gate["all_passed"]
    assert gate["physical_fit_jobs"] == 48
    assert "19618/pool/epn612" not in gate["cells"]
    cell = gate["cells"]["49046/post/epn612"]
    # Historical20s per fold dominates measured7s, then3×1.5 and fixed600s.
    assert cell["projected_fit_seconds_by_fold"] == [90.]*4
    assert cell["projected_seconds"] == 964.
    assert cell["projected_peak_rss_bytes"] == 3*1024**3
    db2 = gate["cells"]["39237/pool/ninapro_db2"]
    assert len(db2["shards"]) == 2 and db2["projected_seconds"] == 1568.
    assert all(gate["shards"][key]["folds"] in ([0, 1], [2, 3]) for key in db2["shards"])
    assert gate["projected_core_hours"] == pytest.approx((32*964+8*1568)*8/3600)


@pytest.mark.parametrize("mutation", ["missing", "sample", "held", "nonconverged", "mixed_numeric", "mixed_cache", "support"])
def test_partial_contaminated_or_incompatible_benchmarks_fail_closed(mutation):
    reports = measurements()
    bench = reports["epn612"]["pre"]["benchmark"]
    if mutation == "missing":
        del reports["uci_emg"]["post"]
    elif mutation == "sample":
        bench["full_fold0_train"] = False
        bench["sample_rows"] = 128
    elif mutation == "held":
        bench["validation_or_held_scored"] = True
    elif mutation == "nonconverged":
        bench["fits"][-1]["converged_by_criterion"] = False
    elif mutation == "mixed_numeric":
        bench["numeric_context"]["threads"] = 2
    elif mutation == "mixed_cache":
        bench["cache_sha256"] = "f"*64
    else:
        bench["fold_train_rows"][-1] += 1
    with pytest.raises(ValueError):
        R.resource_gate(reports)


def test_time_or_memory_blocks_whole_matrix():
    reports = measurements()
    reports["ninapro_db3"]["post"]["historical_timing"]["folds"][0]["fit_seconds"] = 10000.
    gate = R.resource_gate(reports)
    assert not gate["all_passed"]
    assert not gate["cells"]["39237/post/ninapro_db3"]["passed"]
    reports = measurements()
    reports["uci_emg"]["pool"]["peak_rss_bytes"] = 10*1024**3
    assert not R.resource_gate(reports)["all_passed"]


def test_default_benchmark_uses_all_train_rows_and_no_scores(tmp_path, monkeypatch):
    pin, arrays, _, _ = F.cache(tmp_path, step=19618)
    def score_forbidden(*args):
        raise AssertionError("benchmark scored validation/held")
    monkeypatch.setattr(M.TP, "scored", score_forbidden)
    rows = []
    original = M.LogisticRegression.fit
    def capture(self, x, y):
        rows.append(len(y))
        return original(self, x, y)
    monkeypatch.setattr(M.LogisticRegression, "fit", capture)
    bench = M.benchmark_cache(pin["path"], "post", expected=pin)
    assert bench["full_fold0_train"]
    assert rows == [bench["full_train_rows"]]*6
    assert bench["sample_rows"] > 12
    assert bench["subjects"] == bench["fold0_train_subjects"]
    assert bench["fold_train_rows"] == [int(np.isin(arrays["s"], r["train"]).sum())
                                        for r in M.CP.fold_roles(pin["subjects"])]


def test_extract_uses_original_t100g_reference_and_one_teacher(tmp_path, monkeypatch):
    ctx, ds = context(tmp_path), "ninapro_db2"
    ctx["legacy"] = {"inputs": {"splits": {}}, "paths": {"roots": [], "epn_roots": [], "uci_root": "none"}}
    old = {name: {"path": "T100-G-original", "sha256": "b"*64} for name in M.TASKS["p1"]}
    teacher = {"checkpoint": "/teacher.pt", "checkpoint_sha256": "a"*64, "checkpoint_step": 39237, "which": "teacher"}
    loaded, seen = [], []
    monkeypatch.setattr(R, "_historical", lambda ctx: ({}, old))
    monkeypatch.setattr(R.C, "read_pin", lambda pin: {})
    def load(ctx, step, device):
        loaded.append((step, device))
        return SimpleNamespace(), teacher
    monkeypatch.setattr(R.C, "load_teacher", load)
    monkeypatch.setattr(R.C, "finish_output", finish)
    monkeypatch.setattr(R.C, "pinned", lambda pin: None)
    def extract(model, name, **kwargs):
        assert kwargs["batch"] == 64 and kwargs["device"] == "cuda"
        return {"dataset": name}
    def save(path, extraction, identity, reference):
        seen.append((extraction["dataset"], reference, identity))
        Path(path).write_bytes(b"features")
    monkeypatch.setattr(M, "extract_dataset", extract)
    monkeypatch.setattr(M, "save_cache", save)
    ctx["legacy"]["inputs"]["splits"]["sha256"] = "z"*64
    report = R.execute("extract-classification", ctx, tmp_path/"extract", step=39237, task="p1")
    assert loaded == [(39237, "cuda")]
    assert set(report["artifacts"]) == set(M.TASKS["p1"])
    assert all(ref["path"] == "T100-G-original" and identity["step"] == 39237 for _, ref, identity in seen)
    assert report["kind"] == "extract-classification" and report["checkpoint_step"] == 39237


def test_fit_requires_whole_matrix_gate_and_forbids_pool_reuse_or_automatic_shards(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    monkeypatch.setattr(R, "_threads", lambda: {"threads": 1})
    monkeypatch.setattr(R, "_benchmarks", lambda ctx: measurements())
    called = []
    monkeypatch.setattr(M, "fit_cache", lambda *args, **kwargs: called.append(True))
    bad = measurements()
    bad["uci_emg"]["post"]["peak_rss_bytes"] = 20*1024**3
    monkeypatch.setattr(R, "_benchmarks", lambda ctx: bad)
    with pytest.raises(ValueError, match="matrix resource gate failed"):
        R.execute("fit-classification", ctx, tmp_path/"blocked", step=39237, dataset="epn612", view="post")
    assert not called and not (tmp_path/"blocked").exists()
    for step, view, folds in [(19618, "pool", None), (39237, "pre", [0])]:
        with pytest.raises(ValueError, match="logical|unregistered"):
            R.execute("fit-classification", ctx, tmp_path/"invalid", step=step, dataset="epn612", view=view, folds=folds)
    for folds in (None, [0, 1, 2, 3], [0, 2]):
        with pytest.raises(ValueError, match="registered"):
            R.execute("fit-classification", ctx, tmp_path/"invalid-db2", step=39237, dataset="ninapro_db2", view="post", folds=folds)


def test_fit_receipt_covers_real_probe_oof_and_durable_audit(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    pin, _, _, _ = F.cache(tmp_path)
    monkeypatch.setattr(R, "_threads", lambda: {"threads": 1})
    monkeypatch.setattr(R, "_benchmarks", lambda ctx: measurements())
    monkeypatch.setattr(R, "_cache", lambda *args: ({"receipt": {"path": "/extraction.json"}}, pin, {}, {}))
    monkeypatch.setattr(R.C, "finish_output", finish)
    ctx["plan"]["benchmark_receipts"] = {"classification": {"path": "/benchmarkmanifest.json"}}
    out = tmp_path/"fit"
    receipt = R.execute("fit-classification", ctx, out, step=39237, dataset="ninapro_db2", view="pre", folds=[0, 1])
    assert set(receipt["artifacts"]) == {"probe", "oof", "audit"}
    report, oof = M._read_pinned_report(receipt["artifacts"]["probe"])
    assert report["fold_indices"] == receipt["fold_indices"] == [0, 1] and len(report["fit_audit"]) == 14
    assert len(oof["row"]) == 24 and receipt["global_resource_gate_passed"]


def test_receipt_requires_same_reviewed_recipe_and_source(tmp_path):
    ctx = context(tmp_path)
    value = {"protocol": R.C.PROTOCOL, "kind": "fit-classification", "complete": True,
             "source_sha256": ctx["source_sha256"], "recipe_sha256": ctx["plan"]["recipe_sha256"]}
    R._receipt(ctx, value, "fit-classification")
    value["recipe_sha256"] = "bad"
    with pytest.raises(ValueError, match="stage receipt"):
        R._receipt(ctx, value, "fit-classification")


def test_benchmark_manifest_checks_all_receipt_pins_and_same_reviewed_recipe(tmp_path, monkeypatch):
    ctx, values, pins = context(tmp_path), measurements(), {}
    for ds in M.DATASETS:
        pins[ds] = {}
        for view in M.VIEWS:
            report = {**values[ds][view], "protocol": R.C.PROTOCOL, "kind": "benchmark-classification",
                "complete": True, "source_sha256": ctx["source_sha256"], "recipe_sha256": ctx["plan"]["recipe_sha256"],
                "policy": R.BENCHMARK_POLICY}
            path = tmp_path/f"{ds}-{view}.json"
            R.C.write_json(path, report)
            pins[ds][view] = R.C.file_pin(path)
    manifest = tmp_path/"benchmarks.json"
    R.C.write_json(manifest, {"reports": pins})
    ctx["plan"]["benchmark_receipts"] = {"classification": R.C.file_pin(manifest)}
    monkeypatch.setattr(R, "_cache", lambda ctx, step, ds: ({}, {
        "sha256": hashlib.sha256(ds.encode()).hexdigest(), "checkpoint_sha256": "5"*64}, {}, {}))
    assert R.resource_gate(R._benchmarks(ctx))["all_passed"]
    bad = json.loads(Path(pins["epn612"]["post"]["path"]).read_text())
    bad["recipe_sha256"] = "0"*64
    path = tmp_path/"different-recipe.json"
    R.C.write_json(path, bad)
    pins["epn612"]["post"] = R.C.file_pin(path)
    revised = tmp_path/"different-manifest.json"
    R.C.write_json(revised, {"reports": pins})
    ctx["plan"]["benchmark_receipts"]["classification"] = R.C.file_pin(revised)
    with pytest.raises(ValueError, match="stage receipt"):
        R._benchmarks(ctx)


def test_archived_db2_floor_blocks_one4h_job_and_fits_registered_pairs():
    path = ROOT/"results/infra/tokenizer_evaluation_20261103/probe_cpu_19618/completed_aggregates/T100-G/p1.audit.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records = [r for r in records if r["dataset"] == "ninapro_db2"]
    reports = measurements()
    for view in M.VIEWS:
        bench = reports["ninapro_db2"][view]["benchmark"]
        counts, classes, costs = [], [], []
        for f in range(4):
            fits = [r for r in records if r["fold"] == f]
            counts.append(fits[0]["n_train"])
            classes.append(len(fits[0]["classes"]))
            costs.append(sum(r["elapsed_s"] for r in fits))
        bench.update(fold_train_rows=counts, fold_train_classes=classes,
            full_train_rows=counts[0], sample_rows=counts[0], full_train_classes=classes[0], sample_classes=classes[0])
        for fit in bench["fits"]:
            fit["n_train"] = counts[0]
        reports["ninapro_db2"][view]["historical_timing"]["folds"] = [
            {"fold": f, "fit_seconds": costs[f], "train_rows": counts[f], "classes": classes[f]} for f in range(4)]
    gate = R.resource_gate(reports)
    cell = gate["cells"]["39237/pool/ninapro_db2"]
    assert not cell["unsharded_passed"]
    assert cell["full_cell_projected_seconds"] > 13800
    assert cell["passed"]
    assert max(gate["shards"][k]["projected_seconds"] for k in cell["shards"]) < 13800
    # Also charge every observed candidate plus a worst-case candidate refit;
    # the explicit4h pairs pass while a single4h job remains blocked.
    for view in M.VIEWS:
        for fit in reports["ninapro_db2"][view]["benchmark"]["fits"]:
            old = next(r for r in records if r["fold"] == 0 and r["fit_phase"] == "candidate" and r["C"] == fit["C"])
            fit["elapsed_s"] = old["elapsed_s"]
    actual = R.resource_gate(reports)
    assert actual["all_passed"]
    assert not actual["cells"]["39237/pool/ninapro_db2"]["unsharded_passed"]
    assert max(actual["shards"][k]["projected_seconds"] for k in cell["shards"]) < 13800
