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


def measurements(numeric=None):
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
                "numeric_context": copy.deepcopy(numeric or F.historical_numeric_context()), "subjects": ["a", "b"], "fold0_train_subjects": ["a", "b"],
                "fits": [{"C": c, "fit_phase": "benchmark_train_only", "exception": None,
                    "max_iter": 12000, "n_iter": 25, "n_train": 100, "elapsed_s": 1.,
                    "convergence_warnings": 0, "converged_by_criterion": True} for c in M.C_GRID]}
            old = {"dataset": ds, "numeric_sha256": M.jhash(F.historical_numeric_context()), "folds": [{"fold": f, "fit_seconds": 20., "train_rows": n, "classes": 2}
                    for f, n in enumerate(bench["fold_train_rows"])]}
            result[ds][view] = {"benchmark": bench, "historical_timing": old, "peak_rss_bytes": 1024**3,
                "historical_numeric_match": R._numeric_match(bench["numeric_context"], historical_numeric())}
    return result


def historical_numeric():
    return {"numeric_sha256": M.jhash(F.historical_numeric_context()),
            "numeric_context": F.historical_numeric_context(), "reports": {t: [] for t in M.TASKS}}


def table(reports=None):
    """Walltime table as the plan freezes it: the amended gate on the benchmark receipts."""
    shards = R.amended_gate(reports or measurements())["shards"]
    return {key: {k: shard[k] for k in ("step", "view", "dataset", "folds", "projected_seconds", "walltime_seconds")}
            for key, shard in shards.items()}


def context(tmp_path):
    """Amendment A1 context: new outputs carry source f..., listed inputs keep the original source 0..."""
    return {"code_sha": "c"*40, "source_sha256": "f"*64, "plan_pin": {"sha256": "e"*64},
        "amendment": R.C.A.record(),
        "inputs": {"source_sha256": "0"*64, "recipe_sha256": "d"*64, "receipts": [],
                   "phases": {"extract": {"code_sha": "1"*40, "plan_sha256": "2"*64},
                              "benchmark": {"code_sha": "3"*40, "plan_sha256": "4"*64}}},
        "plan": {"output_root": str(tmp_path), "recipe_sha256": "d"*64,
            "classification_benchmark_policy": copy.deepcopy(R.BENCHMARK_POLICY),
            "amendment": {"fit_shards": table()}, "classification_baseline_policy": "declared_historical_cache",
            "extraction_batch": {"classification": 64},
            "endpoints": {str(s): {"checkpoint": {"sha256": "a"*64}} for s in M.DOSES}}}


def admit(ctx, pin, kind, phase="benchmark"):
    """List one receipt pin among the inputs of the amendment, as the manifest does."""
    ctx["inputs"]["receipts"].append({"id": pin["path"], "phase": phase, "kind": kind, "fields": {},
                                      **{k: pin[k] for k in ("path", "sha256", "bytes")}})
    return {k: pin[k] for k in ("path", "sha256", "bytes")}


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


def test_global_gate_accepts_archived_extra_openmp_but_requires_one_new_context():
    preflight = json.loads((ROOT/"results/infra/consolidation_long_20261105/numeric_preflight_20261105.json").read_text())
    current = next(p["numeric_context"] for p in preflight["probes"] if p["probe"] == "with_torch")
    reports = measurements(current)
    gate = R.resource_gate(reports)
    assert gate["all_passed"] and gate["numeric_sha256"] == M.jhash(current)
    assert all(item["historical_numeric_match"]["compatible"] is True
               and item["historical_numeric_match"]["exact_match"] is False
               for views in reports.values() for item in views.values())
    # Both are individually compatible with the old report, but new cells must
    # run in one exact environment rather than silently mixing import contexts.
    old_item = measurements()["epn612"]["post"]
    reports["epn612"]["post"] = old_item
    with pytest.raises(ValueError, match="mixed numeric environments"):
        R.resource_gate(reports)


@pytest.mark.parametrize("active,environment,expected,passes", [
    (8, 8, 8, True), (1, 1, 1, True), (1, 8, 8, False), (8, 1, 8, False)])
def test_threads_match_historical_phase_in_environment_and_loaded_libraries(monkeypatch, active, environment, expected, passes):
    numeric = {"threadpools": [{"internal_api": "openblas", "num_threads": active}]}
    monkeypatch.setattr(M.TP, "numeric_context", lambda: numeric)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, str(environment))
    if passes:
        assert R._threads(expected) == numeric
    else:
        with pytest.raises(ValueError, match="threads|thread count"):
            R._threads(expected)


@pytest.mark.parametrize("mutation", ["timing", "evidence", "all_benchmarks_one_thread"])
def test_resource_gate_rejects_historical_context_mismatch_even_when_new_contexts_agree(mutation):
    reports = measurements()
    if mutation == "timing":
        reports["ninapro_db2"]["pool"]["historical_timing"]["numeric_sha256"] = "a"*64
    elif mutation == "evidence":
        reports["epn612"]["pre"]["historical_numeric_match"]["exact_match"] = False
    else:
        for views in reports.values():
            for item in views.values():
                item["benchmark"]["numeric_context"]["threadpools"][0]["num_threads"] = 1
    with pytest.raises(ValueError, match="numeric"):
        R.resource_gate(reports)


def test_historical_numeric_reference_requires_pinned_plan_recipe_and_matching_p1_p2(tmp_path):
    historical = {"identity": "frozen"}
    ctx = context(tmp_path)
    ctx["plan"]["historical_classification_reports"] = {}
    for task in M.TASKS:
        path = tmp_path/(task+".json")
        R.C.write_json(path, {"complete": True, "protocol": M.TP.PROTOCOL, "kind": "aggregate",
            "step": 19618, "arm": "T100-G", "task": task, "recipe": M.TP.recipe(),
            "plan_content_sha256": M.jhash(historical), "numeric_context": F.historical_numeric_context(),
            "numeric_sha256": M.jhash(F.historical_numeric_context())})
        ctx["plan"]["historical_classification_reports"][task] = R.C.file_pin(path)
    reference = R._historical_numeric(ctx, historical)
    match = R._numeric_match(F.historical_numeric_context(), reference)
    assert match["exact_match"] and match["numeric_sha256"] == match["historical_numeric_sha256"]
    assert set(match["historical_reports"]) == {"p1", "p2"}
    with pytest.raises(ValueError, match="new/historical"):
        R._numeric_match({**F.historical_numeric_context(), "numpy": "different"}, reference)
    with pytest.raises(ValueError, match="historical timing source"):
        R._historical_numeric(ctx, {"identity": "different-plan"})
    pin = ctx["plan"]["historical_classification_reports"]["p2"]
    changed = json.loads(Path(pin["path"]).read_text())
    altered = {**F.historical_numeric_context(), "numpy": "different"}
    changed.update(numeric_context=altered, numeric_sha256=M.jhash(altered))
    different = tmp_path/"different-p2.json"
    R.C.write_json(different, changed)
    ctx["plan"]["historical_classification_reports"]["p2"] = R.C.file_pin(different)
    with pytest.raises(ValueError, match="historical P1/P2"):
        R._historical_numeric(ctx, historical)


@pytest.mark.parametrize("stage,arguments", [
    ("extract-classification", {"step": 39237, "task": "p1"}),
    ("benchmark-classification", {"step": 19618, "dataset": "epn612", "view": "post"})])
def test_closed_stages_are_refused_before_any_work_or_output(tmp_path, monkeypatch, stage, arguments):
    """Amendment A1 reuses the original receipts: this source never extracts or benchmarks again."""
    ctx = context(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("closed stage started")
    for name in ("_policy", "_threads", "_cache", "_historical", "_historical_numeric", "_historical_timing"):
        monkeypatch.setattr(R, name, forbidden)
    for name in ("load_teacher", "begin_output", "read_pin"):
        monkeypatch.setattr(R.C, name, forbidden)
    for name in ("benchmark_cache", "extract_dataset", "save_cache"):
        monkeypatch.setattr(M, name, forbidden)
    with pytest.raises(ValueError, match="closed by amendment A1"):
        R.execute(stage, ctx, tmp_path/"closed", **arguments)
    assert not (tmp_path/"closed").exists()
    assert stage in R.C.A.CLOSED_STAGES


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
        bench["numeric_context"]["threadpools"][0]["num_threads"] = 2
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


def test_fit_requires_whole_matrix_gate_and_forbids_pool_reuse_or_unregistered_shards(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    monkeypatch.setattr(R, "_threads", lambda: F.historical_numeric_context())
    monkeypatch.setattr(R, "_historical", lambda ctx: ({}, {}))
    monkeypatch.setattr(R, "_historical_numeric", lambda ctx, historical: historical_numeric())
    called = []
    monkeypatch.setattr(M, "fit_cache", lambda *args, **kwargs: called.append(True))
    bad = measurements()
    bad["uci_emg"]["post"]["peak_rss_bytes"] = 20*1024**3
    monkeypatch.setattr(R, "_benchmarks", lambda ctx: bad)
    with pytest.raises(ValueError, match="matrix resource gate failed"):
        R.execute("fit-classification", ctx, tmp_path/"blocked", step=39237, dataset="epn612", view="post", folds=[0, 1, 2, 3])
    assert not called and not (tmp_path/"blocked").exists()
    monkeypatch.setattr(R, "_benchmarks", lambda ctx: measurements())
    with pytest.raises(ValueError, match="logical"):
        R.execute("fit-classification", ctx, tmp_path/"invalid", step=19618, dataset="epn612", view="pool", folds=[0, 1, 2, 3])
    # Explicit folds, and only the shapes registered by the amendment: the superseded
    # shapes of DB2 pre/pool, DB3 pre and DB6 pool are no longer accepted.
    for dataset, view, folds in [("epn612", "pre", None), ("epn612", "pre", [0]), ("ninapro_db2", "post", None),
                                 ("ninapro_db2", "post", [0, 1, 2, 3]), ("ninapro_db2", "post", [0, 2]),
                                 ("ninapro_db2", "pre", [0, 1]), ("ninapro_db2", "pool", [2, 3]),
                                 ("ninapro_db3", "pre", [0, 1, 2, 3]), ("ninapro_db3", "pre", [True]),
                                 ("ninapro_db6", "pool", [0, 1, 2, 3]), ("ninapro_db6", "pre", [0, 1])]:
        with pytest.raises(ValueError, match="registered"):
            R.execute("fit-classification", ctx, tmp_path/"invalid", step=39237, dataset=dataset, view=view, folds=folds)
    # A walltime table that is not the one implied by the receipts stops every fit.
    key = R.shard_key(39237, "post", "epn612", [0, 1, 2, 3])
    ctx["plan"]["amendment"]["fit_shards"][key]["walltime_seconds"] += 900
    with pytest.raises(ValueError, match="frozen amendment walltime table differs"):
        R.execute("fit-classification", ctx, tmp_path/"table", step=39237, dataset="epn612", view="post", folds=[0, 1, 2, 3])
    assert not called and not any((tmp_path/name).exists() for name in ("invalid", "table"))


@pytest.mark.parametrize("extra_openmp", [False, True])
@pytest.mark.parametrize("view,folds,fits,rows", [("post", [0, 1], 14, 24), ("pre", [2], 7, 12), ("pool", [3], 7, 12)])
def test_fit_receipt_covers_real_probe_oof_and_durable_audit(tmp_path, monkeypatch, extra_openmp, view, folds, fits, rows):
    ctx = context(tmp_path)
    numeric = F.torch_numeric_context() if extra_openmp else F.historical_numeric_context()
    pin, _, _, _ = F.cache(tmp_path)
    monkeypatch.setattr(R, "_threads", lambda: numeric)
    monkeypatch.setattr(R, "_historical", lambda ctx: ({}, {}))
    monkeypatch.setattr(R, "_historical_numeric", lambda ctx, historical: historical_numeric())
    monkeypatch.setattr(R, "_benchmarks", lambda ctx: measurements(numeric))
    monkeypatch.setattr(R, "_cache", lambda *args: ({"receipt": {"path": "/extraction.json"}}, pin, {}, {}))
    monkeypatch.setattr(R.C, "finish_output", finish)
    ctx["plan"]["benchmark_receipts"] = {"classification": {"path": "/benchmarkmanifest.json"}}
    out = tmp_path/"fit"
    monkeypatch.setattr(M.TP, "numeric_context", lambda: numeric)
    receipt = R.execute("fit-classification", ctx, out, step=39237, dataset="ninapro_db2", view=view, folds=folds)
    assert set(receipt["artifacts"]) == {"probe", "oof", "audit"}
    report, oof = M._read_pinned_report(receipt["artifacts"]["probe"])
    assert report["fold_indices"] == receipt["fold_indices"] == folds and len(report["fit_audit"]) == fits
    assert len(oof["row"]) == rows and receipt["global_resource_gate_passed"]
    assert receipt["historical_numeric_match"]["compatible"] is True
    assert receipt["historical_numeric_match"]["exact_match"] is (not extra_openmp)
    assert report["numeric_sha256"] == M.jhash(numeric)
    assert receipt["historical_numeric_match"]["historical_numeric_sha256"] == M.jhash(F.historical_numeric_context())
    # New outputs carry the amended source, the amendment record and their own frozen walltime.
    frozen = ctx["plan"]["amendment"]["fit_shards"][R.shard_key(39237, view, "ninapro_db2", folds)]
    assert receipt["source_sha256"] == ctx["source_sha256"] != ctx["inputs"]["source_sha256"]
    assert receipt["amendment"] == R.C.A.record()
    assert receipt["resource_gate"]["folds"] == folds and receipt["resource_gate"]["passed"] is True
    assert receipt["resource_gate"]["walltime_seconds"] == frozen["walltime_seconds"]


def test_receipt_requires_same_reviewed_recipe_and_source(tmp_path):
    ctx = context(tmp_path)
    value = {"protocol": R.C.PROTOCOL, "kind": "fit-classification", "complete": True,
             "source_sha256": ctx["source_sha256"], "recipe_sha256": ctx["plan"]["recipe_sha256"]}
    R._receipt(ctx, value, "fit-classification")
    value["recipe_sha256"] = "bad"
    with pytest.raises(ValueError, match="stage receipt"):
        R._receipt(ctx, value, "fit-classification")


def benchmark_receipts(ctx, tmp_path):
    values, pins = measurements(), {}
    for ds in M.DATASETS:
        pins[ds] = {}
        for view in M.VIEWS:
            report = {**values[ds][view], "protocol": R.C.PROTOCOL, "kind": "benchmark-classification",
                "complete": True, "source_sha256": ctx["inputs"]["source_sha256"],
                "recipe_sha256": ctx["plan"]["recipe_sha256"], "policy": R.BENCHMARK_POLICY,
                **ctx["inputs"]["phases"]["benchmark"]}
            path = tmp_path/f"{ds}-{view}.json"
            R.C.write_json(path, report)
            pins[ds][view] = admit(ctx, R.C.file_pin(path), "benchmark-classification")
    return pins


def pinned_manifest(ctx, tmp_path, pins, name):
    R.C.write_json(tmp_path/name, {"reports": pins})
    ctx["plan"]["benchmark_receipts"] = {"classification": R.C.file_pin(tmp_path/name)}


def test_benchmark_manifest_reads_only_listed_original_receipts_with_the_original_policy(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    monkeypatch.setattr(R, "_historical", lambda ctx: ({}, {}))
    monkeypatch.setattr(R, "_historical_numeric", lambda ctx, historical: historical_numeric())
    monkeypatch.setattr(R, "_cache", lambda ctx, step, ds: ({}, {
        "sha256": hashlib.sha256(ds.encode()).hexdigest(), "checkpoint_sha256": "5"*64}, {}, {}))
    pins = benchmark_receipts(ctx, tmp_path)
    pinned_manifest(ctx, tmp_path, pins, "benchmarks.json")
    # The same receipts: rejected matrix under the original packaging is not consulted here,
    # the amended gate is recomputed on them and must equal the frozen table.
    gate = R.amended_gate(R._benchmarks(ctx), ctx["plan"]["amendment"]["fit_shards"])
    assert gate["all_passed"] and gate["physical_fit_jobs"] == 69 and gate["new_fit_cells"] == 40
    original = json.loads(Path(pins["epn612"]["post"]["path"]).read_text())

    def replaced(name, listed=True, **changes):
        path = tmp_path/name
        R.C.write_json(path, {**original, **changes})
        pin = R.C.file_pin(path)
        revised = copy.deepcopy(pins)
        revised["epn612"]["post"] = admit(ctx, pin, "benchmark-classification") if listed else {
            k: pin[k] for k in ("path", "sha256", "bytes")}
        pinned_manifest(ctx, tmp_path, revised, "manifest-"+name)

    replaced("unlisted.json", listed=False)
    with pytest.raises(ValueError, match="not a listed amendment input"):
        R._benchmarks(ctx)
    for name, changes in (("recipe.json", {"recipe_sha256": "0"*64}), ("new-source.json", {"source_sha256": ctx["source_sha256"]}),
                          ("other-source.json", {"source_sha256": "9"*64}), ("code.json", {"code_sha": "9"*40}),
                          ("plan.json", {"plan_sha256": "9"*64}), ("kind.json", {"kind": "fit-classification"})):
        replaced(name, **changes)
        with pytest.raises(ValueError, match="amendment input receipt provenance differs"):
            R._benchmarks(ctx)
    # The receipt keeps the policy it was measured under; the amended admission is a separate object.
    replaced("policy.json", policy={**R.BENCHMARK_POLICY, "physical_fit_jobs": 69})
    with pytest.raises(ValueError, match="benchmark receipt policy differs"):
        R._benchmarks(ctx)
    assert R.BENCHMARK_POLICY["physical_fit_jobs"] == 48 and R.C.A.FIT_POLICY["physical_fit_jobs"] == 69


def test_amended_gate_keeps_original_projection_and_only_repackages_folds():
    reports = measurements()
    original, gate = R.resource_gate(reports), R.amended_gate(reports)
    assert gate["original_gate"] == {"all_passed": True, "physical_fit_jobs": 48, "rejected_shards": []}
    assert gate["new_fit_cells"] == 40 and gate["physical_fit_jobs"] == 69 and gate["all_passed"]
    assert gate["numeric_sha256"] == original["numeric_sha256"] and gate["policy"] == R.C.A.FIT_POLICY
    covered = {}
    for key, shard in gate["shards"].items():
        cell = original["cells"][f"{shard['step']}/{shard['view']}/{shard['dataset']}"]
        assert shard["folds"] in R.C.A.fit_shapes(shard["dataset"], shard["view"])
        # Historical 20 s per fold dominates: 90 s per fold after 3 x 1.5, plus 600 s and 2 x (load + scale).
        assert shard["projected_seconds"] == 90.*len(shard["folds"]) + 604.
        assert shard["projected_seconds"] == sum(cell["projected_fit_seconds_by_fold"][f] for f in shard["folds"]) + 604.
        assert shard["walltime_seconds"] == 1800 and shard["max_projected_seconds"] == 1200.
        assert shard["projected_peak_rss_bytes"] == cell["projected_peak_rss_bytes"] == 3*1024**3
        if key in original["shards"]:
            assert shard["projected_seconds"] == original["shards"][key]["projected_seconds"]
        covered.setdefault((shard["step"], shard["view"], shard["dataset"]), []).extend(shard["folds"])
    assert len(covered) == 40 and all(sorted(v) == [0, 1, 2, 3] for v in covered.values())
    assert gate["allocated_core_hours"] == 69*8*1800/3600
    # Time over the four-hour limit or memory over the original limit blocks the whole matrix.
    slow = measurements()
    slow["ninapro_db3"]["pre"]["historical_timing"]["folds"][0]["fit_seconds"] = 10000.
    blocked = R.amended_gate(slow)
    key = R.shard_key(39237, "pre", "ninapro_db3", [0])
    assert not blocked["all_passed"] and not blocked["shards"][key]["passed"]
    assert blocked["shards"][key]["walltime_seconds"] > R.C.A.FIT_POLICY["max_walltime_seconds"]
    assert blocked["shards"][R.shard_key(39237, "pre", "ninapro_db3", [1])]["passed"]
    large = measurements()
    large["uci_emg"]["pool"]["peak_rss_bytes"] = 10*1024**3
    assert not R.amended_gate(large)["all_passed"]
    # The frozen table must be the one implied by the receipts: walltime, folds and projection.
    frozen = table()
    R.amended_gate(reports, frozen)
    for field, value in (("walltime_seconds", 2700), ("folds", [0, 1]), ("projected_seconds", 700.)):
        changed = copy.deepcopy(frozen)
        changed[R.shard_key(49046, "pool", "ninapro_db2", [0])][field] = value
        with pytest.raises(ValueError, match="frozen amendment walltime table differs"):
            R.amended_gate(reports, changed)
    del changed[R.shard_key(49046, "pool", "ninapro_db2", [1])]
    with pytest.raises(ValueError, match="frozen amendment walltime table differs"):
        R.amended_gate(reports, changed)


def fit_receipts(ctx, tmp_path):
    """One synthetic stage receipt and probe for each of the sixty-nine registered shards."""
    numeric, pins = F.historical_numeric_context(), {}
    for key, shard in ctx["plan"]["amendment"]["fit_shards"].items():
        folder = tmp_path/"fits"/key.replace("/", "_")
        folder.mkdir(parents=True)
        (folder/"probe.oof.npz").write_bytes(b"oof "+key.encode())
        (folder/"probe.audit.jsonl").write_bytes(b"audit "+key.encode())
        oof, audit = R.C.file_pin(folder/"probe.oof.npz"), R.C.file_pin(folder/"probe.audit.jsonl")
        probe = {"dataset": shard["dataset"], "view": shard["view"], "step": shard["step"],
                 "fold_indices": shard["folds"], "checkpoint_sha256": "a"*64, "code_sha": ctx["code_sha"],
                 "plan_sha256": ctx["plan_pin"]["sha256"], "oof": oof["path"], "oof_sha256": oof["sha256"],
                 "audit": audit["path"], "audit_sha256": audit["sha256"],
                 "numeric_context": numeric, "numeric_sha256": M.jhash(numeric)}
        R.C.write_json(folder/"probe.json", probe)
        receipt = {"protocol": R.C.PROTOCOL, "kind": "fit-classification", "complete": True,
                   "source_sha256": ctx["source_sha256"], "recipe_sha256": ctx["plan"]["recipe_sha256"],
                   "amendment": ctx["amendment"], "code_sha": ctx["code_sha"], "plan_sha256": ctx["plan_pin"]["sha256"],
                   "artifacts": {"probe": R.C.file_pin(folder/"probe.json"), "oof": oof, "audit": audit},
                   "historical_numeric_match": R._numeric_match(numeric, historical_numeric()),
                   "checkpoint_step": shard["step"], "view": shard["view"], "dataset": shard["dataset"],
                   "checkpoint_sha256": "a"*64, "fold_indices": shard["folds"],
                   "benchmark_manifest": ctx["plan"]["benchmark_receipts"]["classification"],
                   "resource_gate": {"folds": shard["folds"], "passed": True}, "global_resource_gate_passed": True}
        R.C.write_json(folder/"report.json", receipt)
        pins[key] = R.C.file_pin(folder/"report.json")
    return pins


def test_compare_requires_the_sixty_nine_registered_receipts_of_the_amended_source(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    ctx["plan"]["benchmark_receipts"] = {"classification": {"path": "/benchmarkmanifest.json"}}
    monkeypatch.setattr(R, "_threads", lambda: F.historical_numeric_context())
    monkeypatch.setattr(R, "_historical", lambda ctx: ({}, {}))
    monkeypatch.setattr(R, "_historical_numeric", lambda ctx, historical: historical_numeric())
    monkeypatch.setattr(R, "_historical_reports", lambda ctx, task: [])
    monkeypatch.setattr(R, "_cache", lambda ctx, step, ds: ({}, {"dataset": ds, "step": step}, {}, {}))
    monkeypatch.setattr(R.C, "finish_output", finish)
    calls = []
    def compare(probes, *, out, **kwargs):
        calls.append(list(probes))
        R.C.write_json(out, {"complete": True, "kind": "comparison"})
        return {"numeric_comparison": {"compatible": True}, "reading": {"primary": "unchanged"}}
    monkeypatch.setattr(M, "compare_reports", compare)
    pins = fit_receipts(ctx, tmp_path)

    def run(name, reports):
        R.C.write_json(tmp_path/(name+".json"), {"reports": reports})
        ctx["plan"]["classification_fit_manifest"] = R.C.file_pin(tmp_path/(name+".json"))
        return R.execute("compare-classification", ctx, tmp_path/name)

    def rewritten(key, name, **changes):
        report = {**json.loads(Path(pins[key]["path"]).read_text()), **changes}
        R.C.write_json(tmp_path/(name+"-receipt.json"), report)
        return {**pins, key: R.C.file_pin(tmp_path/(name+"-receipt.json"))}

    result = run("complete", list(pins.values()))
    assert len(calls) == 1 and len(calls[0]) == 69 and result["kind"] == "compare-classification"
    assert result["amendment"] == ctx["amendment"] and result["source_sha256"] == ctx["source_sha256"]
    first, second = R.shard_key(39237, "pre", "ninapro_db2", [0]), R.shard_key(39237, "pre", "ninapro_db2", [1])
    with pytest.raises(ValueError, match="sixty-nine"):
        run("missing", [pin for key, pin in pins.items() if key != first])
    with pytest.raises(ValueError, match="sixty-nine"):
        run("duplicate", [pins[second] if key == first else pin for key, pin in pins.items()])
    with pytest.raises(ValueError, match="sixty-nine"):
        run("extra", list(pins.values())+[pins[first]])
    # A receipt of the original source, without the amendment record or of a superseded shape is not a new fit.
    for name, changes, message in (
            ("old-source", {"source_sha256": ctx["inputs"]["source_sha256"]}, "stage receipt differs"),
            ("no-amendment", {"amendment": None}, "not written under amendment A1"),
            ("other-list", {"amendment": {**ctx["amendment"], "input_source_sha256": "9"*64}}, "not written under amendment A1"),
            ("failed-gate", {"resource_gate": {"folds": [0], "passed": False}}, "probe/stage receipt identity differs")):
        with pytest.raises(ValueError, match=message):
            run(name, list(rewritten(first, name, **changes).values()))
    probe = json.loads(Path(json.loads(Path(pins[first]["path"]).read_text())["artifacts"]["probe"]["path"]).read_text())
    R.C.write_json(tmp_path/"pair-probe.json", {**probe, "fold_indices": [0, 1]})
    receipt = json.loads(Path(pins[first]["path"]).read_text())
    superseded = rewritten(first, "pair", fold_indices=[0, 1], resource_gate={"folds": [0, 1], "passed": True},
                           artifacts={**receipt["artifacts"], "probe": R.C.file_pin(tmp_path/"pair-probe.json")})
    with pytest.raises(ValueError, match="unregistered fit receipt cell/shard"):
        run("superseded-shape", list(superseded.values()))
    assert len(calls) == 1


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
