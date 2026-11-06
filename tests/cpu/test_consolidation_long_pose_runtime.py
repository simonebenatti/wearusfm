"""Runtime stage, artifact provenance and train-only admission contracts."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import consolidation_long_pose_runtime as U
from test_consolidation_long_pose import data


def json_pin(path, value):
    U.C.write_json(path, value)
    return U.C.file_pin(path)


def context(tmp_path):
    plan = {"recipe_sha256": "a"*64, "endpoints": {}}
    caches = {"recipe_sha256": "a"*64, "source_sha256": "b"*64, "pose": {}}
    for step in U.P.DOSES:
        plan["endpoints"][str(step)] = {"checkpoint": {"path": str(tmp_path/f"checkpoint{step}.pt"), "sha256": str(U.P.DOSES.index(step)+1)*64}}
        caches["pose"][str(step)] = {}
        for split in ("train", "val"):
            path = tmp_path/f"{step}.{split}.npz"
            path.write_bytes(f"cache-{step}-{split}".encode())
            cache = U.C.file_pin(path)
            report = {"protocol": U.C.PROTOCOL, "complete": True, "kind": "extract-pose-"+split,
                      "which": "teacher", "checkpoint_unchanged": True, "frozen_teacher_unchanged": True,
                      "source_sha256": "b"*64, "recipe_sha256": "a"*64, "checkpoint_step": step,
                      "checkpoint_sha256": plan["endpoints"][str(step)]["checkpoint"]["sha256"], "artifacts": {"cache": cache}}
            caches["pose"][str(step)][split] = {"cache": cache, "receipt": json_pin(tmp_path/f"{step}.{split}.json", report)}
    plan["cache_manifest"] = json_pin(tmp_path/"caches.json", caches)
    return {"plan": plan, "caches": caches, "source_sha256": "b"*64, "code_sha": "c"*40,
            "pose_reader": {"fold_users": [["a", "c"], ["b", "d"]]}}


def timing():
    return {"read_validate": .01, "fixed_alpha_fit": .01, "predict": .01, "summary": .01,
            "E2_pairing": .001, "bootstrap_diagnostic": .01, "artifact_write": .01,
            "ridge_by_depth": {depth: {"statistics_seconds": .1, "three_eigensystems_seconds": .1} for depth in U.P.DEPTHS}}


def benchmark(ctx):
    values = timing()
    artifact = ctx["caches"]["pose"]["19618"]["train"]["cache"]
    return {"complete": True, "protocol": U.C.PROTOCOL, "kind": "benchmark-pose", "checkpoint_step": 19618,
            "checkpoint_sha256": ctx["plan"]["endpoints"]["19618"]["checkpoint"]["sha256"],
            "source_sha256": ctx["source_sha256"], "recipe_sha256": ctx["plan"]["recipe_sha256"],
            "benchmark_policy": U.BENCHMARK_POLICY, "fit_resources": U.FIT_RESOURCES,
            "cache_manifest": ctx["plan"]["cache_manifest"], "train_cache": ctx["caches"]["pose"]["19618"]["train"],
            "held_payload_opened": False, "held_scored": False, "training_scores_persisted": False,
            "timing_artifacts_only": True,
            "artifacts": {k: artifact for k in ("pre_coefficients", "post_coefficients", "pre_predictions",
                                               "post_predictions", "trained_head_predictions", "report_io_surrogate")},
            "environment": U.environment(), "timings_seconds": values, "peak_rss_bytes": 100_000,
            "projection": U.project_benchmark(values, 100_000)}


def test_projection_exact_policy_and_not_admitted_for_time_or_memory():
    t = timing()
    r = 18548/1907
    expected_floor = 2*2*(.1+27*.1)
    expected_other = 2*(.01*(1+r)+.01+.01*r+.01*r+.001*r*r+.01*r+.01*r)+120
    result = U.project_benchmark(t, 100_000)
    assert result["seconds_per_dose"] == pytest.approx(expected_floor+expected_other)
    assert result["projected_peak_rss_bytes"] == pytest.approx(100_000*r)
    assert result["admitted"] is True
    t["read_validate"] = 1000
    assert U.project_benchmark(t, 100_000)["admitted"] is False
    assert U.project_benchmark(timing(), 10*1024**3)["admitted"] is False
    t["read_validate"] = float("nan")
    with pytest.raises(ValueError, match="measurements"):
        U.project_benchmark(t, 100_000)


@pytest.mark.parametrize("field", ["source_sha256", "recipe_sha256", "checkpoint_step", "cache_manifest", "train_cache",
                                     "benchmark_policy", "fit_resources", "held_payload_opened", "held_scored", "environment", "projection"])
def test_fit_gate_rejects_changed_benchmark_contract(tmp_path, field):
    ctx = context(tmp_path)
    report = benchmark(ctx)
    report[field] = "changed"
    ctx["plan"]["benchmark_receipts"] = {"pose": json_pin(tmp_path/"benchmark.json", report)}
    with pytest.raises(ValueError):
        U._benchmark_gate(ctx)


def test_fit_gate_recomputes_admission_and_checks_file_pin(tmp_path):
    ctx = context(tmp_path)
    report = benchmark(ctx)
    pin = json_pin(tmp_path/"benchmark.json", report)
    ctx["plan"]["benchmark_receipts"] = {"pose": pin}
    assert U._benchmark_gate(ctx) == pin
    report["timings_seconds"]["read_validate"] = 10_000
    report["projection"] = U.project_benchmark(report["timings_seconds"], report["peak_rss_bytes"])
    ctx["plan"]["benchmark_receipts"]["pose"] = json_pin(tmp_path/"large.json", report)
    with pytest.raises(ValueError, match="admission"):
        U._benchmark_gate(ctx)
    Path(pin["path"]).write_text("tampered")
    ctx["plan"]["benchmark_receipts"]["pose"] = pin
    with pytest.raises(ValueError, match="SHA"):
        U._benchmark_gate(ctx)


def test_benchmark_never_opens_held_payload_or_persists_training_scores(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    train = data()
    monkeypatch.setitem(U.BENCHMARK_POLICY, "train_windows", len(train["meta"]))
    reads = []
    def read(c, step, split):
        assert step == 19618 and split == "train"
        reads.append(split)
        return train, c["caches"]["pose"][str(step)][split]
    monkeypatch.setattr(U, "_read", read)
    monkeypatch.setattr(U, "peak_rss_bytes", lambda: 100_000)
    result = U._benchmark(ctx, tmp_path)
    assert reads == ["train"]
    assert result["held_payload_opened"] is result["held_scored"] is result["training_scores_persisted"] is False
    assert set(result["timings_seconds"]["ridge_by_depth"]) == {"pre", "post"}
    assert "analysis" not in json.dumps(result)
    assert result["timing_artifacts_only"] and len(result["artifacts"]) == 6
    assert result["timings_seconds"]["artifact_write"] > 0
    assert result["benchmark_policy"]["reference_step"] == 19618


def test_fit_finishes_both_train_selections_before_held_and_never_fits_head(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    train, held = data(), data(seed=90, users=("held_a", "held_b"))
    events = []
    def read(c, step, split):
        events.append("read:"+split)
        if split == "val":
            assert events == ["read:train", "fit:pre", "fit:post", "read:val"]
        return (train if split == "train" else held), ctx["caches"]["pose"][str(step)][split]
    original = U.P.fit_reader
    def fit(d, depth, folds):
        assert d is train
        events.append("fit:"+depth)
        return original(d, depth, folds)
    monkeypatch.setattr(U, "_read", read)
    monkeypatch.setattr(U.P, "fit_reader", fit)
    out = tmp_path/"fit"
    out.mkdir()
    result = U._fit(ctx, out, 39237, {"path": "benchmark"})
    assert set(result["cells"]) == set(U.P.READERS)
    assert result["held_payload_opened_after_both_ridge_fits"] is True
    assert result["cells"]["trained_head"]["fit"]["supervised_updates"] == 0
    pin = result["cells"]["trained_head"]["artifacts"]["prediction"]
    with np.load(pin["path"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["prediction"], held["pred"][:, 6:-6])
    assert "full-support E2" in result["support_policy"]


def test_extract_adapter_pins_sources_checkpoint_and_original_loader_before_publishing(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    d = data()
    d["config"] = {"pose_stats_sha256": "d"*64, "per_side": False,
                   "n_windows": 10, "window_s": 4., "seed": U.P.PE.SEED_WINDOWS}
    checkpoint = tmp_path/"checkpoint19618.pt"
    checkpoint.write_bytes(b"unchanged checkpoint")
    ctx["plan"]["endpoints"]["19618"]["checkpoint"] = U.C.file_pin(checkpoint)
    ctx["plan"]["extraction_batch"] = {"pose": 32}
    stats = tmp_path/"pose_stats.json"
    stats.write_text("{}")
    ctx["legacy"] = {"inputs": {"pose_stats": U.C.file_pin(stats)}, "paths": {"pose_root": str(tmp_path)}}
    selection = {"n_recordings": d["n_recordings"], "n_windows": len(d["meta"]), "n_users": 4}
    records = [{"recording": i} for i in range(d["n_recordings"])]
    sources = {str(stats): U.C.T.P.SourceGuard.stat(stats)}
    loader = U.C.canonical(asdict(U.C.T.R.sanity_config().loader))
    loader["pose_root"] = str(tmp_path)
    guard = {"checkpoint": str(checkpoint), "checkpoint_step": 19618,
             "checkpoint_sha256": ctx["plan"]["endpoints"]["19618"]["checkpoint"]["sha256"],
             "checkpoint_pose_stats_sha256": "d"*64, "config": {"loader": loader}}
    monkeypatch.setattr(U, "_reference", lambda c, split: (d, selection))
    monkeypatch.setattr(U.C.E, "pose_sources", lambda legacy, split: (records, sources))
    monkeypatch.setattr(U.C, "load_teacher", lambda c, step, device: ("teacher", guard))
    monkeypatch.setattr(U.C.E.PT, "PoseStore", lambda path: "store")
    calls = []
    def extract(model, recs, store, ev, reference, selection):
        assert model == "teacher" and recs == records and store == "store"
        assert ev.device == "cuda" and ev.batch == 32 and ev.n_windows == 10 and ev.window_s == 4.
        assert reference is d and selection["n_windows"] == len(d["meta"])
        assert U.C.canonical(asdict(ev.loader)) == U.C.canonical(asdict(U.C.T.R.sanity_config().loader))
        calls.append("original adapter")
        return copy.deepcopy(d)
    monkeypatch.setattr(U.P, "extract_pose", extract)
    monkeypatch.setattr(U.torch.cuda, "empty_cache", lambda: None)
    out = tmp_path/"extraction"
    out.mkdir()
    result = U._extract(ctx, out, "extract-pose-train", 19618)
    assert calls == ["original adapter"] and result["checkpoint_unchanged"] is True
    assert result["raw_inputs"]["unchanged"] is True
    cached = U.P.read_cache(result["artifacts"]["cache"]["path"])
    assert cached["config"]["checkpoint_step"] == 19618
    assert cached["config"]["checkpoint_sha256"] == guard["checkpoint_sha256"]


def test_fit_leakage_is_fatal(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    train = data()
    monkeypatch.setattr(U, "_read", lambda c, s, split: (train, ctx["caches"]["pose"][str(s)][split]))
    out = tmp_path/"fit"
    out.mkdir()
    with pytest.raises(ValueError, match="leakage"):
        U._fit(ctx, out, 19618, {})


def test_compare_validates_all_extraction_receipts_and_cell_artifacts_before_aggregate(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    ctx["plan"]["benchmark_receipts"] = {"pose": json_pin(tmp_path/"benchmark.json", benchmark(ctx))}
    train, held = data(), data(seed=90, users=("held_a", "held_b"))
    monkeypatch.setattr(U, "_read", lambda c, s, split: (train if split == "train" else held, ctx["caches"]["pose"][str(s)][split]))
    fits = {}
    for step in U.P.DOSES:
        out = tmp_path/str(step)
        out.mkdir()
        fit = U._fit(ctx, out, step, ctx["plan"]["benchmark_receipts"]["pose"])
        fit.update(complete=True, protocol=U.C.PROTOCOL, kind="fit-pose", source_sha256=ctx["source_sha256"], recipe_sha256=ctx["plan"]["recipe_sha256"])
        fits[str(step)] = {"report": json_pin(out/"report.json", fit)}
    ctx["plan"]["pose_fit_manifest"] = json_pin(tmp_path/"fits.json", {"fits": fits})
    calls = []
    monkeypatch.setattr(U.P, "aggregate", lambda cells: calls.append(cells) or {"complete": True})
    result = U._compare(ctx)
    assert len(calls) == 1 and len(calls[0]) == 9 and result["comparison"]["complete"]
    # A validly re-pinned report cannot waive the no-held-before-fit flag.
    pin = fits["39237"]["report"]
    report = U.C.read_pin(pin)
    report["held_payload_opened_after_both_ridge_fits"] = False
    fits["39237"]["report"] = json_pin(tmp_path/"changed-fit.json", report)
    ctx["plan"]["pose_fit_manifest"] = json_pin(tmp_path/"changed-fits.json", {"fits": fits})
    with pytest.raises(ValueError, match="control provenance"):
        U._compare(ctx)
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["checkpoint_sha256", "split", "recipe_sha256", "source_sha256", "dimension", "geometry"])
def test_read_rejects_embedded_cache_identity_or_geometry(tmp_path, monkeypatch, mutation):
    ctx = context(tmp_path)
    ctx["legacy"] = {"inputs": {"pose_stats": {"sha256": "d"*64}}}
    d = data()
    d["dim"] = 384
    d["target"] = np.zeros((len(d["meta"]), 160, 20), np.float32)
    d["config"] = {"which": "teacher", "split": "train", "checkpoint_step": 19618,
                   "checkpoint": ctx["plan"]["endpoints"]["19618"]["checkpoint"]["path"],
                   "checkpoint_sha256": ctx["plan"]["endpoints"]["19618"]["checkpoint"]["sha256"],
                   "source_sha256": ctx["source_sha256"], "recipe_sha256": ctx["plan"]["recipe_sha256"],
                   "pose_stats_sha256": "d"*64, "checkpoint_pose_stats_sha256": "d"*64,
                   "n_windows": 10, "window_s": 4., "seed": U.P.PE.SEED_WINDOWS, "per_side": False}
    monkeypatch.setattr(U, "_reference", lambda ctx, split: ({}, {}))
    monkeypatch.setattr(U.P, "read_cache", lambda *a: d)
    assert U._read(ctx, 19618, "train")[0] is d
    if mutation == "dimension":
        d["dim"] = 4
    elif mutation == "geometry":
        d["target"] = d["target"][:, :159]
    else:
        d["config"][mutation] = "changed"
    with pytest.raises(ValueError, match="identity|geometry"):
        U._read(ctx, 19618, "train")


def test_cache_receipt_rejects_source_recipe_dose_and_uncovered_cache(tmp_path):
    ctx = context(tmp_path)
    assert U._entry(ctx, 19618, "train") == ctx["caches"]["pose"]["19618"]["train"]
    for field, value in (("source_sha256", "0"*64), ("recipe_sha256", "0"*64), ("checkpoint_step", 49046), ("artifacts", {})):
        changed = copy.deepcopy(ctx)
        entry = changed["caches"]["pose"]["19618"]["train"]
        report = U.C.read_pin(entry["receipt"])
        report[field] = value
        entry["receipt"] = json_pin(tmp_path/(field+".json"), report)
        changed["plan"]["cache_manifest"] = json_pin(tmp_path/(field+"-manifest.json"), changed["caches"])
        with pytest.raises(ValueError):
            U._entry(changed, 19618, "train")


@pytest.mark.parametrize("stage,step", [("unknown", None), ("fit-pose", None), ("fit-pose", 30000),
                                      ("extract-pose-train", True), ("benchmark-pose", 19618), ("compare-pose", 49046)])
def test_invalid_stage_or_dose_fails_before_creating_output(tmp_path, monkeypatch, stage, step):
    monkeypatch.setattr(U.C, "begin_output", lambda *a, **k: pytest.fail("output created before stage gate"))
    with pytest.raises(ValueError):
        U.execute(stage, {}, tmp_path/"output", step)


def test_dispatch_uses_shared_begin_finish_and_gates_fit_first(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(U, "_threads", lambda stage: events.append("threads:"+stage) or {"torch_threads": 8})
    monkeypatch.setattr(U, "_benchmark_gate", lambda ctx: events.append("gate") or {"pin": True})
    monkeypatch.setattr(U.C, "begin_output", lambda ctx, out, stage: events.append("begin") or {"kind": stage})
    monkeypatch.setattr(U, "_fit", lambda ctx, out, step, gate: events.append("fit") or {"step": step, "gate": gate})
    monkeypatch.setattr(U.C, "finish_output", lambda ctx, out, report: events.append("finish") or report)
    result = U.execute("fit-pose", {}, tmp_path/"out", 49046)
    assert events == ["threads:fit-pose", "gate", "begin", "fit", "finish"]
    assert result["step"] == 49046 and result["gate"] == {"pin": True}
    assert result["execution_environment"] == {"torch_threads": 8}


@pytest.mark.parametrize("stage,expected", [("extract-pose-train", 1), ("extract-pose-val", 1),
                                           ("benchmark-pose", 8), ("fit-pose", 8), ("compare-pose", 8)])
def test_pose_threads_match_historical_stage_and_actual_libraries(monkeypatch, stage, expected):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, str(expected))
    numeric = {"torch_threads": expected, "threadpools": [{"num_threads": expected}]}
    monkeypatch.setattr(U, "environment", lambda: numeric)
    assert U._threads(stage) == numeric
    numeric["threadpools"][0]["num_threads"] = 3
    with pytest.raises(ValueError, match="active pose numeric"):
        U._threads(stage)
    numeric["threadpools"][0]["num_threads"] = expected
    numeric["torch_threads"] = 3
    with pytest.raises(ValueError, match="active pose numeric"):
        U._threads(stage)
    numeric["torch_threads"] = expected
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    with pytest.raises(ValueError, match="requires"):
        U._threads(stage)


@pytest.mark.parametrize("stage,step", [("benchmark-pose", None), ("fit-pose", 49046)])
def test_wrong_pose_threads_fail_before_benchmark_gate_payload_or_output(monkeypatch, tmp_path, stage, step):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(U, "_benchmark_gate", lambda *a: pytest.fail("gate read before thread policy"))
    monkeypatch.setattr(U, "_read", lambda *a: pytest.fail("payload read before thread policy"))
    monkeypatch.setattr(U.C, "begin_output", lambda *a: pytest.fail("output created before thread policy"))
    with pytest.raises(ValueError, match="requires 8"):
        U.execute(stage, {}, tmp_path/"out", step)
