"""Final 19,618 comparison with real small cache, OOF, audit and pose artifacts.

Derived from test_scaling_compare.py by the replacements declared in scaling_derive19618.py.

Only production selection counts and git identity are replaced; fit algorithms,
file/provenance guards, dimensions and bootstrap calculations remain real.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import scaling_compare19618 as M
import test_scaling_probe19618 as TP

pin, save_json = TP.pin, TP.save_json
CODE = "e"*40


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    root = tmp_path_factory.mktemp("scaling_comparison")
    monkey = pytest.MonkeyPatch()
    monkey.setattr(M.C.E.P, "code_identity", lambda code: code if code == CODE else (_ for _ in ()).throw(ValueError("bad code")))
    selection = {"train": {"n_windows": 40, "n_users": 40, "n_recordings": 40},
                 "val": {"n_windows": 4, "n_users": 2, "n_recordings": 4}}
    monkey.setattr(M.P.Q, "SELECTION", selection)
    projection = M.SP.save(root/"projection")
    probe_root = root/"classification"
    probe_root.mkdir()
    probe = TP.plan.__wrapped__(probe_root, projection)
    probe_pin = save_json(root/"probe_plan.json", probe)
    identity = {"code_sha": CODE, "plan_sha256": probe_pin["sha256"]}
    probe_reports = []
    for cell in M.C.CELLS:
        budgets = {}
        for ds in M.C.DATASETS:
            where = root/(cell+"-"+ds+"-bench")
            M.C.benchmark(probe, cell, ds, where, identity)
            budgets[ds] = pin(where/"benchmark.json")
        for task, datasets in M.C.TASKS.items():
            shard = {"cell": cell, "task": task, "datasets": list(datasets), "folds": [0, 1, 2, 3],
                     "cpus": 1, "memory_gib": 50, "walltime_seconds": 10000}
            resource = save_json(root/(cell+task+"-budget.json"), {"protocol": M.C.RESOURCE_PROTOCOL, "complete": True,
                "plan_content_sha256": M.C.jhash(probe), "code_sha": CODE,
                "caps": {"cpus": 4, "memory_gib": 60, "walltime_seconds": 11000},
                "benchmarks": {cell: budgets}, "shards": [shard]})
            where = root/(cell+task+"-shard")
            M.C.run_shard(probe, cell, task, list(datasets), [0, 1, 2, 3], where, identity, resource)
            report = root/(cell+task+"-aggregate.json")
            M.C.aggregate(probe, cell, task, [where/"shard.json"], report, identity)
            probe_reports.append(pin(report))
    contract = M.ST.check_source_contract(probe["training_source_contract"]["path"])
    data, refs, caches = {}, {}, {a: {} for a in M.P.ARMS}
    train_users = [f"train{i:02}" for i in range(40)]
    rng = np.random.default_rng(5)
    for split, counts in selection.items():
        n = counts["n_windows"]
        users = train_users if split == "train" else ["v0", "v0", "v1", "v1"]
        target = rng.normal(size=(n, 160, 20)).astype(np.float32)
        mask = np.ones_like(target, dtype=bool)
        # Exercise finite fill checks outside supported temporal positions.
        mask[:, 140:] = False
        d = {"target": target, "mask": mask, "std": np.ones((n, 20), np.float32),
             "pose_var": np.resize(np.asarray([.1, .5], np.float32), n),
             "meta": [{"user": u, "recording": i, "start": 0, "n_patches": 140} for i, u in enumerate(users)],
             "n_recordings": n}
        data[split] = d
        ref = root/(split+"-reference.npz")
        np.savez(ref, **{k: d[k] for k in ("target", "mask", "std", "pose_var")},
                 meta_json=np.asarray(json.dumps(d["meta"])), n_recordings=n)
        refs[split] = {**pin(ref), "bytes": ref.stat().st_size, **counts}
        for arm in M.P.ARMS:
            dim = M.P.V.dimension(arm)
            cache, pp, rp = [root/(arm+split+name) for name in (".npz", ".provenance.json", ".receipt.json")]
            cfg = {"which": "teacher", "split": split, "code_sha": "c"*40,
                   "pose_stats_sha256": M.P.Q.E.STATS_SHA, "checkpoint_pose_stats_sha256": M.P.Q.E.STATS_SHA,
                   "n_windows": 10, "window_s": 4., "seed": 20261026, "per_side": False}
            lat = np.zeros((n, 160, dim), np.float16)
            np.savez(cache, **{k: d[k] for k in ("target", "mask", "std", "pose_var")}, lat=lat,
                     meta_json=np.asarray(json.dumps(d["meta"])), config_json=np.asarray(json.dumps(cfg)),
                     n_recordings=n, dim=dim, has_head=True)
            cp = probe["caches"][arm]["epn612"]["checkpoint_sha256"]
            common = {"protocol": M.P.V.PROTOCOL, "arm": arm, "which": "teacher", "checkpoint_step": 19618,
                      "checkpoint_sha256": cp, "code_sha": "c"*40,
                      "training_source_contract_sha256": probe["training_source_contract"]["sha256"]}
            if arm == M.P.ARMS[0]:  # reused tokenizer-campaign pose extraction of the control
                common = {"protocol": M.P.V.E.PROTOCOL, "arm": "T100-G", "which": "teacher", "checkpoint_step": 19618,
                          "checkpoint_sha256": cp, "code_sha": "c"*40}
            model = M.ST.R.config_to_dict(contract["config_30m" if arm == M.P.ARMS[0] else "config_300m"])["model"]
            prov = {**common, **counts, "split": split, "stats_sha256": M.P.Q.E.STATS_SHA, "cache": str(cache),
                    "cache_sha256": M.C.sha(cache), "cache_bytes": cache.stat().st_size, "config": cfg, "model_config": model}
            save_json(pp, prov)
            output = {"path": str(cache), "sha256": M.C.sha(cache), "bytes": cache.stat().st_size,
                      "provenance_path": str(pp), "provenance_sha256": M.C.sha(pp)}
            save_json(rp, {**common, "stage": "extract-pose", "complete": True, "checkpoint_unchanged": True,
                           "postload_recipe_passed": True, "outputs": {split: output}})
            caches[arm][split] = {**output, "receipt_path": str(rp), "receipt_sha256": M.C.sha(rp),
                                 "checkpoint_sha256": cp, "extraction_code_sha": "c"*40}
    pose = M.P.make_plan(caches, refs, [train_users[::2], train_users[1::2]], projection, probe["training_source_contract"])
    pose_pin = save_json(root/"pose_plan.json", pose)
    refs_checked = {s: M.P.Q.validate_reference(refs[s]) for s in refs}
    inspections = {a+"/"+s: M.P.validate_artifact(caches[a][s], a, s, pose, refs_checked[s]) for a in M.P.ARMS for s in refs}
    cells = {}
    for i, (name, (arm, view)) in enumerate(M.P.CELLS.items()):
        dim = 384 if view == "matched" else M.P.V.dimension(arm)
        w, b = np.zeros((dim, 20), np.float64), np.zeros(20, np.float64)
        cp = root/(name+"-coefficients.npz")
        np.savez(cp, weights=w, bias=b)
        val = M.P.Q.read_targets(caches[arm]["val"]["path"])
        pred = (val["target"][:, 6:-6] * (.6 + .1*i)).astype(np.float64)
        predp = root/(name+"-predictions.npy")
        np.save(predp, pred)
        selection = M.P.Q.E.choose_alpha(lambda alpha: float(np.log10(alpha)**2 + 1))
        selection["fold_users"] = pose["fold_users"]
        cells[name] = {"complete": True, "arm": arm, "view": view, "which": "teacher", "checkpoint_step": 19618,
            "checkpoint_sha256": caches[arm]["val"]["checkpoint_sha256"],
            "projection_manifest_sha256": projection["sha256"] if view == "matched" else None,
            "fit": {"n_features": dim, "n_users": 40, "selection": selection,
                    "fit_support_sha256": M.P.Q.arrays_hash(M.P.Q.read_targets(caches[arm]["train"]["path"])["central_mask"]),
                    "coefficients_sha256": M.P.Q.arrays_hash(w, b)},
            "analysis": M.P.Q.summarize(val, pred), "coefficients": pin(cp), "predictions": pin(predp)}
    numeric = M.C.T.numeric_context()
    runtime = {"code_sha": CODE, "sources": M.P.source_hashes(), "versions": {"python": numeric["python"], "numpy": numeric["numpy"]},
               "threadpools": numeric["threadpools"]}
    pose_report = {"protocol": M.P.PROTOCOL, "status": "completed", "complete": True, "step": 19618,
                   "recipe": M.P.recipe(), "plan_sha256": pose_pin["sha256"], "input_pins": caches,
                   "reference_pins": refs, "projection_pin": projection, "training_source_contract": pose["training_source_contract"],
                   "runtime": runtime, "inspections": inspections, "cells": cells, "paired": M.P.paired_comparison(cells)}
    pose_report_pin = save_json(root/"pose_report.json", pose_report)
    inputs = M.make_inputs(probe_pin, pose_pin, probe_reports, pose_report_pin)
    yield {"root": root, "inputs": inputs, "pose": pose, "pose_report": pose_report, "probe": probe}
    monkey.undo()


def test_complete_comparison_recomputes_primary_secondary_and_full_precision(scene, tmp_path):
    result = M.compare(scene["inputs"], tmp_path/"summary.json", CODE)
    assert result["complete"] and result["quality"] == {"artifact_integrity": True, "paired_support": True, "solver_converged": True}
    assert result["pose"]["delta_E_dyn"] == result["pose"]["candidate_E_dyn"] - result["pose"]["control_E_dyn"]
    paired = M.P.paired_comparison(scene["pose_report"]["cells"])
    assert result["pose"]["ci95"] == paired["dynamic"]["contrasts"][M.PRIMARY]["ci95"]
    assert result["reading"]["reading_valid"] and not result["reading"]["further_training_authorized"]
    assert result["reading"]["reading_status"] == "descriptive_endpoint_reading" and "recommendation" not in result["reading"]
    assert result["checkpoint_step"] == 19618 and result["checkpoints"][M.C.ARMS[0]] == M.C.E.CONTROL_CHECKPOINT_SHA256
    assert result["checkpoints"][M.C.ARMS[0]] == scene["probe"]["caches"][M.C.ARMS[0]]["epn612"]["checkpoint_sha256"]


@pytest.mark.parametrize("mutation", ["dose", "code", "runtime", "checkpoint", "projection", "support", "paired", "coeff_nan", "pred_nan_fill", "cv", "incomplete"])
def test_pose_corruption_is_rejected_even_with_new_report_hash(scene, tmp_path, mutation):
    report = copy.deepcopy(scene["pose_report"])
    inputs = copy.deepcopy(scene["inputs"])
    cell = report["cells"][M.CANDIDATE]
    if mutation == "dose": report["step"] = 5000
    if mutation == "code": report["runtime"]["code_sha"] = "a"*40
    if mutation == "runtime": report["runtime"]["versions"]["numpy"] = "different"
    if mutation == "checkpoint": cell["checkpoint_sha256"] = "a"*64
    if mutation == "projection": report["projection_pin"]["sha256"] = "a"*64
    if mutation == "support": cell["analysis"]["support_sha256"] = "a"*64
    if mutation == "paired": report["paired"]["dynamic"]["contrasts"][M.PRIMARY]["delta_ratio"] = .123
    if mutation == "incomplete": report["complete"] = False
    if mutation == "cv": cell["fit"]["selection"]["alpha"] = 100
    if mutation == "coeff_nan":
        with np.load(cell["coefficients"]["path"], allow_pickle=False) as z: w, b = z["weights"], z["bias"]
        w[0, 0] = np.nan
        path = tmp_path/"coefficients.npz"
        np.savez(path, weights=w, bias=b)
        cell["coefficients"] = pin(path)
        cell["fit"]["coefficients_sha256"] = M.P.Q.arrays_hash(w, b)
    if mutation == "pred_nan_fill":
        pred = np.load(cell["predictions"]["path"], allow_pickle=False)
        pred[:, -1, :] = np.nan
        path = tmp_path/"predictions.npy"
        np.save(path, pred)
        cell["predictions"] = pin(path)
    inputs["pose_report"] = save_json(tmp_path/"pose_report.json", report)
    with pytest.raises(ValueError): M.compare(inputs, tmp_path/"bad.json", CODE)
    assert not (tmp_path/"bad.json").exists()


@pytest.mark.parametrize("mutation", ["source", "missing_report", "mixed_plan", "mixed_checkpoint", "mixed_projection", "wrong_probe_code"])
def test_mixed_provenance_or_incomplete_grid_rejected(scene, tmp_path, mutation):
    inputs = copy.deepcopy(scene["inputs"])
    if mutation == "source": inputs["sources"]["scripts/scaling_compare19618.py"] = "a"*64
    if mutation == "missing_report": inputs["probe_reports"].pop()
    if mutation in ("mixed_plan", "mixed_checkpoint", "mixed_projection"):
        pose = copy.deepcopy(scene["pose"])
        if mutation == "mixed_plan": pose["step"] = 5000
        if mutation == "mixed_checkpoint":
            for v in pose["caches"][M.C.ARMS[1]].values(): v["checkpoint_sha256"] = "a"*64
        if mutation == "mixed_projection": pose["projection"]["sha256"] = "a"*64
        inputs["pose_plan"] = save_json(tmp_path/"pose_plan.json", pose)
    if mutation == "wrong_probe_code":
        r = json.loads(Path(inputs["probe_reports"][0]["path"]).read_text())
        r["code_sha"] = "a"*40
        inputs["probe_reports"][0] = save_json(tmp_path/"probe_report.json", r)
    with pytest.raises(ValueError): M.compare(inputs, tmp_path/"bad.json", CODE)


def test_unresolved_probe_keeps_descriptive_metrics_but_disables_reading(scene, tmp_path):
    inputs = copy.deepcopy(scene["inputs"])
    r = json.loads(Path(inputs["probe_reports"][0]["path"]).read_text())
    rows = [json.loads(line) for line in Path(r["audit"]).read_text().splitlines()]
    rows[0]["warnings"].append({"category": "ConvergenceWarning", "message": "fixture", "file": "fixture", "line": 1})
    rows[0]["convergence_warnings"] = 1
    rows[0]["converged_by_criterion"] = False
    rows[0]["anomalous_stop"] = True
    audit = tmp_path/"audit.jsonl"
    audit.write_text("".join(json.dumps(row)+"\n" for row in rows))
    r.update(audit=str(audit), audit_sha256=M.C.sha(audit), audit_summary=M.C.audit_summary(rows), reading_status="solver_unresolved")
    shard = json.loads(Path(r["shards"][0]["path"]).read_text())
    shard.update(audit=str(audit), audit_sha256=M.C.sha(audit), audit_summary=M.C.audit_summary(rows))
    r["shards"] = [save_json(tmp_path/"shard.json", shard)]
    inputs["probe_reports"][0] = save_json(tmp_path/"probe_report.json", r)
    result = M.compare(inputs, tmp_path/"summary.json", CODE)
    assert result["complete"] and not result["quality"]["solver_converged"]
    assert not result["reading"]["reading_valid"] and result["reading"]["reading_status"] == "repair_evaluation_no_scientific_reading"
    assert "recommendation" not in result["reading"] and "primary_reading" not in result["reading"]
    assert np.isfinite(result["pose"]["delta_E_dyn"])


def test_readonly_artifacts_rechecked_before_publication(scene, tmp_path, monkeypatch):
    original = M.C.E.P.check_unchanged
    seen = []
    def checked(states):
        seen.append(set(states))
        return original(states)
    monkeypatch.setattr(M.C.E.P, "check_unchanged", checked)
    M.compare(scene["inputs"], tmp_path/"summary.json", CODE)
    last = seen[-1]
    for cell in scene["pose_report"]["cells"].values():
        assert cell["predictions"]["path"] in last and cell["coefficients"]["path"] in last
    for p in scene["inputs"]["probe_reports"]:
        r = json.loads(Path(p["path"]).read_text())
        assert r["oof"] in last and r["audit"] in last


def test_aggregate_cannot_hide_a_different_shard_fit_audit(scene, tmp_path):
    inputs = copy.deepcopy(scene["inputs"])
    r = json.loads(Path(inputs["probe_reports"][0]["path"]).read_text())
    rows = [json.loads(line) for line in Path(r["audit"]).read_text().splitlines()]
    rows[0]["elapsed_s"] += 10
    audit = tmp_path/"changed-audit.jsonl"
    audit.write_text("".join(json.dumps(row)+"\n" for row in rows))
    r.update(audit=str(audit), audit_sha256=M.C.sha(audit))
    inputs["probe_reports"][0] = save_json(tmp_path/"probe_report.json", r)
    with pytest.raises(ValueError, match="source shards"):
        M.compare(inputs, tmp_path/"summary.json", CODE)


@pytest.mark.parametrize("operation", ["touch", "restore_copy"])
def test_pose_inspection_accepts_same_content_after_filesystem_metadata_changes(scene, tmp_path, operation):
    path = Path(scene["pose"]["caches"][M.C.ARMS[1]]["val"]["path"])
    before, content_sha = path.stat(), M.C.sha(path)
    if operation == "touch":
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        assert path.stat().st_mtime_ns != before.st_mtime_ns
    else:
        restored = path.with_name("restored-cache.npz")
        shutil.copyfile(path, restored)
        restored.replace(path)
        assert path.stat().st_ino != before.st_ino
    assert M.C.sha(path) == content_sha
    result = M.compare(scene["inputs"], tmp_path/"summary.json", CODE)
    assert result["complete"] and result["quality"]["artifact_integrity"]


def test_pose_inspection_still_rejects_changed_cache_bytes(scene, tmp_path):
    path = Path(scene["pose"]["caches"][M.C.ARMS[1]]["val"]["path"])
    original = path.read_bytes()
    altered = bytearray(original)
    altered[-1] ^= 1
    try:
        path.write_bytes(altered)
        with pytest.raises(ValueError, match="SHA"):
            M.compare(scene["inputs"], tmp_path/"summary.json", CODE)
        assert not (tmp_path/"summary.json").exists()
    finally:
        path.write_bytes(original)


def test_pose_inspection_still_rejects_nonvolatile_support_identity_change(scene, tmp_path):
    inputs = copy.deepcopy(scene["inputs"])
    report = copy.deepcopy(scene["pose_report"])
    report["inspections"][M.C.ARMS[1]+"/val"]["targets"]["central_mask"] = "a"*64
    inputs["pose_report"] = save_json(tmp_path/"pose_report.json", report)
    with pytest.raises(ValueError, match="inspection receipt"):
        M.compare(inputs, tmp_path/"summary.json", CODE)
    assert not (tmp_path/"summary.json").exists()
