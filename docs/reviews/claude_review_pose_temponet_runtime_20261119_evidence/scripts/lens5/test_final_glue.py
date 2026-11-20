"""Scratch smoke test (lens 5): drive locked_inference -> metrics -> collection through runner glue."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path("/Users/simonebenatti/dev/wearusfm/.claude/worktrees/review-pose-heads-2f98211")
sys.path.insert(0, str(ROOT / "tests" / "cpu"))
sys.path.insert(0, str(ROOT / "scripts"))
from pose_temponet import execution as E, provenance as P, runner as R, workflow as W, metrics as M, report as REP  # noqa
from test_pose_temponet_provenance import prepared  # noqa: F401  (fixture)
from test_pose_temponet_runner import runtime, initial_chain  # noqa: F401  (fixture)
from test_pose_temponet_data import fixture


def dev_cache():
    data, _ = fixture(users=tuple(f"d{i:02d}" for i in range(15)), windows=2)
    data.pop("lat"); data.pop("pre")
    data["pose_var"][:] = np.tile([M.THRESHOLD, 0.1], 15).astype(np.float32)
    return data


def test_final_stage_glue(runtime, monkeypatch):
    repo, plan, opened, _ = initial_chain(runtime)
    for job in plan["jobs"]:
        if job["stage"] in ("wls_calibration", "controls", "selection", "refit"):
            R.run_stage(plan, job["id"], source_root=repo)
    dev = dev_cache()
    base_loader = P.read_verified_inputs

    def load(p, split, *, stage, depth=None):
        if split == "val":
            out = copy.deepcopy(dev)
            return out
        return base_loader(p, split, stage=stage, depth=depth)
    monkeypatch.setattr(P, "read_verified_inputs", load)

    def lock(selections, refits, *, development_identity):
        cells = {f"{h}/{s}": {"payload_sha256": f"{h}{s}".ljust(64, "0")[:64]} for h in W.HEADS for s in (0, 1, 2)}
        return W._lock({"schema": "x", "complete": True, "cells": cells,
                        "development_identity": W._source(development_identity), "production": True},
                       "inference_lock_sha256")
    monkeypatch.setattr(R.workflow, "lock_inference_manifest", lock)
    rng = np.random.default_rng(3)
    factor = {"L0": 1.0, "L13": .99, "MLP": .98, "TCN_post": .9, "TCN_pre": 1.02}

    def predict(cache, head, seed, state, lk, *, development_identity, device):
        y = cache["target"][:, 6:-6].astype(np.float64)
        pred = (y + factor[head] * (1 + .01 * seed) * rng.normal(size=y.shape) * .3).astype(np.float32)
        receipt = W._lock({"schema": "pose-head-prediction-v1", "complete": True, "head": head, "seed": seed,
                           "inference_lock_sha256": lk["inference_lock_sha256"],
                           "checkpoint_payload_sha256": lk["cells"][f"{head}/{seed}"]["payload_sha256"],
                           "development_identity": lk["development_identity"], "crop": [6, 154],
                           "prediction_array_sha256": W._array_hash(pred)}, "receipt_sha256")
        return {"prediction": pred, "receipt": receipt}
    monkeypatch.setattr(R.workflow, "predict_locked", predict)
    R.run_stage(plan, E.job_id("locked_inference"), source_root=repo)

    y = dev["target"][:, 6:-6].astype(np.float64)
    refs = {"post_ridge_predictions": y + .35 * rng.normal(size=y.shape),
            "trained_head_predictions": y + .34 * rng.normal(size=y.shape)}
    monkeypatch.setattr(R.data, "read_predictions", lambda pin: next(v for k, v in refs.items() if plan["inputs"][k] is pin or plan["inputs"][k] == pin))
    monkeypatch.setattr(R.metrics, "historical_reference_gate", lambda *a: {"passed": True})
    pin = R.run_stage(plan, E.job_id("metrics"), source_root=repo)
    receipt = P.read_json_pin(pin)
    report = P.read_json_pin(receipt["artifacts"]["report"])
    print("metrics receipt status", receipt["status"], "gates", sorted(receipt["gates"]))
    print("details keys", sorted(receipt["details"]))
    print("report status", report["status"], "candidate", report["candidacy"]["candidate"])
    for i in "12345":
        c = report["contrasts"][i]
        print(i, c["comparator"], round(c["relative_E_dyn"]["point"], 4), np.round(c["relative_E_dyn"]["ci"], 4),
              c["statistical_label"], c["scientific_label"], c["al_bordo_MC"])
    assert report["complete"] is True
    pin = R.run_stage(plan, E.job_id("collection"), source_root=repo)
    print("collection status", P.read_json_pin(pin)["status"])


def test_non_diagnostic_closed_form_is_labelled(runtime, monkeypatch):
    repo, plan, opened, _ = initial_chain(runtime)
    for job in plan["jobs"]:
        if job["stage"] == "wls_calibration":
            R.run_stage(plan, job["id"], source_root=repo)
    monkeypatch.setattr(R.controls, "combine_reference_folds", lambda rows: {"status": "control_non_diagnostic"})
    statuses = []
    for job in plan["jobs"]:
        if job["stage"] == "controls":
            with pytest.raises(Exception) as info:
                R.run_stage(plan, job["id"], source_root=repo)
            failure = P.read_json_pin(P.file_pin(P.receipt_path(plan, job["id"])))
            statuses.append((job["id"], job["resources"]["gpus"], failure["status"], type(info.value).__name__,
                             failure["details"].get("error"), sorted(failure["artifacts"])))
    for row in statuses:
        print(row)
