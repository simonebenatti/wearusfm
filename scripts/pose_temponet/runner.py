"""The stage controller. Scientific selection never imports development targets."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from . import budget_amendment as A1, controls, data, execution, heads, metrics, provenance as P, report, resources, train, workflow


GateNotPassed = controls.GateNotPassed


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_artifact(receipt, name):
    return P.read_json_pin(receipt["artifacts"][name])


def _torch_artifact(receipt, name):
    pin = receipt["artifacts"][name]
    check = P.verify_pin(pin)
    # These are our own signed-by-hash numerical artifacts, never arbitrary uploads.
    result = torch.load(pin["path"], map_location="cpu", weights_only=False)
    P.unchanged({pin["path"]: check["state"]})
    return result


def _receipt(plan, logical_id, *, source_root=P.ROOT, verification_cache=None):
    path = P.receipt_path(plan, logical_id)
    pin = plan.get("receipt_pins", {}).get(logical_id)
    if pin is None:
        pin = P.file_pin(path)
    value = P.verified_receipt(pin, plan,source_root=source_root,_memo=verification_cache)
    P.require(value["job_id"] == logical_id, "receipt logical ID mismatch")
    return value, pin


def _ancestors(plan, job, *, source_root=P.ROOT):
    found, pins = {}, {}
    memo = {}
    def visit(identifier):
        if identifier in found:
            return
        expected = execution.find_job(plan, identifier)
        for previous in expected["after"]:
            visit(previous)
        receipt, pin = _receipt(plan, identifier,source_root=source_root,verification_cache=memo)
        declared = sorted(P.read_json_pin(p)["job_id"] for p in receipt["predecessors"])
        P.require(declared == sorted(expected["after"]), "predecessor chain was truncated")
        P.require(receipt["stage"] == expected["stage"], "receipt has wrong planned stage")
        found[identifier], pins[identifier] = receipt, pin
    for identifier in job["after"]:
        visit(identifier)
    return found, pins


def _stage_receipts(receipts, stage):
    return [r for r in receipts.values() if r["stage"] == stage]


def _single(receipts, stage):
    rows = _stage_receipts(receipts, stage)
    P.require(len(rows) == 1, "expected one predecessor for " + stage)
    return rows[0]


def _input_identity(plan):
    return {"execution_sha256": plan["execution_sha256"], "source_sha256": plan["reader"]["source_sha256"],
            "recipe_sha256": plan["recipe_sha256"],
            "cache_sha256": plan["inputs"]["cache_train"]["sha256"],
            "reference_sha256": plan["inputs"]["target_reference_train"]["sha256"]}


def _development_identity(plan):
    return {"cache": plan["inputs"]["cache_val"], "reference": plan["inputs"]["target_reference_val"],
            "producer": plan["producers"]["val"], "stats": plan["inputs"]["pose_stats"]}


def _calibrations(receipts):
    rows = _stage_receipts(receipts, "wls_calibration")
    P.require(len(rows) == 2, "both calibrated folds must exist")
    values = sorted([_torch_artifact(r, "calibration_state") for r in rows], key=lambda x: x["outer_fold"])
    return values


def _control_gate(receipts, artifacts):
    selections = {world: {} for world in execution.WORLDS}
    for row in _stage_receipts(receipts, "controls"):
        selected = _json_artifact(row, "selection")
        world, head = selected["world"], selected["head"]
        P.require(head not in selections[world], "duplicate control selection")
        selections[world][head] = selected
    return workflow.qualify_controls(artifacts, selections)


def _resource_projection(plan, receipts, *, source_root=P.ROOT):
    cpu = _single(receipts, "cpu_benchmark")
    gpu = _single(receipts, "gpu_benchmark")
    checks = plan.get("operator_checks", {})
    required = ("work_free_bytes", "pending_write_bytes")
    P.require(all(type(checks.get(k)) is int for k in required), "current quota/pending-write measurements required")
    measurement = gpu["details"]["benchmark"]
    benchmark_plan = A1.benchmark_identity_plan(plan, source_root=source_root)
    P.require(cpu["details"]["benchmark"]["input_identity"] == measurement["input_identity"] == _input_identity(benchmark_plan),
              "benchmark input identity differs from this execution")
    persistent = measurement["persistent_output_bytes"]
    gate = resources.gate_and_pack(cpu["details"]["benchmark"], measurement,
                  work_free_bytes=checks["work_free_bytes"], pending_write_bytes=checks["pending_write_bytes"],
                  persistent_output_bytes=persistent, budget_caps=A1.effective_caps(plan))
    return gate


def _training_pack(projection):
    return [job for job in projection["jobs"] if job.get("phase") == "neural"]


def measured_plan(plan, cpu_pin, gpu_pin, *, source_root=P.ROOT):
    """Local deterministic phase amendment; no review/source/science changes."""
    import copy
    out = copy.deepcopy(plan)
    cpu = P.verified_receipt(cpu_pin, out, stage="cpu_benchmark",source_root=source_root,
                             verify_artifacts=False,receipt_mirror_dir=Path(cpu_pin["path"]).parent)
    gpu = P.verified_receipt(gpu_pin, out, stage="gpu_benchmark",source_root=source_root,
                             verify_artifacts=False,receipt_mirror_dir=Path(gpu_pin["path"]).parent)
    rows = {cpu["job_id"]: cpu, gpu["job_id"]: gpu}
    gate = _resource_projection(out, rows, source_root=source_root)
    if not gate["all_passed"]:
        raise GateNotPassed("global resource gate failed; no calibration or neural jobs released")
    out["jobs"] = execution.apply_training_pack(out["jobs"], _training_pack(gate),
                                                 budget_caps=A1.effective_caps(out))
    out["resource_gate"] = gate
    out["receipt_pins"] = {**out.get("receipt_pins", {}),
                           **{receipt["job_id"]:{**pin,"path":str(P.receipt_path(out,receipt["job_id"]))}
                              for receipt,pin in ((cpu,cpu_pin),(gpu,gpu_pin))}}
    out["numeric_pins"] = {"cpu8": cpu["numeric_sha256"], "gpu8": gpu["numeric_sha256"]}
    P.validate_plan(out,source_root=source_root,require_review=True)
    return out


def _allocation(job):
    evidence = P.validate_job_allocation(job)
    P.require(os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "requeue/restart forbidden")
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]], text=True)
    fields = dict(token.split("=", 1) for token in raw.split() if "=" in token)
    seconds = job["resources"]["requested_seconds"]
    expected = "%02d:%02d:%02d" % (seconds//3600, seconds//60%60, seconds%60)
    P.require(fields.get("TimeLimit") == expected, "SLURM walltime differs from published allocation")
    P.require(fields.get("Partition") == job["resources"]["partition"], "wrong partition")
    P.require(int(fields.get("NumCPUs","0")) == job["resources"]["cpus"] and
              int(fields.get("NumNodes","0")) == 1, "scontrol CPU/node allocation differs")
    tres = dict(token.split("=",1) for token in fields.get("AllocTRES","").split(",") if "=" in token)
    P.require(int(tres.get("cpu","0")) == job["resources"]["cpus"] and
              int(tres.get("gres/gpu","0")) == job["resources"]["gpus"], "AllocTRES CPU/GPU counts differ")
    memory = tres.get("mem","")
    suffixes = {"K":1024,"M":1024**2,"G":1024**3,"T":1024**4}
    P.require(memory and memory[-1:] in suffixes and
              float(memory[:-1])*suffixes[memory[-1]] == job["resources"]["host_ram_gib"]*1024**3,
              "AllocTRES memory differs")
    P.require(int(os.environ.get("SLURM_MEM_PER_NODE", "0")) == job["resources"]["host_ram_gib"]*1024,
              "host RAM allocation differs")
    if job["resources"]["gpus"]:
        P.require(torch.cuda.device_count() == 1, "one allocated GPU required")
    evidence["scontrol"] = raw.strip()
    return evidence


def _progress(pair):
    # The curve is deliberately incomplete at intermediate checkpoints.
    # pair.curve() validates a completed horizon and must not run here.
    if pair.points:
        record = {"event": "selection_checkpoint", "lr": pair.lr,
                  "point": _plain(asdict(pair.points[-1]))}
        if pair.points[-1].exposure == 0:
            record["initial_sha256_per_fold"] = {str(t.fold): t.initial_sha256 for t in pair.trajectories}
        print(json.dumps(record, allow_nan=False), flush=True)


def _save_results(root, directory, results, artifacts):
    for key, value in results.items():
        if key == "filenames":
            continue
        filename = results.get("filenames", {}).get(key)
        if filename is None:
            continue
        writer = P.atomic_torch if filename.endswith(".pt") else P.atomic_json
        artifacts[key] = writer(directory/filename, value if filename.endswith(".pt") else _plain(value), root=root)


def run_stage(plan, logical_id, *, source_root=P.ROOT):
    """A complete planned stage or an explicit failure receipt. Never submits/retries."""
    job = execution.find_job(plan, logical_id)
    execution.validate_graph(plan["jobs"])
    P.require(job["resource_release"], "job has not passed resource release")
    P.validate_plan(plan,source_root=source_root,require_review=True)
    P.checkout_identity(plan,source_root=source_root)
    allocation = _allocation(job)
    stage = job["stage"]
    root = P.safe_output_root(plan["output_root"])
    directory = root/job["output_directory"]
    P.safe_path(directory,root=root)
    P.require(not directory.exists(), "existing stage output: do not duplicate or retry")
    P.require(not P.receipt_path(plan,logical_id).exists(), "existing stage receipt: do not duplicate or retry")
    if stage not in ("preflight", "collection"):
        train.configure_numerics()
    numeric = P.numeric_context("gpu" if job["resources"]["gpus"] else "cpu")
    started = time.perf_counter()
    receipts, pins = _ancestors(plan, job,source_root=source_root)
    gates, details, artifacts = {}, {"allocation": allocation}, {}
    try:
        P.verify_numeric(plan,stage,numeric)
        if stage != "preflight":
            previous = _single(receipts, "preflight")["details"]["preflight"]
            gates.update(integrity=previous["integrity"] is True,
                         historical_R1=bool(previous["historical_R1"].get("passed", False)))
        if stage not in ("preflight", "cpu_benchmark", "gpu_benchmark"):
            projection = _resource_projection(plan, receipts, source_root=source_root)
            gates["resources"] = projection["all_passed"]
            P.require(projection == plan.get("resource_gate"), "resource plan is not the measured, pinned projection")
            P.require(execution.apply_training_pack(execution.job_graph(), _training_pack(projection),
                          budget_caps=A1.effective_caps(plan)) == plan["jobs"],
                      "allocations differ from measured packing")
        if stage in ("controls", "selection", "refit", "locked_inference", "metrics"):
            calibrated = _calibrations(receipts)
            closed_gate = workflow.combined_closed_form_gate(calibrated)
            artifacts["combined_gate"] = P.atomic_json(directory/"combined_gate.json", _plain(closed_gate), root=root)
            gates.update(closed_references=closed_gate["status"] == "diagnostic",
                         calibration=all(c["calibration"]["status"] == "calibrated" for c in calibrated))
            if not gates["closed_references"]:
                raise GateNotPassed("closed-form controls: "+closed_gate["status"],
                                    status=closed_gate["status"], gate=closed_gate)
        if stage in ("selection", "refit", "locked_inference", "metrics"):
            positive_gate = _control_gate(receipts, calibrated)
            artifacts["positive_gate"] = P.atomic_json(directory/"positive_gate.json", positive_gate, root=root)
            gates["positive_controls"] = positive_gate["status"] == "passed"
            if not gates["positive_controls"]:
                raise GateNotPassed("positive controls: "+positive_gate["status"],
                                    status=positive_gate["status"], gate=positive_gate)
        if stage in ("refit", "locked_inference", "metrics"):
            selections = {_json_artifact(r, "selection")["head"]: _json_artifact(r, "selection")
                          for r in _stage_receipts(receipts, "selection")}
            P.require(set(selections) == set(heads.HEADS), "all real selections must be locked")
            for selection in selections.values():
                workflow._check_selection(selection, production=True)
            gates["selection_locked"] = True
        if stage == "preflight":
            P.check_release(plan,stage,gates=gates,source_root=source_root)
            details["preflight"] = P.preflight(plan)
            gates.update(integrity=True, historical_R1=bool(details["preflight"]["historical_R1"].get("passed", False)))
            P.require(gates["historical_R1"], "historical reference gate did not pass")
        elif stage in ("cpu_benchmark", "gpu_benchmark"):
            from . import benchmark
            P.check_release(plan, stage, gates=gates,source_root=source_root)
            loader = lambda: P.read_verified_inputs(plan, "train", stage=stage)
            cache, first_load = benchmark._load_measurement(loader, _input_identity(plan))
            kwargs = {"input_loader": loader, "input_measurement": first_load, "output_dir": directory/"cost_only",
                      "allowed_train_users": plan["producers"]["train"]["users"],
                      "input_identity": _input_identity(plan)}
            if stage == "gpu_benchmark":
                kwargs["reference_byte_count"] = sum(plan["inputs"][k]["bytes"]
                    for k in ("post_ridge_coefficients", "post_ridge_predictions", "trained_head_predictions"))
                kwargs["reference_proxy_path"] = plan["inputs"]["cache_train"]["path"]
            measurement = getattr(benchmark, "benchmark_"+stage.split("_")[0])(cache, **kwargs)
            artifacts["benchmark"] = P.atomic_json(directory/"benchmark.json", _plain(measurement), root=root)
            details["benchmark"] = _plain(measurement)
        elif stage == "wls_calibration":
            P.check_release(plan, stage, gates=gates,source_root=source_root)
            cache = P.read_verified_inputs(plan, "train", stage=stage)
            cache.pop("pre",None)
            groups = data.alternating_user_folds(cache, previous["train"]["folds"])
            fit_users = set(groups[job["fold"]])
            fit = np.asarray([i for i, m in enumerate(cache["meta"]) if m["user"] in fit_users])
            held = np.asarray([i for i, m in enumerate(cache["meta"]) if m["user"] not in fit_users])
            result = controls.build_positive_control_fold(cache, fit, held, plan["producers"]["train"]["users"],
                                                         outer_fold=job["fold"], block_windows=32)
            artifacts["calibration_state"] = P.atomic_torch(directory/"calibration.pt", result, root=root)
            summary = {k: v for k, v in result.items() if k != "components"}
            artifacts["calibration"] = P.atomic_json(directory/"calibration.json", _plain(summary), root=root)
            details["calibration_status"] = result["calibration"]["status"]
            if result["calibration"]["status"] != "calibrated":
                status = result["calibration"]["status"]
                raise GateNotPassed(status+": stop before neural controls", status=status,
                                    gate={"status": status, "calibration": result["calibration"]})
        elif stage in ("controls", "selection", "refit"):
            P.check_release(plan, stage, gates=gates,source_root=source_root)
            cache = P.read_verified_inputs(plan, "train", stage=stage)
            kwargs = {"allowed_train_users": plan["producers"]["train"]["users"],
                      "source_identity": _input_identity(plan), "expected_folds": previous["train"]["folds"]}
            if stage == "controls":
                bundle = workflow.prepare_control_folds(cache, calibrated, job["world"], **kwargs)
                result = workflow.run_selection(bundle, job["head"], device="cuda", on_checkpoint=_progress)
            else:
                bundle = workflow.prepare_real_folds(cache, **kwargs)
                if stage == "selection":
                    result = workflow.run_selection(bundle, job["head"], control_gate=positive_gate,
                                                    device="cuda", on_checkpoint=_progress)
                else:
                    result = workflow.run_refits(bundle, selections[job["head"]], positive_gate, device="cuda")
            _save_results(root, directory, result, artifacts)
        elif stage == "locked_inference":
            refits = {}
            for row in _stage_receipts(receipts, "refit"):
                manifest = _json_artifact(row, "manifest")
                refits[manifest["head"]] = {"manifest": manifest, "states": _torch_artifact(row, "states")}
            dev = _development_identity(plan)
            lock = workflow.lock_inference_manifest(selections, refits, development_identity=dev)
            artifacts["inference_lock"] = P.atomic_json(directory/"inference_lock.json", lock, root=root)
            gates.update(refits_locked=True)
            P.check_release(plan, stage, gates=gates,source_root=source_root)
            # Only after every lock has been checked do we open development data.
            for depth, names in (("post", [h for h in heads.HEADS if h != "TCN_pre"]), ("pre", ["TCN_pre"])):
                cache = P.read_verified_inputs(plan, "val", stage=stage, depth=depth)
                for head in names:
                    for seed in (0, 1, 2):
                        state = refits[head]["states"]["refit_states"][seed]
                        result = workflow.predict_locked(cache, head, seed, state, lock, development_identity=dev, device="cuda")
                        key = f"{head}_s{seed}"
                        artifacts[key+"_predictions"] = P.atomic_npz(directory/(key+".npz"),
                                                                    {"prediction": result["prediction"]}, root=root)
                        artifacts[key+"_manifest"] = P.atomic_json(directory/(key+".json"), result["receipt"], root=root)
                        del result
                del cache
                gc.collect()
        elif stage == "metrics":
            inference = _single(receipts, "locked_inference")
            P.require(inference["gates"].get("refits_locked") is True, "inference did not descend from locked refits")
            gates.update(refits_locked=True, inference_locked=True)
            P.check_release(plan, stage, gates=gates,source_root=source_root)
            inference_lock = _json_artifact(inference,"inference_lock")
            workflow._verify(inference_lock,"inference_lock_sha256")
            P.require(inference_lock["complete"] is True and
                      inference_lock["development_identity"] == _development_identity(plan), "inference lock differs")
            cache = P.read_verified_inputs(plan, "val", stage=stage, depth="post")
            cache.pop("features", None)
            old = {"historical_post_ridge": data.read_predictions(plan["inputs"]["post_ridge_predictions"]),
                   "historical_trained_head": data.read_predictions(plan["inputs"]["trained_head_predictions"])}
            parity = metrics.historical_reference_gate(cache, old["historical_post_ridge"], old["historical_trained_head"])
            artifacts["historical_R1"] = P.atomic_json(directory/"historical_R1.json", _plain(parity), root=root)
            details["historical_R1"] = _plain(parity)
            P.require(parity["passed"], "historical parity changed before contrasts")
            references = {h: {"complete": True, "analysis": metrics.evaluate(v, cache, batch_windows=32)} for h,v in old.items()}
            del old
            cells = {h: {} for h in heads.HEADS}
            for head in heads.HEADS:
                for seed in (0, 1, 2):
                    key = f"{head}_s{seed}"
                    pin = inference["artifacts"][key+"_predictions"]
                    verified = P.verify_pin(pin)
                    manifest = _json_artifact(inference, key+"_manifest")
                    workflow._verify(manifest,"receipt_sha256")
                    P.require(manifest["complete"] is True and manifest["head"] == head and manifest["seed"] == seed and
                              manifest["crop"] == [6,154] and manifest["development_identity"] == _development_identity(plan) and
                              manifest["inference_lock_sha256"] == inference_lock["inference_lock_sha256"] and
                              manifest["checkpoint_payload_sha256"] == inference_lock["cells"][f"{head}/{seed}"]["payload_sha256"],
                              "prediction manifest cell/development support differs")
                    with np.load(pin["path"], allow_pickle=False) as archive:
                        P.require(set(archive.files) == {"prediction"}, "unexpected prediction payload members")
                        prediction = archive["prediction"]
                    P.unchanged({pin["path"]:verified["state"]})
                    P.require(workflow._array_hash(prediction) == manifest["prediction_array_sha256"], "prediction payload differs")
                    cells[head][seed] = {"complete": True, "analysis": metrics.evaluate(prediction, cache,
                                                                                       batch_windows=32, per_joint=True)}
                    del prediction
            final = report.aggregate(cells, references, assessments=selections,
                                     common_gates={k: gates[k] for k in report.COMMON_GATES})
            final["historical_R1"] = _plain(parity)
            artifacts["report"] = P.atomic_json(directory/"report.json", final, root=root)
            P.require(final["complete"], "aggregate technical_failure")
        elif stage == "collection":
            inventory = []
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    inventory.append(P.file_pin(path))
            P.require(sum(p["bytes"] for p in inventory) <= 8*1024**3, "persistent output exceeds 8GiB")
            artifacts["inventory"] = P.atomic_json(directory/"inventory.json", inventory, root=root)
            details["scientific_results"] = _single(receipts, "metrics")["artifacts"]["report"]
        else:
            raise ValueError("unimplemented stage")
        details["elapsed_seconds"] = time.perf_counter()-started
        details["finished_at"] = datetime.now(timezone.utc).isoformat()
        receipt = P.make_receipt(plan, logical_id, numeric=numeric,source_root=source_root,
                    predecessors=[pins[x] for x in job["after"]], artifacts=artifacts, gates=gates, details=details)
        return P.write_receipt(plan, receipt)
    except Exception as error:
        details.update(elapsed_seconds=time.perf_counter()-started, error_type=type(error).__name__, error=str(error),
                       finished_at=datetime.now(timezone.utc).isoformat())
        try:
            if isinstance(error, GateNotPassed):
                details["gate_status"] = error.status
                if error.gate is not None:
                    artifacts["failed_gate"] = P.atomic_json(directory/"failed_gate.json", _plain(error.gate), root=root)
            receipt = P.make_receipt(plan, logical_id, numeric=numeric,source_root=source_root,
                        predecessors=[pins[x] for x in job["after"]], artifacts=artifacts, gates=gates,
                        details=details, status="gate_failed" if isinstance(error,GateNotPassed) else "technical_failure")
            P.write_receipt(plan, receipt)
        except Exception as recording_error:
            print(json.dumps({"event":"failure_receipt_unavailable","job":logical_id,
                              "original_error":str(error),"recording_error":str(recording_error)}),flush=True)
        raise
