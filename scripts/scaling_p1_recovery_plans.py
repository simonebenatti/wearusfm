"""Pure recovery and mixed-extraction document builders; no remote actions.

One new P1 producer is joined to the original accepted pose/P2 producers.
Every cache keeps its actual code, plan, receipt and job identity. The frozen
builders/readers and source contract are never patched or re-pinned here.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import scaling_p1_recovery as R
import scaling_plans19618 as S

PROTOCOL = "scaling-p1-mixed-extractions-20261116-v1"
HISTORICAL_CODE_SHA = "df2de35c05381e7a5a69edde409171b210a3b5d8"
HISTORICAL_JOBS = {"pose": "64571999", "p2": "64572001"}
require = R.require


def make_recovery_manifest(extraction_plan, diagnostic_pin, diagnostic, cpu_test_pin, tests, sources):
    result = {"protocol": R.PROTOCOL, "arm": R.ARM, "step": R.STEP, "task": R.TASK,
              "logical_job_id": R.LOGICAL_ID, "out": R.OUTPUT, "failed_attempts": copy.deepcopy(R.FAILED_ATTEMPTS),
              "allocation": dict(R.ALLOCATION), "budget": dict(R.BUDGET), "output_barrier": "none",
              "source_sha256": dict(sources), "diagnostic_protocol": R.DIAGNOSTIC_PROTOCOL,
              "diagnostic_source_sha256": copy.deepcopy(diagnostic.get("source_sha256")),
              "extraction_plan": copy.deepcopy(extraction_plan), "diagnostic_report": copy.deepcopy(diagnostic_pin),
              "cpu_tests": copy.deepcopy(cpu_test_pin)}
    R.validate_manifest(result, diagnostic, tests, sources)
    for key in ("extraction_plan", "diagnostic_report", "cpu_tests"):
        S.ST.pin_schema(result[key])
        require(set(result[key]) == {"path", "sha256", "bytes"} and Path(result[key]["path"]).is_absolute(),
                "complete absolute evidence pin required: " + key)
    return result


def _code(value):
    return isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def mixed_candidate_caches(receipt_bytes, receipt_pins, provenances, completed_jobs, checkpoint, extraction_plan_sha256,
                           recovery_code_sha, recovery_manifest_sha256, recovery_sources):
    """Return pose caches, probe caches and lineage from already collected bytes.

Mappings are keyed by pose/p1/p2. provenances[pose] contains train/val bytes;
provenances[p1/p2] contain the corresponding dataset sidecar bytes.
completed_jobs entries are accounting {job_id,state,exit_code,code_sha}.
Receipt bytes are hashed before parsing. This function never adopts a failed
attempt, manufactures a receipt, reads a cache or submits a job.
"""
    tasks = {"pose", "p1", "p2"}
    require(all(set(m) == tasks for m in (receipt_bytes, receipt_pins, provenances, completed_jobs)),
            "exact pose/p1/p2 evidence required")
    require(_code(recovery_code_sha) and recovery_code_sha != HISTORICAL_CODE_SHA and
            R._sha(recovery_manifest_sha256) and R._sha(extraction_plan_sha256), "invalid mixed release identity")
    require(set(recovery_sources) == set(R.SOURCE_PATHS) and all(R._sha(v) for v in recovery_sources.values()),
            "complete recovery source pins required")
    require(isinstance(checkpoint, dict), "checkpoint pin required")
    S.ST.pin_schema({k: checkpoint[k] for k in ("path", "sha256", "bytes") if k in checkpoint})
    require(Path(checkpoint["path"]).is_absolute(), "absolute checkpoint pin required")
    receipts, lineage = {}, {}
    for task in ("pose", "p1", "p2"):
        pin, data, job = receipt_pins[task], receipt_bytes[task], completed_jobs[task]
        S.ST.pin_schema(pin)
        require(set(pin) == {"path", "sha256", "bytes"} and Path(pin["path"]).is_absolute() and
                S.digest(data) == pin["sha256"] and len(data) == pin["bytes"], "collected receipt bytes differ: " + task)
        expected_code = recovery_code_sha if task == "p1" else HISTORICAL_CODE_SHA
        require(job.get("state") == "COMPLETED" and job.get("exit_code") == "0:0" and
                job.get("code_sha") == expected_code and isinstance(job.get("job_id"), str) and
                job["job_id"].isdigit() and int(job["job_id"]) > 0, "unaccepted extraction accounting: " + task)
        if task in HISTORICAL_JOBS:
            require(job["job_id"] == HISTORICAL_JOBS[task], "wrong historical extraction job: " + task)
        else:
            require(job["job_id"] not in [x["job_id"] for x in R.FAILED_ATTEMPTS], "failed P1 cannot be adopted")
        receipt = json.loads(data)
        extraction = {"checkpoint": checkpoint, "code_sha": expected_code, "plan_sha256": extraction_plan_sha256}
        S._extraction_identity(receipt, "extract-pose" if task == "pose" else "extract-classification",
                               extraction, S.E.PROTOCOL, R.STEP, R.ARM)
        if task == "p1":
            for key, value in {"recovery_protocol": R.PROTOCOL, "recovery_manifest_sha256": recovery_manifest_sha256,
                               "recovery_source_sha256": recovery_sources, "output_barrier": "none",
                               "logical_job_id": R.LOGICAL_ID, "slurm_job_id": job["job_id"],
                               "failed_attempts_preserved": R.FAILED_ATTEMPTS}.items():
                require(receipt.get(key) == value, "P1 recovery receipt differs: " + key)
            require(str(Path(pin["path"]).parent) == R.OUTPUT, "P1 recovery receipt outside new output")
        receipts[task] = receipt
        lineage[task] = {"job_id": job["job_id"], "code_sha": expected_code,
                         "plan_sha256": extraction_plan_sha256, "receipt": dict(pin)}
    require(len({j["job_id"] for j in lineage.values()}) == 3, "duplicate extraction job identity")
    model = receipts["pose"].get("model_config")
    require(all(r.get("model_config") == model for r in receipts.values()), "mixed extraction model configurations")
    contract_sha = S.shared("training_source_contract")["sha256"]
    require(all(r.get("training_source_contract_sha256") == contract_sha for r in receipts.values()),
            "mixed training source contracts")
    pose_identity = {"checkpoint": checkpoint, "code_sha": HISTORICAL_CODE_SHA,
                     "plan_sha256": extraction_plan_sha256}
    pose = S.candidate_pose_caches(receipts["pose"], receipt_pins["pose"], provenances["pose"], pose_identity)
    caches = {}
    for task, datasets in S.TASKS.items():
        receipt = receipts[task]
        require(set(receipt["outputs"]) == set(datasets) == set(provenances[task]), "incomplete task cache evidence: " + task)
        for ds in datasets:
            out, raw = receipt["outputs"][ds], provenances[task][ds]
            prov = json.loads(raw)
            require(S.digest(raw) == out["provenance_sha256"], "collected provenance differs from receipt: " + ds)
            require(prov.get("cache") == out["path"] and prov.get("cache_sha256") == out["sha256"] and
                    prov.get("dataset") == ds and prov.get("task") == task and type(out.get("bytes")) is int and
                    prov.get("cache_bytes") == out["bytes"], "provenance/receipt mismatch: " + ds)
            for key in ("protocol", "arm", "which", "checkpoint_step", "checkpoint_sha256", "code_sha", "plan_sha256"):
                require(prov.get(key) == receipt.get(key), "provenance extraction identity mismatch: " + key)
            if task == "p1":
                for key in ("recovery_protocol", "recovery_manifest_sha256", "recovery_source_sha256", "output_barrier"):
                    require(prov.get(key) == receipt[key], "P1 recovery provenance mismatch: " + key)
                audit = prov.get("output_identity_audit")
                require(isinstance(audit, dict) and audit == out.get("output_identity_audit") and
                        audit.get("complete") is True and audit.get("policy") == R.O.POLICY and
                        type(audit.get("extra_reads")) is int and audit["extra_reads"] in (0, 1) and
                        isinstance(audit.get("observations"), list) and len(audit["observations"]) == 2 + audit["extra_reads"],
                        "P1 output finalization audit missing")
            caches[ds] = {"path": out["path"], "sha256": out["sha256"], "bytes": prov["cache_bytes"],
                          "provenance_path": out["provenance_path"], "provenance_sha256": out["provenance_sha256"],
                          "receipt_path": receipt_pins[task]["path"], "receipt_sha256": receipt_pins[task]["sha256"],
                          "checkpoint_path": checkpoint["path"], "checkpoint_sha256": receipt["checkpoint_sha256"],
                          "extraction_code_sha": receipt["code_sha"], "extraction_plan_sha256": receipt["plan_sha256"],
                          **{k: prov[k] for k in ("input_sha256", "splits_sha256", "extraction_recipe")}}
    for cache in [*pose.values(), *caches.values()]:
        folder = str(Path(cache["receipt_path"]).parent)
        require(str(Path(cache["path"]).parent) == folder == str(Path(cache["provenance_path"]).parent),
                "cache outside original receipt folder")
    mixed = {"protocol": PROTOCOL, "checkpoint": copy.deepcopy(checkpoint), "tasks": lineage,
             "recovery_manifest_sha256": recovery_manifest_sha256, "recovery_source_sha256": dict(recovery_sources),
             "training_source_contract_sha256": contract_sha, "scientific_source_contract_unchanged": True,
             "historical_failed_attempts": copy.deepcopy(R.FAILED_ATTEMPTS)}
    return pose, {ds: caches[ds] for ds in S.DATASETS}, mixed
