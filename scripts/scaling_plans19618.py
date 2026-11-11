"""Operational documents of the 19,618 evaluation chain, built only from real evidence.

Pure builders: each receives the reviewed 5,000 document of the same stage plus the receipts,
reports and scheduler accounting collected on the Mac, and returns the document consumed by the
19,618 readers (scaling_*19618.py), or the exact submission lines of a stage. The builders read
nothing on Leonardo, submit no job, open no cache and compute no scientific score; a missing or
inconsistent piece of evidence stops the build. Names, paths and resources are fixed here before
any 19,618 result exists; resources are those of the budget freeze v2. The only remote access is
read-only and bounded: `collect` (small files, file hashes, scheduler accounting) and `gate` (state of
the checkouts, published documents, outputs, serial queue and quota). Every gate leaves its record
among the evidence.
"""
from __future__ import annotations

import argparse
import base64
import calendar
import functools
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import scaling_compare19618 as M  # noqa: E402

C, P = M.C, M.P
E, Q, ST = C.E, P.Q, C.ST
ARMS, STEP, CELLS, TASKS = C.ARMS, C.STEP, tuple(C.CELLS), C.TASKS
ARM = ARMS[1]
DATASETS = tuple(ds for task in TASKS.values() for ds in task)
POSE_RESOURCE_PROTOCOL = "scaling-pose-resources-19618-20261106-v1"
PACKING_PROTOCOL = "scaling-probe-packing-audit-19618-20261106-v1"
GATE_PROTOCOL = "scaling-gate-19618-20261106-v1"
WORK, SCRATCH = "/leonardo_work/IscrB_WearUsFM", "/leonardo_scratch/large/userexternal/sbenatti"
INFRA = "results/infra/scaling_300m_20261104"
PACKAGE = INFRA + "/continuation19618_20261106"
# Recovery of the 5000 -> 19618 training (v3, 09/11/2026): the single source of its identity in this chain. Kept as they are:
# v1 (plan PACKAGE/training_plan.json, checkout 63695634, training 63696864 FAILED at resume on 08/11) and v2 (plan
# continuation19618_20261109_v2, checkout 63974034, pre-check 63974196 passed but refused by its own gate: no training).
RECOVERY = {"plan": INFRA + "/continuation19618_20261109_v3/training_plan.json",
            "campaign": "scaling_300m_continuation19618_20261109_v3", "checkout": "SCALING-300M-cont19618-checkout-20261109-v3",
            "precheck": "SCALING-300M-cont19618-idcheck-20261109-v3", "training": "SCALING-300M-train19618-20261109-v3"}
# Evidence folders of the recovery stages; the v1 checkout (checkout-training) and the v2 checkout and pre-check
# (checkout-training-v2, identity-precheck) keep their records.
EVIDENCE_DIRS = {"checkout-training": "checkout-training-v3", "identity-precheck": "identity-precheck-v3"}
FREEZE = INFRA + "/continuation19618_budget_freeze_20261105_v2.json"
CAMPAIGN = "scaling-300m-20261104"
CHECKOUTS = {"code": "scaling_300m_evaluation19618_20261106_v1", "plans": "scaling_300m_plans19618_20261106_v1",
             "budgets": "scaling_300m_budgets19618_20261106_v1", "inputs": "scaling_300m_inputs19618_20261106_v1"}
# Four checkouts here plus the training one. lrd_all_serial allocates whole cores (two threads): a 1-CPU request gets 2 CPUs
# (job 64194074, 11/11/2026), as the freeze writes (readers.stages.checkout.cpus = 2) and the training checkouts ask. The five
# checkouts at 2 CPUs take 4/3 core-hours with margin instead of the frozen 0.8: amendment CHECKOUT_AMENDMENT (Simone decides).
CHECKOUT = {"cpus": 2, "memory_gib": 4, "seconds": 300}
CHECKOUT_AMENDMENT = INFRA + "/continuation19618_20261111_checkout_cpus/budget_amendment.json"
CHECKOUT_STAGE_CAP = 4 / 3  # core-hours with scheduler margin of the five checkouts of the table, as amended
# A review after an earlier checkout may change only what no job on the cluster reads (rereview_check).
MAC_SIDE = ("scripts/scaling_plans19618.py",)
EXTRACT_ROOT = SCRATCH + "/wearusfm_runs/results/scaling_300m_20261104/evaluation19618_v1/" + ARM
CPU_ROOT = WORK + "/wearusfm_runs/results/scaling_300m_20261104/evaluation19618_v1/cpu"
SHARED = {"splits": "splits/v1/splits_v1.json", "training_source_contract": INFRA + "/training_source_contract.json",
          "projection": INFRA + "/projection_s5/manifest.json"}
DOCUMENTS = {"extraction_plan": ("code", "evaluation/extraction_plan.json"),
             "pose_plan": ("plans", "readers/pose_plan.json"), "probe_plan": ("plans", "readers/probe_plan.json"),
             "pose_budget": ("budgets", "readers/pose_budget.json"),
             "probe_resources": ("budgets", "readers/probe_resources.json"),
             "comparison_inputs": ("inputs", "readers/comparison_inputs.json")}
PYTHON = "/Users/simonebenatti/.venvs/wearusfm-tok/bin/python"
MAC = "/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-scaling-exec"
LEDGER = Path.home() / "wearusfm_local/jobs"
SACCT_FORMAT = "JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,NodeList,Partition,Start,End,Timelimit,AllocTRES"
SSH = ["ssh", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
       "-o", "ProxyCommand=none", "-o", "ControlMaster=no", "-o", "ControlPath=none", "leonardo"]
INLINE_LIMIT, GZIP_FROM = 16 << 20, 1 << 20
LOGS = WORK + "/wearusfm_runs/logs"
SHARD_FILES = ["audit.jsonl", "audit_progress.jsonl", "shard.json", "shard.oof.npz"]
AGGREGATE_FILES = ["aggregate.audit.jsonl", "aggregate.json", "aggregate.oof.npz"]
RSS_GATED = ("pose-validate", "pose-benchmark")  # their measured peak memory enters the pose budget
RUNTIME_KEYS = ("code_sha", "sources", "versions", "threads", "torch_threads", "threadpools")
STAGE_DOCUMENTS = {"extractions": ("extraction_plan",), "pose-validate": ("pose_plan",), "pose-benchmark": ("pose_plan",),
                   "probe-benchmark": ("probe_plan",), "pose-ridge": ("pose_plan", "pose_budget"),
                   "probe-fit": ("probe_plan", "probe_resources"), "probe-aggregate": ("probe_plan",),
                   "compare": ("comparison_inputs", "pose_plan", "probe_plan")}
SERIAL_SUBMIT_LIMIT = 10  # jobs per user in the serial QoS, shared with the other lines
FITS_PER_FOLD = len(C.C_GRID) + 1  # the candidates of the grid and the refit: seven fits per dataset and fold
LINE_SERIAL_LIMIT = 4  # jobs of this line submitted together in the serial queue: prudent default until Simone says otherwise
LINE_JOB_PREFIX = "SCALING-"  # the submitter names every job of this line after its logical ID
GATE_MAX_AGE_SECONDS = 1800  # submission lines come only from a gate read within this time
CLOCK_SKEW_SECONDS = 300  # tolerated difference between the clock of the login node and the one of the Mac
WORK_PROJECT, WORK_FREE_GIB = "20147512", 5  # outputs and checkouts of the whole stage B take about 2 GiB on $WORK


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pin(path, data):
    return {"path": path, "sha256": digest(data), "bytes": len(data)}


def dumps(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def checkout(name):
    return f"{WORK}/wearusfm_campaigns/{CHECKOUTS[name]}"


def document(name, root=ROOT):
    """Remote pin of a published document: its bytes in this repository, its path in its checkout."""
    where, relative = DOCUMENTS[name]
    return pin(f"{checkout(where)}/{PACKAGE}/{relative}", (Path(root) / PACKAGE / relative).read_bytes())


def shared(key, previous=None):
    """One file for both chains: the bytes reviewed at 5,000, read in the frozen 19,618 code checkout."""
    value = pin(f"{checkout('code')}/{SHARED[key]}", (ROOT / SHARED[key]).read_bytes())
    require(previous is None or {k: previous[k] for k in ("sha256", "bytes")} == {k: value[k] for k in ("sha256", "bytes")},
            "shared file differs from the reviewed 5000 plan: " + key)
    return value


def freeze():
    return json.loads((ROOT / FREEZE).read_text())["readers"]


def clock(seconds):
    return "%02d:%02d:%02d" % (seconds // 3600, seconds % 3600 // 60, seconds % 60)


# ---------------------------------------------------------------------------------------------
# Plans

def extraction_plan(previous, training, files, receipt, summary, training_sha256):
    """previous: reviewed 5,000 extraction plan. training: training_plan.json of the continuation, training_sha256 the hash of
    its bytes. files: remote pins of the completed checkpoint, config, receipt and summary; receipt and summary: content.
    The receipt must name this plan and its execution sources (hashes, not only paths) and the identity pre-check it passed."""
    require(previous.get("protocol") == "scaling-evaluation-20261104-v1" and previous.get("step") == 5000 and
            previous.get("arms") == list(ARMS), "not the reviewed 5000 extraction plan")
    run, out = training["run_dir"], training["receipt_dir"]
    expected = {"checkpoint": f"{run}/checkpoint_step{STEP:07d}.pt", "config": out + "/config.json",
                "receipt": out + "/receipt.json", "summary": out + "/summary.json"}
    require(set(files) == set(expected), "need exactly the four completed training files")
    for name, path in expected.items():
        ST.pin_schema(files[name])
        require(files[name]["path"] == path and "bytes" in files[name], "training file is not the planned one: " + name)
    wanted = {"protocol": E.TRAIN_PROTOCOL, "complete": True, "arm": ARM, "checkpoint_step": STEP,
              "checkpoint": files["checkpoint"]["path"], "checkpoint_sha256": files["checkpoint"]["sha256"],
              "config_path": files["config"]["path"], "config_file_sha256": files["config"]["sha256"],
              "start_step": E.START, "additional_steps": STEP - E.START, "scientific_endpoint": True,
              "previous_run_unchanged": True, "further_training_authorized": False,
              "control_checkpoint_sha256": E.CONTROL_CHECKPOINT_SHA256,
              "training_source_contract_sha256": previous["training_source_contract"]["sha256"],
              "plan_sha256": training_sha256, "execution_source_sha256": training["execution_source_sha256"]}
    require(ST.digest(training_sha256), "hash of the training plan required")
    for key, value in wanted.items():
        require(receipt.get(key) == value and type(receipt.get(key)) is type(value), "training receipt differs: " + key)
    precheck = receipt.get("identity_precheck")
    require(isinstance(precheck, dict) and ST.digest(precheck.get("sha256", "")) and
            precheck.get("receipt") == training["identity_precheck"]["receipt_dir"] + "/identity_precheck.json",
            "training receipt differs: identity_precheck")
    require(training["control"]["checkpoint"]["sha256"] == E.CONTROL_CHECKPOINT_SHA256 and
            training["source_contract"]["sha256"] == previous["training_source_contract"]["sha256"],
            "training plan control/contract differs")
    require(summary.get("stopped") == "max_steps" and type(summary.get("steps")) is int and summary["steps"] == STEP,
            "training did not stop at the registered endpoint")
    endpoint = dict(files["checkpoint"])
    for name in ("config", "receipt", "summary"):
        endpoint.update({name + "_path": files[name]["path"], name + "_sha256": files[name]["sha256"]})
    return E.make_plan(previous["comparison"], previous["inputs"], previous["paths"], previous["pose_references"],
                       previous["classification_references"], previous["training_source_contract"], {ARM: endpoint})


def _extraction_identity(receipt, stage, extraction, protocol, step, arm):
    wanted = {"protocol": protocol, "arm": arm, "which": "teacher", "checkpoint_step": step, "complete": True,
              "checkpoint_unchanged": True, "postload_recipe_passed": True, "stage": stage,
              "checkpoint_sha256": extraction["checkpoint"]["sha256"], "code_sha": extraction["code_sha"],
              "plan_sha256": extraction["plan_sha256"]}
    for key, value in wanted.items():
        require(receipt.get(key) == value and type(receipt.get(key)) is type(value), "extraction receipt differs: " + key)


def candidate_pose_caches(receipt, receipt_pin, provenances, extraction, protocol=E.PROTOCOL, step=STEP, arm=ARM):
    """receipt: collected extract-pose receipt; provenances[split]: bytes of the two provenance files;
    extraction: {"checkpoint": pin, "code_sha", "plan_sha256"} of the extraction that produced them."""
    _extraction_identity(receipt, "extract-pose", extraction, protocol, step, arm)
    require(set(receipt.get("outputs", {})) == {"train", "val"} == set(provenances), "need train and val outputs")
    caches = {}
    for split in ("train", "val"):
        out, provenance = receipt["outputs"][split], json.loads(provenances[split])
        require(digest(provenances[split]) == out["provenance_sha256"], "collected provenance differs from the receipt: " + split)
        require(provenance.get("cache") == out["path"] and provenance.get("cache_sha256") == out["sha256"] and
                provenance.get("cache_bytes") == out["bytes"] and provenance.get("split") == split, "provenance/receipt mismatch: " + split)
        caches[split] = {"path": out["path"], "sha256": out["sha256"], "bytes": out["bytes"],
                         "provenance_path": out["provenance_path"], "provenance_sha256": out["provenance_sha256"],
                         "receipt_path": receipt_pin["path"], "receipt_sha256": receipt_pin["sha256"],
                         "checkpoint_sha256": receipt["checkpoint_sha256"], "extraction_code_sha": receipt["code_sha"]}
    return caches


def pose_plan(previous, inventory, candidate):
    """previous: reviewed 5,000 pose plan; inventory: tokenizer pose plan of the 19,618 control; candidate: new caches."""
    require(previous.get("protocol") == "scaling-pose-readout-20261104-v1" and previous.get("step") == 5000,
            "not the reviewed 5000 pose plan")
    require(inventory.get("step") == STEP and inventory.get("stats_sha256") == previous["stats_sha256"] and
            inventory.get("references") == previous["references"] and inventory.get("fold_users") == previous["fold_users"],
            "control inventory has another dose, references or folds")
    control = inventory["caches"]["T100-G"]
    require(all(p["checkpoint_sha256"] == E.CONTROL_CHECKPOINT_SHA256 for p in control.values()), "not the registered control")
    return P.make_plan({ARMS[0]: control, ARM: candidate}, previous["references"], previous["fold_users"],
                       shared("projection", previous["projection"]),
                       shared("training_source_contract", previous["training_source_contract"]))


def candidate_probe_caches(receipts, receipt_pins, provenances, extraction, protocol=E.PROTOCOL, step=STEP, arm=ARM):
    """receipts[task], receipt_pins[task]: the two extract-classification receipts; provenances[dataset]: bytes."""
    require(set(receipts) == set(TASKS) == set(receipt_pins) and set(provenances) == set(DATASETS), "need p1, p2 and five datasets")
    caches = {}
    for task, datasets in TASKS.items():
        receipt = receipts[task]
        _extraction_identity(receipt, "extract-classification", extraction, protocol, step, arm)
        require(set(receipt.get("outputs", {})) == set(datasets), "receipt outputs differ from the task datasets: " + task)
        for ds in datasets:
            out, provenance = receipt["outputs"][ds], json.loads(provenances[ds])
            require(digest(provenances[ds]) == out["provenance_sha256"], "collected provenance differs from the receipt: " + ds)
            require(provenance.get("cache") == out["path"] and provenance.get("cache_sha256") == out["sha256"] and
                    provenance.get("dataset") == ds and provenance.get("task") == task and
                    type(out.get("bytes")) is int and provenance.get("cache_bytes") == out["bytes"],
                    "provenance/receipt mismatch: " + ds)
            caches[ds] = {"path": out["path"], "sha256": out["sha256"], "bytes": provenance["cache_bytes"],
                          "provenance_path": out["provenance_path"], "provenance_sha256": out["provenance_sha256"],
                          "receipt_path": receipt_pins[task]["path"], "receipt_sha256": receipt_pins[task]["sha256"],
                          "checkpoint_path": extraction["checkpoint"]["path"], "checkpoint_sha256": receipt["checkpoint_sha256"],
                          "extraction_code_sha": receipt["code_sha"], "extraction_plan_sha256": receipt["plan_sha256"],
                          **{k: provenance[k] for k in ("input_sha256", "splits_sha256", "extraction_recipe")}}
    return {ds: caches[ds] for ds in DATASETS}


def control_probe_caches(inventory, provenances, checkpoint):
    """inventory: tokenizer probe plan of the 19,618 control; provenances[dataset]: bytes; checkpoint: its pin."""
    require(inventory.get("step") == STEP and checkpoint["sha256"] == E.CONTROL_CHECKPOINT_SHA256 and
            set(inventory["caches"]["T100-G"]) == set(DATASETS) == set(provenances), "not the registered control inventory")
    caches = {}
    for ds in DATASETS:
        source, provenance = inventory["caches"]["T100-G"][ds], json.loads(provenances[ds])
        require(digest(provenances[ds]) == source["provenance_sha256"] and provenance.get("cache") == source["path"] and
                provenance.get("cache_sha256") == source["sha256"] and source["checkpoint_sha256"] == checkpoint["sha256"] and
                type(provenance.get("cache_bytes")) is int, "control provenance differs from the inventory: " + ds)
        caches[ds] = {"path": source["path"], "sha256": source["sha256"], "bytes": provenance["cache_bytes"],
                      "provenance_path": source["provenance_path"], "provenance_sha256": source["provenance_sha256"],
                      "receipt_path": source["receipt_path"], "receipt_sha256": source["receipt_sha256"],
                      "checkpoint_path": checkpoint["path"], "checkpoint_sha256": source["checkpoint_sha256"],
                      "extraction_code_sha": source["extraction_code_sha"],
                      "extraction_plan_sha256": source["extraction_plan_sha256"],
                      **{k: source[k] for k in ("input_sha256", "splits_sha256", "extraction_recipe")}}
    return caches


def probe_plan(previous, control, candidate):
    """previous: reviewed 5,000 probe plan (datasets, references); control, candidate: the two cache grids."""
    require(previous.get("protocol") == "scaling-probe-20261104-v1" and previous.get("step") == 5000,
            "not the reviewed 5000 probe plan")
    plan = {"protocol": C.PROTOCOL, "step": STEP, "recipe": C.recipe(), "sources": C.source_hashes(),
            "caches": {ARMS[0]: control, ARM: candidate}, "datasets": previous["datasets"],
            "references": previous["references"], "splits": shared("splits", previous["splits"]),
            "training_source_contract": shared("training_source_contract", previous["training_source_contract"]),
            "projection": shared("projection", previous["projection"])}
    # The reader validates the same document on Leonardo; here with the identical local bytes of the shared files.
    local = {**plan, **{key: {**plan[key], "path": str(ROOT / SHARED[key])} for key in SHARED}}
    C.validate_plan(local)
    return plan


# ---------------------------------------------------------------------------------------------
# Scheduler accounting and measured budgets

def slurm_seconds(value):
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2})(?::(\d{2}))?", value)
    require(match is not None, "invalid scheduler duration")
    days, a, b, c = match.groups()
    h, m, s = (a, b, c) if c is not None else ("0", a, b)
    return int(days or 0) * 86400 + int(h) * 3600 + int(m) * 60 + int(s)


def byte_count(value):
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]?)", value)
    require(match is not None, "invalid scheduler memory")
    return round(float(match[1]) * 1024 ** ("KMGT".index(match[2]) + 1 if match[2] else 0))


def accounting(raw, job_id, cpus, seconds, gpus=0, raw_name="sacct.txt", need_rss=True):
    """Accounting of one completed job from the original `sacct -P` text; anything but COMPLETED 0:0 stops.
    need_rss: the measured peak memory is required (it feeds a later gate); otherwise it is recorded when present."""
    lines = raw.strip().splitlines()
    header = lines[0].split("|")
    required = {"JobIDRaw", "State", "ExitCode", "ElapsedRaw", "AllocCPUS", "MaxRSS"}
    require(required <= set(header) and len(set(header)) == len(header), "unexpected accounting columns")
    rows = [dict(zip(header, line.split("|"))) for line in lines[1:]]
    require(all(len(line.split("|")) == len(header) for line in lines[1:]), "malformed accounting row")
    job_id = str(job_id)
    require(job_id.isdigit() and rows and all(r["JobIDRaw"] == job_id or r["JobIDRaw"].startswith(job_id + ".") for r in rows),
            "accounting of another job")
    parents = [r for r in rows if r["JobIDRaw"] == job_id]
    require(len(parents) == 1, "one parent accounting row required")
    parent = parents[0]
    require(all(r["State"] == "COMPLETED" and r["ExitCode"] == "0:0" for r in rows), "job or step not COMPLETED 0:0")
    elapsed, allocated = int(parent["ElapsedRaw"]), int(parent["AllocCPUS"])
    require(allocated == cpus and 0 < elapsed <= seconds, "allocated CPUs or elapsed time outside the published job")
    if "Timelimit" in parent:
        require(slurm_seconds(parent["Timelimit"]) == seconds, "scheduler time limit differs from the published job")
    if "AllocTRES" in parent:
        tres = dict(x.split("=", 1) for x in parent["AllocTRES"].split(",") if "=" in x)
        require(tres.get("cpu") == str(cpus) and tres.get("node") == "1" and tres.get("gres/gpu", "0") == str(gpus),
                "allocated resources differ")
    measured = [(r["JobIDRaw"], byte_count(r["MaxRSS"])) for r in rows if r["MaxRSS"]]
    require(not need_rss or (measured and all(value > 0 for _, value in measured)), "no measured MaxRSS")
    measured = [(name, value) for name, value in measured if value > 0]
    peak = max((value for _, value in measured), default=None)
    result = {key: parent[key] for key in ("JobIDRaw", "State", "ExitCode")}
    result.update(ElapsedRaw=elapsed, AllocCPUS=allocated, MaxRSS=parent["MaxRSS"])
    result.update({key: parent[key] for key in ("NodeList", "Partition", "Start", "End", "Timelimit", "AllocTRES") if key in parent})
    result.update(MaxRSS_bytes=peak, MaxRSS_source_rows=[name for name, value in measured if value == peak],
                  core_hours=elapsed * cpus / 3600, cap_core_hours=seconds * cpus / 3600, gpus=gpus,
                  gpu_hours=elapsed * gpus / 3600, raw_sacct={"path": raw_name, "sha256": digest(raw.encode())})
    return result


def _pose_report(report, plan, plan_sha256, code_sha, protocol, step):
    require(report.get("protocol") == protocol and report.get("complete") is True and type(report.get("step")) is int and
            report["step"] == step and report.get("recipe") == plan["recipe"] and report.get("plan_sha256") == plan_sha256 and
            report.get("runtime", {}).get("code_sha") == code_sha and report["runtime"].get("sources") == plan["sources"],
            "pose report provenance differs")
    for key, field in (("input_pins", "caches"), ("reference_pins", "references"), ("projection_pin", "projection"),
                       ("training_source_contract", "training_source_contract")):
        require(report.get(key) == plan.get(field), "pose report inputs differ: " + key)


def pose_budget(plan, plan_sha256, code_sha, validation, benchmarks, protocol=P.PROTOCOL, step=STEP,
                resource_protocol=POSE_RESOURCE_PROTOCOL):
    """validation and benchmarks[cell]: {"report", "report_pin", "accounting", "accounting_pin"} of completed jobs.
    Same checks and formula as the dispatcher, which repeats them with the live runtime before the fit."""
    report, measured = validation["report"], validation["accounting"]
    _pose_report(report, plan, plan_sha256, code_sha, protocol, step)
    require(report.get("status") == "validated_input_metadata" and report.get("scientific_metrics_computed") is False and
            set(report.get("inspections", {})) == {arm + "/" + split for arm in ARMS for split in ("train", "val")},
            "complete four-cache validation required")
    require(measured.get("JobIDRaw") == report.get("slurm_job_id") and measured.get("State") == "COMPLETED" and
            measured.get("ExitCode") == "0:0" and measured.get("AllocCPUS") == 2 and 0 < measured["ElapsedRaw"] <= 1800 and
            0 < measured["MaxRSS_bytes"] <= 30 * 2**30, "validation accounting differs")
    require(set(benchmarks) == set(CELLS), "all three pose benchmarks required")
    estimate, runtimes = 2 * measured["ElapsedRaw"], []
    for cell in CELLS:
        report, measured = benchmarks[cell]["report"], benchmarks[cell]["accounting"]
        _pose_report(report, plan, plan_sha256, code_sha, protocol, step)
        require(measured.get("JobIDRaw") == report.get("slurm_job_id") and measured.get("State") == "COMPLETED" and
                measured.get("ExitCode") == "0:0" and measured.get("AllocCPUS") == 8 and 0 < measured["ElapsedRaw"] <= 600 and
                0 < measured["MaxRSS_bytes"] <= 30 * 2**30, "benchmark accounting differs: " + cell)
        arm = P.CELLS[cell][0]
        require(report.get("status") == "train_only_timing" and report.get("cell") == cell and
                set(report.get("inspections", {})) == {arm + "/train"}, "train-only pose benchmark identity differs")
        timings = report["timings"]
        require(timings.get("n_features") == E.dimension(arm if P.CELLS[cell][1] == "natural" else ARMS[0]) and
                timings.get("n_users") == 40 and timings.get("valid_fit_elements") == 4813620,
                "pose benchmark train support/dimension differs")
        s, e = timings.get("statistics_seconds"), timings.get("three_eigensystems_seconds")
        require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (s, e)), "invalid measured pose timings")
        estimate += 11 * s + 13 * e
        runtime = report["runtime"]
        runtimes.append({key: runtime.get(key) for key in ("versions", "threads", "torch_threads", "threadpools")})
        require(runtime.get("torch_threads") == 8 and all(v == "8" for v in runtime.get("threads", {}).values()) and
                runtime.get("threadpools") and all(t.get("num_threads") == 8 for t in runtime["threadpools"]),
                "eight-thread numeric runtime required")
    require(all(r == runtimes[0] for r in runtimes), "pose benchmarks ran in different numeric runtimes")
    needed = math.ceil((1.5 * estimate + 120) / 60) * 60
    require(needed <= 3600, "measured pose budget exceeds the frozen 3600 seconds: stop before the fit")
    return {"protocol": resource_protocol, "complete": True, "code_sha": code_sha, "plan_sha256": plan_sha256,
            "allocation": {"cpus": 8, "memory_gib": 30, "walltime_seconds": 3600},
            "validation_report": validation["report_pin"], "validation_accounting": validation["accounting_pin"],
            "benchmarks": {cell: benchmarks[cell]["report_pin"] for cell in CELLS},
            "benchmark_accounting": {cell: benchmarks[cell]["accounting_pin"] for cell in CELLS},
            "derived_seconds": needed, "full_live_runtime_gate_pending": True,
            "release_note": "Measured budget only. The dispatcher repeats every check with the live runtime; no job is submitted by this JSON."}


def _exact_cover(rectangles, bits):
    """Minimum number of shards, then minimum total rounded seconds, then lexicographic rectangle indices."""
    by_bit = {bit: [i for i, r in enumerate(rectangles) if r["mask"] & (1 << bit)] for bit in range(bits)}

    def objective(indices):
        return (len(indices), sum(rectangles[i]["walltime_seconds"] for i in indices), indices)

    @functools.lru_cache(None)
    def choose(uncovered):
        if not uncovered:
            return ()
        best = None
        for index in by_bit[(uncovered & -uncovered).bit_length() - 1]:
            mask = rectangles[index]["mask"]
            if mask & uncovered != mask:
                continue
            tail = choose(uncovered ^ mask)
            if tail is None:
                continue
            candidate = tuple(sorted((index,) + tail))
            if best is None or objective(candidate) < objective(best):
                best = candidate
        return best

    solution = choose((1 << bits) - 1)
    require(solution is not None, "no exact cover within the frozen seconds per shard: stop before the fits")
    return [rectangles[i] for i in solution], choose.cache_info().currsize


def _subsets(items):
    for size in range(1, len(items) + 1):
        yield from itertools.combinations(items, size)


def probe_resources(plan, plan_sha256, code_sha, reports, pins, protocol=C.PROTOCOL, step=STEP,
                    resource_protocol=C.RESOURCE_PROTOCOL, packing_protocol=PACKING_PROTOCOL):
    """reports[cell, dataset]: the fifteen completed train-only benchmarks; pins[cell][dataset]: their remote pins.
    Same formula, rounding and exact-cover packing as the reviewed 5,000 release; caps of the budget freeze v2."""
    stage = freeze()["stages"]["probe_fit"]
    cap, most, cpus, memory = stage["maximum_seconds_each"], stage["maximum_shards"], stage["cpus"], 30
    require(set(reports) == set(itertools.product(CELLS, DATASETS)), "all fifteen benchmarks required")
    supports = {}
    for (cell, ds), r in reports.items():
        require(r.get("protocol") == protocol and r.get("kind") == "benchmark" and r.get("complete") is True and
                (r.get("cell"), r.get("dataset")) == (cell, ds) and r.get("train_only") is True and
                r.get("selection_performed") is False and r.get("fold") == 0 and r.get("step") == step, "benchmark identity differs")
        require(r.get("code_sha") == code_sha and r.get("plan_sha256") == plan_sha256 and
                r.get("plan_content_sha256") == C.jhash(plan) and r.get("recipe") == plan["recipe"] and
                r.get("n_features") == plan["recipe"]["widths"][cell], "benchmark plan/code/recipe differs")
        require(r.get("numeric_sha256") == C.jhash(r.get("numeric_context")), "benchmark numeric signature differs")
        fits = r["fits"]
        require([f["C"] for f in fits] == plan["recipe"]["C_grid"] and all(
            f["converged_by_criterion"] is True and f["reached_limit"] is False and f["exception"] is None and
            not f["warnings"] and math.isfinite(f["elapsed_s"]) and f["elapsed_s"] > 0 and
            f["fit_phase"] == "benchmark_train_only" and f["selected"] is None for f in fits), "benchmark fits incomplete or not converged")
        counts = r["fold_train_counts"]
        require(len(counts) == 4 and counts[0] == r["n_train"] and all(type(c) is int and c > 0 for c in counts), "benchmark folds differ")
        support = (counts, r["roles_sha256"], fits[0]["classes"], fits[0]["class_counts"])
        require(supports.setdefault(ds, support) == support, "cells disagree on the train support: " + ds)
        require(type(r["peak_rss_bytes"]) is int and 0 < r["peak_rss_bytes"] * 1.3 <= memory * 2**30, "benchmark memory outside the cap")
        ST.pin_schema(pins[cell][ds])
        require(Path(pins[cell][ds]["path"]).is_absolute(), "benchmark pins must be remote absolute paths")
    numeric = {r["numeric_sha256"] for r in reports.values()}
    require(len(numeric) == 1, "benchmarks ran in different numeric contexts")
    shards, groups = [], []
    for cell in CELLS:
        for task, datasets in TASKS.items():
            rectangles = []
            for selected in _subsets(datasets):
                for folds in _subsets(tuple(range(4))):
                    needed, rss = 120.0, 0
                    for ds in selected:
                        r = reports[cell, ds]
                        times, counts = [f["elapsed_s"] for f in r["fits"]], r["fold_train_counts"]
                        needed += 1.5 * (sum(times) + max(times)) * sum(max(1.0, counts[f] / counts[0]) for f in folds)
                        rss = max(rss, r["peak_rss_bytes"])
                    seconds = math.ceil(needed / 60) * 60
                    if seconds <= cap:
                        mask = sum(1 << (datasets.index(ds) * 4 + f) for ds in selected for f in folds)
                        rectangles.append({"datasets": list(selected), "folds": list(folds), "mask": mask,
                                           "estimated_seconds_with_margin": needed, "walltime_seconds": seconds,
                                           "observed_peak_rss_bytes": rss})
            rectangles.sort(key=lambda r: (r["datasets"], r["folds"]))
            chosen, states = _exact_cover(rectangles, len(datasets) * 4)
            groups.append({"cell": cell, "task": task, "candidate_rectangles": len(rectangles), "dynamic_program_states": states,
                           "minimum_shards": len(chosen), "rounded_seconds": sum(r["walltime_seconds"] for r in chosen)})
            for index, rectangle in enumerate(chosen, 1):
                shard = {k: v for k, v in rectangle.items() if k != "mask"}
                shard.update(cell=cell, task=task, cpus=cpus, memory_gib=memory, shard_name=f"{cell}-{task}-{index:02d}")
                shards.append(shard)
    coverage = [(s["cell"], ds, f) for s in shards for ds in s["datasets"] for f in s["folds"]]
    require(len(coverage) == len(set(coverage)) == 60 and set(coverage) == set(itertools.product(CELLS, DATASETS, range(4))),
            "packing does not cover every cell, dataset and fold exactly once")
    seconds = sum(s["walltime_seconds"] for s in shards)
    require(len(shards) <= most, "more shards than the frozen maximum: stop before the fits")
    require(seconds * cpus <= stage["requested_core_hours"] * 3600, "fits exceed the frozen core-hours: stop before the fits")
    budget = {"protocol": resource_protocol, "complete": True, "code_sha": code_sha, "plan_content_sha256": C.jhash(plan),
              "plan_sha256": plan_sha256,
              "caps": {"cpus": cpus, "memory_gib": memory, "walltime_seconds": cap, "maximum_shards": most,
                       "maximum_allocated_core_hours": stage["requested_core_hours"]},
              "benchmarks": {cell: {ds: pins[cell][ds] for ds in DATASETS} for cell in CELLS},
              "numeric_sha256": next(iter(numeric)), "shards": shards}
    audit = {"protocol": packing_protocol, "complete": True,
             "method": "Exact-cover dynamic programming separately per cell/task; minimum shards, then minimum sum of "
                       "rounded seconds, then lexicographic rectangle indices",
             "formula": freeze()["measured_fit_gates"]["probe_formula"], "allocation_rounding_seconds": 60,
             "benchmarks_complete": len(reports), "covered_cells_dataset_folds": len(coverage), "groups": groups,
             "minimum_shards": len(shards), "total_requested_seconds": seconds, "requested_core_hours": seconds * cpus / 3600,
             "core_hours_with_scheduler_allowance": (seconds + len(shards) * 180) * cpus / 3600,
             "remaining_to_cap_requested_core_hours": stage["requested_core_hours"] - seconds * cpus / 3600,
             "no_scientific_scores_computed": True}
    return budget, audit


def oof_anchors(root=ROOT):
    """Identity of the out-of-fold windows of every dataset, from the six aggregates accepted at 5,000.
    It depends only on rows, labels, subjects and folds: the 19,618 chain must evaluate the same windows."""
    base, anchors = Path(root) / INFRA / "final_stages_20261106/collected_aggregates_20261106_1319", {}
    found = sorted(base.glob("SCALING-probe-aggregate-0?-20261106/aggregate.json"))
    require(len(found) == 6, "the six accepted 5000 aggregates are required")
    for path in found:
        for ds, d in json.loads(path.read_text())["datasets"].items():
            anchor = [d["identity_sha256"], d["n_oof_windows"], d["n_oof_subjects"]]
            require(anchors.setdefault(ds, anchor) == anchor, "5000 aggregates disagree on the evaluated windows: " + ds)
    require(set(anchors) == set(DATASETS), "anchors of the five datasets required")
    return anchors


def fit_state(resources, plan_sha256, code_sha, shards, protocol=C.PROTOCOL, step=STEP):
    """Gate of the reviewed 5,000 chain before the aggregates, on the collected shards.
    shards[name]: {"report", "oof": pin, "audit": pin}. Returns counts only; no score is read."""
    planned, grid = {s["shard_name"]: s for s in resources["shards"]}, FITS_PER_FOLD
    require(set(shards) == set(planned), "collected shards differ from the published fit plan")
    coverage, summaries = [], []
    for name, entry in shards.items():
        r, plan = entry["report"], planned[name]
        require(r.get("protocol") == protocol and r.get("kind") == "shard" and r.get("complete") is True and
                r.get("step") == step and r.get("code_sha") == code_sha and r.get("plan_sha256") == plan_sha256,
                "shard identity differs: " + name)
        folds = {ds: [f["fold"] for f in d["folds"]] for ds, d in r["datasets"].items()}
        require((r.get("cell"), r.get("task")) == (plan["cell"], plan["task"]) and sorted(folds) == sorted(plan["datasets"]) and
                all(f == plan["folds"] for f in folds.values()), "shard cell, task, datasets or folds differ from the fit plan: " + name)
        require(r.get("oof") == entry["oof"]["path"] and r.get("oof_sha256") == entry["oof"]["sha256"] and
                r.get("audit") == entry["audit"]["path"] and r.get("audit_sha256") == entry["audit"]["sha256"],
                "shard report does not pin the files next to it: " + name)
        require(r.get("numeric_sha256") == resources["numeric_sha256"], "shard numeric context differs from the benchmarks: " + name)
        require(r["audit_summary"]["fits"] == grid * sum(len(f) for f in folds.values()), "unexpected number of fits: " + name)
        coverage += [(plan["cell"], ds, f) for ds, fs in folds.items() for f in fs]
        summaries.append(r["audit_summary"])
    require(len(coverage) == len(set(coverage)) == 60 and set(coverage) == set(itertools.product(CELLS, DATASETS, range(4))),
            "fits do not cover every cell, dataset and fold exactly once")
    return {"shards": len(shards), "fits": sum(s["fits"] for s in summaries),
            "convergence_warnings": sum(s["convergence_warnings"] for s in summaries),
            "reached_limit": sum(s["reached_limit"] for s in summaries),
            "unresolved_fits": sum(len(s.get("unresolved_fits") or []) for s in summaries),
            "all_converged_by_criterion": all(s["converged_by_criterion"] is True for s in summaries),
            "numeric_sha256": resources["numeric_sha256"]}


def comparison_inputs(probe, pose, aggregates, pose_report, code_sha, shards, resources, anchors, benchmarks, module=M):
    """probe, pose: remote pins of the two plans. aggregates[cell, task]: {"report", "pin", "oof": pin, "audit": pin,
    "audit_lines"}; shards[name]: pin of each accepted shard; resources: published fit plan; anchors: oof_anchors();
    pose_report: {"report", "pin"}; benchmarks[cell]: the three pose benchmark reports.
    Same checks as the reviewed 5,000 gate before the comparison: identity, shards, audit, windows, convergence, runtime."""
    probes, poses, step, grid = module.C, module.P, module.C.STEP, len(module.C.C_GRID) + 1  # as FITS_PER_FOLD
    require(set(aggregates) == set(itertools.product(CELLS, TASKS)), "all six aggregates required")
    planned = {key: sorted(s["shard_name"] for s in resources["shards"] if (s["cell"], s["task"]) == key) for key in aggregates}
    require(set(shards) == {name for names in planned.values() for name in names}, "collected shards differ from the fit plan")
    contents = set()
    for (cell, task), entry in aggregates.items():
        r, label = entry["report"], "%s %s" % (cell, task)
        require(r.get("protocol") == probes.PROTOCOL and r.get("kind") == "aggregate" and r.get("complete") is True and
                (r.get("cell"), r.get("task")) == (cell, task) and [r.get("arm"), r.get("readout")] == cell.rsplit("-", 1) and
                r.get("step") == step and r.get("code_sha") == code_sha and r.get("plan_sha256") == probe["sha256"],
                "aggregate identity differs: " + label)
        require(r.get("numeric_sha256") == resources["numeric_sha256"], "aggregate numeric context differs from the fits: " + label)
        require(r.get("shards") == [{"path": shards[n]["path"], "sha256": shards[n]["sha256"]} for n in planned[cell, task]],
                "aggregate did not read the accepted shards: " + label)
        require(r.get("oof") == entry["oof"]["path"] and r.get("oof_sha256") == entry["oof"]["sha256"] and
                r.get("audit") == entry["audit"]["path"] and r.get("audit_sha256") == entry["audit"]["sha256"],
                "aggregate report does not pin the files next to it: " + label)
        summary = r["audit_summary"]
        require(summary["fits"] == grid * 4 * len(TASKS[task]) == entry["audit_lines"], "unexpected number of audited fits: " + label)
        require(sorted(r["datasets"]) == sorted(TASKS[task]) and all(
            [f["fold"] for f in d["folds"]] == [0, 1, 2, 3] and
            [d["identity_sha256"], d["n_oof_windows"], d["n_oof_subjects"]] == anchors[ds] for ds, d in r["datasets"].items()),
            "aggregate datasets, folds or evaluated windows differ: " + label)
        require(r.get("reading_status") == "solver_converged_by_criterion" and summary["converged_by_criterion"] is True and
                not summary.get("unresolved_fits"), "solver not converged by criterion, stop before the comparison: " + label)
        contents.add((r.get("plan_content_sha256"), r.get("projection_sha256")))
    require(len(contents) == 1 and all(next(iter(contents))), "aggregates read different plan contents or projections")
    r = pose_report["report"]
    require(r.get("protocol") == poses.PROTOCOL and r.get("complete") is True and r.get("status") == "completed" and
            r.get("step") == step and r.get("plan_sha256") == pose["sha256"] and
            r.get("runtime", {}).get("code_sha") == code_sha and set(r.get("cells", {})) == set(CELLS), "pose report identity differs")
    require(set(benchmarks) == set(CELLS) and all(
        r["runtime"].get(key) is not None and b.get("runtime", {}).get(key) == r["runtime"][key]
        for b in benchmarks.values() for key in RUNTIME_KEYS), "pose ridge ran in a runtime different from its benchmarks")
    inputs = module.make_inputs(probe, pose, [aggregates[cell, task]["pin"] for cell in CELLS for task in TASKS], pose_report["pin"])
    module.validate_inputs(inputs)
    require(all(Path(p["path"]).is_absolute() for p in (probe, pose, pose_report["pin"], *inputs["probe_reports"])),
            "comparison inputs require remote absolute paths")
    return inputs


# ---------------------------------------------------------------------------------------------
# Jobs and submission lines

def _short(cell):
    return cell.replace("T100-G-", "").lower()


def _job(stage, job_id, cpus, seconds, out, local, flags=(), files=None, gpus=0, memory="30G", pinned=None, folder=None,
         exact=None, log="scaling_readers19618"):
    """files: small remote files copied here; pinned: large ones, hashed only; folder: listed, and equal to `exact` if given."""
    return {"stage": stage, "id": job_id, "cpus": cpus, "memory": memory, "seconds": seconds, "gpus": gpus, "out": out,
            "local": local, "flags": list(flags), "files": dict(files if files is not None else {"report.json": out + "/report.json"}),
            "pinned": dict(pinned or {}), "folder": folder if folder is not None else out, "exact": exact, "log": log}


def training_job(training):
    """The reviewed training job as a row of the same table: nine receipts, the endpoint hashed and not copied."""
    out, name, allocation = training["receipt_dir"], f"checkpoint_step{STEP:07d}.pt", training["allocation"]
    names = ["config.json", "launch.json", *(f"rank_{n}.json" for n in range(4)), "receipt.json", "summary.json", "training.log"]
    return _job("training", RECOVERY["training"], allocation["cpus"], allocation["walltime_seconds"], out, "",
                files={n: f"{out}/{n}" for n in names}, gpus=allocation["gpus"], memory=allocation["memory"],
                pinned={name: f"{training['run_dir']}/{name}"}, exact=names, log="scaling_continue")


def precheck_job(training):
    """The CPU identity pre-check of the recovery plan (0 GPU): one receipt, required by the training job of the same plan and
    code (`scaling_continue.py identity-precheck`); collected as its own stage before the training is submitted."""
    gate = training["identity_precheck"]
    out, allocation = gate["receipt_dir"], gate["allocation"]
    return _job("identity-precheck", RECOVERY["precheck"], allocation["cpus"], allocation["walltime_seconds"], out, "",
                files={"identity_precheck.json": out + "/identity_precheck.json"}, memory=allocation["memory"],
                exact=["identity_precheck.json"], log="scaling_identity_precheck")


def checkout_job(name):
    """name: one of CHECKOUTS, or "training" for the frozen checkout of the recovery attempt (v2) of the training."""
    if name == "training":
        return _job("checkout", RECOVERY["checkout"], 2, 300, f"{WORK}/wearusfm_campaigns/{RECOVERY['campaign']}", name,
                    files={}, memory="4G", log="campaign_checkout")
    return _job("checkout", f"SCALING-19618-{name}-checkout-20261106", CHECKOUT["cpus"], CHECKOUT["seconds"], checkout(name), name,
                files={}, memory="%dG" % CHECKOUT["memory_gib"], log="campaign_checkout")


def jobs(stage, resources=None):
    """Job table of a stage: logical ID, allocation of the freeze, new output, flags, files to collect."""
    limits = freeze()["stages"]
    if stage == "extractions":
        table = []
        for row in json.loads((ROOT / FREEZE).read_text())["extractions"]["jobs"]:
            task = row["task"] or "pose"
            names = ("train_subset", "val") if task == "pose" else TASKS[task]
            out = f"{EXTRACT_ROOT}/{task}"
            files = {"receipt.json": out + "/receipt.json",
                     **{f"{n}.npz.provenance.json": f"{out}/{n}.npz.provenance.json" for n in names}}
            table.append(_job(stage, f"SCALING-300M-eval19618-{task}-20261106", row["cpus"], row["walltime_seconds"], out, task,
                              [row["arm"], row["stage"], row["task"] or "none"], files, row["gpus"], row["memory"],
                              log="scaling_evaluation19618"))
        return table
    if stage == "pose-validate":
        s = limits["pose_validate"]
        return [_job(stage, "SCALING-19618-pose-validate-20261106", s["cpus"], s["seconds"], CPU_ROOT + "/pose_validate", "")]
    if stage == "pose-benchmark":
        s = limits["pose_benchmark"]
        return [_job(stage, f"SCALING-19618-pose-bench-{_short(cell)}-20261106", s["cpus"], s["seconds_each"],
                     f"{CPU_ROOT}/pose_benchmark/{cell}", cell, ["--cell", cell]) for cell in CELLS]
    if stage == "probe-benchmark":
        s = limits["probe_benchmark"]
        return [_job(stage, f"SCALING-19618-probe-bench-{_short(cell)}-{ds.replace('_', '-')}-20261106", s["cpus"],
                     s["seconds_by_cell_and_dataset"][cell][ds], f"{CPU_ROOT}/probe_benchmark/{cell}-{ds}", f"{cell}/{ds}",
                     ["--cell", cell, "--dataset", ds],
                     {"benchmark.json": f"{CPU_ROOT}/probe_benchmark/{cell}-{ds}/{ds}/benchmark.json"},
                     folder=f"{CPU_ROOT}/probe_benchmark/{cell}-{ds}/{ds}")
                for cell in CELLS for ds in DATASETS]
    if stage == "pose-ridge":
        s = limits["pose_ridge"]
        return [_job(stage, "SCALING-19618-pose-ridge-20261106", s["cpus"], s["seconds"], CPU_ROOT + "/pose_ridge", "")]
    if stage == "probe-fit":
        require(resources is not None and resources.get("protocol") == C.RESOURCE_PROTOCOL, "published probe resources required")
        ordered = sorted(resources["shards"], key=lambda s: (s["walltime_seconds"], s["shard_name"]))
        return [_job(stage, f"SCALING-19618-probe-fit-{n:02d}-20261106", s["cpus"], s["walltime_seconds"],
                     f"{CPU_ROOT}/probe_fits/{s['shard_name']}", s["shard_name"],
                     ["--cell", s["cell"], "--task", s["task"], "--datasets", *s["datasets"], "--folds", *map(str, s["folds"])],
                     {name: f"{CPU_ROOT}/probe_fits/{s['shard_name']}/{name}" for name in ("shard.json", "audit.jsonl")},
                     pinned={"shard.oof.npz": f"{CPU_ROOT}/probe_fits/{s['shard_name']}/shard.oof.npz"}, exact=SHARD_FILES)
                for n, s in enumerate(ordered, 1)]
    if stage == "probe-aggregate":
        require(resources is not None and resources.get("protocol") == C.RESOURCE_PROTOCOL, "published probe resources required")
        s, table = limits["probe_aggregate"], []
        for n, (cell, task) in enumerate(itertools.product(CELLS, TASKS), 1):
            names = sorted(x["shard_name"] for x in resources["shards"] if (x["cell"], x["task"]) == (cell, task))
            folder = f"{CPU_ROOT}/probe_aggregates/{cell}-{task}"
            table.append(_job(stage, f"SCALING-19618-probe-aggregate-{n:02d}-20261106", s["cpus"], s["seconds_each"],
                              folder + "/aggregate.json", f"{cell}-{task}",
                              ["--cell", cell, "--task", task, "--shards",
                               *(f"{CPU_ROOT}/probe_fits/{name}/shard.json" for name in names)],
                              {name: f"{folder}/{name}" for name in ("aggregate.json", "aggregate.audit.jsonl")},
                              pinned={"aggregate.oof.npz": folder + "/aggregate.oof.npz"}, folder=folder, exact=AGGREGATE_FILES))
        return table
    require(stage == "compare", "unregistered stage")
    s, folder = limits["compare"], CPU_ROOT + "/comparison"
    return [_job(stage, "SCALING-19618-compare-20261106", s["cpus"], s["seconds"], folder + "/report.json", "",
                 files={"report.json": folder + "/report.json"}, folder=folder)]


def stage_table(stage, root=ROOT):
    """Table of any collectable stage, from the published documents of this repository."""
    if stage == "training":
        return [training_job(_load(RECOVERY["plan"], root))]
    if stage == "identity-precheck":
        return [precheck_job(_load(RECOVERY["plan"], root))]
    if stage.startswith("checkout-"):
        return [checkout_job(stage[len("checkout-"):])]
    resources = None
    if stage in ("probe-fit", "probe-aggregate"):
        resources = _load(f"{PACKAGE}/{DOCUMENTS['probe_resources'][1]}", root)
    return jobs(stage, resources)


def _submit(job_id, gpu_hours, sbatch, stdin=None, after=None):
    argv = [PYTHON, MAC + "/scripts/submit_job.py", "--agente", "claude", "--campagna", CAMPAIGN, "--timeout", "120",
            "--id", job_id, "--gpu-ore", "%g" % gpu_hours]
    argv += ["--script-stdin", stdin] if stdin else []
    argv += ["--dopo", after] if after else []
    return shlex.join([*argv, "--", *sbatch])


def checkout_lines(name, commit):
    """Deploy reference (never main, never a checked-out reference) and frozen campaign checkout."""
    require(ST.digest(commit, 40), "full commit SHA required")
    campaign, job = CHECKOUTS[name], checkout_job(name)
    flags = [f"--cpus-per-task={job['cpus']}", f"--mem={job['memory']}", f"--time={clock(job['seconds'])}",
             f"--export=ALL,CAMPAIGN={campaign},CAMPAIGN_SHA={commit},EXPECTED_SHA={commit}", "/dev/stdin"]
    return [shlex.join(["git", "-C", MAC, "push", "leonardo", f"{commit}:refs/heads/deploy/{campaign}"]),
            _submit(job["id"], 0, flags, stdin=MAC + "/scripts/slurm/campaign_checkout.sbatch")]


def recovery_lines(commit, root=ROOT):
    """Submission lines of the recovery attempt, in order, from its plan and its identity: deploy reference (never main), frozen
    checkout, identity pre-check (CPU, 0 GPU) and training (one segment, the frozen 59 GPU-hour cap). Sent one at a time, each
    only after the previous one ended as expected and was read; no afterok chain, no retry."""
    require(ST.digest(commit, 40), "full commit SHA required")
    plan, campaign = _load(RECOVERY["plan"], root), RECOVERY["campaign"]
    code, checkout, precheck, training = f"{WORK}/wearusfm_campaigns/{campaign}", checkout_job("training"), precheck_job(plan), training_job(plan)
    export = f"--export=ALL,CODE_DIR={code},EXPECTED_SHA={commit}"
    return [shlex.join(["git", "-C", MAC, "push", "leonardo", f"{commit}:refs/heads/deploy/{campaign}"]),
            _submit(checkout["id"], 0, [f"--cpus-per-task={checkout['cpus']}", f"--mem={checkout['memory']}",
                                        f"--time={clock(checkout['seconds'])}",
                                        f"--export=ALL,CAMPAIGN={campaign},CAMPAIGN_SHA={commit},EXPECTED_SHA={commit}", "/dev/stdin"],
                    stdin=MAC + "/scripts/slurm/campaign_checkout.sbatch"),
            _submit(precheck["id"], 0, [export, code + "/scripts/slurm/scaling_identity_precheck.sbatch", RECOVERY["plan"],
                                        precheck["out"]]),
            _submit(training["id"], training["gpus"] * training["seconds"] / 3600,
                    [export, code + "/scripts/slurm/scaling_continue.sbatch", RECOVERY["plan"], training["out"]])]


def stage_lines(stage, code_sha, root=ROOT, only=None):
    """Exact submission lines of a stage (or of the jobs `only`), from the published documents. CPU jobs sent together
    form one afterok chain, so the first failure stops the rest; the three extractions are independent."""
    require(ST.digest(code_sha, 40), "full code SHA required")
    code = checkout("code")
    export = f"--export=ALL,CODE_DIR={code},EXPECTED_SHA={code_sha}"
    lines, previous, chain = [], None, stage != "extractions"
    for job in stage_table(stage, root):
        require(job["stage"] not in ("training", "checkout"), "this stage has its own published lines")
        if only is not None and job["id"] not in only:
            continue
        if stage == "extractions":
            sbatch = [f"--time={clock(job['seconds'])}", export, code + "/scripts/slurm/scaling_evaluation19618.sbatch",
                      f"{PACKAGE}/{DOCUMENTS['extraction_plan'][1]}", *job["flags"], job["out"]]
        else:
            name = "comparison_inputs" if stage == "compare" else stage.split("-")[0] + "_plan"
            plan, extra = document(name, root), []
            if stage == "pose-ridge":
                budget = document("pose_budget", root)
                extra = ["--pose-budget", budget["path"], "--pose-budget-sha256", budget["sha256"]]
            elif stage == "probe-fit":
                budget = document("probe_resources", root)
                extra = ["--resource-plan", budget["path"], "--resource-sha256", budget["sha256"]]
            sbatch = [f"--cpus-per-task={job['cpus']}", f"--mem={job['memory']}", f"--time={clock(job['seconds'])}", export,
                      code + "/scripts/slurm/scaling_readers19618.sbatch", stage, plan["path"], plan["sha256"], job["out"],
                      *job["flags"], *extra]
        lines.append(_submit(job["id"], job["seconds"] * job["gpus"] / 3600, sbatch, after=previous if chain else None))
        previous = job["id"]
    return lines


def budget_table(shards=None):
    """Requested and scheduler-margin core-hours of the CPU-only stages against the freeze v2 (caps per stage and total)."""
    limits, rows = freeze(), {}
    overtime = limits["scheduler_overtime_seconds"]

    def row(name, table, cap):
        requested = sum(j["cpus"] * j["seconds"] for j in table) / 3600
        rows[name] = {"jobs": len(table), "requested_core_hours": requested,
                      "core_hours_with_scheduler_margin": sum(j["cpus"] * (j["seconds"] + overtime) for j in table) / 3600,
                      "cap_with_scheduler_margin": cap}

    stages = limits["stages"]
    row("checkout", [checkout_job(name) for name in ("training", *CHECKOUTS)], CHECKOUT_STAGE_CAP)  # frozen 0.8, amended 11/11
    for stage in ("pose-validate", "pose-benchmark", "probe-benchmark", "pose-ridge", "probe-aggregate", "compare"):
        table = jobs(stage, {"protocol": C.RESOURCE_PROTOCOL, "shards": []}) if stage == "probe-aggregate" else jobs(stage)
        row(stage, table, stages[stage.replace("-", "_")]["core_hours_with_scheduler_margin"])
    fit = stages["probe_fit"]
    each = fit["requested_core_hours"] * 3600 // (fit["cpus"] * fit["maximum_shards"])
    worst = [{"cpus": fit["cpus"], "seconds": each}] * fit["maximum_shards"]  # 240 core-hours in the most jobs: largest margin
    row("probe-fit", shards if shards is not None else worst, fit["core_hours_with_scheduler_margin"])
    total = sum(r["core_hours_with_scheduler_margin"] for r in rows.values())
    return {"stages": rows, "core_hours_with_scheduler_margin": total, "cap": limits["cpu_only_core_hours_cap"],
            "within_stage_caps": all(r["core_hours_with_scheduler_margin"] <= r["cap_with_scheduler_margin"] + 1e-9 for r in rows.values()),
            "within_total_cap": total <= limits["cpu_only_core_hours_cap"]}


# ---------------------------------------------------------------------------------------------
# Collection and live gates: bounded read-only access to Leonardo, then local files under PACKAGE/evidence/<stage>/

REMOTE = r'''
import base64, hashlib, json, os, subprocess
P = json.loads(__PAYLOAD__)
def read(path, limit):
    if not os.path.isfile(path):
        return {"path": path, "missing": True}
    h, n, kept = hashlib.sha256(), 0, []
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk); n += len(chunk)
            if n <= limit:
                kept.append(chunk)
    out = {"path": path, "sha256": h.hexdigest(), "bytes": n, "canonical": os.path.realpath(path) == path}
    if n <= limit:
        out["content"] = base64.b64encode(b"".join(kept)).decode()
    return out
def run(cmd):
    cmd = [os.environ.get("USER", "") if x == "@USER" else x for x in cmd]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                       env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
    return {"exit": r.returncode, "stdout": r.stdout, "stderr": r.stderr[-300:]}
print(json.dumps({"files": [read(p, P["limit"]) for p in P["files"]] + [read(p, -1) for p in P["hashed"]],
                  "listings": dict((d, sorted(os.listdir(d)) if os.path.isdir(d) else None) for d in P["listings"]),
                  "exists": dict((p, os.path.lexists(p)) for p in P["exists"]),
                  "sacct": dict((j, run(["sacct", "-P", "-j", j, "--format=" + P["format"]])) for j in P["jobs"]),
                  "commands": dict((name, run(cmd)) for name, cmd in sorted(P["commands"].items())),
                  "time_utc": run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"])["stdout"].strip()}))
'''
QUEUE = ["squeue", "-h", "-u", "@USER", "-p", "lrd_all_serial", "-o", "%i|%j|%T|%C|%r"]
QUOTA = ["lfs", "quota", "-p", WORK_PROJECT, WORK]
GIT_READS = (["rev-parse", "HEAD"], ["status", "--porcelain", "--untracked-files=all"])


def _git(directory, *arguments):
    return ["git", "-C", directory, *arguments]


def remote_read(files=(), slurm_jobs=(), limit=INLINE_LIMIT, ssh=SSH, hashed=(), listings=(), exists=(), commands=None):
    """One read-only request: file hashes (content when small), folder listings, existence, accounting and a fixed
    set of read-only commands (HEAD and status of a checkout, the serial queue of the user, the project quota)."""
    payload = {"files": list(files), "hashed": list(hashed), "listings": list(listings), "exists": list(exists),
               "jobs": [str(j) for j in slurm_jobs], "commands": dict(commands or {}), "limit": limit, "format": SACCT_FORMAT}
    require(all(cmd in (QUEUE, QUOTA) or (cmd[:2] == ["git", "-C"] and cmd[3:] in GIT_READS) for cmd in payload["commands"].values()),
            "only the registered read-only commands are sent")
    script = REMOTE.replace("__PAYLOAD__", repr(json.dumps(payload)))
    done = subprocess.run([*ssh, "python3 -"], input=script, capture_output=True, text=True, timeout=1800)
    require(done.returncode == 0, "remote read failed (%s): %s" % (done.returncode, done.stderr.strip()[-400:]))
    reading = json.loads(done.stdout)
    require([f["path"] for f in reading["files"]] == payload["files"] + payload["hashed"] and
            set(reading["sacct"]) == set(payload["jobs"]) and set(reading["commands"]) == set(payload["commands"]) and
            set(reading["listings"]) == set(payload["listings"]) and set(reading["exists"]) == set(payload["exists"]),
            "remote read answered another request")
    return reading


def ledger_job(job_id, root=LEDGER):
    state = json.loads((Path(root) / job_id / "state.json").read_text())
    require(state.get("stato") == "sottomesso" and str(state.get("job_slurm", "")).isdigit() and
            state.get("agente") == "claude" and state.get("campagna") == CAMPAIGN, "job not submitted by this line: " + job_id)
    return str(state["job_slurm"])


def checkout_commit(name, root=LEDGER):
    """Commit of a campaign checkout, from the receipt of its submission in the common ledger."""
    job = checkout_job(name)
    ledger_job(job["id"], root)
    exports = [x for x in json.loads((Path(root) / job["id"] / "state.json").read_text())["sbatch"] if x.startswith("--export=")]
    require(len(exports) == 1, "unexpected checkout submission: " + job["id"])
    fields = dict(x.split("=", 1) for x in exports[0][len("--export="):].split(",") if "=" in x)
    require(fields.get("CAMPAIGN") == Path(job["out"]).name and fields.get("CAMPAIGN_SHA") == fields.get("EXPECTED_SHA") and
            ST.digest(fields.get("CAMPAIGN_SHA", ""), 40), "unexpected checkout submission: " + job["id"])
    return fields["CAMPAIGN_SHA"]


def log_path(job, slurm):
    return f"{LOGS}/{job['log']}_{slurm}.out"


def collect(stage, table, slurm_ids, reading, directory):
    """Validate a whole stage (every job COMPLETED 0:0 within its allocation, every file present and canonical, every
    output folder as expected) and only then write its evidence: original sacct text, accounting, files, logs and
    manifest. Large files are pinned, not copied. Nothing is overwritten."""
    by_path = {f["path"]: f for f in reading["files"]}
    writes, manifest = {}, {"stage": stage, "remote_time_utc": reading["time_utc"], "jobs": {}, "files": {}}
    for job in table:
        slurm = slurm_ids[job["id"]]
        answer = reading["sacct"][slurm]
        require(answer["exit"] == 0, "sacct failed: " + job["id"])
        prefix = job["local"] + "/" if job["local"] else ""
        record = accounting(answer["stdout"], slurm, job["cpus"], job["seconds"], job["gpus"], need_rss=stage in RSS_GATED)
        writes[prefix + "sacct.txt"] = answer["stdout"].encode()
        writes[prefix + "accounting.json"] = dumps(record)
        listing = reading["listings"][job["folder"]]
        require(listing is not None and (job["exact"] is None or listing == sorted(job["exact"])),
                "unexpected content of the output folder: " + job["folder"])
        manifest["jobs"][job["id"]] = {"job_slurm": slurm, "local": job["local"], "elapsed_seconds": record["ElapsedRaw"],
                                       "core_hours": record["core_hours"], "gpu_hours": record["gpu_hours"],
                                       "folder": job["folder"], "listing": listing}
        wanted = [*((name, path, "copy") for name, path in job["files"].items()),
                  *((name, path, "pin") for name, path in job["pinned"].items()), ("slurm_log.txt", log_path(job, slurm), "log")]
        for name, path, kind in wanted:
            found = by_path[path]
            if kind == "log" and found.get("missing"):
                manifest["files"][prefix + name] = {"path": path, "stored": "absent"}
                continue
            require(not found.get("missing") and found.get("canonical") is True, "missing or non-canonical remote file: " + path)
            entry = {"path": path, "sha256": found["sha256"], "bytes": found["bytes"], "stored": "pin-only"}
            soft = kind == "log" or name.endswith(".log")  # a log too large to travel is pinned, not a failed stage
            if kind != "pin" and ("content" in found or not soft):
                require("content" in found, "remote file larger than the inline limit: " + path)
                data = base64.b64decode(found["content"])
                require(digest(data) == found["sha256"] and len(data) == found["bytes"], "transfer differs from the remote hash: " + path)
                entry["stored"] = "gzip" if len(data) >= GZIP_FROM else "plain"
                writes[prefix + name + (".gz" if entry["stored"] == "gzip" else "")] = (
                    gzip.compress(data, mtime=0) if entry["stored"] == "gzip" else data)
            manifest["files"][prefix + name] = entry
    writes["collection.json"] = dumps(manifest)
    directory = Path(directory)
    require(not directory.exists(), "evidence of this stage already collected: " + str(directory))
    for name, data in writes.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    return manifest


def evidence_dir(stage, root=ROOT):
    """Folder of the collected evidence of a stage; the recovery checkout has its own, the v1 one keeps its record."""
    return Path(root) / PACKAGE / "evidence" / EVIDENCE_DIRS.get(stage, stage)


def evidence(stage, root=ROOT):
    """Collected files of a stage as {name: (original bytes or None, remote pin)}, checked against the manifest."""
    directory = evidence_dir(stage, root)
    manifest, out = json.loads((directory / "collection.json").read_text()), {}
    for name, entry in manifest["files"].items():
        if entry["stored"] == "absent":
            continue
        data = None
        if entry["stored"] != "pin-only":
            raw = (directory / (name + (".gz" if entry["stored"] == "gzip" else ""))).read_bytes()
            data = gzip.decompress(raw) if entry["stored"] == "gzip" else raw
            require(digest(data) == entry["sha256"] and len(data) == entry["bytes"], "collected file differs from its pin: " + name)
        out[name] = (data, {key: entry[key] for key in ("path", "sha256", "bytes")})
    return out, manifest


def accounting_pin(stage, local, root=ROOT):
    """The accounting JSON is distributed by the budgets checkout: remote path there, local bytes here."""
    relative = f"{PACKAGE}/evidence/{EVIDENCE_DIRS.get(stage, stage)}/{local + '/' if local else ''}accounting.json"
    data = (Path(root) / relative).read_bytes()
    return json.loads(data), pin(f"{checkout('budgets')}/{relative}", data)


def lane(reading):
    """Serial queue of the user (shared with the other lines) and free space of the project on $WORK."""
    queue, quota = reading["commands"]["squeue"], reading["commands"]["quota"]
    require(queue["exit"] == 0 and quota["exit"] == 0, "queue or quota reading failed: no count, no gate")
    tokens = quota["stdout"].split()
    require("prj " + WORK_PROJECT in quota["stdout"] and tokens.count(WORK) == 1, "unexpected quota answer")
    used, limit = (int(x.rstrip("*")) for x in tokens[tokens.index(WORK) + 1:][:2])
    require(limit > 0, "unexpected quota answer")
    waiting = [line.split("|") for line in queue["stdout"].splitlines() if line.strip()]
    mine = [row for row in waiting if len(row) > 1 and row[1].startswith(LINE_JOB_PREFIX)]  # this line's jobs already there
    return {"serial_jobs": len(waiting), "serial_room": SERIAL_SUBMIT_LIMIT - len(waiting), "line_jobs": len(mine),
            "work_free_gib": (limit - used) / 2**20}


def _blob_sha256(commit, relative, root=ROOT):
    done = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{relative}"], capture_output=True)
    return digest(done.stdout) if done.returncode == 0 else None


def checkout_pins(value, commits, own, root=ROOT):
    """Every pin of a document that points into a campaign checkout: (path, pinned hash, hash of that file in the commit
    of that checkout). A relative path is read in the checkout `own` of the document. Shared files, plans and
    accounting travel only inside checkouts."""
    found = []
    if isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str) and "sha256" in value:
            if not path.startswith("/"):
                found.append((path, value["sha256"], _blob_sha256(commits[own], path, root)))
            for name in CHECKOUTS:
                if path.startswith(checkout(name) + "/"):
                    found.append((path, value["sha256"], _blob_sha256(commits[name], path[len(checkout(name)) + 1:], root)))
        value = list(value.values())
    for item in value if isinstance(value, list) else []:
        found += checkout_pins(item, commits, own, root)
    return found


def release_check(reviewed, commit, root=ROOT, since="the review"):
    """A checkout created after the review carries the reviewed code: the commits after the reviewed package only add
    evidence and documents of this chain, or edit documentation. The same holds after every earlier checkout of the
    chain: a document already deployed is never rewritten."""
    require(ST.digest(reviewed, 40) and ST.digest(commit, 40), "full commit SHAs required")
    def git(*arguments):
        return subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True)
    require(git("merge-base", "--is-ancestor", reviewed, commit).returncode == 0, "commit does not descend from " + since)
    changed = git("diff", "--name-status", "--no-renames", reviewed, commit)
    require(changed.returncode == 0, "cannot compare with the reviewed package")
    added = tuple(f"{PACKAGE}/{folder}/" for folder in ("evidence", "evaluation", "readers", "submissions"))
    for line in changed.stdout.splitlines():
        status, path = line.split("\t", 1)
        require(path.startswith("docs/") or (status == "A" and path.startswith(added)),
                "change outside the documents of the chain after %s: %s" % (since, line))
    return len(changed.stdout.splitlines())


def _is_ancestor(ancestor, descendant, root=ROOT):
    return subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
                          capture_output=True, text=True).returncode == 0


def rereview_check(previous, reviewed, root=ROOT):
    """A review made after an earlier checkout covers the changes since that checkout only if the cluster side stays as it was:
    the Mac-side operator tool and the CPU tests, documentation, and files added under the campaign folder. A document already
    deployed is never rewritten, and code read by the jobs never changes after the checkout that carries it."""
    require(ST.digest(previous, 40) and ST.digest(reviewed, 40), "full commit SHAs required")
    require(_is_ancestor(previous, reviewed, root), "the review does not descend from the earlier checkout")
    changed = subprocess.run(["git", "-C", str(root), "diff", "--name-status", "--no-renames", previous, reviewed],
                             capture_output=True, text=True)
    require(changed.returncode == 0, "cannot compare the review with the earlier checkout")
    for line in changed.stdout.splitlines():
        status, path = line.split("\t", 1)
        require(path in MAC_SIDE or path.startswith(("tests/cpu/", "docs/")) or (status == "A" and path.startswith(INFRA + "/")),
                "change on the cluster side after an earlier checkout: " + line)
    return len(changed.stdout.splitlines())


def gate_request(stage, root=ROOT, ledger=LEDGER):
    """What the gate of a job stage reads: its checkouts, its published documents, its outputs, queue and quota."""
    table = stage_table(stage, root)
    documents = {name: document(name, root) for name in STAGE_DOCUMENTS[stage]}
    commits = {name: checkout_commit(name, ledger) for name in sorted({"code", *(DOCUMENTS[d][0] for d in documents)})}
    commands = {"squeue": QUEUE, "quota": QUOTA}
    for name in commits:
        commands["head:" + name], commands["status:" + name] = (_git(checkout(name), *arguments) for arguments in GIT_READS)
    request = {"hashed": [p["path"] for p in documents.values()], "exists": [job["out"] for job in table], "commands": commands}
    submitted = {}
    for job in table:
        try:
            submitted[job["id"]] = ledger_job(job["id"], ledger)
        except (ValueError, OSError):
            pass
    return request, commits, documents, table, submitted


def _completed(answer, slurm):
    lines = answer["stdout"].strip().splitlines() if answer["exit"] == 0 else []
    rows = [dict(zip(lines[0].split("|"), line.split("|"))) for line in lines[1:]]
    return any(r.get("JobIDRaw") == slurm and r.get("State") == "COMPLETED" and r.get("ExitCode") == "0:0" for r in rows)


def _collected_checkout(name, ledger, root):
    """The checkout job was collected after COMPLETED 0:0: its accounting is among the evidence with its Slurm ID."""
    try:
        stage = "checkout-" + name
        record = _load(f"{PACKAGE}/evidence/{EVIDENCE_DIRS.get(stage, stage)}/{name}/accounting.json", root)
        return (record.get("JobIDRaw") == ledger_job(checkout_job(name)["id"], ledger) and record.get("State") == "COMPLETED" and
                record.get("ExitCode") == "0:0")
    except (OSError, ValueError):
        return False


def gate(reading, commits, documents, table, submitted, root=ROOT, ledger=LEDGER, line_limit=LINE_SERIAL_LIMIT):
    """Mechanical gate before a tranche of a job stage: (checks, facts). Nothing is submitted here.
    submitted: {logical ID: Slurm ID} of the jobs of the stage already sent; they must have completed.
    The admitted tranche (facts["tranche"]) is what `lines` may emit: the jobs still to submit that fit in the free
    places of the serial queue and in the limit of this line, counting the jobs of this line already there."""
    commands, files, state = reading["commands"], {f["path"]: f for f in reading["files"]}, lane(reading)
    todo = [job for job in table if not (Path(ledger) / job["id"]).exists()]

    pins = [p for name in documents
            for p in checkout_pins(_load(f"{PACKAGE}/{DOCUMENTS[name][1]}", root), commits, DOCUMENTS[name][0], root)]
    checks = [("checkout %s: expected HEAD and clean tree" % name,
               commands["head:" + name]["exit"] == 0 and commands["head:" + name]["stdout"].strip() == commit and
               commands["status:" + name]["exit"] == 0 and commands["status:" + name]["stdout"].strip() == "")
              for name, commit in sorted(commits.items())]
    checks += [("document %s: remote bytes equal to the published ones" % name,
                files.get(p["path"], {}).get("sha256") == p["sha256"] and files[p["path"]].get("bytes") == p["bytes"] and
                files[p["path"]].get("canonical") is True) for name, p in sorted(documents.items())]
    checks += [("files pinned inside the checkouts are in their commits (%d)" % len(pins), all(a == b for _, a, b in pins)),
               ("outputs of the jobs still to submit are absent", all(reading["exists"][job["out"]] is False for job in todo)),
               ("jobs already in the ledger were submitted by this line", all(job["id"] in submitted for job in table if job not in todo)),
               ("jobs already submitted are COMPLETED 0:0 (%d)" % len(submitted),
                all(_completed(reading["sacct"][slurm], slurm) for slurm in submitted.values())),
               ("checkouts used were collected COMPLETED 0:0 (%d)" % len(commits),
                all(_collected_checkout(name, ledger, root) for name in commits)),
               ("free space on $WORK at least %d GiB" % WORK_FREE_GIB, state["work_free_gib"] >= WORK_FREE_GIB)]
    serial = all(job["gpus"] == 0 for job in table)
    room = max(0, min(len(todo), state["serial_room"], line_limit - state["line_jobs"])) if serial else len(todo)
    tranche = [job["id"] for job in todo[:room]]
    checks.append(("a tranche of at least one job is admitted (%d of %d to submit)" % (len(tranche), len(todo)), len(tranche) >= 1))
    return checks, {**state, "to_submit": [job["id"] for job in todo], "line_limit": line_limit, "submittable_now": room,
                    "tranche": tranche}


def checkout_gate(name, commit, reviewed, reading, deploy_ref, root=ROOT, ledger=LEDGER, line_limit=LINE_SERIAL_LIMIT):
    """Gate before the deploy reference and the checkout job of a document commit: (checks, facts)."""
    changed, order = release_check(reviewed, commit, root), list(CHECKOUTS)
    earlier = {c: checkout_commit(c, ledger) for c in order if c != name and (Path(ledger) / checkout_job(c)["id"]).exists()}
    require(set(earlier) == set(order[:order.index(name)]), "checkouts of the chain are created in order, each one once")
    reviewed_after = []
    for earlier_name, previous in earlier.items():
        if previous != reviewed and _is_ancestor(previous, reviewed, root):
            rereview_check(previous, reviewed, root)  # what follows the review is the release_check(reviewed, commit) above
            reviewed_after.append(earlier_name)
        else:
            release_check(previous, commit, root, "an earlier checkout")
    commits = {**earlier, name: commit}
    own = [d for d, (where, _) in DOCUMENTS.items() if where == name]
    blobs = {d: _blob_sha256(commit, f"{PACKAGE}/{DOCUMENTS[d][1]}", root) for d in own}
    local = {d: digest((Path(root) / PACKAGE / DOCUMENTS[d][1]).read_bytes()) for d in own}
    pins = [p for d in own for p in checkout_pins(_load(f"{PACKAGE}/{DOCUMENTS[d][1]}", root), commits, name, root)]
    state, job = lane(reading), checkout_job(name)
    checks = [("commit adds only documents and evidence to the reviewed package (%d paths)" % changed, True),
              ("documents of this checkout are in the commit with the published bytes", bool(own) and blobs == local),
              ("files pinned inside the checkouts are in their commits (%d)" % len(pins), all(a == b for _, a, b in pins)),
              ("deploy reference absent or already at this commit", deploy_ref in (None, commit)),
              ("checkout folder absent", reading["exists"][job["out"]] is False),
              ("logical ID never used in the ledger", not (Path(ledger) / job["id"]).exists()),
              ("one free place in the serial queue", state["serial_room"] >= 1),
              ("line limit %d leaves a place for this checkout" % line_limit, state["line_jobs"] < line_limit),
              ("free space on $WORK at least %d GiB" % WORK_FREE_GIB, state["work_free_gib"] >= WORK_FREE_GIB)]
    return checks, {**state, "line_limit": line_limit, "tranche": [job["id"]], "reviewed_after_checkouts": reviewed_after}


# ---------------------------------------------------------------------------------------------
# Command line: every document is created once, from the fixed local layout

def _write(relative, data, root=ROOT):
    target = Path(root) / PACKAGE / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"written": f"{PACKAGE}/{relative}", "sha256": digest(data), "bytes": len(data)}))


def _load(relative, root=ROOT):
    return json.loads((Path(root) / relative).read_text())


def _extraction(root=ROOT):
    relative = DOCUMENTS["extraction_plan"][1]
    data = (Path(root) / PACKAGE / relative).read_bytes()
    return json.loads(data)["checkpoints"][ARM], digest(data)


def build_extraction_plan(root=ROOT):
    files, manifest = evidence("training", root)
    data = (Path(root) / RECOVERY["plan"]).read_bytes()
    training, receipt = json.loads(data), json.loads(files["receipt.json"][0])
    require(list(manifest["jobs"]) == [RECOVERY["training"]] and
            str(receipt.get("slurm_job_id")) == manifest["jobs"][RECOVERY["training"]]["job_slurm"],
            "the collected training is not the recovery job")
    pins = {name: files[key][1] for name, key in (("checkpoint", f"checkpoint_step{STEP:07d}.pt"), ("config", "config.json"),
                                                  ("receipt", "receipt.json"), ("summary", "summary.json"))}
    plan = extraction_plan(_load(INFRA + "/extraction_plan_5000_20261105.json", root), training, pins, receipt,
                           json.loads(files["summary.json"][0]), digest(data))
    _write(DOCUMENTS["extraction_plan"][1], dumps(plan), root)


def build_reader_plans(code_sha, root=ROOT):
    files, _ = evidence("extractions", root)
    checkpoint, plan_sha = _extraction(root)
    extraction = {"checkpoint": checkpoint, "code_sha": code_sha, "plan_sha256": plan_sha}
    pose = candidate_pose_caches(json.loads(files["pose/receipt.json"][0]), files["pose/receipt.json"][1],
                                 {"train": files["pose/train_subset.npz.provenance.json"][0],
                                  "val": files["pose/val.npz.provenance.json"][0]}, extraction)
    task_of = {ds: task for task, datasets in TASKS.items() for ds in datasets}
    probe = candidate_probe_caches({t: json.loads(files[t + "/receipt.json"][0]) for t in TASKS},
                                   {t: files[t + "/receipt.json"][1] for t in TASKS},
                                   {ds: files[f"{task_of[ds]}/{ds}.npz.provenance.json"][0] for ds in DATASETS}, extraction)
    control_dir = Path(root) / PACKAGE / "control19618_identity"
    control = control_probe_caches(_load("results/infra/tokenizer_evaluation_20261103/probe_plan_19618.json", root),
                                   {ds: (control_dir / f"{ds}.npz.provenance.json").read_bytes() for ds in DATASETS},
                                   _load(RECOVERY["plan"], root)["control"]["checkpoint"])
    reviewed = INFRA + "/cpu_readers_20261105/"
    _write(DOCUMENTS["pose_plan"][1], dumps(pose_plan(
        _load(reviewed + "pose_plan.json", root),
        _load("results/infra/tokenizer_evaluation_20261103/pose_readout_plan_19618.json", root), pose)), root)
    _write(DOCUMENTS["probe_plan"][1], dumps(probe_plan(_load(reviewed + "probe_plan.json", root), control, probe)), root)


def build_budgets(code_sha, root=ROOT):
    def measured(stage, name, local):
        files, _ = evidence(stage, root)
        record, record_pin = accounting_pin(stage, local, root)
        data, report_pin = files[(local + "/" if local else "") + name]
        return {"report": json.loads(data), "report_pin": report_pin, "accounting": record, "accounting_pin": record_pin}

    pose = document("pose_plan", root)
    budget = pose_budget(_load(f"{PACKAGE}/{DOCUMENTS['pose_plan'][1]}", root), pose["sha256"], code_sha,
                         measured("pose-validate", "report.json", ""),
                         {cell: measured("pose-benchmark", "report.json", cell) for cell in CELLS})
    files, _ = evidence("probe-benchmark", root)
    probe = document("probe_plan", root)
    resources, audit = probe_resources(
        _load(f"{PACKAGE}/{DOCUMENTS['probe_plan'][1]}", root), probe["sha256"], code_sha,
        {(cell, ds): json.loads(files[f"{cell}/{ds}/benchmark.json"][0]) for cell in CELLS for ds in DATASETS},
        {cell: {ds: files[f"{cell}/{ds}/benchmark.json"][1] for ds in DATASETS} for cell in CELLS})
    _write(DOCUMENTS["pose_budget"][1], dumps(budget), root)
    _write(DOCUMENTS["probe_resources"][1], dumps(resources), root)
    _write("readers/probe_packing_audit.json", dumps(audit), root)


def build_fit_state(code_sha, root=ROOT):
    """Acceptance of the collected fits against the published fit plan; raises before any aggregate otherwise."""
    files, _ = evidence("probe-fit", root)
    resources = _load(f"{PACKAGE}/{DOCUMENTS['probe_resources'][1]}", root)
    shards = {name: {"report": json.loads(files[name + "/shard.json"][0]), "oof": files[name + "/shard.oof.npz"][1],
                     "audit": files[name + "/audit.jsonl"][1]} for name in (s["shard_name"] for s in resources["shards"])}
    return fit_state(resources, document("probe_plan", root)["sha256"], code_sha, shards)


def build_comparison_inputs(code_sha, root=ROOT):
    build_fit_state(code_sha, root)
    files, _ = evidence("probe-aggregate", root)
    fits, _ = evidence("probe-fit", root)
    ridge, _ = evidence("pose-ridge", root)
    benchmarks, _ = evidence("pose-benchmark", root)
    resources = _load(f"{PACKAGE}/{DOCUMENTS['probe_resources'][1]}", root)
    aggregates = {}
    for cell, task in itertools.product(CELLS, TASKS):
        base = f"{cell}-{task}/aggregate"
        aggregates[cell, task] = {"report": json.loads(files[base + ".json"][0]), "pin": files[base + ".json"][1],
                                  "oof": files[base + ".oof.npz"][1], "audit": files[base + ".audit.jsonl"][1],
                                  "audit_lines": len(files[base + ".audit.jsonl"][0].splitlines())}
    inputs = comparison_inputs(
        document("probe_plan", root), document("pose_plan", root), aggregates,
        {"report": json.loads(ridge["report.json"][0]), "pin": ridge["report.json"][1]}, code_sha,
        {s["shard_name"]: fits[s["shard_name"] + "/shard.json"][1] for s in resources["shards"]}, resources, oof_anchors(root),
        {cell: json.loads(benchmarks[cell + "/report.json"][0]) for cell in CELLS})
    _write(DOCUMENTS["comparison_inputs"][1], dumps(inputs), root)


def run_collect(stage, root=ROOT):
    table = stage_table(stage, root)
    slurm_ids = {job["id"]: ledger_job(job["id"]) for job in table}
    reading = remote_read([path for job in table for path in job["files"].values()] +
                          [log_path(job, slurm_ids[job["id"]]) for job in table], slurm_ids.values(),
                          hashed=[path for job in table for path in job["pinned"].values()],
                          listings=[job["folder"] for job in table])
    manifest = collect(stage, table, slurm_ids, reading, evidence_dir(stage, root))
    print(json.dumps({"collected": stage, "jobs": manifest["jobs"], "files": len(manifest["files"])}, indent=1))


def gate_record(stage, checks, facts, reading, context):
    """What a gate read and concluded, kept among the evidence: (path inside the package, bytes)."""
    stamp = reading["time_utc"].replace("-", "").replace(":", "")
    require(re.fullmatch(r"\d{8}T\d{6}Z", stamp) is not None, "unexpected time of the remote reading")
    record = {"protocol": GATE_PROTOCOL, "stage": stage, "read_at_utc": reading["time_utc"],
              "passed": all(ok for _, ok in checks), "checks": [[name, bool(ok)] for name, ok in checks], "facts": facts, **context}
    return f"evidence/gates/{stage}_{stamp}.json", dumps(record)


def run_gate(stage, commit=None, reviewed=None, root=ROOT, ledger=LEDGER, line_limit=LINE_SERIAL_LIMIT):
    """Reads the live state once (read-only), prints every check and records the outcome among the evidence;
    exit status 1 if any check fails. A refusal raised before the checks leaves its own record and is re-raised."""
    try:
        reading, checks, facts, context = _gate_reading(stage, commit, reviewed, root, ledger, line_limit)
    except Exception as error:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(_now()))
        _write(f"evidence/gates/{stage}_{stamp}_refused.json", dumps({
            "protocol": GATE_PROTOCOL, "stage": stage, "passed": False, "recorded_at_utc": stamp,
            "refused": "%s: %s" % (type(error).__name__, error), "commit": commit, "reviewed": reviewed}), root)
        raise
    for name, ok in checks:
        print("%s  %s" % ("PASS" if ok else "FAIL", name))
    print(json.dumps(facts, indent=1))
    _write(*gate_record(stage, checks, facts, reading, context), root)
    return 0 if all(ok for _, ok in checks) else 1


def _gate_reading(stage, commit, reviewed, root, ledger, line_limit):
    if stage.startswith("checkout-"):
        name = stage[len("checkout-"):]
        require(commit is not None and reviewed is not None, "a checkout gate needs --sha and --reviewed")
        done = subprocess.run(["git", "-C", str(root), "ls-remote", "leonardo", f"refs/heads/deploy/{CHECKOUTS[name]}"],
                              capture_output=True, text=True, timeout=300)
        require(done.returncode == 0, "cannot read the deploy references on leonardo")
        references = [line.split()[0] for line in done.stdout.splitlines() if "\t" in line]
        reference = references[0] if references else None
        reading = remote_read(exists=[checkout(name)], commands={"squeue": QUEUE, "quota": QUOTA})
        checks, facts = checkout_gate(name, commit, reviewed, reading, reference, root, ledger, line_limit)
        return reading, checks, facts, {"commit": commit, "reviewed": reviewed, "deploy_reference": reference}
    request, commits, documents, table, submitted = gate_request(stage, root, ledger)
    if stage == "probe-aggregate":
        fits = build_fit_state(commits["code"], root)
    reading = remote_read(slurm_jobs=submitted.values(), **request)
    checks, facts = gate(reading, commits, documents, table, submitted, root, ledger, line_limit)
    if stage == "probe-aggregate":
        facts["fits"] = fits
    return reading, checks, facts, {"commits": commits, "documents": documents, "submitted": submitted}


def _now():
    return time.time()


def latest_gate(stage, root=ROOT):
    """The most recent record of the gate of a stage, refusals included."""
    found = sorted((Path(root) / PACKAGE / "evidence/gates").glob(f"{stage}_*.json"))
    require(found, "no gate record for %s: run the gate first" % stage)
    record = json.loads(found[-1].read_text())
    require(record.get("protocol") == GATE_PROTOCOL and record.get("stage") == stage, "unexpected gate record: " + found[-1].name)
    return record


def admitted_lines(stage, sha=None, limit=None, root=ROOT, ledger=LEDGER, now=None):
    """Submission lines of the tranche admitted by the latest gate of the stage, or a shorter prefix of it (`limit`):
    never a job outside that tranche, never without a passed gate read recently and still matching the ledger."""
    record = latest_gate(stage, root)
    require(record.get("passed") is True, "the latest gate of %s did not pass: no line to submit" % stage)
    read_at = calendar.timegm(time.strptime(record["read_at_utc"], "%Y-%m-%dT%H:%M:%SZ"))
    age = (_now() if now is None else now) - read_at
    require(-CLOCK_SKEW_SECONDS <= age <= GATE_MAX_AGE_SECONDS,
            "gate record of %s older than %d seconds: run the gate again" % (stage, GATE_MAX_AGE_SECONDS))
    if stage.startswith("checkout-"):
        name = stage[len("checkout-"):]
        require(limit is None, "--max applies to job stages only")
        require(sha is not None, "a checkout needs --sha")
        require(record.get("commit") == sha, "the latest gate of %s was for another commit" % stage)
        require(not (Path(ledger) / checkout_job(name)["id"]).exists(), "checkout already in the ledger: " + name)
        return checkout_lines(name, sha)
    require(sha is None, "--sha applies to checkout stages only")
    table, tranche = stage_table(stage, root), list(record["facts"]["tranche"])
    submitted = {job["id"]: ledger_job(job["id"], ledger) for job in table if (Path(ledger) / job["id"]).exists()}
    require(submitted == record["submitted"] and not any(job_id in submitted for job_id in tranche),
            "the ledger of %s changed after the gate: run the gate again" % stage)
    code = checkout_commit("code", ledger)
    require(code == record["commits"]["code"], "the code checkout changed after the gate")
    require(limit is None or (type(limit) is int and 1 <= limit <= len(tranche)),
            "--max must be between 1 and the admitted tranche (%d)" % len(tranche))
    return stage_lines(stage, code, root, only=tranche[:limit])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("extraction-plan", "reader-plans", "budgets", "comparison-inputs", "budget-table"):
        sub.add_parser(name)
    sub.add_parser("collect").add_argument("stage")
    sub.add_parser("recovery-lines").add_argument("--sha", required=True, help="commit to deploy (code and recovery plan)")
    for name in ("gate", "lines"):
        command = sub.add_parser(name)
        command.add_argument("stage", help="a job stage, or checkout-<code|plans|budgets|inputs>")
        command.add_argument("--sha", help="checkout stages: commit to deploy")
    sub.choices["gate"].add_argument("--reviewed", help="checkout stages: commit of the package reviewed by GPT")
    sub.choices["gate"].add_argument("--line-limit", type=int, default=LINE_SERIAL_LIMIT,
                                     help="jobs of this line submitted together in the serial queue (default %d)" % LINE_SERIAL_LIMIT)
    sub.choices["lines"].add_argument("--max", type=int, help="job stages: a shorter prefix of the admitted tranche")
    args = parser.parse_args(argv)
    if args.command == "extraction-plan":
        build_extraction_plan()
    elif args.command == "reader-plans":
        build_reader_plans(checkout_commit("code"))
    elif args.command == "budgets":
        build_budgets(checkout_commit("code"))
    elif args.command == "comparison-inputs":
        build_comparison_inputs(checkout_commit("code"))
    elif args.command == "collect":
        run_collect(args.stage)
    elif args.command == "recovery-lines":
        print("\n".join(recovery_lines(args.sha)))
    elif args.command == "gate":
        return run_gate(args.stage, args.sha, args.reviewed, ROOT, LEDGER, args.line_limit)
    elif args.command == "lines":
        try:
            text = admitted_lines(args.stage, args.sha, args.max, ROOT, LEDGER)
        except (ValueError, OSError) as error:
            print("nessuna riga: %s" % error, file=sys.stderr)
            return 1
        print("\n".join(text))
    else:
        print(json.dumps(budget_table(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
