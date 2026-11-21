"""A1: exact campaign budget authorization, with three immutable predecessors.

No scoring, timing, model or selection rule is changed here. This is deliberately
specific to the successful 21 November campaign, not a general hash exemption.
"""
from __future__ import annotations

import copy

BASE_PATH = "results/infra/pose_temponet_execution_20261120/plan_released.json"
BASE_SHA256 = "84b82384a2eafb60f11480910d33ebc85832cb75063d3479d42e243351ea23b0"
OLD_SOURCE = "37985d28c75bf82a6a784106d414825b727b9ba78575b675dd2dac1d4ccfc8de"
IDENTIFIER = "TCNH-budget-A1-20261121"
CAMPAIGN = "pose_temponet_20261121_A1"
CODE_DIR = "/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/" + CAMPAIGN
ORIGINAL_CAPS = {"neural_gpu_hours": 20., "gpu_total_hours": 24.,
                 "cpu_only_core_hours": 80., "total_core_hours": 272.}
CAPS = {"neural_gpu_hours": 53.85, "gpu_total_hours": 56.10,
        "cpu_only_core_hours": 115.50, "total_core_hours": 564.30}
LEGACY = {
    "TCNH-preflight-20261120": ("0a3314138a98ea3b9e6a8475297c38d5fed575dc112eb52dc4ae4d14d51ee16b", 14122),
    "TCNH-cpu-benchmark-20261120": ("d6fca52ab2d719003d641a24f1067e25c7c31f9400d928c1d0e0d3757fc33ce8", 70963),
    "TCNH-gpu-benchmark-20261120": ("8228db8c6bed7df2eb45eeeedb2203f002ff4e64f625adca010c53f1b367f306", 76463),
}
CHANGED_SOURCES = {
    "scripts/pose_temponet/budget_amendment.py",
    "scripts/pose_temponet/provenance.py", "scripts/pose_temponet/resources.py",
    "scripts/pose_temponet/execution.py", "scripts/pose_temponet/runner.py",
    "scripts/pose_temponet/__main__.py",
}


def specification():
    return {"id": IDENTIFIER, "base_plan": {"path": BASE_PATH, "sha256": BASE_SHA256},
            "authorized_caps": dict(CAPS), "factor": 1.5,
            "decision": {"date": "2026-11-21", "by": "Simone",
                         "words": ["va bene se aumenti il budget",
                                   "dagli quello che serve moltiplicato per 1.5"]},
            "legacy_receipts": {k: {"sha256": v[0], "bytes": v[1]} for k, v in LEGACY.items()}}


def base_plan(source_root):
    from . import provenance as P
    base = P.read_json_pin({"path": BASE_PATH, "sha256": BASE_SHA256}, source_root=source_root)
    P.require(base["reader"]["source_sha256"] == OLD_SOURCE, "A1 base source differs")
    return base


def effective_caps(plan):
    from . import provenance as P
    if "budget_amendment" not in plan:
        return dict(ORIGINAL_CAPS)
    P.require(plan["budget_amendment"] == specification(), "unauthorized A1 budget specification")
    return dict(CAPS)


def validate(plan, *, source_root):
    from . import provenance as P
    effective_caps(plan)
    if "budget_amendment" not in plan:
        return
    base = base_plan(source_root)
    for key in ("protocol", "proposal", "recipe_sha256", "inputs", "producers",
                "output_root", "resource_policy", "background_monitor"):
        P.require(plan[key] == base[key], "A1 changed scientific/base field: " + key)
    old, new = base["reader"]["source_manifest"], plan["reader"]["source_manifest"]
    changed = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
    P.require(changed <= CHANGED_SOURCES and all(key in new for key in old),
              "A1 changed sources outside resource/provenance scope")
    P.require(len(plan["jobs"]) == len(base["jobs"]), "A1 changed job count")
    for job, previous in zip(plan["jobs"], base["jobs"]):
        for key in set(previous) | set(job):
            if key == "resources":
                P.require({k: v for k, v in job[key].items() if k != "requested_seconds"} ==
                          {k: v for k, v in previous[key].items() if k != "requested_seconds"},
                          "A1 changed per-job allocation profile")
                if previous[key]["requested_seconds"] is not None:
                    P.require(job[key]["requested_seconds"] == previous[key]["requested_seconds"],
                              "A1 changed fixed walltime")
            elif key != "resource_release":
                P.require(job.get(key) == previous.get(key), "A1 changed job graph/identity: " + key)


def prepare(plan, *, source_root):
    """Create an UNRELEASED source plan; authorization never fabricates review."""
    from . import provenance as P
    base = base_plan(source_root)
    P.require(plan == base, "A1 requires the exact pinned released base plan")
    out = copy.deepcopy(base)
    manifest = P.source_manifest(source_root)
    out["reader"] = {"code_sha": None, "source_sha256": P.digest(manifest), "source_manifest": manifest}
    out["budget_amendment"] = specification()
    out["review"]["operational"] = {"status": "REVIEW_REQUIRED", "commit": None, "artifact": None,
                                     "proposal_sha256": out["proposal"]["sha256"]}
    out.update(status="REVIEW_REQUIRED", submission_released=False)
    out["execution_sha256"] = P.execution_identity(out)
    return P.validate_plan(out, source_root=source_root)


def receipt_identity_plan(plan, receipt, pin, *, source_root):
    """Only exact bytes of the three completed jobs may use the old identity."""
    from . import provenance as P
    if "budget_amendment" not in plan:
        return plan
    validate(plan, source_root=source_root)
    identifier = receipt["job_id"]
    if identifier not in LEGACY:
        return plan
    expected_hash, expected_bytes = LEGACY[identifier]
    P.require(pin.get("sha256") == expected_hash and
              ("bytes" not in pin or pin["bytes"] == expected_bytes),
              "A1 predecessor is not the exact pinned completed receipt")
    return base_plan(source_root)


def benchmark_identity_plan(plan, *, source_root):
    if "budget_amendment" not in plan:
        return plan
    validate(plan, source_root=source_root)
    return base_plan(source_root)
