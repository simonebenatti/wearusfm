"""Per-task lineage joins, using the existing accepted-evidence fixtures."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import scaling_p1_recovery_plans as M

spec = importlib.util.spec_from_file_location("p1_recovery_existing_fixtures", ROOT / "tests/cpu/test_scaling_plans19618.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


@pytest.fixture
def inputs():
    extraction, (class_receipts, class_pins, class_prov), (pose_receipt, pose_pin, pose_prov) = F.new_candidates()
    sources, new_code, release_sha, new_job = {k: "e" * 64 for k in M.R.SOURCE_PATHS}, "d" * 40, "f" * 64, "64600000"
    old_folder = str(Path(class_pins["p1"]["path"]).parent)
    receipts = {"pose": pose_receipt, **class_receipts}
    pins = {"pose": pose_pin, **class_pins}
    prov = {"pose": pose_prov, **{task: {ds: class_prov[ds] for ds in datasets} for task, datasets in M.S.TASKS.items()}}
    recovery = {"recovery_protocol": M.R.PROTOCOL, "recovery_manifest_sha256": release_sha,
                "recovery_source_sha256": sources, "output_barrier": "none"}
    audit = {"policy": M.R.O.POLICY, "complete": True, "extra_reads": 0, "observations": [{}, {}]}
    for task in ("pose", "p1", "p2"):
        code = new_code if task == "p1" else M.HISTORICAL_CODE_SHA
        if task == "p1":
            receipts[task] = json.loads(json.dumps(receipts[task]).replace(old_folder, M.R.OUTPUT))
            pins[task]["path"] = pins[task]["path"].replace(old_folder, M.R.OUTPUT)
        receipt = receipts[task]
        receipt["code_sha"] = code
        if task == "p1":
            receipt.update(**recovery, logical_job_id=M.R.LOGICAL_ID, slurm_job_id=new_job,
                           failed_attempts_preserved=M.R.FAILED_ATTEMPTS)
        for name in prov[task]:
            data = prov[task][name]
            if task == "p1":
                data = data.replace(old_folder.encode(), M.R.OUTPUT.encode())
            value = json.loads(data)
            value["code_sha"] = code
            if task == "p1":
                value.update(**recovery, output_identity_audit=audit)
                receipt["outputs"][name]["output_identity_audit"] = audit
            prov[task][name] = F.body(value)
            receipt["outputs"][name]["provenance_sha256"] = F.sha(prov[task][name])
    raw = {task: F.body(receipt) for task, receipt in receipts.items()}
    pins = {task: M.S.pin(pin["path"], raw[task]) for task, pin in pins.items()}
    jobs = {task: {"job_id": new_job if task == "p1" else M.HISTORICAL_JOBS[task], "state": "COMPLETED",
                   "exit_code": "0:0", "code_sha": new_code if task == "p1" else M.HISTORICAL_CODE_SHA}
            for task in receipts}
    return {"receipt_bytes": raw, "receipt_pins": pins, "provenances": prov, "completed_jobs": jobs,
            "checkpoint": extraction["checkpoint"], "extraction_plan_sha256": extraction["plan_sha256"],
            "recovery_code_sha": new_code, "recovery_manifest_sha256": release_sha, "recovery_sources": sources}


def test_mixed_plan_keeps_each_actual_producer_and_is_accepted_by_existing_readers(inputs):
    pose, candidate, lineage = M.mixed_candidate_caches(**inputs)
    assert {p["extraction_code_sha"] for p in pose.values()} == {M.HISTORICAL_CODE_SHA}
    for task, datasets in M.S.TASKS.items():
        expected = inputs["recovery_code_sha"] if task == "p1" else M.HISTORICAL_CODE_SHA
        assert {candidate[ds]["extraction_code_sha"] for ds in datasets} == {expected}
    assert lineage["tasks"]["p1"]["job_id"] == "64600000"
    assert lineage["tasks"]["pose"]["job_id"] == "64571999"
    assert lineage["tasks"]["p2"]["job_id"] == "64572001"
    control = M.S.control_probe_caches(F.inventory("probe"), F.control_provenances(), F.TRAINING["control"]["checkpoint"])
    plan = M.S.probe_plan(F.load("cpu_readers_20261105/probe_plan.json"), control, candidate)
    assert plan["caches"][M.R.ARM] == candidate
    pose_plan = M.S.pose_plan(F.load("cpu_readers_20261105/pose_plan.json"), F.inventory("pose"), pose)
    assert pose_plan["caches"][M.R.ARM] == pose


def mutate_receipt(inputs, task, change):
    value = json.loads(inputs["receipt_bytes"][task])
    change(value)
    raw = F.body(value)
    inputs["receipt_bytes"][task] = raw
    inputs["receipt_pins"][task] = M.S.pin(inputs["receipt_pins"][task]["path"], raw)


@pytest.mark.parametrize("change", [
    lambda a: a["completed_jobs"]["p1"].update(state="FAILED"),
    lambda a: a["completed_jobs"]["p1"].update(job_id="64596915"),
    lambda a: a["completed_jobs"]["pose"].update(job_id="64600001"),
    lambda a: a["completed_jobs"]["p2"].update(code_sha="d" * 40),
    lambda a: a.update(recovery_code_sha=M.HISTORICAL_CODE_SHA),
    lambda a: a.update(recovery_sources={}),
    lambda a: a["receipt_bytes"].update(p1=a["receipt_bytes"]["p1"] + b" "),
    lambda a: a["receipt_pins"]["p1"].update(path="/old-r1/receipt.json"),
    lambda a: mutate_receipt(a, "p1", lambda r: r.update(complete=False)),
    lambda a: mutate_receipt(a, "p1", lambda r: r.update(checkpoint_step=5000)),
    lambda a: mutate_receipt(a, "p1", lambda r: r.update(code_sha=M.HISTORICAL_CODE_SHA)),
    lambda a: mutate_receipt(a, "p1", lambda r: r.update(recovery_manifest_sha256="a" * 64)),
    lambda a: mutate_receipt(a, "p1", lambda r: r.update(recovery_source_sha256={})),
    lambda a: mutate_receipt(a, "p1", lambda r: r.update(slurm_job_id="64572000")),
    lambda a: mutate_receipt(a, "pose", lambda r: r.update(training_source_contract_sha256="a" * 64)),
    lambda a: mutate_receipt(a, "p2", lambda r: r.update(model_config={"dim": 384})),
])
def test_mixed_builder_rejects_wrong_attempt_code_or_incomplete_evidence(inputs, change):
    change(inputs)
    with pytest.raises(ValueError):
        M.mixed_candidate_caches(**inputs)


def test_missing_output_audit_and_foreign_sidecars_are_rejected(inputs):
    ds = M.S.TASKS["p1"][0]
    prov = json.loads(inputs["provenances"]["p1"][ds])
    prov["output_identity_audit"] = {}
    raw = F.body(prov)
    inputs["provenances"]["p1"][ds] = raw
    def change(receipt):
        receipt["outputs"][ds]["provenance_sha256"] = F.sha(raw)
        receipt["outputs"][ds]["output_identity_audit"] = {}
    mutate_receipt(inputs, "p1", change)
    with pytest.raises(ValueError, match="audit missing"):
        M.mixed_candidate_caches(**inputs)
