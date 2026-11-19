"""Finite job graph and reviewable submitter commands; never submits a job."""
from __future__ import annotations

import re
import shlex
from pathlib import Path

from .heads import HEADS

CAMPAIGN = "pose_temponet_20261119"
CODE_DIR = "/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/" + CAMPAIGN
LOG_ROOT = "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs"
CONTROL_HEADS = ("L0", "MLP", "TCN_post")
WORLDS = ("linear", "temporal")
GRACE = 180


def job_id(stage, *, head=None, world=None, fold=None):
    token = stage.replace("_", "-")
    parts = ["TCNH", token]
    parts.extend(str(x) for x in (world, head, fold) if x is not None)
    parts.append("20261119")
    value = "-".join(parts)
    if len(value) > 64 or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError("invalid logical job ID")
    return value


def _job(stage, *, after=(), head=None, world=None, fold=None,
         requested_seconds=None, cpus=8, gpus=0):
    return {"id": job_id(stage, head=head, world=world, fold=fold), "stage": stage,
            "head": head, "world": world, "fold": fold, "after": list(after),
            "output_directory": "/".join(str(x) for x in (stage, world, head, fold) if x is not None),
            "resources": {"requested_seconds": requested_seconds, "grace_seconds": GRACE,
                          "cpus": cpus, "gpus": gpus, "host_ram_gib": 32 if gpus else 24,
                          "partition": "boost_usr_prod" if gpus else "lrd_all_serial"},
            "resource_release": requested_seconds is not None,
            "operator": "2-TCN head launcher", "retry": "forbidden_without_new_decision"}


def job_graph():
    """All cells specified before measurement; unknown walls cannot be submitted."""
    jobs = [_job("preflight", requested_seconds=720, cpus=2)]
    first = jobs[0]["id"]
    jobs.extend([_job("cpu_benchmark", after=[first], requested_seconds=1620),
                 _job("gpu_benchmark", after=[first], requested_seconds=1620, gpus=1)])
    benches = [j["id"] for j in jobs[1:]]
    for fold in (0, 1):
        job = _job("wls_calibration", fold=fold, after=benches, requested_seconds=14220)
        job["resource_release"] = False
        jobs.append(job)
    calibration = [j["id"] for j in jobs if j["stage"] == "wls_calibration"]
    for world in WORLDS:
        for head in CONTROL_HEADS:
            jobs.append(_job("controls", world=world, head=head, after=calibration, gpus=1))
    controls = [j["id"] for j in jobs if j["stage"] == "controls"]
    for head in HEADS:
        jobs.append(_job("selection", head=head, after=controls, gpus=1))
    selections = [j["id"] for j in jobs if j["stage"] == "selection"]
    for head in HEADS:
        # Lock all five selections before any full-training refit.
        jobs.append(_job("refit", head=head, after=selections, gpus=1))
    refits = [j["id"] for j in jobs if j["stage"] == "refit"]
    jobs.append(_job("locked_inference", after=refits, requested_seconds=3420, gpus=1))
    jobs[-1]["resource_release"] = False
    jobs.append(_job("metrics", after=[jobs[-1]["id"]], requested_seconds=3420))
    jobs[-1]["resource_release"] = False
    jobs.append(_job("collection", after=[jobs[-1]["id"]], requested_seconds=720, cpus=2))
    jobs[-1]["resource_release"] = False
    validate_graph(jobs)
    return jobs


def validate_graph(jobs):
    identifiers, outputs, seen = set(), set(), set()
    for j in jobs:
        if j["id"] in identifiers or j["output_directory"] in outputs:
            raise ValueError("duplicate job/output")
        if not set(j["after"]) <= seen:
            raise ValueError("dependency absent or graph not topological")
        if j["operator"] != "2-TCN head launcher":
            raise ValueError("only one operator is allowed")
        r = j["resources"]
        seconds = r["requested_seconds"]
        if r["gpus"] not in (0, 1) or r["cpus"] not in (2, 8) or r["grace_seconds"] != GRACE:
            raise ValueError("allocation differs from frozen profile")
        if seconds is not None and (type(seconds) is not int or seconds <= 0 or
                                   seconds + GRACE > (86400 if r["gpus"] else 14400)):
            raise ValueError("job exceeds inclusive lane walltime")
        identifiers.add(j["id"]); outputs.add(j["output_directory"]); seen.add(j["id"])
    if sum(j["resources"]["gpus"] > 0 for j in jobs) > 40 or sum(j["resources"]["gpus"] == 0 for j in jobs) > 7:
        raise ValueError("job count exceeds frozen cap")
    return True


def apply_training_pack(jobs, pack):
    """Bind measured times, preserving every cell and dependency of the graph."""
    import copy
    out = copy.deepcopy(jobs)
    required = {(j["stage"], j["head"], j["world"]) for j in out
                if j["stage"] in ("controls", "selection", "refit")}
    rows = {(p["stage"], p["head"], p.get("world")): p for p in pack}
    if len(rows) != len(pack) or set(rows) != required:
        raise ValueError("measured pack differs from the complete frozen matrix")
    for j in out:
        key = (j["stage"], j["head"], j["world"])
        if key in rows:
            row = rows[key]
            if any(row.get(k, v) != v for k, v in (("cpus", 8), ("gpus", 1), ("host_ram_gib", 32))):
                raise ValueError("training pack changes allocation")
            j["resources"]["requested_seconds"] = row["requested_seconds"]
        j["resource_release"] = True
    validate_graph(out)
    gpu = sum((j["resources"]["requested_seconds"] + GRACE)/3600 for j in out if j["resources"]["gpus"])
    cpu = sum((j["resources"]["requested_seconds"] + GRACE)*j["resources"]["cpus"]/3600
              for j in out if not j["resources"]["gpus"])
    fit = sum((j["resources"]["requested_seconds"] + GRACE)/3600 for j in out
              if j["stage"] in ("controls", "selection", "refit"))
    if fit > 20 or gpu > 24 or cpu > 80 or gpu*8+cpu > 272:
        raise ValueError("inclusive allocation exceeds approved phase/campaign caps")
    return out


def find_job(plan, logical_id):
    rows = [j for j in plan["jobs"] if j["id"] == logical_id]
    if len(rows) != 1:
        raise ValueError("job is absent or duplicated in execution plan")
    return rows[0]


def verify_completed_predecessors(plan, proofs):
    """Accept only archived receipt + sacct 0:0, never disappearance from squeue."""
    from . import provenance as P
    accepted = set()
    for identifier, proof in proofs.items():
        expected = find_job(plan, identifier)
        receipt = P.verified_receipt(proof["receipt"], plan, stage=expected["stage"], verify_artifacts=False,
                                     receipt_mirror_dir=Path(proof["receipt"]["path"]).parent)
        if receipt["job_id"] != identifier:
            raise ValueError("completed proof has another logical ID")
        slurm_id = str(receipt["details"]["allocation"]["slurm_job_id"])
        check = P.verify_pin(proof["sacct"])
        rows = [line.split("|") for line in Path(check["pin"]["path"]).read_text().splitlines()]
        matches = [row for row in rows if len(row) == 3 and row[0] == slurm_id]
        P.unchanged({check["pin"]["path"]: check["state"]})
        if matches != [[slurm_id, "COMPLETED", "0:0"]]:
            raise ValueError("completed predecessor lacks unambiguous sacct COMPLETED 0:0")
        accepted.add(identifier)
    return accepted


def submit_command(plan, job, *, checkout_sha, plan_path, plan_sha256,
                   code_dir=CODE_DIR, after_slurm=None, completed_proofs=None):
    """Return argv only. The operator checks ledger, SLURM and reviews first.

    ``after_slurm`` may pin active IDs when Slurm has already purged a completed
    predecessor. Omitting a dependency is never an automatic fallback.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", checkout_sha) or not re.fullmatch(r"[0-9a-f]{64}", plan_sha256):
        raise ValueError("full checkout and plan hashes are required")
    if plan.get("submission_released") is not True or not job["resource_release"]:
        raise ValueError("reviews/resources have not released this job")
    r = job["resources"]
    seconds = r["requested_seconds"]
    if seconds is None:
        raise ValueError("unmeasured allocation")
    time = "%02d:%02d:%02d" % (seconds//3600, seconds//60%60, seconds%60)
    command = ["python3", "scripts/submit_job.py", "--id", job["id"], "--agente", "gpt_tcn_launcher",
               "--campagna", CAMPAIGN, "--gpu-ore", str((seconds+GRACE)*r["gpus"]/3600)]
    completed = verify_completed_predecessors(plan, completed_proofs or {})
    remaining = [p for p in job["after"] if p not in completed]
    if after_slurm is None:
        for predecessor in remaining:
            command += ["--dopo", predecessor]
    elif set(after_slurm) != set(remaining) or any(not str(v).isdigit() for v in after_slurm.values()):
        raise ValueError("explicit dependencies must cover every predecessor")
    launcher = "pose_temponet_checkout.sbatch" if job["stage"] == "preflight" else "pose_temponet.sbatch"
    export = (f"ALL,CODE_DIR={code_dir},EXPECTED_SHA={checkout_sha},TCN_PLAN={plan_path},"
              f"TCN_PLAN_SHA256={plan_sha256},TCN_JOB_ID={job['id']}")
    if job["stage"] == "preflight":
        export += f",CAMPAIGN={Path(code_dir).name},CAMPAIGN_SHA={checkout_sha}"
    if any(c in str(plan_path) for c in (",", "\n")):
        raise ValueError("plan path cannot be represented in SLURM export")
    command += ["--script-stdin", "scripts/slurm/"+launcher, "--", "--account=IscrB_WearUsFM",
                "--partition="+r["partition"], "--nodes=1", "--ntasks=1", "--cpus-per-task="+str(r["cpus"]),
                "--mem="+str(r["host_ram_gib"])+"G", "--time="+time, "--no-requeue", "--export="+export,
                f"--output={LOG_ROOT}/{job['id']}_%j.out", f"--error={LOG_ROOT}/{job['id']}_%j.out"]
    if r["gpus"]:
        command += ["--gres=gpu:1", "--qos=normal"]
    if after_slurm:
        command.append("--dependency=afterok:"+":".join(str(after_slurm[k]) for k in remaining))
    return command


def shell_command(*args, **kwargs):
    return shlex.join(submit_command(*args, **kwargs))
