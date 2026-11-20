"""Complete cell graph, inclusive allocation and non-submitting command generation."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from pose_temponet import execution as E


def measured_pack(jobs):
    return [{"stage": j["stage"], "head": j["head"], "world": j["world"],
             "requested_seconds": 1800, "cpus": 8, "gpus": 1, "host_ram_gib": 32}
            for j in jobs if j["stage"] in ("controls", "selection", "refit")]


def test_second_delivery_never_reuses_the_identity_of_the_failed_first_preflight():
    ids = [j["id"] for j in E.job_graph()]
    assert len(set(ids)) == 24 and all(i.startswith("TCNH-") and i.endswith("-20261120") for i in ids)
    assert E.job_id("preflight") == "TCNH-preflight-20261120" and "TCNH-preflight-20261119" not in ids
    assert E.CAMPAIGN == "pose_temponet_20261120"
    assert E.CODE_DIR == "/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose_temponet_20261120"


def test_preflight_command_names_the_new_campaign_everywhere():
    """pose_temponet_checkout.sbatch refuses a CODE_DIR that is not $WORK/wearusfm_campaigns/$CAMPAIGN."""
    output_root = "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pose_temponet_20261117"
    command = E.submit_command({"submission_released": True}, E.job_graph()[0], checkout_sha="a"*40,
                               plan_path=output_root+"/manifests/"+"b"*64+".json", plan_sha256="b"*64)
    assert command[command.index("--id")+1] == "TCNH-preflight-20261120"
    assert command[command.index("--campagna")+1] == "pose_temponet_20261120"
    export = dict(item.split("=", 1) for item in next(w for w in command if w.startswith("--export="))[9:].split(",")[1:])
    assert export["CAMPAIGN"] == "pose_temponet_20261120" and export["TCN_JOB_ID"] == "TCNH-preflight-20261120"
    assert export["CODE_DIR"] == "/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/"+export["CAMPAIGN"]
    assert export["CAMPAIGN_SHA"] == export["EXPECTED_SHA"] == "a"*40
    assert command[command.index("--script-stdin")+1] == "scripts/slurm/pose_temponet_checkout.sbatch"
    assert "--time=00:12:00" in command and "--mem=24G" in command and "--cpus-per-task=2" in command


def test_graph_has_no_duplicate_cells_and_keeps_global_scientific_dependencies():
    jobs = E.job_graph()
    assert len(jobs) == 24
    assert sum(j["resources"]["gpus"] for j in jobs) == 18
    assert sum(not j["resources"]["gpus"] for j in jobs) == 6
    bystage = {s: [j for j in jobs if j["stage"] == s] for s in {j["stage"] for j in jobs}}
    assert len(bystage["controls"]) == 6
    assert len(bystage["selection"]) == len(bystage["refit"]) == 5
    for j in bystage["selection"]:
        assert set(j["after"]) == {c["id"] for c in bystage["controls"]}
    for j in bystage["refit"]:
        assert set(j["after"]) == {c["id"] for c in bystage["selection"]}
    assert set(bystage["locked_inference"][0]["after"]) == {c["id"] for c in bystage["refit"]}
    assert sum(j["resource_release"] for j in jobs) == 3


def test_measured_pack_cannot_drop_a_control_or_change_a_comparator():
    jobs = E.job_graph(); pack = measured_pack(jobs)
    ready = E.apply_training_pack(jobs, pack)
    assert all(j["resource_release"] for j in ready)
    assert all(a["after"] == b["after"] for a, b in zip(jobs, ready))
    with pytest.raises(ValueError, match="complete frozen matrix"):
        E.apply_training_pack(jobs, pack[:-1])
    pack[0]["head"] = "unapproved"
    with pytest.raises(ValueError, match="complete frozen matrix"):
        E.apply_training_pack(jobs, pack)


def test_overhead_counts_toward_twenty_hour_phase_cap():
    jobs = E.job_graph(); pack = measured_pack(jobs)
    for p in pack: p["requested_seconds"] = 4500  # 16*4500=20h without grace
    with pytest.raises(ValueError, match="phase/campaign"):
        E.apply_training_pack(jobs, pack)
    jobs[3]["resources"]["requested_seconds"] = 14400
    with pytest.raises(ValueError, match="inclusive lane"):
        E.validate_graph(jobs)


def test_commands_require_releases_and_use_only_common_submitter():
    jobs = E.job_graph(); plan = {"jobs": jobs, "submission_released": False}
    kwargs = dict(checkout_sha="a"*40, plan_path="/work/p.json", plan_sha256="b"*64)
    with pytest.raises(ValueError, match="not released"):
        E.submit_command(plan, jobs[0], **kwargs)
    plan["submission_released"] = True
    cmd = E.submit_command(plan, jobs[0], **kwargs)
    assert cmd[:2] == ["python3", "scripts/submit_job.py"]
    assert "scripts/slurm/pose_temponet_checkout.sbatch" in cmd
    assert "--time=00:12:00" in cmd and "--no-requeue" in cmd
    assert any("TCN_PLAN_SHA256="+"b"*64 in s for s in cmd)
    cmd = E.submit_command(plan, jobs[2], **kwargs)
    assert "--dopo" in cmd and jobs[0]["id"] in cmd and "--gres=gpu:1" in cmd
    with pytest.raises(ValueError, match="not released"):
        E.submit_command(plan, jobs[5], **kwargs)


def test_dependency_override_never_silently_omits_a_prerequisite():
    jobs=E.job_graph(); plan={"submission_released":True}
    kwargs=dict(checkout_sha="a"*40,plan_path="/work/p.json",plan_sha256="b"*64)
    with pytest.raises(ValueError,match="every predecessor"):
        E.submit_command(plan,jobs[1],after_slurm={},**kwargs)
    cmd=E.submit_command(plan,jobs[1],after_slurm={jobs[0]["id"]:"12345"},**kwargs)
    assert "--dependency=afterok:12345" in cmd and "--dopo" not in cmd
    broken=copy.deepcopy(jobs);broken[1]["after"]=["unknown"]
    with pytest.raises(ValueError,match="dependency"):
        E.validate_graph(broken)
