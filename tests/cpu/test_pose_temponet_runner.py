"""Exercise the normal controller/CLI with small mocked compute, never SLURM."""
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
from pose_temponet import benchmark, execution as E, provenance as P, runner as R
from pose_temponet import __main__ as CLI
from test_pose_temponet_provenance import prepared,release,numeric_fixture
from test_pose_temponet_resources import receipts as cost_receipts


@pytest.fixture
def runtime(prepared,monkeypatch):
    repo,plan = prepared
    plan = release(repo,plan,monkeypatch)
    plan["jobs"] = E.job_graph()
    plan["operator_checks"] = {"work_free_bytes":13*1024**3,"pending_write_bytes":0}
    original_validate = P.validate_plan
    monkeypatch.setattr(P,"validate_plan",lambda p,**kw:original_validate(p,**{**kw,"source_root":repo}))
    monkeypatch.setattr(R,"_allocation",lambda job:{"slurm_job_id":"fixture","cpus":job["resources"]["cpus"]})
    monkeypatch.setattr(R.train,"configure_numerics",lambda:None)
    monkeypatch.setattr(P,"numeric_context",lambda kind:numeric_fixture(kind))
    users = plan["producers"]["train"]["users"]
    halves = [users[::2],users[1::2]]
    preflight = {"integrity":True,"historical_R1":{"passed":True},"train":{"folds":halves}}
    monkeypatch.setattr(P,"preflight",lambda p:copy.deepcopy(preflight))
    opened = []
    def load(p,split,*,stage,depth=None):
        assert "cache_"+split in P.stage_inputs(p,stage)
        opened.append((stage,split,depth))
        members = p["producers"][split]["users"]
        return {"meta":[{"user":u} for u in members],"lat":np.zeros((len(members),1,1)),
                "pre":np.ones((len(members),1,1)),"depth":depth}
    monkeypatch.setattr(P,"read_verified_inputs",load)
    cpu,gpu = cost_receipts()
    for measure in (cpu,gpu): measure["input_identity"] = R._input_identity(plan)
    gpu["persistent_output_bytes"] = 7*1024**3
    def measure_gpu(cache,**kw):
        assert "reference_paths" not in kw
        assert kw["reference_proxy_path"] == plan["inputs"]["cache_train"]["path"]
        assert kw["reference_byte_count"] == sum(plan["inputs"][k]["bytes"] for k in
            ("post_ridge_coefficients","post_ridge_predictions","trained_head_predictions"))
        return copy.deepcopy(gpu)
    monkeypatch.setattr(benchmark,"benchmark_cpu",lambda cache,**kw:copy.deepcopy(cpu))
    monkeypatch.setattr(benchmark,"benchmark_gpu",measure_gpu)
    def calibrate(cache,fit,held,allowed,*,outer_fold,**kw):
        assert "lat" in cache and "pre" not in cache and len(fit) == len(held) == 20
        return {"outer_fold":outer_fold,"components":{"fixture":np.ones(1)},"calibration":{"status":"calibrated"}}
    monkeypatch.setattr(R.controls,"build_positive_control_fold",calibrate)
    monkeypatch.setattr(R.controls,"combine_reference_folds",lambda rows:{"status":"diagnostic"})
    monkeypatch.setattr(R.workflow,"prepare_control_folds",lambda c,a,world,**kw:{"world":world})
    monkeypatch.setattr(R.workflow,"prepare_real_folds",lambda c,**kw:{"world":"real"})
    def select(bundle,head,**kw):
        return {"selection":{"head":head,"world":bundle["world"],"selection_locked":True,
                             "optimizer_alarms":[],"objective_tradeoff":{"flag":False}},
                "states":{"fixture":np.arange(2)},"filenames":{"selection":"selection.json","states":"selection_states.pt"}}
    monkeypatch.setattr(R.workflow,"run_selection",select)
    def positive(artifacts,selections):
        assert len(artifacts) == 2 and set(selections) == set(E.WORLDS)
        assert all(set(v) == set(E.CONTROL_HEADS) for v in selections.values())
        return {"status":"passed","complete":True}
    monkeypatch.setattr(R.workflow,"qualify_controls",positive)
    monkeypatch.setattr(R.workflow,"_check_selection",lambda s,**kw:P.require(s["selection_locked"],"not locked"))
    def refit(bundle,selected,gate,**kw):
        return {"manifest":{"head":selected["head"]},"states":{"refit_states":{s:{"seed":s} for s in (0,1,2)}},
                "filenames":{"manifest":"refits.json","states":"refit_states.pt"}}
    monkeypatch.setattr(R.workflow,"run_refits",refit)
    return repo,plan,opened


def initial_chain(runtime):
    repo,plan,opened = runtime
    pins = {}
    for stage in ("preflight","cpu_benchmark","gpu_benchmark"):
        pins[stage] = R.run_stage(plan,E.job_id(stage),source_root=repo)
    plan = R.measured_plan(plan,pins["cpu_benchmark"],pins["gpu_benchmark"],source_root=repo)
    return repo,plan,opened,pins


def test_normal_cli_preflight_and_train_only_benchmarks_then_measured_phase_plan(runtime,monkeypatch,tmp_path):
    repo,plan,opened = runtime
    path = tmp_path/"phase.json"
    pin = P.atomic_json(path,plan,root=tmp_path)
    identifier = E.job_id("preflight")
    monkeypatch.setenv("TCN_JOB_ID",identifier)
    assert CLI.main(["run","--plan",str(path),"--plan-sha256",pin["sha256"],"--job-id",identifier]) == 0
    bench_pins = [R.run_stage(plan,E.job_id(stage),source_root=repo) for stage in ("cpu_benchmark","gpu_benchmark")]
    assert opened == [("cpu_benchmark","train",None),("gpu_benchmark","train",None)]
    measured = R.measured_plan(plan,*bench_pins,source_root=repo)
    assert all(j["resource_release"] for j in measured["jobs"])
    assert len(R._training_pack(measured["resource_gate"])) == 16 and len(measured["jobs"]) == 24
    for job in measured["jobs"]:
        assert job["after"] == E.find_job(plan,job["id"])["after"]


def test_review_and_submitted_plan_hash_block_before_allocation_or_compute(runtime,monkeypatch,tmp_path):
    repo,plan,opened = runtime
    plan["submission_released"] = False
    monkeypatch.setattr(R,"_allocation",lambda j:pytest.fail("allocation checked before review"))
    with pytest.raises(ValueError,match="explicitly released"):
        R.run_stage(plan,E.job_id("preflight"),source_root=repo)
    path = tmp_path/"badplan.json";P.atomic_json(path,plan,root=tmp_path)
    with pytest.raises(ValueError,match="submitted hash"):
        CLI.main(["run","--plan",str(path),"--plan-sha256","1"*64,"--job-id",E.job_id("preflight")])
    assert not opened


def test_local_benchmark_mirrors_are_verified_but_pinned_as_remote_paths(runtime,tmp_path):
    repo,plan,_,pins = initial_chain(runtime)
    mirror = tmp_path/"mac-receipts";mirror.mkdir()
    for pin in pins.values(): shutil.copyfile(pin["path"],mirror/Path(pin["path"]).name)
    local = [P.file_pin(mirror/Path(pins[s]["path"]).name) for s in ("cpu_benchmark","gpu_benchmark")]
    # Local planning does not depend on large remote benchmark artifacts.
    for stage in ("cpu_benchmark","gpu_benchmark"):
        receipt = P.read_json_pin(pins[stage])
        Path(receipt["artifacts"]["benchmark"]["path"]).unlink()
    measured = R.measured_plan(plan,*local,source_root=repo)
    assert all(str(mirror) not in p["path"] for p in measured["receipt_pins"].values())
    for s in ("cpu_benchmark","gpu_benchmark"):
        assert measured["receipt_pins"][E.job_id(s)]["sha256"] == pins[s]["sha256"]


def test_calibration_keeps_lat_and_gate_failure_preserves_evidence_without_fits(runtime,monkeypatch):
    repo,plan,opened,_ = initial_chain(runtime)
    first = R.run_stage(plan,E.job_id("wls_calibration",fold=0),source_root=repo)
    receipt = P.verified_receipt(first,plan,source_root=repo)
    assert {"calibration","calibration_state"} <= set(receipt["artifacts"])
    def fail(cache,*args,**kwargs):
        assert "lat" in cache and "pre" not in cache
        return {"outer_fold":1,"components":{},"calibration":{"status":"control_non_calibrated"}}
    monkeypatch.setattr(R.controls,"build_positive_control_fold",fail)
    identifier = E.job_id("wls_calibration",fold=1)
    with pytest.raises(R.GateNotPassed):
        R.run_stage(plan,identifier,source_root=repo)
    failure = P.read_json_pin(P.file_pin(P.receipt_path(plan,identifier)))
    assert failure["status"] == "gate_failed" and not failure["complete"]
    assert "calibration_state" in failure["artifacts"] and "contrasts" not in failure["details"]
    assert all(split == "train" for _,split,_ in opened)
    with pytest.raises(ValueError,match="predecessor did not complete"):
        R.run_stage(plan,E.job_id("controls",head="L0",world="linear"),source_root=repo)


def test_numeric_mismatch_is_recorded_before_cache_load_or_fit(runtime,monkeypatch):
    repo,plan,opened,_ = initial_chain(runtime)
    opened.clear()
    bad = numeric_fixture();bad["numpy"] = "different"
    monkeypatch.setattr(P,"numeric_context",lambda kind:bad)
    identifier = E.job_id("wls_calibration",fold=0)
    with pytest.raises(ValueError,match="differs from benchmark"):
        R.run_stage(plan,identifier,source_root=repo)
    failure = P.read_json_pin(P.file_pin(P.receipt_path(plan,identifier)))
    assert failure["status"] == "technical_failure" and not opened
    assert failure["gates"]["numeric_context_matches_benchmark"] is False


def test_all_training_stages_preserve_artifacts_and_lock_before_any_development_read(runtime,monkeypatch):
    repo,plan,opened,_ = initial_chain(runtime)
    for job in plan["jobs"]:
        if job["stage"] in ("wls_calibration","controls","selection","refit"):
            R.run_stage(plan,job["id"],source_root=repo)
    assert all(split == "train" for _,split,_ in opened)
    assert len([r for r in Path(plan["output_root"]).rglob("refits.json")]) == 5
    assert len([r for r in Path(plan["output_root"]).rglob("selection_states.pt")]) == 11
    def bad_lock(selections,refits,**kw):
        assert set(selections) == set(refits) == set(R.heads.HEADS)
        assert all(set(v["states"]["refit_states"]) == {0,1,2} for v in refits.values())
        raise ValueError("incomplete lock fixture")
    monkeypatch.setattr(R.workflow,"lock_inference_manifest",bad_lock)
    with pytest.raises(ValueError,match="incomplete lock"):
        R.run_stage(plan,E.job_id("locked_inference"),source_root=repo)
    assert all(split == "train" for _,split,_ in opened)


def test_duplicate_preflight_receipt_cannot_be_reexecuted(runtime):
    repo,plan,_ = runtime
    R.run_stage(plan,E.job_id("preflight"),source_root=repo)
    with pytest.raises(ValueError,match="existing stage receipt"):
        R.run_stage(plan,E.job_id("preflight"),source_root=repo)


@pytest.mark.parametrize("damage",["time","memory","cpus","gpu","restart"])
def test_scontrol_and_environment_allocation_guard(monkeypatch,damage):
    job = next(j for j in E.job_graph() if j["stage"] == "gpu_benchmark")
    environment = {"SLURM_JOB_ID":"123","SLURM_CPUS_PER_TASK":"8","SLURM_GPUS_ON_NODE":"1",
                   "SLURM_NNODES":"1","SLURM_MEM_PER_NODE":"32768","SLURM_RESTART_COUNT":"0"}
    for key,value in environment.items(): monkeypatch.setenv(key,value)
    fields = {"TimeLimit":"00:27:00","Partition":"boost_usr_prod","NumCPUs":"8","NumNodes":"1",
              "AllocTRES":"cpu=8,mem=32G,node=1,gres/gpu=1"}
    monkeypatch.setattr(R.subprocess,"check_output",lambda *a,**kw:" ".join(k+"="+v for k,v in fields.items()))
    monkeypatch.setattr(R.torch.cuda,"device_count",lambda:1)
    assert R._allocation(job)["gpus"] == 1
    if damage == "time": fields["TimeLimit"] = "00:28:00"
    if damage == "memory": fields["AllocTRES"] = "cpu=8,mem=64G,node=1,gres/gpu=1"
    if damage == "cpus": fields["NumCPUs"] = "16"
    if damage == "gpu": fields["AllocTRES"] = "cpu=8,mem=32G,node=1,gres/gpu=2"
    if damage == "restart": monkeypatch.setenv("SLURM_RESTART_COUNT","1")
    with pytest.raises(ValueError): R._allocation(job)
