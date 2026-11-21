"""Fail-closed review, source, filesystem and receipt-chain tests; no cluster I/O."""
import copy
import hashlib
import json
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
from pose_temponet import controls as C, provenance as P

REPO = Path(__file__).resolve().parents[2]
# IDs read from meta_json of the two pinned caches on Leonardo (provenance inside the file).
REAL_CACHE_USERS = json.loads((REPO/"tests/cpu/data/pose_temponet_cache_users_20261120.json").read_text())


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    repo = tmp_path/"repo"
    for path in ("src/example.py","scripts/legacy.py","scripts/pose_temponet/example.py",
                 "scripts/slurm/lib_code_dir.sh","scripts/slurm/pose_temponet.sbatch",
                 "scripts/slurm/campaign_checkout.sbatch"):
        destination = repo/path
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text("# immutable source\n")
    proposal_path = "results/infra/pose_temponet_proposal_20261117/proposal.json"
    copies = [proposal_path,P.DOCUMENTARY_REVIEW["artifact"]["path"]]
    copies += [pin["local_copy"] for pin in P.HISTORICAL_RECEIPTS.values()]
    for path in copies:
        destination = repo/path
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO/path,destination)
    monkeypatch.setattr(P,"OUTPUT_ROOT",str(tmp_path/"out"))
    plan = P.build_plan(proposal_path,source_root=repo,jobs=[
        {"id":"preflight","stage":"preflight","after":[],"resources":{"cpus":2,"gpus":0}},
        {"id":"bench","stage":"cpu_benchmark","after":["preflight"],"resources":{"cpus":8,"gpus":0}},
    ])
    return repo,plan


def release(repo,plan,monkeypatch):
    plan = copy.deepcopy(plan)
    op = repo/"docs/reviews/test_operational_review.txt"
    op.write_text("TEST FIXTURE: external operational review evidence\n")
    pin = P.file_pin(op)
    pin["path"] = pin["local_copy"] = str(op.relative_to(repo))
    plan["review"]["operational"] = {"status":"claude_review_converged","commit":"f"*40,
                                     "artifact":pin,"source_sha256":plan["reader"]["source_sha256"],
                                     "proposal_sha256":plan["proposal"]["sha256"]}
    plan.update(status="claude_review_converged",submission_released=True)
    plan["reader"]["code_sha"] = "e"*40
    original = subprocess.check_output
    def fake_git_head(args,*a,**kw):
        if args == ["git","rev-parse","HEAD"]:
            return "e"*40+"\n"
        return original(args,*a,**kw)
    monkeypatch.setattr(P.subprocess,"check_output",fake_git_head)
    monkeypatch.setattr(P,"checkout_identity",lambda plan,**kw:{"code_sha":"e"*40,"reviewed_code_sha":"e"*40})
    return plan


def numeric_fixture(kind="cpu"):
    return {"python":"3.fixture","numpy":"1.fixture","scipy":"1.fixture","sklearn":"1.fixture","torch":"2.fixture",
            "threads":{"intraop":8,"interop":1,"environment":{k:"8" for k in
                       ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}},
            "threadpools":[{"prefix":"libopenblas","num_threads":8}],"hardware":{"kind":kind},
            "flags":{"deterministic":True,"deterministic_warn_only":False,"cudnn_deterministic":True,
                     "cudnn_benchmark":False,"cudnn_tf32":False,"matmul_tf32":False,"CUBLAS_WORKSPACE_CONFIG":":4096:8"}}


def test_real_original_producers_and_documentary_review_remain_distinct_from_reader(prepared):
    repo,plan = prepared
    assert plan["status"] == "REVIEW_REQUIRED" and not plan["submission_released"]
    assert plan["review"]["documentary"]["commit"] == "ad9f5141711ba60ea1014bd9ba842af168ff0a37"
    assert plan["review"]["operational"]["status"] == "REVIEW_REQUIRED"
    assert plan["producers"]["train"]["metadata"]["code_sha"] == "7d276c98fc54d66dab1226226eb6c5e447422c69"
    assert plan["producers"]["train"]["metadata"]["source_sha256"] != plan["reader"]["source_sha256"]
    assert len(plan["producers"]["train"]["users"]) == 40
    assert len(plan["producers"]["val"]["users"]) == 15
    assert len(plan["inputs"]) == 8
    # The producer's SCRATCH artifact identifies the byte-identical WORK copy.
    assert plan["producers"]["train"]["original_cache"]["path"].startswith("/leonardo_scratch/")
    assert plan["inputs"]["cache_train"]["path"].startswith("/leonardo_work/")
    with pytest.raises(ValueError,match="explicitly released"):
        P.validate_plan(plan,source_root=repo,require_review=True)


def test_real_producer_users_are_the_ids_stored_in_the_real_caches(prepared):
    """Preflight 64931573 compared storage directories ("u"+ID) with cache IDs."""
    _,plan = prepared
    for split,count in (("train",40),("val",15)):
        real = REAL_CACHE_USERS[split]
        producer = plan["producers"][split]
        assert hashlib.sha256(json.dumps(real["users"]).encode()).hexdigest() == real["users_json_sha256"]
        assert real["cache_pin_sha256"] == plan["inputs"]["cache_"+split]["sha256"]
        assert real["windows"] == plan["inputs"]["cache_"+split]["windows"]
        assert len(real["users"]) == count and producer["users"] == real["users"]
        assert producer["storage_directories"] == ["u"+user for user in real["users"]]
        assert not set(producer["users"]) & set(producer["storage_directories"])


def test_storage_directory_maps_back_only_where_the_id_is_unique(monkeypatch):
    assert P.storage_directory("03d49b393e") == "u03d49b393e"
    assert P.cache_user("u03d49b393e") == "03d49b393e"
    assert P.cache_user("uu1.a@b-c") == "u1.a@b-c"
    # An unsafe character and a literal underscore share one directory: no unique ID.
    assert P.storage_directory("ab/cd") == P.storage_directory("ab_cd") == "uab_cd"
    for name in ("03d49b393e","u","","uab_cd","uab cd","Uabc",None,"u03d49b393e/x"):
        with pytest.raises(ValueError,match="no unique cache user ID"):
            P.cache_user(name)
    with pytest.raises(ValueError,match="invalid cache user ID"):
        P.storage_directory("")
    # If the producer naming rule drifted, the stored directory no longer round-trips.
    monkeypatch.setattr(P,"_STORAGE_UNSAFE",re.compile(r"[^0-9a-f]"))
    with pytest.raises(ValueError,match="does not round-trip"):
        P.cache_user("uABC")


def test_producer_with_ambiguous_storage_directory_is_refused(prepared):
    repo,plan = prepared
    proposal = json.loads((repo/plan["proposal"]["path"]).read_text())
    receipt = json.loads((repo/P.HISTORICAL_RECEIPTS["train"]["local_copy"]).read_text())
    arguments = {"split":"train","cache_pin":plan["inputs"]["cache_train"],"stats_sha256":proposal["stats_sha256"]}
    expected = {k:v for k,v in plan["producers"]["train"].items() if k != "receipt"}
    assert P.historical_producer(receipt,**arguments) == expected
    first = expected["storage_directories"][0]
    states = receipt["raw_inputs"]["file_states"]
    receipt["raw_inputs"]["file_states"] = {name.replace("/"+first+"/","/"+first[:4]+"_"+first[5:]+"/"):state
                                            for name,state in states.items()}
    assert len(receipt["raw_inputs"]["file_states"]) == len(states)
    with pytest.raises(ValueError,match="no unique cache user ID"):
        P.historical_producer(receipt,**arguments)


def test_verified_inputs_compare_cache_ids_and_directories_with_the_real_user_ids(prepared,monkeypatch):
    _,plan = prepared
    producer = plan["producers"]["train"]
    real = REAL_CACHE_USERS["train"]["users"]
    served = {}
    monkeypatch.setattr(P,"verify_pin",lambda pin,**kwargs:{"pin":pin,"state":{}})
    monkeypatch.setattr(P,"unchanged",lambda states:None)
    monkeypatch.setattr(P.D,"read_cache_pin",lambda pin,**kwargs:served["data"])
    def cache(users):
        return {"meta":[{"user":user} for user in users for _ in range(2)],
                "input_verification":copy.deepcopy(producer["details"])}
    served["data"] = cache(real)
    data = P.read_verified_inputs(plan,"train",stage="preflight")
    assert data is served["data"]
    # The same list is the allow-list of the control partitions (the other consumers are exercised on
    # real-format files in test_pose_temponet_preflight_real_layout.py).
    half = len(data["meta"])//2
    partition = C.validate_outer_partition(data["meta"],range(half),range(half,2*half),producer["users"])
    assert sorted(partition["fitting_users"]+partition["held_users"]) == real
    with pytest.raises(ValueError,match="non-training users"):
        C.validate_outer_partition(data["meta"],range(half),range(half,2*half),producer["storage_directories"])
    # The exact failure of preflight 64931573: directory names offered as IDs.
    served["data"] = cache(producer["storage_directories"])
    with pytest.raises(ValueError,match="cache user population differs"):
        P.read_verified_inputs(plan,"train",stage="preflight")
    served["data"] = cache(real[:-1]+["ffffffffff"])
    with pytest.raises(ValueError,match="cache user population differs"):
        P.read_verified_inputs(plan,"train",stage="preflight")
    # IDs alone are not enough when the plan's directory list disagrees with them.
    drifted = copy.deepcopy(plan)
    drifted["producers"]["train"]["storage_directories"] = ["u"+user for user in real[:-1]]+["uffffffffff"]
    served["data"] = cache(real)
    with pytest.raises(ValueError,match="do not map onto the producer storage directories"):
        P.read_verified_inputs(drifted,"train",stage="preflight")


def test_validate_plan_rejects_rewritten_user_lists_even_with_recomputed_identity(prepared):
    """A plan carrying the first delivery's list is refused by validation itself, not only at read time."""
    repo,plan = prepared
    for split in ("train","val"):
        for field in ("users","storage_directories"):
            bad = copy.deepcopy(plan)
            if field == "users":  # the first delivery's content
                bad["producers"][split]["users"] = list(bad["producers"][split]["storage_directories"])
            else:
                bad["producers"][split]["storage_directories"][0] = "uffffffffff"
            bad["execution_sha256"] = P.execution_identity(bad)
            with pytest.raises(ValueError,match="^historical producer identity was rewritten$"):
                P.validate_plan(bad,source_root=repo)


def test_cache_with_one_user_missing_and_receipt_with_one_directory_missing_are_refused(prepared,monkeypatch):
    repo,plan = prepared
    real = REAL_CACHE_USERS["train"]["users"]
    monkeypatch.setattr(P,"verify_pin",lambda pin,**kwargs:{"pin":pin,"state":{}})
    monkeypatch.setattr(P,"unchanged",lambda states:None)
    served = {"meta":[{"user":user} for user in real[:-1] for _ in range(2)],
              "input_verification":copy.deepcopy(plan["producers"]["train"]["details"])}
    monkeypatch.setattr(P.D,"read_cache_pin",lambda pin,**kwargs:served)
    with pytest.raises(ValueError,match="^cache user population differs from original producer$"):
        P.read_verified_inputs(plan,"train",stage="preflight")
    # Equality, not inclusion, also when the plan is built: 39 storage directories are not the 40 of the cache.
    proposal = json.loads((repo/plan["proposal"]["path"]).read_text())
    receipt = json.loads((repo/P.HISTORICAL_RECEIPTS["train"]["local_copy"]).read_text())
    last = plan["producers"]["train"]["storage_directories"][-1]
    states = receipt["raw_inputs"]["file_states"]
    receipt["raw_inputs"]["file_states"] = {name:state for name,state in states.items() if "/emg2pose/"+last+"/" not in name}
    assert 0 < len(receipt["raw_inputs"]["file_states"]) < len(states)
    with pytest.raises(ValueError,match="^historical cache population differs$"):
        P.historical_producer(receipt,split="train",cache_pin=plan["inputs"]["cache_train"],
                              stats_sha256=proposal["stats_sha256"])


def test_each_source_category_and_launcher_enters_new_reader_identity(prepared):
    repo,plan = prepared
    for path in plan["reader"]["source_manifest"]:
        file = repo/path
        original = file.read_text()
        file.write_text(original+"# changed\n")
        with pytest.raises(ValueError,match="reader sources changed"):
            P.validate_plan(plan,source_root=repo)
        file.write_text(original)
    assert P.source_identity(repo) == plan["reader"]["source_sha256"]


@pytest.mark.parametrize("mutation",["producer","input","recipe","resource","identity"])
def test_rewriting_historical_metadata_or_scientific_pins_fails(prepared,mutation):
    repo,plan = prepared
    if mutation == "producer":
        plan["producers"]["val"]["metadata"]["source_sha256"] = plan["reader"]["source_sha256"]
    elif mutation == "input":
        plan["inputs"]["cache_val"]["sha256"] = "1"*64
    elif mutation == "recipe":
        plan["recipe_sha256"] = "1"*64
    elif mutation == "resource":
        plan["resource_policy"]["gpu_hours_cap"] = 25
    else:
        plan["execution_sha256"] = "1"*64
    with pytest.raises(ValueError):
        P.validate_plan(plan,source_root=repo)


def test_explicit_review_binds_actual_artifact_and_source_without_fake_documentary_code(prepared,monkeypatch):
    repo,plan = prepared
    plan = release(repo,plan,monkeypatch)
    assert P.check_release(plan,"preflight",gates={},source_root=repo)
    assert "source_sha256" not in plan["review"]["documentary"]
    with pytest.raises(ValueError,match="stage gates"):
        P.check_release(plan,"cpu_benchmark",gates={},source_root=repo)
    bad = copy.deepcopy(plan)
    bad["review"]["operational"]["source_sha256"] = "1"*64
    with pytest.raises(ValueError,match="other sources"):
        P.validate_plan(bad,source_root=repo,require_review=True)
    artifact = repo/plan["review"]["operational"]["artifact"]["path"]
    artifact.write_text("unreviewed replacement")
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        P.validate_plan(plan,source_root=repo,require_review=True)


def test_train_only_stages_never_open_or_verify_validation(prepared,monkeypatch):
    _,plan = prepared
    calls = []
    monkeypatch.setattr(P,"verify_pin",lambda pin,**kwargs:calls.append(pin["path"]) or {})
    monkeypatch.setattr(P.D,"read_cache_pin",lambda *a,**kw:pytest.fail("val was opened"))
    for stage in P.TRAIN_STAGES:
        calls.clear()
        P.verify_inputs(plan,stage)
        assert set(calls) == {plan["inputs"][k]["path"] for k in P.TRAIN_INPUTS}
        with pytest.raises(ValueError,match="not permitted"):
            P.read_verified_inputs(plan,"val",stage=stage)
    assert len(P.stage_inputs(plan,"preflight")) == 8
    assert set(P.stage_inputs(plan,"locked_inference")) == {"cache_val","target_reference_val","pose_stats"}


def test_file_guards_detect_replacement_and_symlinks_at_every_component(tmp_path):
    directory = tmp_path/"plain";directory.mkdir()
    path = directory/"file.json";path.write_text("{}")
    pin = P.file_pin(path)
    checked = P.verify_pin(pin)
    path.unlink();path.write_text("{}")
    with pytest.raises(ValueError,match="changed while in use"):
        P.unchanged({str(path):checked["state"]})
    link = tmp_path/"linked";link.symlink_to(directory,target_is_directory=True)
    with pytest.raises(ValueError,match="symlink component"):
        P.verify_pin({**pin,"path":str(link/"file.json")})
    dangling = tmp_path/"dangling";dangling.symlink_to(tmp_path/"absent")
    with pytest.raises(ValueError,match="symlink component"):
        P.safe_path(dangling/"out.json")
    with pytest.raises(ValueError,match="escapes"):
        P.safe_path(path,root=tmp_path/"different")


def test_atomic_json_torch_npz_never_replace_and_leave_no_partial_file(tmp_path):
    root = tmp_path/"out"
    path = root/"nested/result.json"
    pin = P.atomic_json(path,{"answer":1},root=root)
    assert P.read_json_pin(pin) == {"answer":1}
    with pytest.raises(FileExistsError):
        P.atomic_json(path,{"answer":2},root=root)
    assert json.loads(path.read_text()) == {"answer":1}
    tpin = P.atomic_torch(root/"weights.pt",{"x":torch.tensor([1.,2.])},root=root)
    assert torch.equal(torch.load(tpin["path"],weights_only=True)["x"],torch.tensor([1.,2.]))
    npin = P.atomic_npz(root/"values.npz",{"x":np.arange(3)},root=root)
    with np.load(npin["path"],allow_pickle=False) as data:
        np.testing.assert_array_equal(data["x"],np.arange(3))
    def fail_writer(stream):
        stream.write(b"partial")
        raise RuntimeError("interruption")
    with pytest.raises(RuntimeError):
        P._atomic(root/"partial.json",fail_writer,root=root)
    assert not (root/"partial.json").exists() and not list(root.rglob("*.tmp"))


def test_receipts_link_fixed_code_numeric_context_and_complete_predecessors(prepared,monkeypatch):
    repo,plan = prepared
    plan = release(repo,plan,monkeypatch)
    numeric = {"fixture":"cpu2"}
    first = P.make_receipt(plan,"preflight",numeric=numeric,gates={"integrity":True,"historical_R1":True},source_root=repo)
    firstpin = P.write_receipt(plan,first)
    assert P.verified_receipt(firstpin,plan,source_root=repo)["job_id"] == "preflight"
    benchmark = P.make_receipt(plan,"bench",numeric=numeric_fixture(),predecessors=[firstpin],
                               gates={"integrity":True,"historical_R1":True},source_root=repo)
    pin = P.write_receipt(plan,benchmark)
    assert P.verified_receipt(pin,plan,stage="cpu_benchmark",source_root=repo)["predecessors"] == [firstpin]
    with pytest.raises(FileExistsError):
        P.write_receipt(plan,benchmark)
    with pytest.raises(ValueError,match="predecessor chain"):
        P.make_receipt(plan,"bench",numeric={},gates={"integrity":True,"historical_R1":True},source_root=repo)
    changed = copy.deepcopy(plan)
    changed["numeric_pins"]["cpu8"] = "1"*64
    with pytest.raises(ValueError,match="numerical context differs"):
        P.verified_receipt(pin,changed,source_root=repo)
    # Preflight's CPU2 context is deliberately not compared to CPU8 benchmark.
    assert P.verified_receipt(firstpin,changed,source_root=repo)["stage"] == "preflight"


@pytest.mark.parametrize("mutation",["code","job","stage","chain","payload"])
def test_forged_receipt_even_with_recomputed_payload_digest_is_rejected(prepared,monkeypatch,mutation):
    repo,plan = prepared
    plan = release(repo,plan,monkeypatch)
    receipt = P.make_receipt(plan,"preflight",numeric={},source_root=repo)
    if mutation == "code": receipt["reviewed_code_sha"] = "a"*40
    if mutation == "job": receipt["job_id"] = "not-planned"
    if mutation == "stage": receipt["stage"] = "metrics"
    if mutation == "chain":
        receipt["job_id"] = "bench";receipt["stage"] = "cpu_benchmark"
    if mutation == "payload": receipt["details"]["forged"] = True
    if mutation != "payload":
        receipt["receipt_sha256"] = P.digest({k:v for k,v in receipt.items() if k != "receipt_sha256"})
    pin = P.atomic_json(Path(plan["output_root"])/"forged.json",receipt,root=plan["output_root"])
    with pytest.raises(ValueError):
        P.verified_receipt(pin,plan,source_root=repo)


def test_actual_allocation_cpu_gpu_nodes_match_planned_resources():
    job = {"resources":{"cpus":8,"gpus":1,"requested_seconds":1000}}
    env = {"SLURM_JOB_ID":"123","SLURM_CPUS_PER_TASK":"8","SLURM_GPUS_ON_NODE":"1","SLURM_NNODES":"1"}
    assert P.validate_job_allocation(job,environment=env)["gpus"] == 1
    for key,value in (("SLURM_CPUS_PER_TASK","4"),("SLURM_GPUS_ON_NODE","0"),("SLURM_NNODES","2")):
        with pytest.raises(ValueError):
            P.validate_job_allocation(job,environment={**env,key:value})


def test_numeric_census_names_all_libraries_and_hardware_without_mutating_threads():
    before = torch.get_num_threads(),torch.get_num_interop_threads()
    numeric = P.numeric_context("cpu")
    assert all(numeric[k] for k in ("python","numpy","scipy","sklearn","torch"))
    assert numeric["threadpools"] and numeric["hardware"]["kind"] == "cpu"
    assert (torch.get_num_threads(),torch.get_num_interop_threads()) == before


def test_metadata_descendant_records_both_shas_but_source_archive_must_match(prepared,monkeypatch):
    repo,plan = prepared
    plan["reader"]["code_sha"] = "e"*40
    calls = []
    bad = [False]
    def git_output(args,**kwargs):
        if args == ["git","rev-parse","HEAD"]:
            return "d"*40+"\n"
        assert args[:3] == ["git","archive","--format=tar"] and args[3] == "e"*40
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer,mode="w") as archive:
            for i,path in enumerate(plan["reader"]["source_manifest"]):
                content = (repo/path).read_bytes()+(b"tampered" if bad[0] and i == 0 else b"")
                member = tarfile.TarInfo(path);member.size = len(content)
                archive.addfile(member,io.BytesIO(content))
        return buffer.getvalue()
    monkeypatch.setattr(P.subprocess,"check_output",git_output)
    monkeypatch.setattr(P.subprocess,"check_call",lambda args,**kwargs:calls.append(args) or 0)
    monkeypatch.setenv("EXPECTED_SHA","d"*40)
    assert P.checkout_identity(plan,source_root=repo) == {"code_sha":"d"*40,"reviewed_code_sha":"e"*40}
    assert calls == [["git","merge-base","--is-ancestor","e"*40,"d"*40]]
    bad[0] = True
    with pytest.raises(ValueError,match="does not contain"):
        P.checkout_identity(plan,source_root=repo)


def test_numerical_context_checked_before_fit_and_failed_receipt_can_preserve_mismatch(prepared,monkeypatch):
    repo,plan = prepared
    plan = release(repo,plan,monkeypatch)
    numeric = numeric_fixture()
    plan["numeric_pins"]["cpu8"] = P.digest(numeric)
    assert P.verify_numeric(plan,"cpu_benchmark",numeric)
    changed = copy.deepcopy(numeric);changed["numpy"] = "different"
    with pytest.raises(ValueError,match="differs from benchmark"):
        P.verify_numeric(plan,"cpu_benchmark",changed)
    changed["threads"]["intraop"] = 1
    with pytest.raises(ValueError,match="thread settings"):
        P.verify_numeric(plan,"cpu_benchmark",changed)
