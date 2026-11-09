"""Continuation 300M 5000 -> 19618: offline contracts of the wrapper and exact-resume continuity on CPU.

No cluster, GPU or submission. The continuity tests run the unchanged trainer: a run stopped at a kept
checkpoint and resumed in a NEW directory seeded by the wrapper must equal an uninterrupted run, bit for
bit, in weights, teacher, AdamW moments, random generators and records of the new updates.
"""
import copy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
mp = pytest.importorskip("torch.multiprocessing")

from wearusfm.training import run as R  # noqa: E402

from test_pretraining_loader import _tree  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("scaling_continue_test", ROOT / "scripts/scaling_continue.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
BASE = ROOT / "results/infra/scaling_300m_20261104"
CONTRACT = BASE / "training_source_contract.json"
PLAN = BASE / "continuation19618_20261109_v2/training_plan.json"  # v2 (09/11): import order fixed, identity pre-check gate
PLAN_V1 = BASE / "continuation19618_20261106/training_plan.json"  # superseded: job 63696864 FAILED at resume (08/11)
PRECHECK_JOB = "122"
SIGNATURE = {"test": "data"}
CODE, JOB, PLAN_SHA, PANEL = "d" * 40, "123", "e" * 64, "c" * 64
CONTROL_CONFIG = ROOT / "results/infra/tokenizer_execution_20261103/19618/T100-G/config.json"


@pytest.fixture(scope="module")
def context():
    return C.M.check_source_contract(CONTRACT)


@pytest.fixture
def plan():
    return json.loads(PLAN.read_text())


def file_pin(path):
    return {"sha256": R.sha256_file(path), "bytes": Path(path).stat().st_size}


# ---------------------------------------------------------------- published plan and recipe


def test_published_plan_is_valid_and_bound_to_the_current_sources(context, plan, capsys):
    C.validate_plan(plan, context)
    assert plan["execution_source_sha256"] == {p: R.sha256_file(ROOT / p) for p in C.EXECUTION_SOURCES}
    C.check_budget_freeze(plan, {})
    inventories = [json.loads((ROOT / plan["control"][k]["path"]).read_text()) for k in ("pose_inventory", "classification_inventory")]
    assert plan["control"]["artifacts"] == C.control_pins(*inventories) and len(plan["control"]["artifacts"]) == 19
    assert sum(name.endswith(("_train", "_val")) and name.startswith("pose_") and "reference" not in name
               for name in plan["control"]["artifacts"]) == 2
    assert sum(name.split("_", 1)[1] in C.CONTROL_DATASETS for name in plan["control"]["artifacts"]) == 5
    receipt = json.loads((ROOT / plan["previous"]["receipt_copy"]["path"]).read_text())
    C.check_previous_receipt(receipt, plan["previous"], context, plan["source_contract"]["sha256"])
    files = plan["previous"]["files"]
    assert files["checkpoint.pt"]["sha256"] == files["checkpoint_step0005000.pt"]["sha256"] == receipt["checkpoint_sha256"]
    assert plan["control"]["checkpoint"]["sha256"] == "a31c895061c1f273eaac2d796fc00cf76cffa4318b521a20be4d6ba65b63a056"
    assert 4 * plan["allocation"]["walltime_seconds"] / 3600 == 59.0
    assert C.main(["check-plan", "--plan", str(PLAN)]) == 0
    assert json.loads(capsys.readouterr().out)["runtime_readiness_checked"] is False


def test_v2_plan_supersedes_the_failed_v1_attempt_with_the_same_science(context, plan):
    """Same arm, dose, recipe, 5000 run, control and budget freeze; new execution sources (wrapper with torch first and the
    pre-check launcher), new output directories, the pre-check gate and a later retention horizon. The v1 plan is refused."""
    v1 = json.loads(PLAN_V1.read_text())
    with pytest.raises(ValueError, match="execution source pins"):  # it pins neither the pre-check launcher nor the new wrapper
        C.validate_plan(v1, context)
    with pytest.raises(ValueError, match="identity pre-check gate"):  # and it has no pre-check gate
        C.validate_plan({**v1, "execution_source_sha256": plan["execution_source_sha256"]}, context)
    assert {k for k in set(plan) | set(v1) if plan.get(k) != v1.get(k)} == {"execution_source_sha256", "identity_precheck",
                                                                               "superseded_attempt", "run_dir", "receipt_dir", "control"}
    assert {k for k in plan["control"] if plan["control"][k] != v1["control"][k]} == {"required_until_utc"}
    assert plan["control"]["required_until_utc"] == "2026-12-01T00:00:00Z" and v1["control"]["required_until_utc"] == "2026-11-22T00:00:00Z"
    assert plan["previous"] == v1["previous"] and plan["allocation"] == v1["allocation"] and plan["budget_freeze"] == v1["budget_freeze"]
    assert plan["run_dir"].endswith("/T100-G-300M-19618-from5000-v2") and plan["receipt_dir"].endswith("/continuation19618_v2")
    assert plan["identity_precheck"] == {"protocol": C.IDENTITY_PROTOCOL, "allocation": C.PRECHECK_ALLOCATION,
                                         "receipt_dir": plan["receipt_dir"] + "_identity_precheck"}
    assert plan["identity_precheck"]["allocation"] == {"partition": "lrd_all_serial", "cpus": 4, "memory": "24G", "walltime_seconds": 1200}
    superseded = plan["superseded_attempt"]
    assert (superseded["slurm_job_id"], superseded["run_dir"], superseded["receipt_dir"]) == ("63696864", v1["run_dir"], v1["receipt_dir"])
    assert superseded["plan_sha256"] == R.sha256_file(PLAN_V1) and (ROOT / superseded["diagnosis"]).is_file()
    assert set(plan["execution_source_sha256"]) - set(v1["execution_source_sha256"]) == {"scripts/slurm/scaling_identity_precheck.sbatch"}
    unchanged = {"scripts/scaling_train.py", "scripts/scaling_preflight.py", "scripts/slurm/scaling_continue.sbatch"}
    assert all(plan["execution_source_sha256"][p] == v1["execution_source_sha256"][p] for p in unchanged)
    assert plan["execution_source_sha256"]["scripts/scaling_continue.py"] != v1["execution_source_sha256"]["scripts/scaling_continue.py"]


def test_recipe_changes_only_output_dose_and_operational_timeout(context, plan):
    previous, cfg, argv = C.validate_plan(plan, context)
    old, new = R.config_to_dict(previous), R.config_to_dict(cfg)
    assert previous.max_steps == 5000 and cfg.max_steps == 19618
    assert {k for k in set(old) | set(new) if old.get(k) != new.get(k)} == {"max_steps"}
    assert cfg.model == previous.model and cfg.batch_size == 32 and cfg.accum_steps == 1 and cfg.seed == 0
    assert cfg.schedule.total_steps == 49046 and cfg.warmup_steps == 1962 and cfg.schedule.decay_start == 39237
    # No new warmup and no decay inside the tranche: every new update runs at the peak learning rate.
    assert {R._lr_at(step, cfg) for step in range(5000, 19618)} == {cfg.lr}
    assert all(R._lr_at(step, cfg) == R._lr_at(step, previous) for step in (0, 1961, 4999))
    R._check_resume_recipe(cfg, {"config": old})  # the trainer's own resume guard accepts exactly this change
    with pytest.raises(ValueError):
        R._check_resume_recipe(replace(cfg, lr=cfg.lr * 2), {"config": old})
    assert argv[0] == "scripts/train_jepa.py" and argv[-2:] == ["--time-limit-s", "51300"]
    assert not {"--resume", "--init-from", "--branch-from"} & set(argv)
    assert argv[argv.index("--out-dir") + 1] == plan["run_dir"]
    # Against the command really executed for the 5000 phase: only output, dose and timeout differ.
    executed = json.loads((BASE / "scientific5000_completion_20261105/launch.json").read_text())["command"]
    assert executed[:3] == ["torchrun", "--standalone", "--nproc_per_node=4"]
    recovered, executed = list(argv), executed[3:]
    for flag in ("--out-dir", "--max-steps", "--time-limit-s"):
        recovered[recovered.index(flag) + 1] = executed[executed.index(flag) + 1]
    assert recovered == executed


def test_control_configuration_is_the_one_verified_by_the_tokenizer_campaign_at_19618(context):
    control = replace(context["config_30m"], max_steps=19618)
    command = C.T.command_for(context["comparison"], "T100-G", 19618, 1000)  # the reviewed command of the control tranche
    assert command[3] == "scripts/train_jepa.py"
    assert C.CLI.build_config(C.CLI.build_parser().parse_args(command[4:])) == control
    assert R.config_to_dict(control) == json.loads(CONTROL_CONFIG.read_text())
    same = {k for k, v in R.config_to_dict(control).items() if v == R.config_to_dict(replace(context["config_300m"], max_steps=19618))[k]}
    assert set(R.config_to_dict(control)) - same == {"model"}  # candidate and control differ only in the model


def _mutations():
    def end(p): p["end_step"] = 49046
    def start(p): p["start_step"] = 0
    def added(p): p["additional_steps"] = 14617
    def gpus(p): p["allocation"]["gpus"] = 8
    def time(p): p["allocation"]["walltime_seconds"] = 86400
    def qos(p): p["allocation"]["qos"] = "boost_qos_bprod"
    def memory(p): p["memory"]["allocated_fraction_limit"] = 1.0
    def missing(p): del p["previous"]["files"]["metrics.jsonl"]
    def extra(p): p["previous"]["files"]["STOP"] = {"sha256": "a" * 64, "bytes": 1}
    def same(p): p["run_dir"] = p["previous"]["run_dir"]
    def inside(p): p["run_dir"] = p["previous"]["run_dir"] + "/continuation"
    def receipts(p): p["receipt_dir"] = p["run_dir"]
    def relative(p): p["run_dir"] = "runs/new"
    def sources(p): del p["execution_source_sha256"]["scripts/slurm/scaling_continue.sbatch"]
    def science(p): p["scientific_source_sha256"]["src/wearusfm/training/run.py"] = "a" * 64
    def contract(p): p["source_contract"]["sha256"] = "a" * 64
    def control(p): p["control"]["artifacts"] = {}
    def horizon(p): del p["control"]["required_until_utc"]
    def copy_(p): p["previous"]["receipt_copy"]["sha256"] = "a" * 64
    def precheck_missing(p): del p["identity_precheck"]
    def precheck_protocol(p): p["identity_precheck"]["protocol"] = "scaling-continuation-identity-precheck-20261106-v0"
    def precheck_same(p): p["identity_precheck"]["receipt_dir"] = p["receipt_dir"]
    def precheck_inside(p): p["identity_precheck"]["receipt_dir"] = p["run_dir"] + "/precheck"
    def precheck_in_5000(p): p["identity_precheck"]["receipt_dir"] = p["previous"]["run_dir"] + "/precheck"
    def precheck_relative(p): p["identity_precheck"]["receipt_dir"] = "results/precheck"
    def precheck_cpus(p): p["identity_precheck"]["allocation"]["cpus"] = 8
    def precheck_partition(p): p["identity_precheck"]["allocation"]["partition"] = "boost_usr_prod"
    def precheck_extra(p): p["identity_precheck"]["slurm_job_id"] = "1"
    return [end, start, added, gpus, time, qos, memory, missing, extra, same, inside, receipts, relative, sources, science,
            contract, control, horizon, copy_, precheck_missing, precheck_protocol, precheck_same, precheck_inside, precheck_in_5000,
            precheck_relative, precheck_cpus, precheck_partition, precheck_extra]


@pytest.mark.parametrize("change", _mutations(), ids=lambda f: f.__name__)
def test_plan_rejects_other_doses_resources_outputs_and_incomplete_pins(context, plan, change):
    change(plan)
    with pytest.raises(ValueError):
        C.validate_plan(plan, context)


@pytest.mark.parametrize("change", ["walltime", "segments", "cap", "control", "inventory", "protocol"])
def test_plan_must_request_exactly_the_frozen_budget_and_control(plan, tmp_path, change):
    freeze = json.loads((ROOT / plan["budget_freeze"]["path"]).read_text())
    if change == "walltime": freeze["training"]["allocation"]["walltime_seconds"] = 86400
    if change == "segments": freeze["training"]["segments"].append({"start_step": 19618, "end_step": 49046, "walltime_seconds": 1})
    if change == "cap": freeze["training"]["gpu_hours_cap"] = 58.0
    if change == "control": plan["control"]["checkpoint"]["sha256"] = "a" * 64
    if change == "inventory": plan["control"]["pose_inventory"]["sha256"] = "a" * 64
    if change == "protocol": freeze["protocol"] = "scaling-continuation19618-budget-freeze-20261105"
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(freeze))
    plan["budget_freeze"] = {"path": str(path), "sha256": R.sha256_file(path)}
    with pytest.raises(ValueError):
        C.check_budget_freeze(plan, {})


@pytest.mark.parametrize("change", [None, "gpu", "memory", "time", "node", "qos", "partition", "account"])
def test_actual_allocation_must_match_plan_including_explicit_qos(change):
    fields = {"NumNodes": "1", "NumCPUs": "32", "TimeLimit": "14:45:00", "AllocTRES": "cpu=32,mem=480G,node=1,gres/gpu=4",
              "Partition": "boost_usr_prod", "QOS": "normal", "Account": "iscrb_wearusfm"}
    if change == "gpu": fields["AllocTRES"] = "cpu=32,mem=480G,node=1,gres/gpu=8"
    if change == "memory": fields["AllocTRES"] = "cpu=32,mem=500G,node=1,gres/gpu=4"
    if change == "time": fields["TimeLimit"] = "1-00:00:00"
    if change == "node": fields["NumNodes"] = "2"
    if change == "qos": fields["QOS"] = "boost_qos_bprod"
    if change == "partition": fields["Partition"] = "boost_usr_dbg"
    if change == "account": fields["Account"] = "another_account"
    if change:
        with pytest.raises(ValueError): C.check_allocation_fields(fields, C.ALLOCATION, 4)
    else:
        C.check_allocation_fields(fields, C.ALLOCATION, 4)


# ---------------------------------------------------------------- the completed 5000 run is read-only evidence


@pytest.fixture
def old_run(tmp_path):
    directory = tmp_path / "old"
    directory.mkdir()
    for name in C.PREVIOUS_FILES:
        (directory / name).write_text("content of " + name)
    return {"run_dir": str(directory), "files": {name: file_pin(directory / name) for name in C.PREVIOUS_FILES}}


def test_previous_run_inventory_is_complete_and_seed_copies_are_byte_identical(old_run, tmp_path):
    states = C.previous_states(old_run)
    assert set(states) == {"previous:" + name for name in C.PREVIOUS_FILES}
    new = tmp_path / "new"
    seeded = C.seed_run_dir(old_run, new)
    assert sorted(p.name for p in new.iterdir()) == sorted(C.SEED_FILES) == sorted(seeded)
    assert all((new / name).read_bytes() == (Path(old_run["run_dir"]) / name).read_bytes() for name in C.SEED_FILES)
    C.P.check_unchanged(states)  # seeding reads the 5000 run, never writes it
    with pytest.raises(FileExistsError):
        C.seed_run_dir(old_run, new)


@pytest.mark.parametrize("change", ["extra", "stop", "missing", "content", "symlink", "seed_content"])
def test_changed_or_incomplete_previous_run_stops_before_any_new_output(old_run, tmp_path, change):
    directory = Path(old_run["run_dir"])
    if change == "extra": (directory / "checkpoint_step0019618.pt").write_text("unexpected")
    if change == "stop": (directory / "STOP").write_text("alarm")
    if change == "missing": (directory / "pose_panel.jsonl").unlink()
    if change == "content": (directory / "metrics.jsonl").write_text("content of metrics.jsonX")
    if change == "symlink":
        target = tmp_path / "elsewhere.pt"
        target.write_text("content of checkpoint.pt")
        (directory / "checkpoint.pt").unlink()
        (directory / "checkpoint.pt").symlink_to(target)
    if change == "seed_content":
        old_run["files"]["origin_sessions.json"]["sha256"] = "a" * 64
        with pytest.raises(ValueError):
            C.seed_run_dir(old_run, tmp_path / "new")
        return
    with pytest.raises(ValueError):
        C.previous_states(old_run)


@pytest.mark.parametrize("change", ["protocol", "incomplete", "dose", "checkpoint", "resume", "path", "contract", "sources"])
def test_previous_receipt_must_attest_the_pinned_scientific_phase(context, plan, change):
    receipt = json.loads((ROOT / plan["previous"]["receipt_copy"]["path"]).read_text())
    if change == "protocol": receipt["protocol"] = "scaling-production-memory-20261104-v1"
    if change == "incomplete": receipt["complete"] = False
    if change == "dose": receipt["checkpoint_step"] = 4999
    if change == "checkpoint": receipt["checkpoint_sha256"] = "a" * 64
    if change == "resume": receipt["resume_checkpoint_sha256"] = "a" * 64
    if change == "path": receipt["resume_checkpoint"] = "/elsewhere/checkpoint.pt"
    if change == "contract": receipt["training_source_contract_sha256"] = "a" * 64
    if change == "sources": receipt["scientific_source_sha256"]["src/wearusfm/training/run.py"] = "a" * 64
    with pytest.raises(ValueError):
        C.check_previous_receipt(receipt, plan["previous"], context, plan["source_contract"]["sha256"])


def test_scratch_retention_is_computed_from_real_modification_times(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "SCRATCH_PREFIX", str(tmp_path) + "/scratch/")
    (tmp_path / "scratch").mkdir()
    fresh, old, work = tmp_path / "scratch/fresh.npz", tmp_path / "scratch/old.npz", tmp_path / "work.json"
    for path in (fresh, old, work):
        path.write_text("x")
    now = datetime(2026, 11, 6, 18, tzinfo=timezone.utc)
    os.utime(fresh, (now.timestamp(), (now - timedelta(days=3)).timestamp()))
    os.utime(old, (now.timestamp(), (now - timedelta(days=30)).timestamp()))
    os.utime(work, (now.timestamp(), (now - timedelta(days=400)).timestamp()))  # outside scratch: no purge
    horizon = now + timedelta(days=16)
    assert C.check_retention([fresh, work], horizon, now, 53100) == now - timedelta(days=3) + timedelta(days=40)
    with pytest.raises(ValueError, match="retention"):
        C.check_retention([fresh, old], horizon, now, 53100)
    with pytest.raises(ValueError, match="horizon"):
        C.check_retention([fresh], now + timedelta(hours=10), now, 53100)


# ---------------------------------------------------------------- exact resume in a new directory equals an uninterrupted run


def _config(max_steps, **extra):
    return replace(R.small_config(datasets=None, max_steps=max_steps), stream_segment=4, ckpt_every=100, fixed_eval_every=2,
                   log_batch_origin=True, schedule=R.Schedule(kind="wsd", total_steps=20, keep_checkpoints=(4, 8)), **extra)


NAMES = ("checkpoint.pt", "fixed_eval_set.json", "origin_sessions.json")  # the tiny data have no pose panel


def _tree_hash(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(directory).iterdir())}


def _records(directory, name="metrics.jsonl"):
    """Records without the wall-clock fields (t_data_s, t_step_s, elapsed_s): everything else must be identical."""
    return [{k: v for k, v in r.items() if not k.startswith("t_") and k != "elapsed_s"} for r in C.read_lines(Path(directory) / name)]


def _assert_same_state(a, c, cfg):
    assert a["step"] == c["step"] == 8 and tuple(a["alarm_counts"]) == tuple(c["alarm_counts"])
    for part in ("student", "teacher"):
        assert list(a[part]) == list(c[part]) and all(torch.equal(a[part][k], c[part][k]) for k in a[part]), part
    assert a["optimizer"]["param_groups"] == c["optimizer"]["param_groups"]
    assert list(a["optimizer"]["state"]) == list(c["optimizer"]["state"])
    for index, moments in a["optimizer"]["state"].items():
        assert all(torch.equal(moments[k], c["optimizer"]["state"][index][k]) for k in ("step", "exp_avg", "exp_avg_sq")), index
    if "rank_rng" in a:  # one generator per DDP process
        assert len(a["rank_rng"]) == len(c["rank_rng"]) == a["world_size"]
        assert all(torch.equal(x["torch_rng"], y["torch_rng"]) for x, y in zip(a["rank_rng"], c["rank_rng"]))
    else:
        assert torch.equal(a["torch_rng"], c["torch_rng"])
    before = C.optimizer_mapping(a, cfg)
    assert before["adam_step_max"] == 8 and before["tensors_with_moments"] > 0


def _check_continuation(tmp_path, run, config=None, names=NAMES, logs=("fixed_eval.jsonl",)):
    """run(config, directory) trains with the unchanged trainer; a = uninterrupted, b = stopped at 4, c = b resumed elsewhere."""
    a, b, c = (tmp_path / name for name in "abc")
    config = config or _config
    cfg = config(8)
    run(cfg, a)
    run(config(4), b)
    assert (b / "checkpoint_step0000004.pt").read_bytes() == (b / "checkpoint.pt").read_bytes()  # as in the real 5000 run
    frozen = _tree_hash(b)
    previous = {"run_dir": str(b), "files": {name: file_pin(b / name) for name in names}}
    old = C.optimizer_mapping(torch.load(b / "checkpoint.pt", weights_only=False), config(4))
    assert old["adam_step_max"] == 4
    assert sorted(C.seed_run_dir(previous, c, names)) == sorted(names)
    run(cfg, c)
    assert _tree_hash(b) == frozen  # the first phase is untouched
    assert sorted(p.name for p in c.iterdir()) == sorted(["checkpoint.pt", "checkpoint_step0000008.pt", "config.json", "metrics.jsonl",
                                                          "summary.json", *names[1:], *logs])
    state_a, state_c = (torch.load(d / "checkpoint.pt", weights_only=False) for d in (a, c))
    _assert_same_state(state_a, state_c, cfg)
    named = torch.load(c / "checkpoint_step0000008.pt", weights_only=False)
    assert all(torch.equal(named["student"][k], state_c["student"][k]) for k in named["student"])
    # Records of the new updates only, identical to the same updates of the uninterrupted run.
    assert [r["step"] for r in _records(c)] == [5, 6, 7, 8] and _records(c) == _records(a)[4:]
    for name in (name for name in logs if name.endswith(".jsonl")):  # fixed diagnostics: no step 0 again, same values
        assert [r["step"] for r in _records(c, name)] == [6, 8], name
        assert _records(c, name) == [r for r in _records(a, name) if r["step"] > 4], name
    report = C.check_tranche_logs(c, b, cfg, start=4, end=8)
    assert report["metrics_records"] == 4 and report["lr_min"] == report["lr_max"] == cfg.lr
    C.check_optimizer_continuity(old, C.optimizer_mapping(state_c, cfg), added=4)
    with pytest.raises(ValueError):  # an optimizer restarted from zero would count only the new updates
        C.check_optimizer_continuity(old, C.optimizer_mapping(torch.load(b / "checkpoint.pt", weights_only=False), config(4)), added=4)
    assert json.loads((c / "summary.json").read_text())["steps"] == 8
    return a, b, c, cfg, report


@pytest.mark.parametrize("num_workers", [0, 2])
def test_resume_in_new_directory_equals_uninterrupted_run(tmp_path, num_workers):
    root, mpath = _tree(tmp_path)

    def run(cfg, directory):
        R.train(cfg, mpath, [root], directory, log=lambda m: None, num_workers=num_workers)

    a, b, c, cfg, _ = _check_continuation(tmp_path, run)
    if num_workers:
        return
    with pytest.raises(ValueError, match="ricetta diversa"):  # the trainer refuses a changed recipe on the seeded checkpoint
        R.train(replace(cfg, max_steps=10, lr=cfg.lr * 2), mpath, [root], c, log=lambda m: None, num_workers=num_workers)
    for name, text in (("fixed_eval_set.json", "[]"), ("origin_sessions.json", "[]")):
        d = tmp_path / ("bad_" + name)
        C.seed_run_dir({"run_dir": str(b), "files": {n: file_pin(b / n) for n in NAMES}}, d, NAMES)
        (d / name).write_text(text)
        with pytest.raises(ValueError, match="diverso da quello del run"):  # identity files are verified, not rewritten
            R.train(cfg, mpath, [root], d, log=lambda m: None, num_workers=num_workers)


def _ddp_worker(rank, world, store, root, mpath, out, max_steps):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        R.train(_config(max_steps, batch_size=4, prefetch=True), Path(mpath), [Path(root)], Path(out), log=lambda m: None,
                dist=R.DistContext(rank, world, rank))
    finally:
        dist.destroy_process_group()


def test_ddp_resume_in_new_directory_equals_uninterrupted_run_with_prefetch(tmp_path):
    root, mpath = _tree(tmp_path)

    def run(cfg, directory):
        mp.spawn(_ddp_worker, args=(2, str(tmp_path / ("store_" + directory.name)), str(root), str(mpath), str(directory), cfg.max_steps),
                 nprocs=2, join=True)

    a, _, c, _, _ = _check_continuation(tmp_path, run)
    state = torch.load(c / "checkpoint.pt", weights_only=False)
    assert state["world_size"] == 2 and not torch.equal(state["rank_rng"][0]["torch_rng"], state["rank_rng"][1]["torch_rng"])
    assert json.loads((c / "summary.json").read_text())["prefetch"] is True


def test_resume_with_pose_head_and_pose_panel_seeds_exactly_the_four_files_of_the_real_job(tmp_path):
    """The real recipe trains the pose head and logs the fixed pose panel: same continuity, with the real set of seeded files."""
    from test_pose_panel import _base, _multi_tree
    root, mpath, pose_root = _multi_tree(tmp_path)

    def config(max_steps):
        return replace(_base(pose_root), max_steps=max_steps, ckpt_every=100, warmup_steps=2, log_batch_origin=True, fixed_eval_every=2,
                       pose_panel_every=2, pose_panel_users=2, pose_panel_recordings=2, pose_panel_windows=2,
                       schedule=R.Schedule(kind="wsd", total_steps=20, keep_checkpoints=(4, 8)))

    def run(cfg, directory):
        R.train(cfg, mpath, [root], directory, log=lambda m: None)

    assert config(8).model.pose_head and config(8).jepa.pose_weight > 0
    _, b, c, cfg, report = _check_continuation(tmp_path, run, config, C.SEED_FILES,
                                               ("fixed_eval.jsonl", "pose_panel.jsonl", "pose_provenance.json"))
    assert report["pose_panel.jsonl"] == 2 and report["fixed_eval.jsonl"] == 2
    assert {r["panel_sha256"] for r in C.read_lines(b / "pose_panel.jsonl")} == {report["panel_sha256"]}
    assert [r["step"] for r in C.read_lines(b / "pose_panel.jsonl")] == [0, 2, 4]  # the step-0 panel stays only in the first phase
    for name, text in (("pose_panel_set.json", "{}"),):
        d = tmp_path / "bad_panel"
        C.seed_run_dir({"run_dir": str(b), "files": {n: file_pin(b / n) for n in C.SEED_FILES}}, d)
        (d / name).write_text(text)
        with pytest.raises(ValueError, match="diverso da quello del run"):  # a panel that is not the 5000 one is refused by the trainer
            R.train(cfg, mpath, [root], d, log=lambda m: None)


def test_optimizer_mapping_rejects_moments_that_do_not_belong_to_the_parameters(tmp_path):
    root, mpath = _tree(tmp_path)
    cfg = replace(R.small_config(datasets=None, max_steps=2), ckpt_every=100)
    R.train(cfg, mpath, [root], tmp_path / "run", log=lambda m: None)
    good = torch.load(tmp_path / "run/checkpoint.pt", weights_only=False)
    report = C.optimizer_mapping(good, cfg)
    assert report["adam_step_max"] == 2 and report["trainable_tensors"] >= report["tensors_with_moments"] > 0
    assert report["parameters"] == R.count_parameters(cfg.model) == json.loads((tmp_path / "run/summary.json").read_text())["parameters"]
    with pytest.raises(ValueError, match="model size"):
        C.optimizer_mapping(good, cfg, expected_parameters=C.PARAMETERS)
    for change in ("shape", "order", "group", "teacher", "student_key", "step"):
        state = copy.deepcopy(good)
        moments, index = state["optimizer"]["state"], next(iter(good["optimizer"]["state"]))
        if change == "shape": moments[index]["exp_avg"] = torch.zeros(7, 5)
        if change == "order":
            first, second = [i for i in moments if moments[i]["exp_avg"].shape != moments[index]["exp_avg"].shape][0], index
            moments[first], moments[second] = moments[second], moments[first]
        if change == "group": state["optimizer"]["param_groups"][0]["params"].pop()
        if change == "teacher": state["teacher"].pop(next(iter(state["teacher"])))
        if change == "student_key": state["student"]["unexpected"] = torch.zeros(1)
        if change == "step":
            for values in moments.values(): values["step"] = torch.tensor(1.)
        with pytest.raises(ValueError):
            C.optimizer_mapping(state, cfg)


# ---------------------------------------------------------------- complete endpoint, tranche records and per-rank memory


def tiny_state(cfg, step, signature=SIGNATURE):
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([parameter], lr=cfg.lr, weight_decay=cfg.weight_decay)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.state[parameter]["step"].fill_(step)
    return {"step": step, "config": R.config_to_dict(cfg), "world_size": 4,
            "stream": {"num_workers": 6, "segment": cfg.stream_segment, "world_size": 4},
            "rank_rng": [{"torch_rng": torch.Generator().manual_seed(i).get_state(),
                          "cuda_rng": torch.zeros(16, dtype=torch.uint8)} for i in range(4)],
            "pose_stats_sha256": "757f2aeebf688063747d50e55d983dd6b3c4cea268ce43d2bae06f3e99524a7e",
            "optimizer": optimizer.state_dict(), "data_signature": signature, "alarm_counts": (0, 0),
            "student": {"weight": torch.ones(3)}, "teacher": {"weight": torch.zeros(3)}}


def stub_mapping(state, cfg, expected_parameters=None):
    return {"parameters": 1, "trainable_tensors": 1, "tensors_with_moments": 1, "tensors_without_moments": [],
            "adam_step_min": state["step"], "adam_step_max": state["step"], "adam_steps": {"weight": state["step"]}}


def write_endpoint(run, out, cfg, signature=SIGNATURE, plan_sha=PLAN_SHA):
    """What the four-rank trainer leaves at 19618 in the seeded directory, with tiny checkpoints."""
    steps = range(C.START + 1, C.END + 1)
    (run / "metrics.jsonl").write_text("".join(json.dumps({"step": s, "lr": R._lr_at(s - 1, cfg)}) + "\n" for s in steps))
    (run / "fixed_eval.jsonl").write_text("".join(json.dumps({"step": s}) + "\n" for s in steps if s % cfg.fixed_eval_every == 0))
    (run / "pose_panel.jsonl").write_text("".join(json.dumps({"step": s, "panel_sha256": PANEL}) + "\n"
                                                    for s in steps if s % cfg.pose_panel_every == 0))
    (run / "pose_provenance.json").write_text("{}")
    (run / "config.json").write_text(json.dumps(R.config_to_dict(cfg)))
    (run / "summary.json").write_text(json.dumps({"stopped": "max_steps", "steps": C.END, "world_size": 4, "parameters": C.PARAMETERS,
                                                  "accum_steps": 1, "micro_batch": 8, "batch_per_process": 8, "gpu_max_memory_gib": 40.0}))
    payload = tiny_state(cfg, C.END, signature)
    for name in ("checkpoint_step0019618.pt", "checkpoint.pt"):
        torch.save(payload, run / name)
    for rank in range(4):
        (out / f"rank_{rank}.json").write_text(json.dumps({
            "protocol": C.RANK_PROTOCOL, "rank": rank, "local_rank": rank, "world_size": 4, "returncode": 0, "code_sha": CODE,
            "slurm_job_id": JOB, "plan_sha256": plan_sha, "peak_allocated_bytes": (40 if rank == 2 else 36) * 2**30,
            "peak_reserved_bytes": 52 * 2**30, "device_memory_bytes": 64 * 2**30}))


def old_panel(old):
    (old / "pose_panel.jsonl").write_text("".join(json.dumps({"step": s, "panel_sha256": PANEL}) + "\n" for s in (0, 500, 5000)))


def precheck_path(plan_path):
    return Path(json.loads(plan_path.read_text())["identity_precheck"]["receipt_dir"]) / C.PRECHECK_RECEIPT


def write_precheck(plan_path, **changes):
    """What a passed identity pre-check of this plan and code leaves (the fields the GPU job requires)."""
    plan = json.loads(plan_path.read_text())
    receipt = {"protocol": C.IDENTITY_PROTOCOL, "identical": True, "code_sha": CODE, "plan_sha256": R.sha256_file(plan_path),
               "slurm_job_id": PRECHECK_JOB, "node": "lrdn0001", "import_order": {"torch_before_numpy": True},
               "previous_identity_sha256": {name: plan["previous"]["files"][name]["sha256"] for name in C.IDENTITY_FILES},
               "comparison": {name: {"equal": True} for name in C.IDENTITY_FILES}, **changes}
    path = precheck_path(plan_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt))
    return path


@pytest.fixture
def endpoint(context, plan, tmp_path, monkeypatch):
    """Synthetic complete endpoint with tiny checkpoints; the 300M model is never built in tests."""
    monkeypatch.setattr(C, "optimizer_mapping", stub_mapping)
    for name in ("run_dir", "receipt_dir"):
        plan[name] = str(tmp_path / name)
    plan["previous"]["run_dir"] = str(tmp_path / "old")
    _, cfg, _ = C.validate_plan(plan, context)
    run, old, out = Path(plan["run_dir"]), Path(plan["previous"]["run_dir"]), Path(plan["receipt_dir"])
    for directory in (run, old, out):
        directory.mkdir()
    old_panel(old)
    for name in C.IDENTITY_FILES:
        (run / name).write_text(json.dumps([name]))
        plan["previous"]["files"][name] = file_pin(run / name)
    (run / C.MARKER).write_text("{}")
    write_endpoint(run, out, cfg)
    before = stub_mapping({"step": C.START}, None)
    return lambda: C.verify_outputs(plan, run, out, cfg, SIGNATURE, before, CODE, JOB, PLAN_SHA), run, out, cfg


def rewrite(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_complete_endpoint_reports_resumed_optimizer_tranche_and_memory_of_every_rank(endpoint):
    verify, run, _, cfg = endpoint
    evidence = verify()
    assert evidence["summary"]["steps"] == 19618 and evidence["named"].name == "checkpoint_step0019618.pt"
    assert evidence["tranche"] == {"metrics_records": 14618, "first_step": 5001, "last_step": 19618, "lr_min": cfg.lr, "lr_max": cfg.lr,
                                   "fixed_eval.jsonl": 29, "pose_panel.jsonl": 29, "panel_sha256": PANEL}
    memory = evidence["memory"]
    assert memory["peak_allocated_gib"] == 40.0 and memory["allocated_fraction_by_rank"] == [.5625, .5625, .625, .5625]
    assert memory["reserved_fraction_by_rank"] == [.8125] * 4 and memory["allocated_fraction_limit"] == .9
    assert evidence["optimizer"]["adam_step_max"] == 19618 and "adam_steps" not in evidence["optimizer"]


@pytest.mark.parametrize("change", ["time_limit", "stop", "old_checkpoint", "leftover", "configuration", "dose", "teacher",
                                     "missing_named", "identity", "microbatch"])
def test_no_endpoint_for_partial_restarted_or_altered_outputs(endpoint, change):
    verify, run, _, _ = endpoint
    if change == "time_limit": rewrite(run / "summary.json", lambda s: s.update(stopped="limite di tempo", steps=19000))
    if change == "stop": (run / "STOP").write_text("allarme")
    if change == "old_checkpoint": (run / "checkpoint_step0005000.pt").write_text("the 5000 update was executed again")
    if change == "leftover": (run / "checkpoint.tmp").write_text("partial")
    if change == "configuration": rewrite(run / "config.json", lambda c: c.update(lr=3e-4))
    if change == "microbatch": rewrite(run / "summary.json", lambda s: s.update(accum_steps=2, micro_batch=4))
    if change == "missing_named": (run / "checkpoint_step0019618.pt").unlink()
    if change == "identity": (run / "pose_panel_set.json").write_text("[]")
    if change in ("dose", "teacher"):
        payload = torch.load(run / "checkpoint.pt", weights_only=False)
        if change == "dose": payload["step"] = 19617
        if change == "teacher": payload["teacher"]["weight"][0] = 1
        torch.save(payload, run / "checkpoint.pt")
    with pytest.raises((ValueError, FileNotFoundError)):
        verify()


@pytest.mark.parametrize("change", ["gap", "repeat_5000", "short", "warmup", "fixed_extra", "panel_changed", "panel_missing"])
def test_tranche_records_must_be_exactly_the_new_updates_on_the_original_schedule(endpoint, change):
    verify, run, _, _ = endpoint
    metrics = (run / "metrics.jsonl").read_text().splitlines()
    if change == "gap": del metrics[100]
    if change == "repeat_5000": metrics.insert(0, json.dumps({"step": 5000, "lr": 1e-4}))
    if change == "short": metrics = metrics[:-1]
    if change == "warmup": metrics[0] = json.dumps({"step": 5001, "lr": 1e-4 / 1962})
    (run / "metrics.jsonl").write_text("\n".join(metrics) + "\n")
    if change == "fixed_extra": (run / "fixed_eval.jsonl").write_text(json.dumps({"step": 0}) + "\n" + (run / "fixed_eval.jsonl").read_text())
    if change == "panel_changed":
        (run / "pose_panel.jsonl").write_text((run / "pose_panel.jsonl").read_text().replace(PANEL, "f" * 64, 1))
    if change == "panel_missing":
        (run / "pose_panel.jsonl").write_text("\n".join((run / "pose_panel.jsonl").read_text().splitlines()[1:]) + "\n")
    with pytest.raises(ValueError):
        verify()


@pytest.mark.parametrize("change", ["allocated", "reserved_below", "reserved_above", "summary", "job", "code", "plan", "failed", "missing",
                                     "float"])
def test_memory_gate_needs_allocated_and_reserved_peaks_of_every_rank(endpoint, change):
    verify, _, out, _ = endpoint
    path = out / "rank_3.json"
    if change == "allocated": rewrite(path, lambda r: r.update(peak_allocated_bytes=58 * 2**30, peak_reserved_bytes=60 * 2**30))
    if change == "reserved_below": rewrite(path, lambda r: r.update(peak_reserved_bytes=30 * 2**30))
    if change == "reserved_above": rewrite(path, lambda r: r.update(peak_reserved_bytes=65 * 2**30))
    if change == "summary": rewrite(path, lambda r: r.update(peak_allocated_bytes=41 * 2**30))
    if change == "job": rewrite(path, lambda r: r.update(slurm_job_id="124"))
    if change == "code": rewrite(path, lambda r: r.update(code_sha="a" * 40))
    if change == "plan": rewrite(path, lambda r: r.update(plan_sha256="a" * 64))
    if change == "failed": rewrite(path, lambda r: r.update(returncode=1))
    if change == "float": rewrite(path, lambda r: r.update(peak_reserved_bytes=52.0 * 2**30))
    if change == "missing": path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        verify()


def test_optimizer_continuity_rejects_reset_lost_or_overrun_history():
    before = stub_mapping({"step": 5000}, None)
    C.check_optimizer_continuity(before, stub_mapping({"step": 19618}, None))
    partial = {**stub_mapping({"step": 19618}, None), "adam_steps": {"weight": 19618, "unused_head": 3}}
    C.check_optimizer_continuity({**before, "adam_steps": {"weight": 5000}}, partial)  # a tensor may get its first update later
    for after in (stub_mapping({"step": 14618}, None), stub_mapping({"step": 19619}, None),
                  {**stub_mapping({"step": 19618}, None), "adam_steps": {}},
                  {**stub_mapping({"step": 19618}, None), "trainable_tensors": 2}):
        with pytest.raises(ValueError):
            C.check_optimizer_continuity(before, after)


# ---------------------------------------------------------------- one DDP rank around the unchanged entry point


@pytest.fixture
def rank(context, plan, tmp_path, monkeypatch):
    for name in ("run_dir", "receipt_dir"):
        plan[name] = str(tmp_path / name)
        Path(plan[name]).mkdir()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    _, _, argv = C.validate_plan(plan, context)
    identity = {"protocol": C.PROTOCOL, "plan_sha256": R.sha256_file(plan_path), "code_sha": CODE, "slurm_job_id": JOB}
    launch, marker = Path(plan["receipt_dir"]) / "launch.json", Path(plan["run_dir"]) / C.MARKER
    launch.write_text(json.dumps({**identity, "trainer_argv": argv}))
    marker.write_text(json.dumps(identity))
    calls = []
    monkeypatch.setattr(C.P, "code_identity", lambda expected: expected)
    monkeypatch.setattr(C.CLI, "main", lambda arguments: calls.append(arguments) or 0)
    monkeypatch.setattr(C.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(C.torch.cuda, "max_memory_allocated", lambda device: (30 + device) * 2**30)
    monkeypatch.setattr(C.torch.cuda, "max_memory_reserved", lambda device: 50 * 2**30)
    monkeypatch.setattr(C.torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=64 * 2**30))
    for key, value in {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "4", "SLURM_JOB_ID": JOB}.items():
        monkeypatch.setenv(key, value)
    return plan_path, launch, marker, argv, calls, Path(plan["receipt_dir"])


def test_rank_runs_only_the_unchanged_trainer_entry_point_and_records_its_peaks(rank):
    plan_path, launch, _, argv, calls, receipts = rank
    C.worker(plan_path, CODE, R.sha256_file(launch))
    assert calls == [argv[1:]]
    record = json.loads((receipts / "rank_1.json").read_text())
    assert record["rank"] == record["local_rank"] == 1 and record["returncode"] == 0 and record["protocol"] == C.RANK_PROTOCOL
    assert record["peak_allocated_bytes"] == 31 * 2**30 and record["peak_reserved_bytes"] == 50 * 2**30
    assert record["device_memory_bytes"] == 64 * 2**30 and record["plan_sha256"] == R.sha256_file(plan_path)
    with pytest.raises(FileExistsError):  # a second trainer in the same allocation cannot replace the record
        C.worker(plan_path, CODE, R.sha256_file(launch))


@pytest.mark.parametrize("change", ["launch_sha", "marker_job", "marker_missing", "stop", "argv", "local_rank", "world", "no_job"])
def test_rank_refuses_an_unprepared_directory_or_another_launch_before_training(rank, monkeypatch, change):
    plan_path, launch, marker, _, calls, receipts = rank
    sha = R.sha256_file(launch)
    if change == "launch_sha": sha = "a" * 64
    if change == "marker_job": rewrite(marker, lambda m: m.update(slurm_job_id="124"))
    if change == "marker_missing": marker.unlink()
    if change == "stop": (marker.parent / "STOP").write_text("allarme")
    if change == "argv":
        rewrite(launch, lambda r: r["trainer_argv"].extend(["--init-from", "/elsewhere.pt"]))
        sha = R.sha256_file(launch)
    if change == "local_rank": monkeypatch.setenv("LOCAL_RANK", "0")
    if change == "world": monkeypatch.setenv("WORLD_SIZE", "8")
    if change == "no_job": monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises((ValueError, FileNotFoundError)):
        C.worker(plan_path, CODE, sha)
    assert calls == [] and not list(receipts.glob("rank_*.json"))


# ---------------------------------------------------------------- the 30M control at 19618, re-verified before training


@pytest.fixture
def control(context, plan, monkeypatch):
    """Receipt and inventories are the real repository files; only the files that live on the cluster are replaced by recorders."""
    real, seen, identities = C.M.resolve_pin, [], []

    def resolve(pin, root=C.M.ROOT):
        if not Path(pin["path"]).is_absolute():
            return real(pin, root)
        seen.append(pin["path"])
        return {"path": pin["path"], "sha256": pin["sha256"]}

    monkeypatch.setattr(C.M, "resolve_pin", resolve)
    monkeypatch.setattr(C.T, "checkpoint_identity", lambda *arguments: identities.append(arguments))
    monkeypatch.setattr(C, "SCRATCH_PREFIX", "/not-a-local-path/")  # retention from real modification times has its own test
    now = datetime(2026, 11, 6, 18, tzinfo=timezone.utc)
    return lambda states: C.check_control(plan, context, SIGNATURE, states, now), plan["control"], seen, identities


def test_control_is_reverified_from_reviewed_receipt_inventories_checkpoint_and_all_caches(context, control):
    check, pins, seen, identities = control
    states = {}
    assert check(states) == {"required_until_utc": "2026-12-01T00:00:00Z", "earliest_scratch_expiry_utc": None}
    (path, cfg, step, signature), = identities
    assert path == pins["checkpoint"]["path"] and step == 19618 and signature == SIGNATURE
    assert R.config_to_dict(cfg) == json.loads(CONTROL_CONFIG.read_text())
    assert sorted(seen) == sorted([pins["checkpoint"]["path"], *(pin["path"] for pin in pins["artifacts"].values())]) and len(seen) == 20
    assert set(states) == {"control_receipt", "control_pose_inventory", "control_classification_inventory", "control_checkpoint",
                           *("control:" + name for name in pins["artifacts"])}


@pytest.mark.parametrize("change", ["arm", "incomplete", "dose", "code", "path", "checkpoint", "cache", "missing_cache", "horizon"])
def test_control_with_other_identity_caches_or_too_short_horizon_stops_before_training(control, tmp_path, change):
    check, pins, _, _ = control
    receipt = json.loads((ROOT / pins["receipt"]["path"]).read_text())
    if change == "arm": receipt["arm"] = "T100-L"
    if change == "incomplete": receipt["complete"] = False
    if change == "dose": receipt["checkpoint_step"] = 5000
    if change == "code": receipt["code_sha"] = "a" * 40
    if change == "path": receipt["checkpoint"] = "/elsewhere/checkpoint_step0019618.pt"
    if change == "checkpoint": receipt["checkpoint_sha256"] = "a" * 64
    if change == "cache": pins["artifacts"]["p2_epn612"]["sha256"] = "a" * 64
    if change == "missing_cache": del pins["artifacts"]["pose_val"]
    if change == "horizon": pins["required_until_utc"] = "2026-11-06T20:00:00Z"  # before the end of a 14:45 allocation
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    pins["receipt"] = {"path": str(path), "sha256": R.sha256_file(path)}
    with pytest.raises(ValueError):
        check({})


# ---------------------------------------------------------------- the whole job, with the GPU trainer simulated


@pytest.fixture
def cluster(context, plan, tmp_path, monkeypatch):
    """execute() end to end on local stand-ins. Replaced: scheduler, git identity, the remote inputs, the 300M model, the
    control artifacts (verified in their own test) and the GPU trainer, simulated by writing what it leaves."""
    inputs = {}
    for name in context["contract"]["inputs"]:
        path = tmp_path / f"input_{name}.json"
        path.write_text(json.dumps({"scales": {}, "window_s": {}, "name": name}))
        inputs[name] = {"path": str(path), "sha256": R.sha256_file(path)}
    local = {**context, "contract": {**context["contract"], "inputs": inputs}}
    controls = []
    monkeypatch.setattr(C.M, "check_source_contract", lambda path: local)
    monkeypatch.setattr(C.P, "code_identity", lambda expected: expected)
    monkeypatch.setattr(C.R, "check_input_files", lambda *arguments: None)
    monkeypatch.setattr(C, "verify_allocation", lambda plan: JOB)
    monkeypatch.setattr(C, "optimizer_mapping", stub_mapping)
    monkeypatch.setattr(C, "check_previous_receipt", lambda *arguments: None)  # tested on the real receipt above
    monkeypatch.setattr(C, "check_control", lambda plan, context, signature, states: controls.append(signature) or
                        {"required_until_utc": plan["control"]["required_until_utc"], "earliest_scratch_expiry_utc": None})
    monkeypatch.setenv("SLURM_JOB_ID", JOB)
    previous_cfg, cfg = context["config_300m"], replace(context["config_300m"], max_steps=C.END)
    signature = json.loads(json.dumps(R.calibration_signature(cfg, inputs["manifest"]["path"], {}, {})))
    old, run, out, receipt = tmp_path / "old", tmp_path / "run", tmp_path / "receipts", tmp_path / "receipt5000.json"
    old.mkdir()
    torch.save(tiny_state(previous_cfg, C.START, signature), old / "checkpoint.pt")
    shutil.copyfile(old / "checkpoint.pt", old / "checkpoint_step0005000.pt")
    old_panel(old)
    for name in C.PREVIOUS_FILES - {"checkpoint.pt", "checkpoint_step0005000.pt", "pose_panel.jsonl"}:
        (old / name).write_text(json.dumps([name]))
    shutil.copyfile(ROOT / plan["previous"]["receipt_copy"]["path"], receipt)
    plan["previous"].update(run_dir=str(old), files={name: file_pin(old / name) for name in C.PREVIOUS_FILES},
                            receipt={**plan["previous"]["receipt"], "path": str(receipt)})
    plan.update(run_dir=str(run), receipt_dir=str(out))
    plan["identity_precheck"]["receipt_dir"] = str(tmp_path / "precheck")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    write_precheck(plan_path)
    behaviour = {}

    def trainer(command, cwd, env, stdout, stderr, check):
        assert command == [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                           str(ROOT / "scripts/scaling_continue.py"), "worker", "--plan", str(plan_path), "--expected-code-sha", CODE,
                           "--launch-sha256", R.sha256_file(out / "launch.json")]
        assert cwd == C.ROOT and env["WEARUSFM_CODE_SHA"] == CODE and Path(stdout.name) == out / "training.log" and check is True
        # What every rank finds: the byte-identical 5000 state and identity files, in a directory that is not the 5000 run.
        assert sorted(p.name for p in run.iterdir()) == sorted([*C.SEED_FILES, C.MARKER])
        assert all((run / name).read_bytes() == (old / name).read_bytes() for name in C.SEED_FILES)
        if behaviour.get("fail"):
            raise subprocess.CalledProcessError(1, command)
        write_endpoint(run, out, cfg, signature, R.sha256_file(plan_path))
        if behaviour.get("after"):
            behaviour["after"]()

    monkeypatch.setattr(C.subprocess, "run", trainer)
    return plan_path, old, run, out, behaviour, controls


def test_job_seeds_a_new_directory_trains_once_and_certifies_the_endpoint_with_the_5000_run_untouched(cluster):
    plan_path, old, run, out, _, controls = cluster
    frozen = _tree_hash(old)
    receipt = C.execute(plan_path, CODE, out)
    assert _tree_hash(old) == frozen and len(controls) == 1
    assert sorted(p.name for p in out.iterdir()) == ["config.json", "launch.json", "rank_0.json", "rank_1.json", "rank_2.json",
                                                      "rank_3.json", "receipt.json", "summary.json", "training.log"]
    assert json.loads((out / "receipt.json").read_text())["trained_state_sha256"] == receipt["trained_state_sha256"]
    assert receipt["protocol"] == C.PROTOCOL and receipt["complete"] is True and receipt["arm"] == "T100-G-300M"
    assert (receipt["start_step"], receipt["checkpoint_step"], receipt["additional_steps"]) == (5000, 19618, 14618)
    assert receipt["checkpoint"] == str(run / "checkpoint_step0019618.pt") and receipt["resume_checkpoint"] == str(run / "checkpoint.pt")
    assert receipt["previous_run_dir"] == str(old) and receipt["previous_checkpoint_sha256"] == R.sha256_file(old / "checkpoint.pt")
    assert receipt["previous_run_unchanged"] is True and receipt["previous_trained_state_sha256"] != receipt["trained_state_sha256"]
    assert receipt["scientific_endpoint"] is True and receipt["automatic_continuation"] is False
    assert receipt["further_training_authorized"] is False and receipt["slurm_job_id"] == JOB and receipt["code_sha"] == CODE
    assert receipt["trainer_argv"][-2:] == ["--time-limit-s", "51300"] and receipt["tranche"]["metrics_records"] == 14618
    assert receipt["memory"]["peak_allocated_gib"] == 40.0 and receipt["optimizer"]["adam_step_max"] == 19618
    launch, marker = json.loads((out / "launch.json").read_text()), json.loads((run / C.MARKER).read_text())
    assert launch["trainer_argv"] == receipt["trainer_argv"] and set(launch["seeded_files"]) == set(C.SEED_FILES) == set(marker["seeded_files"])
    assert {"previous:checkpoint.pt", "previous_receipt", "budget_freeze", "input:manifest", "execution_plan",
            "identity_precheck"} <= set(launch["verified_inputs"])
    assert marker["previous_run_dir"] == str(old) and marker["slurm_job_id"] == JOB and launch["previous_optimizer"]["adam_step_max"] == 5000
    precheck = precheck_path(plan_path)  # the GPU job records which pre-check admitted it
    assert launch["identity_precheck"] == receipt["identity_precheck"] == {
        "receipt": str(precheck), "sha256": R.sha256_file(precheck), "slurm_job_id": PRECHECK_JOB, "node": "lrdn0001",
        "comparison": {name: {"equal": True} for name in C.IDENTITY_FILES}}
    assert launch["verified_inputs"]["identity_precheck"]["sha256"] == R.sha256_file(precheck)
    with pytest.raises(ValueError, match="no overwrite"):  # a second job cannot reuse or replace the endpoint
        C.execute(plan_path, CODE, out)


@pytest.mark.parametrize("change", ["stop_in_5000", "extra_in_5000", "existing_run", "existing_receipts", "input", "wrong_out",
                                    "precheck_missing", "precheck_failed", "precheck_protocol", "precheck_code", "precheck_plan",
                                    "precheck_order", "precheck_identity", "precheck_job"])
def test_job_stops_before_creating_any_output(cluster, tmp_path, change):
    plan_path, old, run, out, _, _ = cluster
    if change == "stop_in_5000": (old / "STOP").write_text("allarme")
    if change == "extra_in_5000": (old / "checkpoint_step0019618.pt").write_text("unexpected")
    if change == "existing_run": run.mkdir()
    if change == "existing_receipts": out.mkdir()
    if change == "input": (tmp_path / "input_scales.json").write_text(json.dumps({"scales": {"x": 2.0}, "window_s": {}}))
    if change == "precheck_missing": precheck_path(plan_path).unlink()
    if change == "precheck_failed": write_precheck(plan_path, identical=False)  # the pre-check found a difference: no GPU job
    if change == "precheck_protocol": write_precheck(plan_path, protocol="scaling-diag-fixedeval-20261108-v1")
    if change == "precheck_code": write_precheck(plan_path, code_sha="a" * 40)  # passed on another checkout
    if change == "precheck_plan": write_precheck(plan_path, plan_sha256="a" * 64)  # passed on another plan
    if change == "precheck_order": write_precheck(plan_path, import_order={"torch_before_numpy": False})
    if change == "precheck_identity": write_precheck(plan_path, previous_identity_sha256={n: "a" * 64 for n in C.IDENTITY_FILES})
    if change == "precheck_job": write_precheck(plan_path, slurm_job_id="")
    with pytest.raises(ValueError):
        C.execute(plan_path, CODE, tmp_path / "other" if change == "wrong_out" else out)
    assert not (out / "launch.json").exists() and not (run / C.MARKER).exists() and not (tmp_path / "other").exists()


@pytest.mark.parametrize("change", ["trainer_fails", "time_limit", "5000_modified", "5000_gains_file", "missing_rank"])
def test_job_without_a_complete_untouched_endpoint_writes_no_receipt_and_cannot_be_retried_blindly(cluster, change):
    plan_path, old, run, out, behaviour, _ = cluster
    if change == "trainer_fails": behaviour["fail"] = True
    if change == "time_limit": behaviour["after"] = lambda: rewrite(run / "summary.json", lambda s: s.update(stopped="limite di tempo"))
    if change == "5000_modified": behaviour["after"] = lambda: (old / "metrics.jsonl").write_text("rewritten")
    if change == "5000_gains_file": behaviour["after"] = lambda: (old / "checkpoint_step0019618.pt").write_text("misplaced")
    if change == "missing_rank": behaviour["after"] = lambda: (out / "rank_3.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError, subprocess.CalledProcessError)):
        C.execute(plan_path, CODE, out)
    assert not (out / "receipt.json").exists() and (out / "launch.json").exists()
    with pytest.raises(ValueError, match="no overwrite"):
        C.execute(plan_path, CODE, out)


# ---------------------------------------------------------------- v2: the CPU identity pre-check before the GPU job


def test_precheck_argv_is_the_plan_recipe_on_one_cpu_process_without_ddp(context, plan):
    _, cfg, argv = C.validate_plan(plan, context)
    cli = C.precheck_argv(argv, Path("/elsewhere/trainer_out"))
    assert cli[0] == "scripts/train_jepa.py" and "--distributed" in argv and "--distributed" not in cli
    for flag, value in (("--out-dir", "/elsewhere/trainer_out"), ("--num-workers", "0"), ("--device", "cpu")):
        assert cli.count(flag) == 1 and cli[cli.index(flag) + 1] == value, flag
    kept = [a for a in argv if a != "--distributed"]
    kept[kept.index("--out-dir") + 1], kept[kept.index("--num-workers") + 1] = "/elsewhere/trainer_out", "0"
    assert cli == kept + ["--device", "cpu"]  # the 5000 command leaves --device to its default: here it is explicit
    # Same RunConfig as the GPU job: device, workers, DDP and output directory are arguments of train(), not of the recipe.
    assert R.config_to_dict(C.CLI.build_config(C.CLI.build_parser().parse_args(cli[1:]))) == R.config_to_dict(cfg)
    explicit = C.precheck_argv(argv + ["--device", "cuda"], Path("/x"))  # an explicit device is replaced, not kept
    assert explicit.count("--device") == 1 and explicit[explicit.index("--device") + 1] == "cpu"
    for flag, value in (("--num-workers", "3"), ("--out-dir", "/y")):
        with pytest.raises(ValueError, match="duplicate argument"):
            C.precheck_argv(argv + [flag, value], Path("/x"))
    with pytest.raises(ValueError, match="duplicate argument"):
        C.precheck_argv(argv + ["--device", "cuda", "--device", "cuda"], Path("/x"))
    with pytest.raises(ValueError, match="missing output directory"):
        C.precheck_argv([a for a in argv if a not in ("--out-dir", plan["run_dir"])], Path("/x"))


def test_compare_identity_reports_where_the_sets_differ():
    expected = {"seed": [0, 5151], "windows": [["a", None, 0, 10], ["b", 1, 5, 15]], "batch_sha256": ["x", "y"]}
    assert C.compare_identity(expected, json.loads(json.dumps(expected))) == {
        "equal": True, "keys_differing": [], "batch_sha256_equal": [2, 2, 2], "windows_equal": [2, 2, 2]}
    assert C.compare_identity(expected, {**expected, "batch_sha256": ["x", "z"]}) == {  # the failure of 08/11: same windows, other content
        "equal": False, "keys_differing": ["batch_sha256"], "batch_sha256_equal": [1, 2, 2], "windows_equal": [2, 2, 2]}
    assert C.compare_identity(expected, {**expected, "windows": expected["windows"][:1], "extra": 1}) == {
        "equal": False, "keys_differing": ["extra", "windows"], "batch_sha256_equal": [2, 2, 2], "windows_equal": [1, 2, 1]}
    assert C.compare_identity(["s1", "s2", "s3"], ["s1", "s9", "s3"]) == {"equal": False, "length": [3, 3], "first_difference": 1}
    assert C.compare_identity(["s1"], ["s1"]) == {"equal": True, "length": [1, 1], "first_difference": None}
    assert C.compare_identity(["s1"], {"a": 1}) == {"equal": False}


def test_identity_sets_are_exactly_the_files_the_unchanged_trainer_checks_at_resume(tmp_path):
    """On tiny data: what identity_sets computes without a model is byte for byte what train() wrote and will verify at resume."""
    root, mpath = _tree(tmp_path)
    cfg = _config(4)
    R.train(cfg, mpath, [root], tmp_path / "run", log=lambda m: None, num_workers=0)
    computed = C.identity_sets(cfg, mpath, [root], {}, None)
    assert set(computed) == {"fixed_eval_set.json", "origin_sessions.json"}  # the tiny data have no pose panel
    for name, value in computed.items():
        written = json.loads((tmp_path / "run" / name).read_text())
        assert written == value and C.compare_identity(written, value)["equal"], name
    assert computed["fixed_eval_set.json"]["seed"] == [cfg.seed, R.FIXED_EVAL_SEED] and len(computed["origin_sessions.json"]) > 1
    assert C.identity_sets(replace(cfg, fixed_eval_every=0, log_batch_origin=False), mpath, [root], None, None) == {}
    # Another seed or batch size is another fixed set: the comparison sees it, as the trainer did on 08/11.
    other = C.identity_sets(replace(cfg, seed=cfg.seed + 1), mpath, [root], {}, None)["fixed_eval_set.json"]
    report = C.compare_identity(computed["fixed_eval_set.json"], other)
    assert report["equal"] is False and "seed" in report["keys_differing"]


def test_import_order_and_numeric_libraries_describe_this_process():
    order, modules = C.import_order(), list(sys.modules)
    assert order["positions"]["torch"] == modules.index("torch") and order["positions"]["numpy"] == modules.index("numpy")
    assert order["torch_before_numpy"] == (modules.index("torch") < modules.index("numpy"))
    libraries = C.numeric_libraries()
    assert libraries is None or (isinstance(libraries, list) and all(isinstance(x, str) for x in libraries))  # Linux only


PRECHECK_FIELDS = {"JobId": PRECHECK_JOB, "JobState": "RUNNING", "Partition": "lrd_all_serial", "Account": "iscrb_wearusfm",
                   "NumNodes": "1", "NumCPUs": "4", "TimeLimit": "00:20:00", "AllocTRES": "cpu=4,mem=24G,node=1"}


@pytest.mark.parametrize("change", [None, {"Partition": "boost_usr_prod"}, {"Account": "other_account"}, {"NumNodes": "2"},
                                    {"NumCPUs": "8"}, {"TimeLimit": "00:30:00"}, {"TimeLimit": "04:00:00"},
                                    {"AllocTRES": "cpu=4,mem=24G,node=1,gres/gpu=1"}, {"AllocTRES": "cpu=8,mem=24G,node=1"},
                                    {"AllocTRES": "cpu=4,mem=32G,node=1"}, {"JobState": "PENDING"}, {"JobId": "9"}])
def test_actual_precheck_allocation_must_be_the_small_serial_one_of_the_plan(plan, monkeypatch, change):
    fields = {**PRECHECK_FIELDS, **(change or {})}
    monkeypatch.setenv("SLURM_JOB_ID", PRECHECK_JOB)
    monkeypatch.setattr(C.subprocess, "check_output", lambda command, text: " ".join(f"{k}={v}" for k, v in fields.items()) + "\n"
                        if command == ["scontrol", "show", "job", "-o", PRECHECK_JOB] else pytest.fail(str(command)))
    if change is None:
        assert C.verify_precheck_allocation(plan) == PRECHECK_JOB
        C.check_precheck_fields({**fields, "AllocTRES": "cpu=4,mem=24576M,node=1"}, plan)  # the other spelling of 24G
    else:
        with pytest.raises(ValueError):
            C.verify_precheck_allocation(plan)
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(ValueError, match="real SLURM allocation"):
        C.verify_precheck_allocation(plan)


@pytest.fixture
def precheck(context, plan, tmp_path, monkeypatch):
    """identity_precheck() end to end on local stand-ins: scheduler, git identity, the remote inputs and the identity files of the
    5000 run. The trainer CLI is replaced by what train_jepa.main does up to train() (parse, configure, call), as in the real job
    where train() itself is replaced; the identity sets by the published values (their computation has its own test above)."""
    inputs = {}
    for name in context["contract"]["inputs"]:
        path = tmp_path / f"input_{name}.json"
        path.write_text(json.dumps({"scales": {}, "window_s": {}, "name": name}))
        inputs[name] = {"path": str(path), "sha256": R.sha256_file(path)}
    local = {**context, "contract": {**context["contract"], "inputs": inputs}}
    monkeypatch.setattr(C.M, "check_source_contract", lambda path: local)
    monkeypatch.setattr(C.P, "code_identity", lambda expected: expected)
    monkeypatch.setattr(C.R, "check_input_files", lambda *arguments: None)
    monkeypatch.setattr(C, "verify_precheck_allocation", lambda plan: PRECHECK_JOB)
    monkeypatch.setattr(C, "import_order", lambda: {"torch_before_numpy": True, "positions": {"torch": 1, "numpy": 2, "scipy": None}})
    old, out = tmp_path / "old", tmp_path / "precheck"
    old.mkdir()
    sets = {"fixed_eval_set.json": {"seed": [0, 5151], "windows": [["a/1/1", None, 0, 10]], "batch_sha256": ["x"]},
            "pose_panel_set.json": {"seed": [0, 7272], "batch_sha256": ["p"]}, "origin_sessions.json": ["a/1/1", "b/2/1"]}
    for name, value in sets.items():
        (old / name).write_text(json.dumps(value))
    plan["previous"].update(run_dir=str(old), files={**plan["previous"]["files"], **{name: file_pin(old / name) for name in sets}})
    plan.update(run_dir=str(tmp_path / "run"), receipt_dir=str(tmp_path / "receipts"))
    plan["identity_precheck"]["receipt_dir"] = str(out)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    calls, behaviour = [], {}

    def cli_main(argv):
        args = C.CLI.build_parser().parse_args(argv)
        cfg = C.CLI.build_config(args)
        calls.append(list(argv))
        if behaviour.get("seed"): cfg = replace(cfg, seed=cfg.seed + 1)  # noqa: E701
        if behaviour.get("input"): Path(inputs["scales"]["path"]).write_text("{}")  # noqa: E701
        C.R.train(cfg, args.manifest, args.root, args.out_dir, scales={}, device=args.device, num_workers=args.num_workers,
                  time_limit_s=args.time_limit_s, log=print, window_s={}, init_from=args.init_from, dist=None,
                  branch_from=args.branch_from)
        return 0

    def identity_sets(cfg, manifest, roots, scales, window_s):
        computed = json.loads(json.dumps(sets))
        if behaviour.get("differ"):  # the failure of 08/11: same windows, other content of the batches
            computed["fixed_eval_set.json"]["batch_sha256"] = ["y"]
        return computed

    monkeypatch.setattr(C.CLI, "main", cli_main)
    monkeypatch.setattr(C, "identity_sets", identity_sets)
    return plan_path, out, behaviour, calls


def test_precheck_runs_the_trainer_cli_on_cpu_and_writes_the_receipt_the_gpu_job_accepts(precheck):
    plan_path, out, _, calls = precheck
    trainer = C.R.train
    receipt = C.identity_precheck(plan_path, CODE, out)
    assert C.R.train is trainer and len(calls) == 1 and sorted(p.name for p in out.iterdir()) == [C.PRECHECK_RECEIPT]
    assert json.loads((out / C.PRECHECK_RECEIPT).read_text()) == json.loads(json.dumps(receipt))
    plan = json.loads(plan_path.read_text())
    assert receipt["protocol"] == C.IDENTITY_PROTOCOL and receipt["identical"] is True and receipt["code_sha"] == CODE
    assert receipt["plan_sha256"] == R.sha256_file(plan_path) and receipt["slurm_job_id"] == PRECHECK_JOB
    assert receipt["previous_identity_sha256"] == {n: plan["previous"]["files"][n]["sha256"] for n in C.IDENTITY_FILES}
    assert all(receipt["comparison"][n]["equal"] for n in C.IDENTITY_FILES)
    assert receipt["cli_call"] == {"device": "cpu", "num_workers": 0, "distributed": False}
    assert calls[0] == receipt["precheck_argv"][1:] and "--distributed" not in calls[0]
    assert receipt["precheck_argv"][receipt["precheck_argv"].index("--out-dir") + 1] == str(out / "trainer_out")
    assert {"execution_plan", "input:manifest", "previous:fixed_eval_set.json"} <= set(receipt["verified_inputs"])
    # The consumer of the GPU job accepts exactly this receipt, for this plan and code only.
    states = {}
    admitted = C.check_identity_precheck(plan, CODE, C.P.file_identity(plan_path), states)
    assert admitted["slurm_job_id"] == PRECHECK_JOB and admitted["sha256"] == R.sha256_file(out / C.PRECHECK_RECEIPT)
    assert states["identity_precheck"]["sha256"] == admitted["sha256"]
    with pytest.raises(ValueError, match="not passed"):
        C.check_identity_precheck(plan, "a" * 40, C.P.file_identity(plan_path), {})
    with pytest.raises(ValueError, match="no overwrite"):  # one pre-check per plan: no second run over the receipt
        C.identity_precheck(plan_path, CODE, out)


@pytest.mark.parametrize("change", ["differ", "seed", "input", "existing", "elsewhere", "order"])
def test_precheck_refuses_differences_other_recipes_changed_inputs_and_reused_outputs(precheck, tmp_path, monkeypatch, change):
    plan_path, out, behaviour, calls = precheck
    trainer, plan = C.R.train, json.loads(plan_path.read_text())
    if change in ("differ", "seed", "input"):
        behaviour[change] = True
    if change == "existing":
        out.mkdir()
    if change == "order":  # a process where numpy came first: the receipt says so and the GPU job refuses it
        monkeypatch.setattr(C, "import_order", lambda: {"torch_before_numpy": False, "positions": {"torch": 9, "numpy": 2, "scipy": None}})
        C.identity_precheck(plan_path, CODE, out)
        with pytest.raises(ValueError, match="not passed"):
            C.check_identity_precheck(plan, CODE, C.P.file_identity(plan_path), {})
        return
    message = {"differ": "identity sets differ", "seed": "CLI configuration differs", "input": "input changed",
               "existing": "no overwrite", "elsewhere": "wrong pre-check directory"}[change]
    with pytest.raises(ValueError, match=message):
        C.identity_precheck(plan_path, CODE, tmp_path / "elsewhere" if change == "elsewhere" else out)
    assert C.R.train is trainer  # the trainer entry point is restored whatever happened
    if change == "differ":  # the receipt records where the sets differ, and the GPU job refuses it
        receipt = json.loads((out / C.PRECHECK_RECEIPT).read_text())
        assert receipt["identical"] is False and receipt["comparison"]["fixed_eval_set.json"]["batch_sha256_equal"] == [0, 1, 1]
        assert receipt["comparison"]["fixed_eval_set.json"]["windows_equal"] == [1, 1, 1]
        with pytest.raises(ValueError, match="not passed"):
            C.check_identity_precheck(plan, CODE, C.P.file_identity(plan_path), {})
    else:
        assert not (out / C.PRECHECK_RECEIPT).exists()
    if change in ("existing", "elsewhere"):
        assert calls == []  # refused before computing anything
