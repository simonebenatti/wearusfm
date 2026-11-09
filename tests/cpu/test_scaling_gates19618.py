"""Mechanical gates of the 19618 chain between its stages: no review is repeated, so every gate is tested here.

The gates before the aggregates and before the comparison are those reviewed for the 5000 chain; they are run on
the states really collected there and must accept them, and reproduce the reviewed comparison manifest. The live
gates (checkouts, published documents, outputs, queue, space) and the rule that later checkouts only add documents
run on a scratch repository and a scratch ledger. The hand-over from the training wrapper to the extraction is
exercised on the files really written by the wrapper. Nothing reaches Leonardo or the common ledger.
"""
import calendar
import copy
from dataclasses import replace
import importlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


plans = module("plans19618_fixture", "tests/cpu/test_scaling_plans19618.py")
continuation = module("continuation19618_fixture", "tests/cpu/test_scaling_continue.py")
context, plan, cluster = continuation.context, continuation.plan, continuation.cluster  # fixtures of the training stage
S, load, sha, body, INFRA, PACKAGE = plans.S, plans.load, plans.sha, plans.body, plans.INFRA, plans.PACKAGE
ARM, CODE, OLD = plans.ARM, plans.CODE, plans.OLD
OLD_CODE = "aef665ba7ea66fc80291846487907bb3ef9107df"
FINAL = "final_stages_20261106/"
QUOTA = """Disk quotas for prj 20147512 (pid 20147512):
     Filesystem  kbytes   quota   limit   grace   files   quota   limit   grace
/leonardo_work/IscrB_WearUsFM
                1048868200  1073741824       0       - 1512642       0       0       -
"""
QUEUE = "63693570|LONG-FIT-G39237-pre-ninapro_db2-F2-20261105-A1|RUNNING|8|None\n63695634|SCALING-300M-cont19618-checkout-20261106|PENDING|2|QOSMaxCpuPerUserLimit\n"


# ---------------------------------------------------------------------------------------------
# Fits and aggregates: the reviewed 5000 gates on the states collected at 5000

def old_fit_state():
    state, resources = load(FINAL + "live_state_20261106_fits_complete.json"), load("cpu_fit_release_20261105/probe_fit_resources.json")
    shards = {}
    for s in state["shards"]:
        report = {**s["report"], "datasets": {ds: {"folds": [{"fold": f} for f in folds]} for ds, folds in s["datasets"].items()},
                  "audit_summary": {**s["audit_summary"], "unresolved_fits": []}}
        shards[s["name"]] = {"report": report, "oof": s["oof"], "audit": s["audit"]}
    return resources, state["plan"]["sha256"], shards


def run_fit_state(change=None):
    resources, plan_sha, shards = old_fit_state()
    if change:
        change(resources, shards)
    return S.fit_state(resources, plan_sha, OLD_CODE, shards, protocol=OLD["probe"], step=5000)


def test_fit_gate_accepts_the_fourteen_fits_accepted_at_5000():
    facts = run_fit_state()
    assert facts == {"shards": 14, "fits": 420, "convergence_warnings": 0, "reached_limit": 0, "unresolved_fits": 0,
                     "all_converged_by_criterion": True, "numeric_sha256": load("cpu_fit_release_20261105/probe_fit_resources.json")["numeric_sha256"]}
    first = "T100-G-30M-natural-p1-01"
    warned = run_fit_state(lambda r, s: s[first]["report"]["audit_summary"].update(
        converged_by_criterion=False, convergence_warnings=2, reached_limit=1, unresolved_fits=[{"C": 100.0}]))
    # Convergence is reported here and stops the chain at the aggregates, as in the reviewed 5000 chain.
    assert (warned["all_converged_by_criterion"], warned["convergence_warnings"], warned["reached_limit"], warned["unresolved_fits"]) == (False, 2, 1, 1)


FIRST = "T100-G-30M-natural-p1-01"


@pytest.mark.parametrize("change, message", [
    (lambda r, s: s.pop(FIRST), "differ from the published fit plan"),
    (lambda r, s: s.update(extra=s[FIRST]), "differ from the published fit plan"),
    (lambda r, s: s[FIRST]["report"].update(complete=False), "shard identity differs"),
    (lambda r, s: s[FIRST]["report"].update(kind="benchmark"), "shard identity differs"),
    (lambda r, s: s[FIRST]["report"].update(step=19618), "shard identity differs"),
    (lambda r, s: s[FIRST]["report"].update(code_sha=CODE), "shard identity differs"),
    (lambda r, s: s[FIRST]["report"].update(plan_sha256="d" * 64), "shard identity differs"),
    (lambda r, s: s[FIRST]["report"].update(protocol=S.C.PROTOCOL), "shard identity differs"),
    (lambda r, s: s[FIRST]["report"].update(cell=S.CELLS[1]), "differ from the fit plan"),
    (lambda r, s: s[FIRST]["report"]["datasets"].update(extra={"folds": []}), "differ from the fit plan"),
    (lambda r, s: next(iter(s[FIRST]["report"]["datasets"].values()))["folds"].pop(), "differ from the fit plan"),
    (lambda r, s: s[FIRST]["oof"].update(sha256="d" * 64), "does not pin the files next to it"),
    (lambda r, s: s[FIRST]["audit"].update(path="/elsewhere/audit.jsonl"), "does not pin the files next to it"),
    (lambda r, s: s[FIRST]["report"].update(numeric_sha256="d" * 64), "numeric context differs"),
    (lambda r, s: s[FIRST]["report"]["audit_summary"].update(fits=6), "unexpected number of fits"),
    (lambda r, s: [x.update(folds=x["folds"][:-1]) for x in r["shards"][:1]] and
     [next(iter(s[r["shards"][0]["shard_name"]]["report"]["datasets"].values()))["folds"].pop()] and
     s[r["shards"][0]["shard_name"]]["report"]["audit_summary"].update(fits=s[r["shards"][0]["shard_name"]]["report"]["audit_summary"]["fits"] - 7),
     "exactly once")])
def test_fit_gate_stops_before_the_aggregates(change, message):
    with pytest.raises(ValueError, match=message):
        run_fit_state(change)


def old_comparison(change=None, relabel=False):
    """Arguments of comparison_inputs from the aggregates, shards, ridge and benchmarks collected at 5000."""
    state, reviewed = load(FINAL + "aggregates_state_20261106_1319.json"), load(FINAL + "comparison_inputs.json")
    text = (lambda data: data) if not relabel else (lambda data: plans_relabel(data, reviewed))
    aggregates = {}
    for a in state["aggregates"]:
        folder = INFRA / FINAL / "collected_aggregates_20261106_1319" / a["id"]
        report = json.loads(text((folder / "aggregate.json").read_text()))
        assert sha(folder / "aggregate.json") == a["report"]["sha256"] and sha(folder / "aggregate.audit.jsonl") == a["audit"]["sha256"]
        aggregates[report["cell"], report["task"]] = {
            "report": report, "pin": {k: a["report"][k] for k in ("path", "sha256", "bytes")}, "oof": a["oof"], "audit": a["audit"],
            "audit_lines": len((folder / "aggregate.audit.jsonl").read_bytes().splitlines())}
    ridge = INFRA / "cpu_fit_release_20261105/submissions/collected_jobs/SCALING-pose-ridge-20261105/report.json"
    assert sha(ridge) == reviewed["pose_report"]["sha256"]
    arguments = dict(
        probe=reviewed["probe_plan"], pose=reviewed["pose_plan"], aggregates=aggregates,
        pose_report={"report": json.loads(text(ridge.read_text())), "pin": reviewed["pose_report"]},
        code_sha=CODE if relabel else OLD_CODE, shards={s["name"]: s["shard"] for s in state["shards"]},
        resources=load("cpu_fit_release_20261105/probe_fit_resources.json"), anchors=S.oof_anchors(),
        benchmarks={c: json.loads(text((INFRA / "cpu_readers_20261105/pose_benchmark" / c / "report.json").read_text())) for c in S.CELLS},
        module=S.M if relabel else importlib.import_module("scaling_compare"))
    if change:
        change(arguments)
    return S.comparison_inputs(**arguments)


def plans_relabel(text, reviewed):
    """A report of the 5000 chain with the identity of the 19618 chain; plans, shards and windows keep their pins."""
    for old, new in ((OLD["probe"], S.C.PROTOCOL), (OLD["pose"], S.P.PROTOCOL), ('"step": 5000', '"step": 19618'), (OLD_CODE, CODE)):
        text = text.replace(old, new)
    return text


def test_comparison_gate_reproduces_the_manifest_reviewed_at_5000():
    assert old_comparison() == load(FINAL + "comparison_inputs.json")


def test_comparison_gate_builds_the_19618_manifest_from_reports_of_the_19618_chain():
    inputs, old = old_comparison(relabel=True), load(FINAL + "comparison_inputs.json")
    assert inputs["protocol"] == S.M.INPUT_PROTOCOL and inputs["checkpoint_step"] == 19618 and inputs["sources"] == S.M.source_hashes()
    assert list(inputs) == list(old) and {k: inputs[k] for k in ("probe_plan", "pose_plan", "probe_reports", "pose_report")} == {
        k: old[k] for k in ("probe_plan", "pose_plan", "probe_reports", "pose_report")}
    assert [Path(p["path"]).parent.name for p in inputs["probe_reports"]] == [f"{c}-{t}" for c in S.CELLS for t in S.TASKS]
    with pytest.raises(ValueError, match="aggregate identity differs"):  # a 5000 report is not a 19618 report
        old_comparison(lambda a: a.update(module=S.M))


def test_evaluated_windows_are_those_accepted_at_5000():
    anchors = S.oof_anchors()
    assert anchors["uci_emg"] == ["132e4e8dbed63a9340585f235c8b68375364da0a0310827249446bddd85bd3fd", 364, 19]  # Review 10
    assert list(map(lambda ds: anchors[ds][2], S.DATASETS)) == [32, 8, 8, 314, 19] and set(anchors) == set(S.DATASETS)
    with pytest.raises(ValueError, match="six accepted 5000 aggregates"):
        S.oof_anchors(ROOT / "tests")


KEY = (S.CELLS[1], "p1")


@pytest.mark.parametrize("change, message", [
    (lambda a: a["aggregates"].pop(KEY), "all six aggregates"),
    (lambda a: a["shards"].pop(FIRST), "collected shards differ from the fit plan"),
    (lambda a: a["aggregates"][KEY]["report"].update(complete=False), "aggregate identity differs"),
    (lambda a: a["aggregates"][KEY]["report"].update(kind="shard"), "aggregate identity differs"),
    (lambda a: a["aggregates"][KEY]["report"].update(step=19618), "aggregate identity differs"),
    (lambda a: a["aggregates"][KEY]["report"].update(readout="matched"), "aggregate identity differs"),
    (lambda a: a["aggregates"][KEY]["report"].update(arm=S.ARMS[0]), "aggregate identity differs"),
    (lambda a: a["aggregates"][KEY]["report"].update(code_sha=CODE), "aggregate identity differs"),
    (lambda a: a["aggregates"][KEY]["report"].update(plan_sha256="d" * 64), "aggregate identity differs"),
    (lambda a: a["aggregates"][KEY]["report"].update(numeric_sha256="d" * 64), "numeric context differs from the fits"),
    (lambda a: a["aggregates"][KEY]["report"]["shards"].reverse(), "did not read the accepted shards"),
    (lambda a: a["aggregates"][KEY]["report"]["shards"][0].update(sha256="d" * 64), "did not read the accepted shards"),
    (lambda a: a["shards"][FIRST].update(sha256="d" * 64), "did not read the accepted shards"),
    (lambda a: a["aggregates"][KEY]["oof"].update(sha256="d" * 64), "does not pin the files next to it"),
    (lambda a: a["aggregates"][KEY]["report"].update(audit_sha256="d" * 64), "does not pin the files next to it"),
    (lambda a: a["aggregates"][KEY].update(audit_lines=83), "unexpected number of audited fits"),
    (lambda a: a["aggregates"][KEY]["report"]["audit_summary"].update(fits=56), "unexpected number of audited fits"),
    (lambda a: a["aggregates"][KEY]["report"]["datasets"].pop("ninapro_db6"), "datasets, folds or evaluated windows differ"),
    (lambda a: a["aggregates"][KEY]["report"]["datasets"]["ninapro_db3"]["folds"].pop(), "datasets, folds or evaluated windows differ"),
    (lambda a: a["aggregates"][KEY]["report"]["datasets"]["ninapro_db3"].update(n_oof_windows=1), "datasets, folds or evaluated windows differ"),
    (lambda a: a["aggregates"][KEY]["report"]["datasets"]["ninapro_db3"].update(identity_sha256="d" * 64), "datasets, folds or evaluated windows differ"),
    (lambda a: a["anchors"].update(ninapro_db2=["d" * 64, 38544, 32]), "datasets, folds or evaluated windows differ"),
    (lambda a: a["aggregates"][KEY]["report"].update(reading_status="solver_not_converged"), "not converged by criterion"),
    (lambda a: a["aggregates"][KEY]["report"]["audit_summary"].update(converged_by_criterion=False), "not converged by criterion"),
    (lambda a: a["aggregates"][KEY]["report"]["audit_summary"].update(unresolved_fits=[{"C": 1.0}]), "not converged by criterion"),
    (lambda a: a["aggregates"][KEY]["report"].update(projection_sha256="d" * 64), "different plan contents or projections"),
    (lambda a: a["aggregates"][KEY]["report"].update(plan_content_sha256="d" * 64), "different plan contents or projections"),
    (lambda a: a["pose_report"]["report"].update(status="train_only_timing"), "pose report identity differs"),
    (lambda a: a["pose_report"]["report"].update(plan_sha256="d" * 64), "pose report identity differs"),
    (lambda a: a["pose_report"]["report"]["cells"].pop(S.CELLS[2]), "pose report identity differs"),
    (lambda a: a["pose_report"]["report"]["runtime"].update(code_sha=CODE), "pose report identity differs"),
    (lambda a: a["benchmarks"].pop(S.CELLS[0]), "runtime different from its benchmarks"),
    (lambda a: a["benchmarks"][S.CELLS[0]]["runtime"].update(torch_threads=4), "runtime different from its benchmarks"),
    (lambda a: a["benchmarks"][S.CELLS[2]]["runtime"]["versions"].update(numpy="other"), "runtime different from its benchmarks"),
    (lambda a: a["aggregates"][KEY]["pin"].update(path="aggregate.json"), "absolute"),
    (lambda a: a["aggregates"][KEY].update(pin=a["aggregates"][(S.CELLS[1], "p2")]["pin"]), "six distinct|duplicate")])
def test_comparison_gate_stops_before_the_manifest(change, message):
    with pytest.raises(ValueError, match=message):
        old_comparison(change)


# ---------------------------------------------------------------------------------------------
# Hand-over from the training wrapper to the extraction, on the files written by the wrapper itself

def test_endpoint_written_by_the_training_wrapper_is_the_one_accepted_by_the_extraction(cluster, context, monkeypatch):
    plan_path, _, run, out, _, _ = cluster
    continuation.C.execute(plan_path, continuation.CODE, out)
    listing = sorted(p.name for p in out.iterdir())
    training = json.loads(plan_path.read_text())
    job = S.training_job(training)
    assert listing == sorted(job["exact"]) and set(job["files"]) == set(listing)  # the nine files the collection expects
    assert job["pinned"] == {"checkpoint_step0019618.pt": str(run / "checkpoint_step0019618.pt")} and Path(job["pinned"]["checkpoint_step0019618.pt"]).is_file()
    paths = {"checkpoint": run / "checkpoint_step0019618.pt", "config": out / "config.json", "receipt": out / "receipt.json",
             "summary": out / "summary.json"}
    files = {name: S.pin(str(path), path.read_bytes()) for name, path in paths.items()}
    receipt, summary = json.loads(paths["receipt"].read_text()), json.loads(paths["summary"].read_text())
    built = S.extraction_plan(load("extraction_plan_5000_20261105.json"), training, files, receipt, summary, sha(plan_path))
    endpoint = built["checkpoints"][ARM]
    # The receipt names the plan that ran (hash), its execution sources and the identity pre-check that admitted the job:
    # the plan, the sources or the pre-check of another attempt (the failed v1 above all) are refused, not only other paths.
    assert receipt["plan_sha256"] == sha(plan_path) and receipt["execution_source_sha256"] == training["execution_source_sha256"]
    assert receipt["identity_precheck"]["receipt"] == training["identity_precheck"]["receipt_dir"] + "/identity_precheck.json"
    v1 = json.loads((PACKAGE / "training_plan.json").read_text())
    for plan_doc, plan_sha, message in ((training, sha(PACKAGE / "training_plan.json"), "receipt differs: plan_sha256"),
                                        ({**training, "execution_source_sha256": v1["execution_source_sha256"]}, sha(plan_path),
                                         "receipt differs: execution_source_sha256"),
                                        ({**training, "identity_precheck": {**training["identity_precheck"],
                                                                            "receipt_dir": training["receipt_dir"] + "_v1"}},
                                         sha(plan_path), "receipt differs: identity_precheck"),
                                        (v1, sha(PACKAGE / "training_plan.json"), "not the planned one")):
        with pytest.raises(ValueError, match=message):
            S.extraction_plan(load("extraction_plan_5000_20261105.json"), plan_doc, files, receipt, summary, plan_sha)
    assert endpoint["sha256"] == receipt["checkpoint_sha256"] == sha(paths["checkpoint"]) and receipt["protocol"] == S.E.TRAIN_PROTOCOL

    # The extractor reads those four files: configuration, summary and receipt pass, up to the load of the weights.
    class Reached(Exception):
        pass

    def stop(*arguments, **options):
        raise Reached

    monkeypatch.setattr(S.E.torch, "load", stop)
    contract = {**context["contract"], "sha256": sha(INFRA / "training_source_contract.json")}
    expected = continuation.R.config_to_dict(replace(context["config_300m"], max_steps=19618))
    with pytest.raises(Reached):
        S.E.load_teacher_endpoint(endpoint, expected, ARM, "0" * 64, {}, contract)
    for field, value, message in (("previous_run_unchanged", False, "exact continuation"), ("start_step", 0, "exact continuation"),
                                  ("protocol", "scaling-training-20261104-v1", "incomplete/wrong training receipt"),
                                  ("training_source_contract_sha256", "d" * 64, "scientific training source identity"),
                                  ("checkpoint_step", 5000, "training receipt mismatch")):
        altered = out.parent / ("receipt_%s.json" % field)
        altered.write_bytes(body({**receipt, field: value}))
        with pytest.raises(ValueError, match=message):
            S.E.load_teacher_endpoint({**endpoint, "receipt_path": str(altered), "receipt_sha256": sha(altered)}, expected, ARM,
                                      "0" * 64, {}, contract)
    with pytest.raises(ValueError, match="source config differs"):
        S.E.load_teacher_endpoint(endpoint, {**expected, "max_steps": 5000}, ARM, "0" * 64, {}, contract)


# ---------------------------------------------------------------------------------------------
# Live gates on a scratch repository and a scratch ledger

def git(repo, *arguments):
    done = subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                           "-c", "commit.gpgsign=false", *arguments], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def commit(repo, files, message):
    for relative, data in files.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def ledger_entry(ledger, job_id, slurm, sbatch=(), **fields):
    (ledger / job_id).mkdir(parents=True)
    (ledger / job_id / "state.json").write_text(json.dumps({"stato": "sottomesso", "job_slurm": slurm, "agente": "claude",
                                                            "campagna": S.CAMPAIGN, "sbatch": list(sbatch), **fields}))


def export(name, sha1):
    return f"--export=ALL,CAMPAIGN={S.CHECKOUTS[name]},CAMPAIGN_SHA={sha1},EXPECTED_SHA={sha1}"


def accounting_bytes(slurm, state="COMPLETED", exit_code="0:0"):
    return json.dumps({"JobIDRaw": str(slurm), "State": state, "ExitCode": exit_code}).encode()


def collected(repo, name, slurm, state="COMPLETED", exit_code="0:0"):
    """What `collect checkout-<name>` leaves after the checkout job: its accounting, read by the gates of the readers."""
    target = repo / S.PACKAGE / f"evidence/checkout-{name}/{name}/accounting.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(accounting_bytes(slurm, state, exit_code))


@pytest.fixture
def chain(tmp_path):
    """Reviewed package X, code checkout Y (extraction plan), plans checkout Z (reader plans, collected accounting of Y);
    ledger with the code checkout."""
    repo, ledger = tmp_path / "repo", tmp_path / "ledger"
    repo.mkdir()
    git(repo, "init", "-q")
    shared = {relative: (ROOT / relative).read_bytes() for relative in S.SHARED.values()}
    reviewed = commit(repo, {**shared, S.FREEZE: (ROOT / S.FREEZE).read_bytes(), "scripts/reader.py": b"reviewed = True\n",
                             "results/infra/tokenizer_review_20261103/comparison_configurations.json":
                             (ROOT / "results/infra/tokenizer_review_20261103/comparison_configurations.json").read_bytes()}, "reviewed package")
    package = S.PACKAGE + "/"
    code = commit(repo, {package + S.DOCUMENTS["extraction_plan"][1]: S.dumps(plans.new_extraction_plan()),
                         package + "evidence/training/collection.json": b"{}\n"}, "extraction plan")
    documents = commit(repo, {package + S.DOCUMENTS["pose_plan"][1]: S.dumps(plans.new_pose_plan()),
                              package + S.DOCUMENTS["probe_plan"][1]: S.dumps(plans.new_probe_plan()),
                              package + "evidence/extractions/collection.json": b"{}\n", "docs/decisioni.md": b"consuntivo\n",
                              package + "evidence/checkout-code/code/accounting.json": accounting_bytes(700)}, "reader plans")
    ledger_entry(ledger, S.checkout_job("code")["id"], 700, ["sbatch", export("code", code), "/dev/stdin"])
    return repo, ledger, reviewed, code, documents


def live(commits, documents, table, **changes):
    """Answer of the read-only request of a gate when everything is in place."""
    commands = {"squeue": {"exit": 0, "stdout": QUEUE, "stderr": ""}, "quota": {"exit": 0, "stdout": QUOTA, "stderr": ""}}
    for name, sha1 in commits.items():
        commands["head:" + name] = {"exit": 0, "stdout": sha1 + "\n", "stderr": ""}
        commands["status:" + name] = {"exit": 0, "stdout": "", "stderr": ""}
    reading = {"commands": commands, "sacct": {}, "listings": {}, "exists": {job["out"]: False for job in table},
               "files": [{**p, "canonical": True} for p in documents.values()], "time_utc": "2026-11-08T10:00:00Z"}
    for key, value in changes.items():
        target, _, field = key.partition("__")
        if field:
            reading[target][field.replace("_", ":", 1) if target == "commands" and field.count("_") else field].update(value)
        else:
            reading[target] = value
    return reading


def test_lane_reads_the_shared_serial_queue_and_the_project_quota():
    reading = {"commands": {"squeue": {"exit": 0, "stdout": QUEUE}, "quota": {"exit": 0, "stdout": QUOTA}}}
    state = S.lane(reading)
    assert state["serial_jobs"] == 2 and state["serial_room"] == 8 and state["work_free_gib"] == pytest.approx(23.72, abs=.01)
    assert state["line_jobs"] == 1  # the jobs of this line are named after their logical IDs: SCALING-...
    mixed = "1|SCALING-19618-pose-validate-20261106|RUNNING|2|None\n2|LOGSQ-a|PENDING|8|x\n3|SCALING-x|PENDING|1|x\n4||PENDING|1|x\n"
    assert S.lane({"commands": {"squeue": {"exit": 0, "stdout": mixed}, "quota": {"exit": 0, "stdout": QUOTA}}})["line_jobs"] == 2
    over = QUOTA.replace("1048868200 ", "1073741900*")
    assert S.lane({"commands": {"squeue": {"exit": 0, "stdout": ""}, "quota": {"exit": 0, "stdout": over}}}) == {
        "serial_jobs": 0, "serial_room": 10, "line_jobs": 0, "work_free_gib": pytest.approx(-76 / 2**20)}
    for squeue, quota in (({"exit": 1, "stdout": ""}, {"exit": 0, "stdout": QUOTA}), ({"exit": 0, "stdout": QUEUE}, {"exit": 1, "stdout": ""}),
                          ({"exit": 0, "stdout": QUEUE}, {"exit": 0, "stdout": QUOTA.replace("20147512", "1")}),
                          ({"exit": 0, "stdout": QUEUE}, {"exit": 0, "stdout": QUOTA.replace(S.WORK, "/other")}),
                          ({"exit": 0, "stdout": QUEUE}, {"exit": 0, "stdout": QUOTA.replace("1073741824", "0")})):
        with pytest.raises(ValueError, match="no gate|unexpected quota"):  # a failed reading is not an empty queue
            S.lane({"commands": {"squeue": squeue, "quota": quota}})
    assert S.QUEUE[:4] == ["squeue", "-h", "-u", "@USER"] and S.QUOTA == ["lfs", "quota", "-p", "20147512", S.WORK]


def test_checkout_commit_comes_from_the_receipt_of_the_common_ledger(tmp_path):
    receipt = PACKAGE / "submissions/SCALING-300M-cont19618-checkout-20261106.json"  # real submission of 06/11 (failed v1 attempt)
    target = tmp_path / "SCALING-300M-cont19618-checkout-20261106"
    target.mkdir()
    (target / "state.json").write_bytes(receipt.read_bytes())
    assert S.ledger_job("SCALING-300M-cont19618-checkout-20261106", tmp_path) == "63695634"
    with pytest.raises(OSError):  # the v1 checkout (73524b6, job 63695634) is not the training checkout of the recovery
        S.checkout_commit("training", tmp_path)
    recovery = S.RECOVERY
    ledger_entry(tmp_path, recovery["checkout"], 900, ["sbatch", f"--export=ALL,CAMPAIGN={recovery['campaign']},CAMPAIGN_SHA={CODE},"
                                                       f"EXPECTED_SHA={CODE}", "/dev/stdin"])
    assert S.checkout_commit("training", tmp_path) == CODE
    shutil.rmtree(tmp_path / recovery["checkout"])
    ledger_entry(tmp_path, recovery["checkout"], 901, ["sbatch", "--export=ALL,CAMPAIGN=scaling_300m_continuation19618_20261106_v1,"
                                                       f"CAMPAIGN_SHA={CODE},EXPECTED_SHA={CODE}", "/dev/stdin"])
    with pytest.raises(ValueError, match="unexpected checkout submission"):  # the v1 checkout folder under the recovery ID
        S.checkout_commit("training", tmp_path)
    ledger_entry(tmp_path, S.checkout_job("plans")["id"], 1, ["sbatch", export("code", CODE)])
    with pytest.raises(ValueError, match="unexpected checkout submission"):  # the reference of another checkout
        S.checkout_commit("plans", tmp_path)
    ledger_entry(tmp_path, S.checkout_job("budgets")["id"], 1, ["sbatch", export("budgets", "abc")])
    with pytest.raises(ValueError, match="unexpected checkout submission"):
        S.checkout_commit("budgets", tmp_path)
    with pytest.raises(OSError):
        S.checkout_commit("inputs", tmp_path)


def test_later_checkouts_only_add_documents_to_the_reviewed_package(chain):
    repo, _, reviewed, code, documents = chain
    assert S.release_check(reviewed, code, repo) == 2 and S.release_check(reviewed, documents, repo) == 7
    changed = commit(repo, {"scripts/reader.py": b"reviewed = False\n"}, "code after the review")
    with pytest.raises(ValueError, match="change outside the documents of the chain after the review: M\tscripts/reader.py"):
        S.release_check(reviewed, changed, repo)
    for files, message in (({"tests/cpu/test_new.py": b"\n"}, "A\ttests/cpu/test_new.py"),
                           ({S.PACKAGE + "/training_plan.json": b"{}\n"}, "training_plan.json"),
                           ({S.RECOVERY["plan"]: b"{}\n"}, "continuation19618_20261109_v2/training_plan.json"),
                           ({S.SHARED["splits"]: b"{}\n"}, "M\tsplits/v1/splits_v1.json")):
        git(repo, "checkout", "-q", "-B", "side", documents)
        with pytest.raises(ValueError, match=message):
            S.release_check(reviewed, commit(repo, files, "side"), repo)
    # A deployed document rewritten later is still an addition for the reviewed package, not for the checkout that carries it.
    git(repo, "checkout", "-q", "-B", "side", documents)
    plan = S.PACKAGE + "/" + S.DOCUMENTS["pose_plan"][1]
    rewritten = commit(repo, {plan: b"{}\n"}, "side")
    assert S.release_check(reviewed, rewritten, repo) == 7 and S.release_check(code, documents, repo, "an earlier checkout") == 5
    # Submissions and evidence of the recovery attempt go where the chain already admits additions; the allowlist is unchanged.
    git(repo, "checkout", "-q", "-B", "side", documents)
    attempt = commit(repo, {f"{S.PACKAGE}/submissions/{S.RECOVERY['precheck']}.json": b"{}\n",
                            str(S.evidence_dir("identity-precheck", "")) + "/identity_precheck.json": b"{}\n",
                            str(S.evidence_dir("checkout-training", "")) + "/training/accounting.json": b"{}\n"}, "attempt")
    assert S.release_check(reviewed, attempt, repo) == 10
    with pytest.raises(ValueError, match="after an earlier checkout: M\t" + plan):
        S.release_check(documents, rewritten, repo, "an earlier checkout")
    with pytest.raises(ValueError, match="does not descend from the review"):
        S.release_check(code, reviewed, repo)
    with pytest.raises(ValueError, match="full commit SHAs"):
        S.release_check(reviewed[:12], code, repo)


def test_checkout_gate_before_the_deploy_of_a_document_commit(chain):
    repo, ledger, reviewed, code, documents = chain
    job = S.checkout_job("plans")
    assert (job["cpus"], job["memory"], job["seconds"], job["out"]) == (1, "4G", 300, S.checkout("plans"))

    def run(deploy=None, **changes):
        reading = live({}, {}, [job], **changes)
        return S.checkout_gate("plans", documents, reviewed, reading, deploy, repo, ledger)

    checks, facts = run()
    assert all(ok for _, ok in checks) and len(checks) == 9 and facts["serial_room"] == 8
    assert facts["tranche"] == [job["id"]] and facts["line_limit"] == 4 and facts["line_jobs"] == 1
    assert all(ok for _, ok in run(documents)[0])  # the reference is already at this commit: the push is a no-op
    failed = lambda result: [name for name, ok in result[0] if not ok]  # noqa: E731
    assert failed(run("e" * 40)) == ["deploy reference absent or already at this commit"]
    assert failed(run(exists={job["out"]: True})) == ["checkout folder absent"]
    full = "\n".join("%d|x|PENDING|8|Priority" % n for n in range(10)) + "\n"
    assert failed(run(commands__squeue={"stdout": full})) == ["one free place in the serial queue"]
    crowded = "\n".join("%d|SCALING-19618-probe-bench-%d-20261106|PENDING|8|Priority" % (n, n) for n in range(4)) + "\n"
    assert failed(run(commands__squeue={"stdout": crowded})) == ["line limit 4 leaves a place for this checkout"]
    wider = S.checkout_gate("plans", documents, reviewed, live({}, {}, [job], commands__squeue={"stdout": crowded}), None, repo, ledger, 5)
    assert all(ok for _, ok in wider[0]) and wider[1]["line_limit"] == 5
    low = QUOTA.replace("1048868200", str(1073741824 - 4 * 2**20))
    assert failed(run(commands__quota={"stdout": low})) == ["free space on $WORK at least 5 GiB"]
    (repo / S.PACKAGE / S.DOCUMENTS["pose_plan"][1]).write_bytes(b"{}\n")  # local document differs from the committed one
    assert "documents of this checkout are in the commit with the published bytes" in failed(run())
    git(repo, "checkout", "-q", "--", ".")
    with pytest.raises(ValueError, match="created in order"):  # the budgets checkout comes after the plans checkout
        S.checkout_gate("budgets", documents, reviewed, live({}, {}, [S.checkout_job("budgets")]), None, repo, ledger)
    ledger_entry(ledger, job["id"], 701, ["sbatch", export("plans", documents)])
    assert failed(run()) == ["logical ID never used in the ledger"]
    rewritten = commit(repo, {S.PACKAGE + "/" + S.DOCUMENTS["pose_plan"][1]: b"{}\n"}, "plan rewritten after its checkout")
    with pytest.raises(ValueError, match="after an earlier checkout: M\t"):
        S.checkout_gate("budgets", rewritten, reviewed, live({}, {}, [S.checkout_job("budgets")]), None, repo, ledger)
    with pytest.raises(ValueError, match="after the review: M\tscripts/reader.py"):
        S.checkout_gate("budgets", commit(repo, {"scripts/reader.py": b"x\n"}, "code"), reviewed,
                        live({}, {}, [S.checkout_job("budgets")]), None, repo, ledger)


def test_gate_before_a_tranche_reads_checkouts_documents_outputs_queue_and_space(chain):
    repo, ledger, _, code, documents = chain
    ledger_entry(ledger, S.checkout_job("plans")["id"], 701, ["sbatch", export("plans", documents)])
    collected(repo, "plans", 701)
    request, commits, published, table, submitted = S.gate_request("pose-benchmark", repo, ledger)
    assert commits == {"code": code, "plans": documents} and submitted == {} and list(published) == ["pose_plan"]
    assert request["hashed"] == [f"{S.checkout('plans')}/{S.PACKAGE}/readers/pose_plan.json"] and request["exists"] == [j["out"] for j in table]
    assert request["commands"]["head:plans"] == ["git", "-C", S.checkout("plans"), "rev-parse", "HEAD"]
    assert request["commands"]["status:code"] == ["git", "-C", S.checkout("code"), "status", "--porcelain", "--untracked-files=all"]

    def run(**changes):
        return S.gate(live(commits, published, table, **changes), commits, published, table, submitted, repo, ledger)

    checks, facts = run()
    assert all(ok for _, ok in checks), checks
    assert facts["to_submit"] == [j["id"] for j in table] and facts["submittable_now"] == 3 and facts["serial_jobs"] == 2
    assert facts["tranche"] == [j["id"] for j in table] and facts["line_limit"] == 4 and facts["line_jobs"] == 1
    assert [name for name, _ in checks] == [
        "checkout code: expected HEAD and clean tree", "checkout plans: expected HEAD and clean tree",
        "document pose_plan: remote bytes equal to the published ones",
        "files pinned inside the checkouts are in their commits (2)",  # projection and source contract, in the code checkout
        "outputs of the jobs still to submit are absent", "jobs already in the ledger were submitted by this line",
        "jobs already submitted are COMPLETED 0:0 (0)", "checkouts used were collected COMPLETED 0:0 (2)",
        "free space on $WORK at least 5 GiB", "a tranche of at least one job is admitted (3 of 3 to submit)"]
    failed = lambda result: [name for name, ok in result[0] if not ok]  # noqa: E731
    assert failed(run(commands__head_code={"stdout": "e" * 40 + "\n"})) == ["checkout code: expected HEAD and clean tree"]
    assert failed(run(commands__status_plans={"stdout": "?? new_file\n"})) == ["checkout plans: expected HEAD and clean tree"]
    assert failed(run(commands__head_plans={"exit": 128})) == ["checkout plans: expected HEAD and clean tree"]
    other = [{**p, "sha256": "d" * 64, "canonical": True} for p in published.values()]
    assert failed(run(files=other)) == ["document pose_plan: remote bytes equal to the published ones"]
    assert failed(run(files=[])) == ["document pose_plan: remote bytes equal to the published ones"]
    assert failed(run(exists={**{j["out"]: False for j in table}, table[1]["out"]: True})) == ["outputs of the jobs still to submit are absent"]
    low = QUOTA.replace("1048868200", str(1073741824 - 4 * 2**20))
    assert failed(run(commands__quota={"stdout": low})) == ["free space on $WORK at least 5 GiB"]
    full = "\n".join("%d|x|PENDING|8|Priority" % n for n in range(9)) + "\n"
    assert run(commands__squeue={"stdout": full})[1]["submittable_now"] == 1
    collected(repo, "plans", 702)  # the accounting of another job, or none at all: the checkout was not collected
    assert failed(run()) == ["checkouts used were collected COMPLETED 0:0 (2)"]
    collected(repo, "plans", 701, state="FAILED", exit_code="1:0")
    assert failed(run()) == ["checkouts used were collected COMPLETED 0:0 (2)"]
    collected(repo, "plans", 701)

    # After the first job: a new tranche needs it COMPLETED 0:0; any other ledger state stops (no blind retry).
    ledger_entry(ledger, table[0]["id"], 900)
    request, commits, published, table, submitted = S.gate_request("pose-benchmark", repo, ledger)
    assert submitted == {table[0]["id"]: "900"}
    done = {"900": {"exit": 0, "stdout": plans.HEADER + "\n900|COMPLETED|0:0|33|8||n|p|a|b|00:10:00|cpu=8\n", "stderr": ""}}
    checks, facts = run(sacct=done)
    assert all(ok for _, ok in checks) and facts["to_submit"] == [j["id"] for j in table[1:]]
    for state in ("RUNNING", "PENDING", "TIMEOUT", "FAILED"):
        bad = {"900": {**done["900"], "stdout": done["900"]["stdout"].replace("COMPLETED", state)}}
        assert failed(run(sacct=bad)) == ["jobs already submitted are COMPLETED 0:0 (1)"]
    assert failed(run(sacct={"900": {"exit": 1, "stdout": "", "stderr": "x"}})) == ["jobs already submitted are COMPLETED 0:0 (1)"]
    (ledger / table[1]["id"]).mkdir()
    (ledger / table[1]["id"] / "state.json").write_text(json.dumps({"stato": "rifiutato", "agente": "claude", "campagna": S.CAMPAIGN}))
    request, commits, published, table, submitted = S.gate_request("pose-benchmark", repo, ledger)
    assert failed(run(sacct=done)) == ["jobs already in the ledger were submitted by this line"]


def test_gate_of_the_extractions_reads_the_plan_in_the_code_checkout(chain):
    repo, ledger, _, code, _ = chain
    request, commits, published, table, submitted = S.gate_request("extractions", repo, ledger)
    assert commits == {"code": code} and request["hashed"] == [f"{S.checkout('code')}/{S.PACKAGE}/evaluation/extraction_plan.json"]
    checks, facts = S.gate(live(commits, published, table), commits, published, table, submitted, repo, ledger)
    assert all(ok for _, ok in checks), checks
    assert facts["submittable_now"] == 3 and facts["tranche"] == [j["id"] for j in table]  # GPU partition: not limited by the serial queue
    full = "\n".join("%d|SCALING-19618-x-%d|PENDING|8|Priority" % (n, n) for n in range(10)) + "\n"
    checks, facts = S.gate(live(commits, published, table, commands__squeue={"stdout": full}), commits, published, table, submitted, repo, ledger)
    assert all(ok for _, ok in checks) and facts["tranche"] == [j["id"] for j in table] and facts["serial_room"] == 0
    # Relative pins of the extraction plan (contract, comparison, splits) are read in the commit of the code checkout.
    assert [name for name, _ in checks][2] == "files pinned inside the checkouts are in their commits (3)"
    with pytest.raises(OSError):  # no stage starts before its checkouts exist in the ledger
        S.gate_request("pose-validate", repo, ledger)


# ---------------------------------------------------------------------------------------------
# Remote reading and collection

def test_remote_read_sends_only_registered_read_only_commands(tmp_path):
    bins = tmp_path / "bin"
    bins.mkdir()
    for name in ("sacct", "squeue", "lfs", "git"):
        (bins / name).write_text("#!/bin/sh\necho %s \"$@\" \"locks=$GIT_OPTIONAL_LOCKS\"\n" % name)
    (bins / "ssh").write_text("#!/bin/sh\nUSER=tester PATH=%s:$PATH exec %s -\n" % (bins, sys.executable))
    for path in bins.iterdir():
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    folder, big = tmp_path / "out", tmp_path / "out" / "cache.npz"
    folder.mkdir()
    big.write_bytes(b"x" * 4096)
    (folder / "report.json").write_bytes(b"{}\n")
    commands = {"squeue": S.QUEUE, "quota": S.QUOTA, "head:code": S._git("/checkout", *S.GIT_READS[0]),
                "status:code": S._git("/checkout", *S.GIT_READS[1])}
    got = S.remote_read([str(folder / "report.json")], ["7"], ssh=[str(bins / "ssh")], hashed=[str(big)],
                        listings=[str(folder), str(tmp_path / "none")], exists=[str(big), str(tmp_path / "none")], commands=commands)
    assert [("content" in f, f["bytes"]) for f in got["files"]] == [(True, 3), (False, 4096)] and got["files"][1]["sha256"] == sha(big)
    assert got["listings"] == {str(folder): ["cache.npz", "report.json"], str(tmp_path / "none"): None}
    assert got["exists"] == {str(big): True, str(tmp_path / "none"): False}
    assert got["commands"]["squeue"]["stdout"].split()[:5] == ["squeue", "-h", "-u", "tester", "-p"]  # the user of the login node
    assert got["commands"]["head:code"]["stdout"].strip() == "git -C /checkout rev-parse HEAD locks=0"  # status never writes the index
    assert got["commands"]["quota"]["stdout"].split()[:4] == ["lfs", "quota", "-p", "20147512"]
    for command in (["rm", "-rf", "/x"], ["git", "-C", "/checkout", "reset", "--hard"], ["git", "-C", "/checkout", "status"],
                    ["squeue", "-u", "other"], ["scancel", "1"], ["sbatch", "x"], ["lfs", "setquota"]):
        with pytest.raises(ValueError, match="only the registered read-only commands"):
            S.remote_read(ssh=[str(bins / "ssh")], commands={"x": command})


def test_accounting_without_a_memory_gate_accepts_a_job_too_short_to_be_sampled():
    text = plans.sacct(("1500K", ""), ("900K", ""))
    with pytest.raises(ValueError, match="no measured MaxRSS"):
        S.accounting(text, "77", 2, 1800)
    got = S.accounting(text, "77", 2, 1800, need_rss=False)
    assert got["MaxRSS_bytes"] is None and got["MaxRSS_source_rows"] == [] and got["ElapsedRaw"] == 40
    assert S.accounting(plans.sacct(("1500K", "0")), "77", 2, 1800, need_rss=False)["MaxRSS_bytes"] == 900 * 1024
    assert S.RSS_GATED == ("pose-validate", "pose-benchmark")


def resources_fixture():
    return {"protocol": S.C.RESOURCE_PROTOCOL, "shards": [
        {"cell": c, "task": t, "datasets": list(d), "folds": [0, 1, 2, 3], "cpus": 8, "memory_gib": 30, "walltime_seconds": 3600,
         "shard_name": f"{c}-{t}-01"} for c, (t, d) in itertools.product(S.CELLS, S.TASKS.items())]}


def test_collection_of_fits_pins_the_predictions_copies_reports_and_checks_the_folder(tmp_path):
    table = S.jobs("probe-fit", resources_fixture())[:2]
    slurm = {job["id"]: str(800 + n) for n, job in enumerate(table)}
    assert all(sorted(job["exact"]) == ["audit.jsonl", "audit_progress.jsonl", "shard.json", "shard.oof.npz"] and
               set(job["files"]) == {"shard.json", "audit.jsonl"} and set(job["pinned"]) == {"shard.oof.npz"} for job in table)
    reading = plans.reading_for(table, slurm, {})
    extra = copy.deepcopy(reading)
    extra["listings"][table[0]["folder"]].append("failed.json")
    with pytest.raises(ValueError, match="unexpected content of the output folder"):
        S.collect("probe-fit", table, slurm, extra, tmp_path / "a")
    absent = copy.deepcopy(reading)
    absent["listings"][table[1]["folder"]] = None
    with pytest.raises(ValueError, match="unexpected content of the output folder"):
        S.collect("probe-fit", table, slurm, absent, tmp_path / "a")
    lost = copy.deepcopy(reading)
    lost["files"] = [({"path": f["path"], "missing": True} if f["path"].endswith(".oof.npz") else f) for f in lost["files"]]
    with pytest.raises(ValueError, match="missing or non-canonical remote file"):
        S.collect("probe-fit", table, slurm, lost, tmp_path / "a")
    quiet = copy.deepcopy(reading)  # a missing scheduler log is recorded, a large one is pinned: neither fails the stage
    quiet["files"] = [({"path": f["path"], "missing": True} if f["path"] == S.log_path(table[0], "800") else
                       {k: v for k, v in f.items() if k != "content"} if f["path"] == S.log_path(table[1], "801") else f) for f in quiet["files"]]
    manifest = S.collect("probe-fit", table, slurm, quiet, tmp_path / "a")
    names = [table[0]["local"], table[1]["local"]]
    assert manifest["files"][names[0] + "/slurm_log.txt"] == {"path": S.log_path(table[0], "800"), "stored": "absent"}
    assert manifest["files"][names[1] + "/slurm_log.txt"]["stored"] == "pin-only"
    assert manifest["files"][names[0] + "/shard.oof.npz"]["stored"] == "pin-only" and manifest["files"][names[0] + "/shard.json"]["stored"] == "plain"
    assert not (tmp_path / "a" / names[0] / "shard.oof.npz").exists() and not (tmp_path / "a" / names[0] / "slurm_log.txt").exists()
    large = copy.deepcopy(reading)
    large["files"] = [({k: v for k, v in f.items() if k != "content"} if f["path"].endswith("shard.json") else f) for f in large["files"]]
    with pytest.raises(ValueError, match="larger than the inline limit"):
        S.collect("probe-fit", table, slurm, large, tmp_path / "b")


def test_checkouts_and_training_are_rows_of_the_same_table():
    training = S.checkout_job("training")  # the frozen checkout of the recovery attempt (v2), not the v1 one
    assert (training["id"], training["cpus"], training["seconds"], training["log"]) == (
        "SCALING-300M-cont19618-checkout-20261109-v2", 2, 300, "campaign_checkout")
    assert training["out"] == S.WORK + "/wearusfm_campaigns/scaling_300m_continuation19618_20261109_v2"
    plan = json.loads((ROOT / S.RECOVERY["plan"]).read_text())
    precheck, job = S.precheck_job(plan), S.training_job(plan)
    assert (precheck["stage"], precheck["id"], precheck["cpus"], precheck["memory"], precheck["seconds"], precheck["gpus"], precheck["log"]) == (
        "identity-precheck", "SCALING-300M-cont19618-idcheck-20261109-v2", 4, "24G", 1200, 0, "scaling_identity_precheck")
    assert precheck["out"] == plan["identity_precheck"]["receipt_dir"] and precheck["exact"] == ["identity_precheck.json"]
    assert precheck["files"] == {"identity_precheck.json": precheck["out"] + "/identity_precheck.json"} and precheck["pinned"] == {}
    assert job["id"] == "SCALING-300M-train19618-20261109-v2" and job["out"] == plan["receipt_dir"]
    assert S.stage_table("training") == [job] and S.stage_table("identity-precheck") == [precheck]
    # The v1 checkout keeps its collected record; the recovery checkout is collected elsewhere and never over it.
    v1 = json.loads((PACKAGE / "evidence/checkout-training/training/accounting.json").read_text())
    assert v1["JobIDRaw"] == "63695634" and S.evidence_dir("checkout-training") == PACKAGE / "evidence/checkout-training-v2"
    assert not S.evidence_dir("checkout-training").exists() and S.evidence_dir("training") == PACKAGE / "evidence/training"
    assert all(S.evidence_dir(stage) == PACKAGE / "evidence" / stage for stage in ("extractions", "probe-fit", "checkout-code"))
    assert [S.stage_table("checkout-" + name) for name in S.CHECKOUTS] == [[S.checkout_job(name)] for name in S.CHECKOUTS]
    assert all(job["files"] == {} and job["pinned"] == {} and job["exact"] is None for name in S.CHECKOUTS for job in [S.checkout_job(name)])
    assert S.log_path(S.jobs("extractions")[0], "5") == S.LOGS + "/scaling_evaluation19618_5.out"
    for launcher, name in (("scaling_continue.sbatch", "scaling_continue"), ("scaling_identity_precheck.sbatch", "scaling_identity_precheck"),
                           ("scaling_evaluation19618.sbatch", "scaling_evaluation19618"),
                           ("scaling_readers19618.sbatch", "scaling_readers19618"), ("campaign_checkout.sbatch", "campaign_checkout")):
        assert f"#SBATCH -o {S.LOGS}/{name}_%j.out" in (ROOT / "scripts/slurm" / launcher).read_text()
    assert set(S.STAGE_DOCUMENTS) == {"extractions", "pose-validate", "pose-benchmark", "probe-benchmark", "pose-ridge", "probe-fit",
                                      "probe-aggregate", "compare"} and all(d in S.DOCUMENTS for ds in S.STAGE_DOCUMENTS.values() for d in ds)


def test_every_gate_leaves_its_record_among_the_evidence(chain, monkeypatch, capsys):
    repo, ledger, reviewed, code, documents = chain
    bare = repo.parent / "leonardo.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    git(repo, "remote", "add", "leonardo", str(bare))
    answers, asked = [], []

    def read(**request):
        asked.append(request)
        return answers.pop(0)

    monkeypatch.setattr(S, "remote_read", read)
    gates = repo / S.PACKAGE / "evidence/gates"

    # Checkout of the reader plans: the reference is absent, then already at another commit.
    job = S.checkout_job("plans")
    answers.append(live({}, {}, [job]))
    assert S.run_gate("checkout-plans", documents, reviewed, repo, ledger) == 0
    record = json.loads((gates / "checkout-plans_20261108T100000Z.json").read_text())
    assert record == {"protocol": S.GATE_PROTOCOL, "stage": "checkout-plans", "read_at_utc": "2026-11-08T10:00:00Z", "passed": True,
                      "checks": record["checks"], "facts": record["facts"], "commit": documents, "reviewed": reviewed,
                      "deploy_reference": None}
    assert len(record["checks"]) == 9 and all(ok is True for _, ok in record["checks"]) and record["facts"]["serial_room"] == 8
    assert asked[-1] == {"exists": [S.checkout("plans")], "commands": {"squeue": S.QUEUE, "quota": S.QUOTA}}
    git(repo, "push", "-q", "leonardo", f"{code}:refs/heads/deploy/{S.CHECKOUTS['plans']}")
    answers.append(live({}, {}, [job], time_utc="2026-11-08T10:01:00Z"))
    assert S.run_gate("checkout-plans", documents, reviewed, repo, ledger) == 1
    record = json.loads((gates / "checkout-plans_20261108T100100Z.json").read_text())
    assert record["passed"] is False and record["deploy_reference"] == code
    assert [name for name, ok in record["checks"] if not ok] == ["deploy reference absent or already at this commit"]

    # Tranche of a job stage: the same record, with the commits of its checkouts and the jobs already sent.
    ledger_entry(ledger, job["id"], 701, ["sbatch", export("plans", documents)])
    collected(repo, "plans", 701)
    _, commits, published, table, _ = S.gate_request("pose-benchmark", repo, ledger)
    answers.append(live(commits, published, table, time_utc="2026-11-08T10:02:00Z"))
    capsys.readouterr()
    assert S.run_gate("pose-benchmark", root=repo, ledger=ledger) == 0
    out = capsys.readouterr().out
    assert out.count("PASS  ") == 10 and "FAIL" not in out and '"submittable_now": 3' in out
    record = json.loads((gates / "pose-benchmark_20261108T100200Z.json").read_text())
    assert record["passed"] is True and record["commits"] == {"code": code, "plans": documents} and record["submitted"] == {}
    assert record["documents"] == published and record["facts"]["to_submit"] == [j["id"] for j in table]
    assert record["facts"]["tranche"] == [j["id"] for j in table] and record["facts"]["line_limit"] == 4
    assert list(asked[-1]["slurm_jobs"]) == [] and asked[-1]["exists"] == [j["out"] for j in table]
    answers.append(live(commits, published, table, exists={j["out"]: True for j in table}, time_utc="2026-11-08T10:03:00Z"))
    assert S.run_gate("pose-benchmark", root=repo, ledger=ledger) == 1
    record = json.loads((gates / "pose-benchmark_20261108T100300Z.json").read_text())
    assert [name for name, ok in record["checks"] if not ok] == ["outputs of the jobs still to submit are absent"]
    assert sorted(p.name for p in gates.iterdir()) == [
        "checkout-plans_20261108T100000Z.json", "checkout-plans_20261108T100100Z.json", "pose-benchmark_20261108T100200Z.json",
        "pose-benchmark_20261108T100300Z.json"]
    # The records are additions under evidence/: a later checkout may carry them.
    later = commit(repo, {}, "gate records")
    assert S.release_check(reviewed, later, repo) == 12 and S.release_check(documents, later, repo, "an earlier checkout") == 5
    with pytest.raises(ValueError, match="unexpected time of the remote reading"):
        S.gate_record("compare", [], {}, {"time_utc": ""}, {})


def test_tranche_admitted_by_the_gate_respects_the_queue_and_the_line_limit(chain):
    """B-1 of the GPT review (c2df17e): admission and generation are bound to the same tranche, number and IDs."""
    repo, ledger, _, code, documents = chain
    ledger_entry(ledger, S.checkout_job("plans")["id"], 701, ["sbatch", export("plans", documents)])
    collected(repo, "plans", 701)
    _, commits, published, table, submitted = S.gate_request("pose-benchmark", repo, ledger)
    ids = [j["id"] for j in table]
    others = lambda n: "".join("%d|LOGSQ-%d|PENDING|8|Priority\n" % (n, n) for n in range(n))  # noqa: E731  another line
    mine = lambda n: "".join("%d|SCALING-19618-probe-bench-%d-20261106|PENDING|8|Priority\n" % (900 + n, n) for n in range(n))  # noqa: E731
    failed = lambda result: [name for name, ok in result[0] if not ok]  # noqa: E731

    def run(queue, limit=S.LINE_SERIAL_LIMIT, table=table, published=published, commits=commits, submitted=submitted):
        reading = live(commits, published, table, commands__squeue={"stdout": queue})
        return S.gate(reading, commits, published, table, submitted, repo, ledger, limit)

    none = "a tranche of at least one job is admitted (0 of 3 to submit)"
    checks, facts = run(others(10))  # full queue: nothing admitted, and the gate fails on that alone
    assert failed((checks, facts)) == [none] and facts["tranche"] == [] and facts["submittable_now"] == 0
    checks, facts = run(others(9))  # one place: the first job of the table alone
    assert all(ok for _, ok in checks) and facts["tranche"] == ids[:1] and facts["submittable_now"] == 1
    checks, facts = run("")  # free queue: the whole stage, within the line limit
    assert all(ok for _, ok in checks) and facts["tranche"] == ids and facts["line_jobs"] == 0 and facts["line_limit"] == 4
    checks, facts = run(mine(4))  # four jobs of this line already there: no place, even with six free in the queue
    assert failed((checks, facts)) == [none] and facts["line_jobs"] == 4 and facts["serial_room"] == 6
    assert run(mine(2))[1]["tranche"] == ids[:2] and run(mine(2) + others(7))[1]["tranche"] == ids[:1]
    assert run(mine(4), limit=10)[1]["tranche"] == ids and run(mine(4), limit=10)[1]["line_limit"] == 10
    # Six aggregates: at most four together, minus the jobs of this line already present.
    aggregates = S.jobs("probe-aggregate", {"protocol": S.C.RESOURCE_PROTOCOL, "shards": []})
    assert len(aggregates) == 6
    few = lambda queue, limit=S.LINE_SERIAL_LIMIT: run(queue, limit, aggregates, {}, {"code": code}, {})  # noqa: E731
    assert few("")[1]["tranche"] == [j["id"] for j in aggregates[:4]]
    assert few(mine(1))[1]["tranche"] == [j["id"] for j in aggregates[:3]]
    assert few(others(8))[1]["tranche"] == [j["id"] for j in aggregates[:2]]
    assert few("", 6)[1]["tranche"] == [j["id"] for j in aggregates]
    assert failed(few(others(10))) == ["a tranche of at least one job is admitted (0 of 6 to submit)"]


def test_lines_come_only_from_the_tranche_admitted_by_the_latest_gate(chain, monkeypatch, capsys):
    """B-1: `lines` reads the record of its gate; zero places give zero lines; --max only shortens the admitted tranche."""
    repo, ledger, reviewed, code, documents = chain
    bare = repo.parent / "leonardo.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    git(repo, "remote", "add", "leonardo", str(bare))
    answers = []
    monkeypatch.setattr(S, "remote_read", lambda **request: answers.pop(0))
    monkeypatch.setattr(S, "ROOT", repo)
    monkeypatch.setattr(S, "LEDGER", ledger)
    monkeypatch.setattr(S, "_now", lambda: calendar.timegm((2026, 11, 8, 10, 5, 0)))  # five minutes after the readings
    others = lambda n: "".join("%d|LOGSQ-%d|PENDING|8|Priority\n" % (n, n) for n in range(n))  # noqa: E731
    refused = lambda text: (1, [], "nessuna riga: " + text)  # noqa: E731

    def lines(*argv):
        capsys.readouterr()
        status = S.main(["lines", *argv])
        out, err = capsys.readouterr()
        return status, out.splitlines(), err.strip()

    # Checkout stage: its two lines need the passed gate of that very commit, before its submission.
    assert lines("checkout-plans", "--sha", documents) == refused("no gate record for checkout-plans: run the gate first")
    job = S.checkout_job("plans")
    answers.append(live({}, {}, [job]))
    assert S.main(["gate", "checkout-plans", "--sha", documents, "--reviewed", reviewed]) == 0
    assert lines("checkout-plans", "--sha", documents) == (0, S.checkout_lines("plans", documents), "")
    assert lines("checkout-plans") == refused("a checkout needs --sha")
    assert lines("checkout-plans", "--sha", code) == refused("the latest gate of checkout-plans was for another commit")
    assert lines("checkout-plans", "--sha", documents, "--max", "1") == refused("--max applies to job stages only")
    ledger_entry(ledger, job["id"], 701, ["sbatch", export("plans", documents)])
    assert lines("checkout-plans", "--sha", documents) == refused("checkout already in the ledger: plans")
    # A refusal raised before the checks leaves its own record, which is then the latest record of its stage.
    with pytest.raises(ValueError, match="needs --sha and --reviewed"):
        S.main(["gate", "checkout-plans", "--sha", documents])
    record = json.loads((repo / S.PACKAGE / "evidence/gates/checkout-plans_20261108T100500Z_refused.json").read_text())
    assert record["passed"] is False and record["refused"] == "ValueError: a checkout gate needs --sha and --reviewed"
    assert record["stage"] == "checkout-plans" and record["commit"] == documents and record["reviewed"] is None
    assert lines("checkout-plans", "--sha", documents) == refused("the latest gate of checkout-plans did not pass: no line to submit")

    # Job stage: no gate yet, failed gate, one place, free queue.
    collected(repo, "plans", 701)
    _, commits, published, table, _ = S.gate_request("pose-benchmark", repo, ledger)
    ids, whole = [j["id"] for j in table], S.stage_lines("pose-benchmark", code, repo)
    assert lines("pose-benchmark") == refused("no gate record for pose-benchmark: run the gate first")
    answers.append(live(commits, published, table, commands__squeue={"stdout": others(10)}, time_utc="2026-11-08T10:01:00Z"))
    assert S.main(["gate", "pose-benchmark"]) == 1
    assert lines("pose-benchmark") == refused("the latest gate of pose-benchmark did not pass: no line to submit")
    answers.append(live(commits, published, table, commands__squeue={"stdout": others(9)}, time_utc="2026-11-08T10:02:00Z"))
    assert S.main(["gate", "pose-benchmark"]) == 0
    assert lines("pose-benchmark") == (0, whole[:1], "") and lines("pose-benchmark", "--max", "1") == (0, whole[:1], "")
    for value in ("2", "3", "0", "-1"):
        assert lines("pose-benchmark", "--max", value) == refused("--max must be between 1 and the admitted tranche (1)")
    answers.append(live(commits, published, table, time_utc="2026-11-08T10:03:00Z"))  # QUEUE: one job of this line, eight places
    assert S.main(["gate", "pose-benchmark"]) == 0
    assert lines("pose-benchmark") == (0, whole, "") and len(whole) == 3  # the commands, resources and chain of the plan
    assert lines("pose-benchmark", "--max", "2") == (0, whole[:2], "")
    assert "--dopo" not in whole[0] and f"--dopo {ids[0]}" in whole[1] and f"--dopo {ids[1]}" in whole[2]
    assert lines("pose-benchmark", "--max", "4") == refused("--max must be between 1 and the admitted tranche (3)")
    assert lines("pose-benchmark", "--sha", code) == refused("--sha applies to checkout stages only")
    # The record binds the moment and the ledger: a stale gate, or a job sent after it, ask for a new gate.
    monkeypatch.setattr(S, "_now", lambda: calendar.timegm((2026, 11, 8, 10, 33, 1)))  # 1801 s after the reading
    assert lines("pose-benchmark") == refused("gate record of pose-benchmark older than 1800 seconds: run the gate again")
    monkeypatch.setattr(S, "_now", lambda: calendar.timegm((2026, 11, 8, 10, 0, 0)))  # the Mac three minutes behind: tolerated
    assert lines("pose-benchmark") == (0, whole, "")
    ledger_entry(ledger, ids[0], 900)
    assert lines("pose-benchmark") == refused("the ledger of pose-benchmark changed after the gate: run the gate again")
