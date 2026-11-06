"""Operational documents of the 19618 chain: the builders reproduce the reviewed 5000 documents.

The mechanics do not depend on the dose. Fed with the receipts, benchmark reports and scheduler accounting
collected for the reviewed 5000 chain, the builders must return its reviewed pins, budgets and packing.
The 19618 documents are then built from the real control evidence and from candidate fixtures obtained by
re-labelling those real 5000 files, and are checked by the 19618 readers. No cluster, cache or fit is used.
"""
import base64
import copy
import gzip
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shlex
import stat
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import scaling_plans19618 as S

INFRA = ROOT / S.INFRA
PACKAGE = ROOT / S.PACKAGE
ARM, CONTROL = S.ARM, S.ARMS[0]
OLD = {"evaluation": "scaling-evaluation-20261104-v1", "pose": "scaling-pose-readout-20261104-v1",
       "probe": "scaling-probe-20261104-v1", "pose_resources": "scaling-pose-resources-20261104-v1",
       "probe_resources": "scaling-probe-resources-20261104-v1"}
CODE = "c" * 40
STATIC = ("extractions", "pose-validate", "pose-benchmark", "probe-benchmark", "pose-ridge", "compare")
spec = importlib.util.spec_from_file_location("reader19618_fixture", ROOT / "tests/cpu/test_scaling_readers19618_launcher.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def load(relative, base=INFRA):
    return json.loads((base / relative).read_text())


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else Path(data).read_bytes()).hexdigest()


def body(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def old_classification(arm):
    base, manifest = INFRA / "classification_completed_20261105", load("classification_completed_20261105/collection_manifest.json")
    receipts = {t: json.loads(gzip.decompress((base / arm / t / "receipt.json.gz").read_bytes())) for t in S.TASKS}
    pins = {t: {"path": manifest[f"{arm}/{t}/receipt.json"]["remote_path"], **{k: manifest[f"{arm}/{t}/receipt.json"][k]
                for k in ("sha256", "bytes")}} for t in S.TASKS}
    provenances = {ds: (base / arm / t / f"{ds}.npz.provenance.json").read_bytes() for t, names in S.TASKS.items() for ds in names}
    return receipts, pins, provenances


def old_pose():
    base, manifest = INFRA / "pose_completed_20261105", load("pose_completed_20261105/collection_manifest.json")
    reviewed = load("cpu_readers_20261105/pose_plan.json")["caches"][ARM]["train"]
    receipt = json.loads(gzip.decompress((base / "receipt.json.gz").read_bytes()))
    pin = {"path": reviewed["receipt_path"], **{k: manifest["receipt.json"][k] for k in ("sha256", "bytes")}}
    provenances = {"train": (base / "train_subset.npz.provenance.json").read_bytes(), "val": (base / "val.npz.provenance.json").read_bytes()}
    return receipt, pin, provenances


def old_extraction(arm=ARM):
    first = load("cpu_readers_20261105/probe_plan.json")["caches"][arm]["ninapro_db2"]
    return {"checkpoint": {"path": first["checkpoint_path"], "sha256": first["checkpoint_sha256"]},
            "code_sha": first["extraction_code_sha"], "plan_sha256": first["extraction_plan_sha256"]}


# ---------------------------------------------------------------------------------------------
# Names, allocations and budget

def test_job_tables_are_the_allocations_of_the_budget_freeze():
    freeze = load("continuation19618_budget_freeze_20261105_v2.json")
    assert sha(ROOT / S.FREEZE) == load("continuation19618_20261106/training_plan.json")["budget_freeze"]["sha256"]
    stages = freeze["readers"]["stages"]
    extractions = S.jobs("extractions")
    assert [(j["flags"], j["seconds"], j["gpus"], j["cpus"], j["memory"]) for j in extractions] == [
        ([ARM, "extract-pose", "none"], 9000, 1, 8, "100G"), ([ARM, "extract-classification", "p1"], 7200, 1, 8, "100G"),
        ([ARM, "extract-classification", "p2"], 9000, 1, 8, "100G")]
    assert sum(j["seconds"] * j["gpus"] for j in extractions) / 3600 == freeze["extractions"]["gpu_hours_cap"] == 7.0
    assert sum(j["seconds"] * j["cpus"] for j in extractions) / 3600 == freeze["extractions"]["allocated_cpu_core_hours_cap"]
    assert [(j["cpus"], j["seconds"]) for j in S.jobs("pose-validate")] == [(2, 1800)]
    assert [(j["flags"], j["cpus"], j["seconds"]) for j in S.jobs("pose-benchmark")] == [(["--cell", c], 8, 600) for c in S.CELLS]
    benchmarks = S.jobs("probe-benchmark")
    table = stages["probe_benchmark"]["seconds_by_cell_and_dataset"]
    assert [(j["flags"], j["cpus"], j["seconds"]) for j in benchmarks] == [
        (["--cell", c, "--dataset", d], 8, table[c][d]) for c in S.CELLS for d in S.DATASETS]
    assert len(benchmarks) == stages["probe_benchmark"]["jobs"] == 15
    assert sum(j["cpus"] * j["seconds"] for j in benchmarks) / 3600 == pytest.approx(stages["probe_benchmark"]["requested_core_hours"])
    assert [(j["cpus"], j["seconds"]) for j in S.jobs("pose-ridge") + S.jobs("compare")] == [(8, 3600), (8, 1800)]
    assert tuple(S.C.DATASETS) == S.DATASETS and S.POSE_RESOURCE_PROTOCOL in (ROOT / "scripts/slurm/scaling_readers19618.sbatch").read_text()


def test_cpu_budget_stays_inside_every_stage_cap_and_the_350_core_hours():
    freeze = load("continuation19618_budget_freeze_20261105_v2.json")["readers"]
    table = S.budget_table()
    assert table["within_stage_caps"] is True and table["within_total_cap"] is True and table["cap"] == 350
    # Five checkouts instead of three, in the same core-hours: the training one at 2 CPUs, four here at 1 CPU.
    checkout = table["stages"]["checkout"]
    assert checkout["jobs"] == 5 and checkout["requested_core_hours"] == pytest.approx(0.5)
    assert checkout["core_hours_with_scheduler_margin"] == pytest.approx(freeze["stages"]["checkout"]["core_hours_with_scheduler_margin"])
    assert "--cpus-per-task=2 --mem=4G --time=00:05:00" in (PACKAGE / "training_submit_lines.txt").read_text()
    for name, row in table["stages"].items():
        if name not in ("checkout", "probe-fit"):
            assert row["core_hours_with_scheduler_margin"] == pytest.approx(row["cap_with_scheduler_margin"]), name
    fifteen = [{"cpus": 8, "seconds": 7200}] * 15  # 240 core-hours in the largest number of shards
    worst = S.budget_table(fifteen)
    assert worst["stages"]["probe-fit"]["core_hours_with_scheduler_margin"] == pytest.approx(246)
    assert worst["core_hours_with_scheduler_margin"] == pytest.approx(freeze["budgeted_core_hours_with_scheduler_margin"])
    assert worst["within_stage_caps"] and worst["within_total_cap"]
    assert S.budget_table([{"cpus": 8, "seconds": 10800}] * 11)["within_stage_caps"] is False


def test_logical_ids_and_outputs_are_unique_new_and_outside_the_code():
    resources = {"protocol": S.C.RESOURCE_PROTOCOL, "shards": [
        {"cell": c, "task": t, "datasets": list(d), "folds": [0, 1, 2, 3], "cpus": 8, "memory_gib": 30, "walltime_seconds": 3600 + 60 * n,
         "shard_name": f"{c}-{t}-01"} for n, (c, (t, d)) in enumerate(itertools.product(S.CELLS, S.TASKS.items()))]}
    table = [j for stage in STATIC for j in S.jobs(stage)] + S.jobs("probe-fit", resources) + S.jobs("probe-aggregate", resources)
    ids, outs = [j["id"] for j in table], [j["out"] for j in table]
    assert len(set(ids)) == len(ids) == 3 + 1 + 3 + 15 + 1 + 1 + 6 + 6 and len(set(outs)) == len(outs)
    assert all(i.endswith("-20261106") and "19618" in i for i in ids)
    known = (Path.home() / "wearusfm_local/jobs/ledger.tsv")
    if known.exists():  # no logical ID of this chain is already reserved in the common ledger
        assert not [i for i in ids if i in known.read_text()]
    for j in table:
        root = S.EXTRACT_ROOT if j["stage"] == "extractions" else S.CPU_ROOT
        assert j["out"].startswith(root + "/") and j["folder"].startswith(str(Path(j["out"]).parent))
        assert all(p.startswith(j["folder"] + "/") for p in [*j["files"].values(), *j["pinned"].values()])
        assert "/wearusfm_campaigns/" not in j["out"] and not j["out"].startswith(S.WORK + "/wearusfm/")
    assert S.CPU_ROOT.startswith(S.WORK + "/wearusfm_runs/") and S.EXTRACT_ROOT.startswith(S.SCRATCH + "/wearusfm_runs/")
    assert "evaluation19618_v1" in S.CPU_ROOT and "evaluation19618_v1" in S.EXTRACT_ROOT
    # Fits follow the pose ridge by ascending allocated seconds; aggregates list the shards of their cell and task.
    assert [j["seconds"] for j in S.jobs("probe-fit", resources)] == sorted(s["walltime_seconds"] for s in resources["shards"])
    aggregate = S.jobs("probe-aggregate", resources)[0]
    assert aggregate["flags"] == ["--cell", S.CELLS[0], "--task", "p1", "--shards", f"{S.CPU_ROOT}/probe_fits/{S.CELLS[0]}-p1-01/shard.json"]
    with pytest.raises(ValueError, match="unregistered stage"):
        S.jobs("probe-fit-again")
    with pytest.raises(ValueError, match="published probe resources required"):
        S.jobs("probe-fit")


@pytest.fixture
def harness(tmp_path):
    fixture = launcher.harness.__wrapped__(tmp_path)
    h = next(fixture)
    try:
        yield h
    finally:
        try:
            next(fixture)
        except StopIteration:
            pass


@pytest.mark.parametrize("stage", ["pose-validate", "pose-benchmark", "probe-benchmark", "pose-ridge", "compare"])
def test_real_dispatcher_guards_accept_every_job_of_the_table(harness, stage):
    """Flags, CPUs and time limit of each generated job pass the real shell guards of the 19618 launcher."""
    extra = {"pose-ridge": ["--pose-budget", harness["gate_pin"]["path"], "--pose-budget-sha256", harness["gate_pin"]["sha256"]]}
    for n, job in enumerate(S.jobs(stage)):
        got = harness["run"](stage, flags=job["flags"] + extra.get(stage, []), fields={"TimeLimit": S.clock(job["seconds"])},
                             out=harness["work"].parent / f"out{n}")
        assert got.returncode == 0, job["id"] + got.stdout + got.stderr
        assert job["cpus"] == (2 if stage == "pose-validate" else 8) and job["memory"] == "30G"
    wrong = harness["run"](stage, flags=S.jobs(stage)[0]["flags"] + extra.get(stage, []),
                           fields={"TimeLimit": S.clock(S.jobs(stage)[0]["seconds"] + 60)}, out=harness["work"].parent / "other")
    assert wrong.returncode != 0


# ---------------------------------------------------------------------------------------------
# Regression on the real 5000 evidence

@pytest.mark.parametrize("raw, reviewed, cpus, seconds, gpus", [
    ("cpu_readers_20261105/checkout_sacct.txt", "cpu_readers_20261105/checkout_accounting.json", 2, 300, 0),
    ("cpu_readers_20261105/pose_validate/sacct.txt", "cpu_readers_20261105/pose_validate/accounting.json", 2, 1800, 0),
    ("cpu_readers_20261105/pose_benchmark/T100-G-300M-natural/sacct.txt",
     "cpu_readers_20261105/pose_benchmark/T100-G-300M-natural/accounting.json", 8, 600, 0),
    ("pose_completed_20261105/sacct.txt", "pose_completed_20261105/accounting.json", 8, 9000, 1)])
def test_accounting_reproduces_the_reviewed_5000_accounting(raw, reviewed, cpus, seconds, gpus):
    old, text = load(reviewed), (INFRA / raw).read_bytes().decode()
    got = S.accounting(text, old["JobIDRaw"], cpus, seconds, gpus, raw_name=Path(raw).name)
    for key in ("JobIDRaw", "State", "ExitCode", "ElapsedRaw", "AllocCPUS", "MaxRSS_bytes", "MaxRSS_source_rows"):
        assert got[key] == old[key] and type(got[key]) is type(old[key]), key
    assert got["raw_sacct"] == old["raw_sacct"] and got["core_hours"] == old["ElapsedRaw"] * cpus / 3600
    assert got["gpu_hours"] == old.get("gpu_hours", 0) and got["cap_core_hours"] == seconds * cpus / 3600


HEADER = S.SACCT_FORMAT.replace(",", "|")
ROWS = ["77|COMPLETED|0:0|40|2||login08|lrd_all_serial|2026-11-07T10:00:00|2026-11-07T10:00:40|00:30:00|billing=2,cpu=2,mem=30G,node=1",
        "77.batch|COMPLETED|0:0|40|2|1500K|login08||2026-11-07T10:00:00|2026-11-07T10:00:40||cpu=2,mem=30G,node=1",
        "77.extern|COMPLETED|0:0|40|2|900K|login08||2026-11-07T10:00:00|2026-11-07T10:00:40||billing=2,cpu=2,mem=30G,node=1"]


def sacct(*changes, rows=ROWS):
    text = "\n".join([HEADER, *rows]) + "\n"
    for old, new in changes:
        assert old in text
        text = text.replace(old, new)
    return text


def test_accounting_reads_the_registered_columns_and_the_peak_memory():
    got = S.accounting(sacct(), "77", 2, 1800)
    assert got["ElapsedRaw"] == 40 and got["MaxRSS_bytes"] == 1500 * 1024 and got["MaxRSS_source_rows"] == ["77.batch"]
    assert got["Timelimit"] == "00:30:00" and got["Partition"] == "lrd_all_serial" and got["gpus"] == 0
    assert got["raw_sacct"] == {"path": "sacct.txt", "sha256": sha(sacct().encode())}
    assert S.slurm_seconds("1-02:03:04") == 93784 and S.slurm_seconds("14:45:00") == 53100 and S.slurm_seconds("05:00") == 300
    assert S.byte_count("2G") == 2**31 and S.byte_count("1.5M") == 1572864 and S.byte_count("12") == 12
    gpu = sacct(("cpu=2,mem=30G,node=1\n77.batch", "cpu=2,gres/gpu=1,mem=30G,node=1\n77.batch"))
    assert S.accounting(gpu, "77", 2, 1800, gpus=1)["gpu_hours"] == 40 / 3600


@pytest.mark.parametrize("text, message", [
    (sacct(("77|COMPLETED|0:0", "77|TIMEOUT|0:0")), "not COMPLETED"), (sacct(("77.batch|COMPLETED|0:0", "77.batch|FAILED|1:0")), "not COMPLETED"),
    (sacct(("77|COMPLETED|0:0", "77|COMPLETED|0:9")), "not COMPLETED"), (sacct(("77.extern", "78.extern")), "another job"),
    (sacct(("|40|2||login08", "|1801|2||login08")), "outside the published job"), (sacct(("|40|2||login08", "|0|2||login08")), "outside the published job"),
    (sacct(("|40|2||login08", "|40|8||login08")), "outside the published job"), (sacct(("00:30:00", "00:31:00")), "time limit differs"),
    (sacct(("billing=2,cpu=2,mem=30G,node=1\n77.batch", "billing=2,cpu=2,gres/gpu=1,mem=30G,node=1\n77.batch")), "resources differ"),
    (sacct(("billing=2,cpu=2,mem=30G,node=1\n77.batch", "billing=2,cpu=2,mem=30G,node=2\n77.batch")), "resources differ"),
    (sacct(("1500K", ""), ("900K", "")), "no measured MaxRSS"), (sacct(rows=ROWS[1:]), "one parent"),
    (sacct(("JobIDRaw|State", "JobID|State")), "unexpected accounting columns"), (sacct(("77.batch|COMPLETED|0:0|40", "77.batch|COMPLETED|0:0")), "malformed"),
    (sacct(("1500K", "1500Q")), "invalid scheduler memory"), (sacct(("00:30:00", "UNLIMITED")), "invalid scheduler duration")])
def test_accounting_stops_on_anything_but_a_completed_job_inside_its_allocation(text, message):
    with pytest.raises(ValueError, match=message):
        S.accounting(text, "77", 2, 1800)


@pytest.mark.parametrize("arm", S.ARMS)
def test_classification_pins_reproduce_the_reviewed_5000_probe_plan(arm):
    receipts, pins, provenances = old_classification(arm)
    got = S.candidate_probe_caches(receipts, pins, provenances, old_extraction(arm), protocol=OLD["evaluation"], step=5000, arm=arm)
    reviewed = load("cpu_readers_20261105/probe_plan.json")["caches"][arm]
    assert got == reviewed and list(got) == list(reviewed) == list(S.DATASETS)
    assert all(list(got[ds]) == list(reviewed[ds]) for ds in got)


def test_pose_pins_reproduce_the_reviewed_5000_pose_plan():
    receipt, pin, provenances = old_pose()
    extraction = {**old_extraction(), "plan_sha256": receipt["plan_sha256"]}
    got = S.candidate_pose_caches(receipt, pin, provenances, extraction, protocol=OLD["evaluation"], step=5000)
    reviewed = load("cpu_readers_20261105/pose_plan.json")["caches"][ARM]
    assert got == reviewed and all(set(got[s]) == S.Q.CACHE_FIELDS for s in got)
    assert receipt["plan_sha256"] == sha(INFRA / "extraction_plan_5000_20261105.json")


def old_benchmarks():
    reports, pins = {}, {c: {} for c in S.CELLS}
    for entry in load("cpu_recovery_20261105/accepted_benchmarks.json")["reports"]:
        data = (ROOT / entry["local_report"]["path"]).read_bytes()
        assert sha(data) == entry["report"]["sha256"]
        reports[entry["cell"], entry["dataset"]] = json.loads(data)
        pins[entry["cell"]][entry["dataset"]] = {"path": entry["report"]["remote_path"], **{k: entry["report"][k] for k in ("sha256", "bytes")}}
    return reports, pins


def old_resources(reports=None, pins=None, **changes):
    found, found_pins = old_benchmarks()
    reviewed = load("cpu_fit_release_20261105/probe_fit_resources.json")
    arguments = dict(plan=load("cpu_readers_20261105/probe_plan.json"), plan_sha256=sha(INFRA / "cpu_readers_20261105/probe_plan.json"),
                     code_sha=reviewed["code_sha"], reports=reports or found, pins=pins or found_pins, protocol=OLD["probe"], step=5000,
                     resource_protocol=OLD["probe_resources"])
    return S.probe_resources(**{**arguments, **changes})


def test_packing_reproduces_the_reviewed_5000_fit_release():
    budget, audit = old_resources()
    reviewed, old_audit = load("cpu_fit_release_20261105/probe_fit_resources.json"), load("cpu_fit_release_20261105/probe_fit_packing_audit.json")
    assert {k: v for k, v in reviewed.items() if k != "accepted_benchmark_manifest"} == budget
    assert audit["groups"] == old_audit["groups"] and audit["minimum_shards"] == old_audit["minimum_shards"] == 14
    for key in ("total_requested_seconds", "requested_core_hours", "core_hours_with_scheduler_allowance", "covered_cells_dataset_folds"):
        assert audit[key] == old_audit[key], key
    assert audit["remaining_to_cap_requested_core_hours"] == pytest.approx(240 - 205.86666666666667)
    # The dispatcher of the fits accepts exactly this shape: full coverage, caps, core-hours.
    cover = [(s["cell"], d, f) for s in budget["shards"] for d in s["datasets"] for f in s["folds"]]
    assert len(cover) == len(set(cover)) == 60 and all(0 < s["walltime_seconds"] <= 10800 and s["walltime_seconds"] % 60 == 0 and
                                                       s["walltime_seconds"] >= s["estimated_seconds_with_margin"] for s in budget["shards"])


def mutated_benchmarks(change):
    reports, pins = old_benchmarks()
    reports = copy.deepcopy(reports)
    change(reports)
    return reports, pins


@pytest.mark.parametrize("change, message", [
    (lambda r: r.pop((S.CELLS[1], "epn612")), "all fifteen"),
    (lambda r: r[S.CELLS[0], "uci_emg"].update(complete=False), "identity differs"),
    (lambda r: r[S.CELLS[0], "uci_emg"].update(step=19618), "identity differs"),
    (lambda r: r[S.CELLS[0], "uci_emg"].update(selection_performed=True), "identity differs"),
    (lambda r: r[S.CELLS[0], "uci_emg"].update(code_sha="d" * 40), "plan/code/recipe"),
    (lambda r: r[S.CELLS[0], "uci_emg"].update(plan_sha256="d" * 64), "plan/code/recipe"),
    (lambda r: r[S.CELLS[2], "ninapro_db3"]["fits"][3].update(converged_by_criterion=False), "not converged"),
    (lambda r: r[S.CELLS[2], "ninapro_db3"]["fits"][3].update(reached_limit=True), "not converged"),
    (lambda r: r[S.CELLS[2], "ninapro_db3"]["fits"].pop(), "not converged"),
    (lambda r: r[S.CELLS[1], "ninapro_db6"].update(fold_train_counts=[1, 2, 3]), "folds differ"),
    (lambda r: r[S.CELLS[1], "ninapro_db6"].update(roles_sha256="e" * 64), "train support"),
    (lambda r: r[S.CELLS[1], "ninapro_db6"].update(peak_rss_bytes=24 * 2**30), "memory outside"),
    (lambda r: r[S.CELLS[1], "ninapro_db6"].update(numeric_sha256="e" * 64), "numeric signature"),
    (lambda r: [f.update(elapsed_s=f["elapsed_s"] * 40) for f in r[S.CELLS[1], "epn612"]["fits"]], "no exact cover"),
    (lambda r: [f.update(elapsed_s=f["elapsed_s"] * 1.6) for x in r.values() for f in x["fits"]], "core-hours|more shards")])
def test_packing_stops_before_the_fits_on_incomplete_slow_or_inconsistent_benchmarks(change, message):
    reports, pins = mutated_benchmarks(change)
    with pytest.raises(ValueError, match=message):
        old_resources(reports, pins)


def test_packing_refuses_another_numeric_context_or_local_pins():
    def other(reports):
        report = reports[S.CELLS[0], "uci_emg"]
        report["numeric_context"] = {**report["numeric_context"], "extra": 1}
        report["numeric_sha256"] = S.C.jhash(report["numeric_context"])
    with pytest.raises(ValueError, match="different numeric contexts"):
        old_resources(*mutated_benchmarks(other))
    reports, pins = old_benchmarks()
    pins = copy.deepcopy(pins)
    pins[S.CELLS[0]]["uci_emg"]["path"] = "relative/benchmark.json"
    with pytest.raises(ValueError, match="remote absolute"):
        old_resources(reports, pins)


def old_pose_budget(change=None):
    reviewed = load("cpu_readers_20261105/pose_budget.json")

    def measured(directory, report_pin, accounting_pin):
        report, record = (INFRA / directory / "report.json").read_bytes(), (INFRA / directory / "accounting.json").read_bytes()
        assert sha(report) == report_pin["sha256"] and sha(record) == accounting_pin["sha256"]
        return {"report": json.loads(report), "report_pin": report_pin, "accounting": json.loads(record), "accounting_pin": accounting_pin}

    validation = measured("cpu_readers_20261105/pose_validate", reviewed["validation_report"], reviewed["validation_accounting"])
    benchmarks = {c: measured("cpu_readers_20261105/pose_benchmark/" + c, reviewed["benchmarks"][c], reviewed["benchmark_accounting"][c])
                  for c in S.CELLS}
    if change:
        change(validation, benchmarks)
    return S.pose_budget(load("cpu_readers_20261105/pose_plan.json"), sha(INFRA / "cpu_readers_20261105/pose_plan.json"),
                         reviewed["code_sha"], validation, benchmarks, protocol=OLD["pose"], step=5000,
                         resource_protocol=OLD["pose_resources"]), reviewed


def test_pose_budget_reproduces_the_reviewed_5000_budget():
    got, reviewed = old_pose_budget()
    assert {k: v for k, v in got.items() if k != "release_note"} == {k: v for k, v in reviewed.items() if k != "release_note"}
    assert got["derived_seconds"] == 660 and list(got) == list(reviewed)


@pytest.mark.parametrize("change, message", [
    (lambda v, b: v["accounting"].update(State="TIMEOUT"), "validation accounting"),
    (lambda v, b: v["accounting"].update(JobIDRaw="1"), "validation accounting"),
    (lambda v, b: v["report"].update(scientific_metrics_computed=True), "four-cache validation"),
    (lambda v, b: v["report"]["inspections"].pop("T100-G-30M/val"), "four-cache validation"),
    (lambda v, b: v["report"].update(plan_sha256="d" * 64), "provenance differs"),
    (lambda v, b: b[S.CELLS[1]]["report"]["runtime"].update(code_sha="d" * 40), "provenance differs"),
    (lambda v, b: b[S.CELLS[1]]["report"].update(input_pins={}), "inputs differ"),
    (lambda v, b: b[S.CELLS[2]]["accounting"].update(ElapsedRaw=601), "benchmark accounting"),
    (lambda v, b: b[S.CELLS[2]]["accounting"].update(MaxRSS_bytes=31 * 2**30), "benchmark accounting"),
    (lambda v, b: b[S.CELLS[2]]["report"].update(cell=S.CELLS[1]), "benchmark identity"),
    (lambda v, b: b[S.CELLS[2]]["report"]["timings"].update(n_features=896), "support/dimension"),
    (lambda v, b: b[S.CELLS[0]]["report"]["timings"].update(valid_fit_elements=1), "support/dimension"),
    (lambda v, b: b[S.CELLS[0]]["report"]["timings"].update(statistics_seconds=0.0), "invalid measured"),
    (lambda v, b: b[S.CELLS[0]]["report"]["runtime"].update(torch_threads=4), "eight-thread"),
    (lambda v, b: b[S.CELLS[0]]["report"]["runtime"]["versions"].update(numpy="other"), "different numeric runtimes"),
    (lambda v, b: b.pop(S.CELLS[0]), "all three"),
    (lambda v, b: b[S.CELLS[1]]["report"]["timings"].update(three_eigensystems_seconds=200.0), "exceeds the frozen 3600")])
def test_pose_budget_stops_before_the_fit_on_any_failed_gate(change, message):
    with pytest.raises(ValueError, match=message):
        old_pose_budget(change)


# ---------------------------------------------------------------------------------------------
# 19618 documents: real control evidence, candidate fixtures re-labelled from the real 5000 files

NEW_CHECKPOINT = "9" * 64
TRAINING = load("continuation19618_20261106/training_plan.json")


def training_files():
    run, out = TRAINING["run_dir"], TRAINING["receipt_dir"]
    files = {"checkpoint": {"path": f"{run}/checkpoint_step0019618.pt", "sha256": NEW_CHECKPOINT, "bytes": 4673783740},
             "config": {"path": out + "/config.json", "sha256": "1" * 64, "bytes": 3000},
             "summary": {"path": out + "/summary.json", "sha256": "2" * 64, "bytes": 1900}}
    receipt = {"protocol": S.E.TRAIN_PROTOCOL, "complete": True, "arm": ARM, "checkpoint_step": 19618,
               "checkpoint": files["checkpoint"]["path"], "checkpoint_sha256": NEW_CHECKPOINT, "config_path": files["config"]["path"],
               "config_file_sha256": files["config"]["sha256"], "start_step": 5000, "additional_steps": 14618,
               "scientific_endpoint": True, "previous_run_unchanged": True, "further_training_authorized": False,
               "control_checkpoint_sha256": S.E.CONTROL_CHECKPOINT_SHA256, "code_sha": CODE,
               "training_source_contract_sha256": TRAINING["source_contract"]["sha256"]}
    files["receipt"] = S.pin(out + "/receipt.json", body(receipt))
    return files, receipt, {"stopped": "max_steps", "steps": 19618}


def new_extraction_plan(change=None):
    files, receipt, summary = training_files()
    training = copy.deepcopy(TRAINING)
    if change:
        change(files, receipt, summary, training)
    return S.extraction_plan(load("extraction_plan_5000_20261105.json"), training, files, receipt, summary)


def test_extraction_plan_pins_only_the_certified_300m_endpoint_and_keeps_the_reviewed_inputs():
    plan, old = new_extraction_plan(), load("extraction_plan_5000_20261105.json")
    assert plan["protocol"] == S.E.PROTOCOL and plan["step"] == 19618 and plan["arms"] == list(S.ARMS)
    assert plan["control_checkpoint_sha256"] == S.E.CONTROL_CHECKPOINT_SHA256 and set(plan["checkpoints"]) == {ARM}
    for key in ("comparison", "inputs", "paths", "pose_references", "classification_references", "training_source_contract"):
        assert plan[key] == old[key], key
    pin = plan["checkpoints"][ARM]
    assert list(pin) == list(old["checkpoints"][ARM]) and Path(pin["path"]).name == "checkpoint_step0019618.pt"
    assert pin["path"].startswith(TRAINING["run_dir"] + "/") and pin["receipt_path"] == TRAINING["receipt_dir"] + "/receipt.json"
    assert old["checkpoints"][ARM]["path"] not in json.dumps(plan) and old["checkpoints"][CONTROL]["path"] not in json.dumps(plan)
    # The shell guard of the extraction launcher accepts exactly this shape.
    text = (ROOT / "scripts/slurm/scaling_evaluation19618.sbatch").read_text()
    assert "plan.get('protocol') != '%s'" % plan["protocol"] in text and "set(plan.get('checkpoints', {})) != {arms[1]}" in text


@pytest.mark.parametrize("change, message", [
    (lambda f, r, s, t: r.update(protocol="scaling-training-20261104-v1"), "receipt differs: protocol"),
    (lambda f, r, s, t: r.update(complete=False), "receipt differs: complete"),
    (lambda f, r, s, t: r.update(arm=CONTROL), "receipt differs: arm"),
    (lambda f, r, s, t: r.update(checkpoint_step=5000), "receipt differs: checkpoint_step"),
    (lambda f, r, s, t: r.update(checkpoint_step=19618.0), "receipt differs: checkpoint_step"),
    (lambda f, r, s, t: r.update(checkpoint=r["checkpoint"].replace("0019618", "0005000")), "receipt differs: checkpoint"),
    (lambda f, r, s, t: r.update(checkpoint_sha256="8" * 64), "receipt differs: checkpoint_sha256"),
    (lambda f, r, s, t: r.update(config_file_sha256="8" * 64), "receipt differs: config_file_sha256"),
    (lambda f, r, s, t: r.update(start_step=0), "receipt differs: start_step"),
    (lambda f, r, s, t: r.update(additional_steps=19618), "receipt differs: additional_steps"),
    (lambda f, r, s, t: r.update(scientific_endpoint=False), "receipt differs: scientific_endpoint"),
    (lambda f, r, s, t: r.update(previous_run_unchanged=False), "receipt differs: previous_run_unchanged"),
    (lambda f, r, s, t: r.update(further_training_authorized=True), "receipt differs: further_training_authorized"),
    (lambda f, r, s, t: r.update(control_checkpoint_sha256=NEW_CHECKPOINT), "receipt differs: control_checkpoint_sha256"),
    (lambda f, r, s, t: r.update(training_source_contract_sha256="8" * 64), "receipt differs: training_source_contract"),
    (lambda f, r, s, t: s.update(stopped="time_limit"), "registered endpoint"),
    (lambda f, r, s, t: s.update(steps=19617), "registered endpoint"),
    (lambda f, r, s, t: f["checkpoint"].update(path=f["checkpoint"]["path"].replace("-from5000", "")), "not the planned one"),
    (lambda f, r, s, t: f["checkpoint"].pop("bytes"), "not the planned one"),
    (lambda f, r, s, t: f.pop("summary"), "exactly the four"),
    (lambda f, r, s, t: t["control"]["checkpoint"].update(sha256="8" * 64), "control/contract differs")])
def test_extraction_plan_needs_the_completed_exact_continuation(change, message):
    with pytest.raises(ValueError, match=message):
        new_extraction_plan(change)
    with pytest.raises(ValueError, match="reviewed 5000 extraction plan"):
        S.extraction_plan(new_extraction_plan(), TRAINING, *training_files())


def relabel(text, old_plan_sha, plan_sha):
    """A real 5000 receipt or provenance of the 300M with the identity of a 19618 extraction."""
    old = old_extraction()
    for a, b in ((OLD["evaluation"], S.E.PROTOCOL), ('"checkpoint_step": 5000', '"checkpoint_step": 19618'),
                 (old["checkpoint"]["sha256"], NEW_CHECKPOINT), (old["code_sha"], CODE), (old_plan_sha, plan_sha),
                 ("/scaling_300m_20261105/5000/T100-G-300M/", S.EXTRACT_ROOT.replace(S.SCRATCH + "/wearusfm_runs/results", "") + "/")):
        assert a in text, a
        text = text.replace(a, b)
    return text


def new_candidates(plan_sha="7" * 64):
    """Receipts and provenance as the three 19618 extractions will write them, with consistent hashes."""
    extraction = {"checkpoint": new_extraction_plan()["checkpoints"][ARM], "code_sha": CODE, "plan_sha256": plan_sha}
    old_sha = old_extraction()["plan_sha256"]
    receipts, _, provenances = old_classification(ARM)
    provenances = {ds: relabel(data.decode(), old_sha, plan_sha).encode() for ds, data in provenances.items()}
    new_receipts, pins = {}, {}
    for task, receipt in receipts.items():
        receipt = json.loads(relabel(json.dumps(receipt), old_sha, plan_sha))
        for ds in S.TASKS[task]:
            receipt["outputs"][ds]["provenance_sha256"] = sha(provenances[ds])
        new_receipts[task], pins[task] = receipt, S.pin(f"{S.EXTRACT_ROOT}/{task}/receipt.json", body(receipt))
    pose_receipt, _, pose_provenances = old_pose()
    pose_provenances = {s: relabel(data.decode(), old_sha, plan_sha).encode() for s, data in pose_provenances.items()}
    pose_receipt = json.loads(relabel(json.dumps(pose_receipt), old_sha, plan_sha))
    for split in pose_provenances:
        pose_receipt["outputs"][split]["provenance_sha256"] = sha(pose_provenances[split])
    pose = (pose_receipt, S.pin(S.EXTRACT_ROOT + "/pose/receipt.json", body(pose_receipt)), pose_provenances)
    return extraction, (new_receipts, pins, provenances), pose


def control_provenances():
    return {ds: (PACKAGE / "control19618_identity" / f"{ds}.npz.provenance.json").read_bytes() for ds in S.DATASETS}


def inventory(kind):
    return load({"pose": "pose_readout_plan_19618.json", "probe": "probe_plan_19618.json"}[kind],
                ROOT / "results/infra/tokenizer_evaluation_20261103")


def test_control_identity_evidence_is_the_one_pinned_by_the_training_plan_and_accepted_by_the_readers():
    artifacts, freeze = TRAINING["control"]["artifacts"], load("continuation19618_budget_freeze_20261105_v2.json")["control19618"]
    names = {**{f"{t}_{ds}": ds for t, names in S.TASKS.items() for ds in names}, "pose_train": "train_subset", "pose_val": "val"}
    context = S.ST.check_source_contract(str(INFRA / "training_source_contract.json"))
    model = S.ST.R.config_to_dict(context["config_30m"])["model"]
    for key, name in names.items():
        data = (PACKAGE / "control19618_identity" / f"{name}.npz.provenance.json").read_bytes()
        record = json.loads(data)
        assert sha(data) == artifacts[key + "_provenance"]["sha256"], key
        assert record["cache"] == artifacts[key]["path"] and record["cache_sha256"] == artifacts[key]["sha256"]
        # Identity required by scaling_probe19618.load_cache and scaling_pose_readout19618.validate_artifact for the control.
        assert {k: record[k] for k in ("protocol", "arm", "which", "checkpoint_step", "checkpoint_sha256", "postload_recipe_passed")} == {
            "protocol": S.E.E.PROTOCOL, "arm": "T100-G", "which": "teacher", "checkpoint_step": 19618,
            "checkpoint_sha256": S.E.CONTROL_CHECKPOINT_SHA256, "postload_recipe_passed": True}
        assert "training_source_contract_sha256" not in record and record["model_config"] == model
        S.E.model_recipe(record["model_config"], CONTROL)
    assert len(list((PACKAGE / "control19618_identity").iterdir())) == 7
    for kind, key in (("pose", "pose_inventory"), ("probe", "classification_inventory")):
        assert sha(ROOT / freeze[key]["path"]) == freeze[key]["sha256"] == TRAINING["control"][key]["sha256"]
    for ds, pin in inventory("probe")["caches"]["T100-G"].items():
        record = json.loads(control_provenances()[ds])
        assert {k: record[k] for k in ("code_sha", "plan_sha256", "input_sha256", "splits_sha256", "extraction_recipe")} == {
            "code_sha": pin["extraction_code_sha"], "plan_sha256": pin["extraction_plan_sha256"],
            **{k: pin[k] for k in ("input_sha256", "splits_sha256", "extraction_recipe")}}
        assert record["reference_cache_sha256"] == load("extraction_plan_5000_20261105.json")["classification_references"][ds]["sha256"]


def new_probe_plan(change=None):
    extraction, (receipts, pins, provenances), _ = new_candidates()
    control = S.control_probe_caches(inventory("probe"), control_provenances(), TRAINING["control"]["checkpoint"])
    candidate = S.candidate_probe_caches(receipts, pins, provenances, extraction)
    if change:
        change(control, candidate)
    return S.probe_plan(load("cpu_readers_20261105/probe_plan.json"), control, candidate)


def test_probe_plan_joins_the_reused_control_and_the_new_candidate_on_the_reviewed_datasets():
    plan, old = new_probe_plan(), load("cpu_readers_20261105/probe_plan.json")
    assert plan["protocol"] == S.C.PROTOCOL and plan["step"] == 19618 and list(plan) == list(old)
    assert plan["datasets"] == old["datasets"] and plan["references"] == old["references"]
    assert plan["recipe"] == {**old["recipe"], "checkpoint_step": 19618} and plan["sources"] == S.C.source_hashes()
    control, candidate = plan["caches"][CONTROL], plan["caches"][ARM]
    assert all(list(control[ds]) == list(candidate[ds]) == list(old["caches"][ARM][ds]) for ds in S.DATASETS)
    assert {p["checkpoint_sha256"] for p in control.values()} == {S.E.CONTROL_CHECKPOINT_SHA256}
    assert {p["checkpoint_path"] for p in control.values()} == {TRAINING["control"]["checkpoint"]["path"]}
    assert all("/tokenizer_evaluation_20261103/19618/T100-G/" in p["path"] for p in control.values())
    assert all(p["path"].startswith(S.EXTRACT_ROOT + "/") and p["checkpoint_sha256"] == NEW_CHECKPOINT for p in candidate.values())
    for key in S.SHARED:  # same bytes as at 5000, read in the frozen 19618 code checkout
        assert plan[key] == {**old[key], "path": S.checkout("code") + "/" + S.SHARED[key]}
    for ds, pin in control.items():  # control sizes are those measured by its extraction
        assert pin["bytes"] == json.loads(control_provenances()[ds])["cache_bytes"] > 0


@pytest.mark.parametrize("change, message", [
    (lambda c, n: c["epn612"].update(checkpoint_sha256=NEW_CHECKPOINT), "mixed checkpoints|wrong registered"),
    (lambda c, n: [p.update(checkpoint_sha256=NEW_CHECKPOINT) for p in c.values()], "wrong registered 30M 19618 control"),
    (lambda c, n: [p.update(checkpoint_sha256=S.E.CONTROL_CHECKPOINT_SHA256) for p in n.values()], "duplicate cache/checkpoint"),
    (lambda c, n: n["uci_emg"].update(input_sha256="8" * 64), "cache dataset identity differs"),
    (lambda c, n: n["ninapro_db2"].update(extraction_recipe={}), "cache dataset identity differs"),
    (lambda c, n: n.pop("ninapro_db6"), "incomplete cache grid"),
    (lambda c, n: n["ninapro_db3"].update(path=c["ninapro_db3"]["path"]), "duplicate cache/checkpoint"),
    (lambda c, n: n["ninapro_db3"].update(checkpoint_path="relative.pt"), "absolute input paths")])
def test_probe_plan_is_validated_by_the_19618_reader(change, message):
    with pytest.raises(ValueError, match=message):
        new_probe_plan(change)


def test_cache_pins_need_complete_receipts_of_the_published_extraction():
    extraction, (receipts, pins, provenances), (pose_receipt, pose_pin, pose_provenances) = new_candidates()
    cases = [({"complete": False}, "complete"), ({"stage": "extract-pose"}, "stage"), ({"checkpoint_step": 5000}, "checkpoint_step"),
             ({"code_sha": "d" * 40}, "code_sha"), ({"plan_sha256": "d" * 64}, "plan_sha256"), ({"arm": CONTROL}, "arm"),
             ({"protocol": OLD["evaluation"]}, "protocol"), ({"checkpoint_unchanged": False}, "checkpoint_unchanged"),
             ({"postload_recipe_passed": False}, "postload_recipe_passed"), ({"checkpoint_sha256": "d" * 64}, "checkpoint_sha256")]
    for change, key in cases:
        with pytest.raises(ValueError, match="extraction receipt differs: " + key):
            S.candidate_probe_caches({**receipts, "p2": {**receipts["p2"], **change}}, pins, provenances, extraction)
    for change, key in cases:
        change = {"stage": "extract-classification"} if key == "stage" else change
        with pytest.raises(ValueError, match="extraction receipt differs: " + key):
            S.candidate_pose_caches({**pose_receipt, **change}, pose_pin, pose_provenances, extraction)
    with pytest.raises(ValueError, match="collected provenance differs"):
        S.candidate_probe_caches(receipts, pins, {**provenances, "epn612": provenances["epn612"] + b" "}, extraction)
    with pytest.raises(ValueError, match="collected provenance differs"):
        S.candidate_pose_caches(pose_receipt, pose_pin, {**pose_provenances, "val": pose_provenances["val"] + b" "}, extraction)
    with pytest.raises(ValueError, match="five datasets"):
        S.candidate_probe_caches(receipts, pins, {k: v for k, v in provenances.items() if k != "uci_emg"}, extraction)
    with pytest.raises(ValueError, match="train and val"):
        S.candidate_pose_caches(pose_receipt, pose_pin, {"train": pose_provenances["train"]}, extraction)
    swapped = copy.deepcopy(receipts)
    swapped["p1"]["outputs"]["ninapro_db2"]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="provenance/receipt mismatch: ninapro_db2"):
        S.candidate_probe_caches(swapped, pins, provenances, extraction)
    wrong = {**control_provenances(), "uci_emg": control_provenances()["uci_emg"] + b" "}
    with pytest.raises(ValueError, match="control provenance differs from the inventory: uci_emg"):
        S.control_probe_caches(inventory("probe"), wrong, TRAINING["control"]["checkpoint"])
    with pytest.raises(ValueError, match="not the registered control inventory"):
        S.control_probe_caches(inventory("probe"), control_provenances(), {**TRAINING["control"]["checkpoint"], "sha256": NEW_CHECKPOINT})
    with pytest.raises(ValueError, match="not the registered control inventory"):
        S.control_probe_caches(load("cpu_readers_20261105/probe_plan.json"), control_provenances(), TRAINING["control"]["checkpoint"])


def new_pose_plan(change=None):
    extraction, _, (receipt, pin, provenances) = new_candidates()
    candidate, source = S.candidate_pose_caches(receipt, pin, provenances, extraction), copy.deepcopy(inventory("pose"))
    if change:
        change(source, candidate)
    return S.pose_plan(load("cpu_readers_20261105/pose_plan.json"), source, candidate)


def test_pose_plan_joins_the_reused_control_and_the_new_candidate_on_the_reviewed_references():
    plan, old = new_pose_plan(), load("cpu_readers_20261105/pose_plan.json")
    assert plan["protocol"] == S.P.PROTOCOL and plan["step"] == 19618 and list(plan) == list(old)
    for key in ("recipe", "stats_sha256", "fold_users", "references"):
        assert plan[key] == old[key], key
    assert plan["caches"][CONTROL] == inventory("pose")["caches"]["T100-G"] and plan["sources"] == S.P.source_hashes()
    assert all(p["path"].startswith(S.EXTRACT_ROOT + "/pose/") and p["checkpoint_sha256"] == NEW_CHECKPOINT and
               p["extraction_code_sha"] == CODE for p in plan["caches"][ARM].values())
    probe = new_probe_plan()
    for key in ("projection", "training_source_contract"):  # the comparison requires the same pins in both plans
        assert plan[key] == probe[key] == {**old[key], "path": S.checkout("code") + "/" + S.SHARED[key]}
    for arm in S.ARMS:  # and the same checkpoint per arm
        assert {p["checkpoint_sha256"] for p in plan["caches"][arm].values()} == {p["checkpoint_sha256"] for p in probe["caches"][arm].values()}


@pytest.mark.parametrize("change, message", [
    (lambda i, c: i.update(step=5000), "another dose"), (lambda i, c: i["fold_users"][0].reverse(), "another dose"),
    (lambda i, c: i["references"]["val"].update(sha256="8" * 64), "another dose"),
    (lambda i, c: i["caches"]["T100-G"]["val"].update(checkpoint_sha256="8" * 64), "not the registered control"),
    (lambda i, c: [p.update(checkpoint_sha256=S.E.CONTROL_CHECKPOINT_SHA256) for p in c.values()], "duplicate scaling checkpoints"),
    (lambda i, c: c["val"].update(checkpoint_sha256="8" * 64), "train/val checkpoint mismatch"),
    (lambda i, c: c["train"].pop("bytes"), "cache pin fields mismatch"),
    (lambda i, c: c["train"].update(path=c["val"]["path"]), "duplicate cache splits")])
def test_pose_plan_is_validated_by_the_19618_reader(change, message):
    with pytest.raises(ValueError, match=message):
        new_pose_plan(change)
    with pytest.raises(ValueError, match="reviewed 5000 pose plan"):
        S.pose_plan(new_pose_plan(), inventory("pose"), new_pose_plan()["caches"][ARM])
    with pytest.raises(ValueError, match="reviewed 5000 probe plan"):
        S.probe_plan(new_probe_plan(), {}, {})


# ---------------------------------------------------------------------------------------------
# Collection, documents on disk and submission lines

def reading_for(table, slurm, texts, state="COMPLETED", extra=()):
    """The answer of the remote reading for a completed stage: copied files, pinned files, logs, folders, accounting."""
    files, listings = [], {}
    for job in table:
        log = S.log_path(job, slurm[job["id"]])
        for path in [*job["files"].values(), log]:
            data = texts.get(path, b"log of %s\n" % job["id"].encode())
            files.append({"path": path, "sha256": sha(data), "bytes": len(data), "canonical": True,
                          "content": base64.b64encode(data).decode()})
        listings[job["folder"]] = sorted(job["exact"] or [Path(p).name for p in [*job["files"].values(), *job["pinned"].values(), *extra]])
    for job in table:
        files += [{"path": path, "sha256": "5" * 64, "bytes": 4673783740, "canonical": True} for path in job["pinned"].values()]
    rows = {}
    for job in table:
        j, cpus = slurm[job["id"]], job["cpus"]
        gres = "gres/gpu=%d," % job["gpus"] if job["gpus"] else ""
        rows[j] = {"exit": 0, "stderr": "", "stdout": "\n".join([
            HEADER, f"{j}|{state}|0:0|33|{cpus}||n|p|a|b|{S.clock(job['seconds'])}|cpu={cpus},{gres}mem=30G,node=1",
            f"{j}.batch|{state}|0:0|33|{cpus}|2048K|n||a|b||cpu={cpus},{gres}mem=30G,node=1"]) + "\n"}
    return {"files": files, "sacct": rows, "listings": listings, "exists": {}, "commands": {}, "time_utc": "2026-11-07T09:00:00Z"}


def test_collection_writes_a_stage_only_when_every_job_completed_and_reads_it_back(tmp_path):
    table = S.jobs("pose-benchmark")
    slurm = {job["id"]: str(900 + n) for n, job in enumerate(table)}
    texts = {path: body({"cell": job["flags"][1], "big": "x" * (S.GZIP_FROM if n == 1 else 10)})
             for n, job in enumerate(table) for path in job["files"].values()}
    target = tmp_path / S.PACKAGE / "evidence" / "pose-benchmark"
    failed = reading_for(table, slurm, texts)
    failed["sacct"]["901"]["stdout"] = failed["sacct"]["901"]["stdout"].replace("COMPLETED", "TIMEOUT")
    for bad, message in ((failed, "not COMPLETED"), ({**reading_for(table, slurm, texts), "files": [
            {**f, "canonical": False} for f in reading_for(table, slurm, texts)["files"]]}, "non-canonical"),
            ({**reading_for(table, slurm, texts), "files": [{"path": f["path"], "missing": True} for f in reading_for(table, slurm, texts)["files"]]}, "missing"),
            ({**reading_for(table, slurm, texts), "files": [{**f, "sha256": "0" * 63 + "1"} for f in reading_for(table, slurm, texts)["files"]]}, "transfer differs")):
        with pytest.raises(ValueError, match=message):
            S.collect("pose-benchmark", table, slurm, bad, target)
        assert not target.exists()  # nothing is written for a stage that did not pass
    manifest = S.collect("pose-benchmark", table, slurm, reading_for(table, slurm, texts), target)
    assert sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()) == sorted(
        ["collection.json", *(f"{c}/{n}" for c in S.CELLS for n in ("accounting.json", "sacct.txt", "slurm_log.txt",
                                                                    "report.json.gz" if c == S.CELLS[1] else "report.json"))])
    assert manifest["jobs"][table[0]["id"]]["listing"] == ["report.json"] and manifest["jobs"][table[0]["id"]]["folder"] == table[0]["out"]
    assert manifest["files"][S.CELLS[0] + "/slurm_log.txt"]["path"] == f"{S.LOGS}/scaling_readers19618_900.out"
    assert [manifest["jobs"][j["id"]]["job_slurm"] for j in table] == ["900", "901", "902"]
    files, again = S.evidence("pose-benchmark", tmp_path)
    assert again == manifest and all(files[f"{j['flags'][1]}/report.json"] == (texts[p], {"path": p, "sha256": sha(texts[p]), "bytes": len(texts[p])})
                                     for j in table for p in j["files"].values())
    assert files[S.CELLS[2] + "/slurm_log.txt"][0] == b"log of %s\n" % table[2]["id"].encode()
    record, pin = S.accounting_pin("pose-benchmark", S.CELLS[1], tmp_path)
    assert record["JobIDRaw"] == "901" and record["AllocCPUS"] == 8 and record["MaxRSS_bytes"] == 2048 * 1024
    assert pin["path"] == f"{S.checkout('budgets')}/{S.PACKAGE}/evidence/pose-benchmark/{S.CELLS[1]}/accounting.json"
    assert pin["sha256"] == sha(target / S.CELLS[1] / "accounting.json")
    with pytest.raises(ValueError, match="already collected"):
        S.collect("pose-benchmark", table, slurm, reading_for(table, slurm, texts), target)
    (target / S.CELLS[0] / "report.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="differs from its pin"):
        S.evidence("pose-benchmark", tmp_path)


def test_remote_read_is_a_bounded_read_only_script(tmp_path):
    small, large = tmp_path / "small.json", tmp_path / "large.bin"
    small.write_bytes(b'{"a": 1}\n')
    large.write_bytes(b"x" * 5000)
    bins = tmp_path / "bin"
    bins.mkdir()
    (bins / "sacct").write_text("#!/bin/sh\necho \"$@\"\n")
    (bins / "ssh").write_text("#!/bin/sh\nPATH=%s:$PATH exec %s -\n" % (bins, sys.executable))
    for name in ("sacct", "ssh"):
        (bins / name).chmod((bins / name).stat().st_mode | stat.S_IXUSR)
    got = S.remote_read([str(small), str(large), str(tmp_path / "absent")], ["41", "42"], limit=1000, ssh=[str(bins / "ssh")])
    first, second, third = got["files"]
    assert base64.b64decode(first["content"]) == small.read_bytes() and first["sha256"] == sha(small) and first["canonical"] is True
    assert "content" not in second and second["sha256"] == sha(large) and second["bytes"] == 5000  # hashed, not transferred
    assert third == {"path": str(tmp_path / "absent"), "missing": True}
    assert got["sacct"]["42"]["stdout"].strip() == "-P -j 42 --format=" + S.SACCT_FORMAT and got["time_utc"].endswith("Z")
    for forbidden in ("open(path, \"w", "os.remove", "unlink", "rename", "sbatch", "scancel", "mkdir", "chmod"):
        assert forbidden not in S.REMOTE
    assert S.SSH[-1] == "leonardo" and "ControlMaster=no" in S.SSH and "ProxyCommand=none" in S.SSH
    (bins / "ssh").write_text("#!/bin/sh\nexit 255\n")
    with pytest.raises(ValueError, match="remote read failed"):
        S.remote_read([str(small)], [], ssh=[str(bins / "ssh")])


def test_ledger_job_needs_a_submission_of_this_line(tmp_path):
    def state(**fields):
        (tmp_path / "J").mkdir(exist_ok=True)
        (tmp_path / "J" / "state.json").write_text(json.dumps({"stato": "sottomesso", "job_slurm": 123, "agente": "claude",
                                                               "campagna": S.CAMPAIGN, **fields}))
    state()
    assert S.ledger_job("J", tmp_path) == "123"
    for fields in ({"stato": "prenotato"}, {"stato": "rifiutato"}, {"job_slurm": None}, {"agente": "gpt"}, {"campagna": "linea-2"}):
        state(**fields)
        with pytest.raises(ValueError, match="not submitted by this line"):
            S.ledger_job("J", tmp_path)


def parse(line):
    argv = shlex.split(line)
    cut = argv.index("--")
    options = dict(zip(argv[2:cut:2], argv[3:cut:2]))
    return argv[:2], options, argv[cut + 1:]


def published(tmp_path):
    """The published documents of the chain in a scratch repository root."""
    base = tmp_path / S.PACKAGE
    resources = {"protocol": S.C.RESOURCE_PROTOCOL, "shards": [
        {"cell": c, "task": t, "datasets": list(d), "folds": [0, 1, 2, 3], "cpus": 8, "memory_gib": 30,
         "walltime_seconds": 7200 - 60 * n, "shard_name": f"{c}-{t}-01"} for n, (c, (t, d)) in enumerate(itertools.product(S.CELLS, S.TASKS.items()))]}
    for name, value in (("extraction_plan", new_extraction_plan()), ("pose_plan", new_pose_plan()), ("probe_plan", new_probe_plan()),
                        ("pose_budget", {"protocol": S.POSE_RESOURCE_PROTOCOL}), ("probe_resources", resources),
                        ("comparison_inputs", {"protocol": S.M.INPUT_PROTOCOL})):
        target = base / S.DOCUMENTS[name][1]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(S.dumps(value))
    return base


def test_submission_lines_use_the_common_submitter_the_frozen_code_and_the_published_pins(tmp_path):
    base, code = published(tmp_path), S.checkout("code")
    export = f"--export=ALL,CODE_DIR={code},EXPECTED_SHA={CODE}"
    seen = []
    for stage in (*STATIC, "probe-fit", "probe-aggregate"):
        table = S.jobs(stage, json.loads((base / S.DOCUMENTS["probe_resources"][1]).read_text()))
        lines = S.stage_lines(stage, CODE, tmp_path)
        assert len(lines) == len(table)
        for n, (job, line) in enumerate(zip(table, lines)):
            program, options, sbatch = parse(line)
            assert program == [S.PYTHON, S.MAC + "/scripts/submit_job.py"] and "--script-stdin" not in options
            chained = {"--dopo": table[n - 1]["id"]} if n and stage != "extractions" else {}
            assert options == {"--agente": "claude", "--campagna": "scaling-300m-20261104", "--timeout": "120", "--id": job["id"],
                               "--gpu-ore": "%g" % (job["seconds"] * job["gpus"] / 3600), **chained}
            seen.append(job["id"])
            if stage == "extractions":
                assert sbatch == [f"--time={S.clock(job['seconds'])}", export, code + "/scripts/slurm/scaling_evaluation19618.sbatch",
                                  f"{S.PACKAGE}/evaluation/extraction_plan.json", *job["flags"], job["out"]]
                continue
            name = "comparison_inputs" if stage == "compare" else stage.split("-")[0] + "_plan"
            where, relative = S.DOCUMENTS[name]
            assert sbatch[:9] == [f"--cpus-per-task={job['cpus']}", "--mem=30G", f"--time={S.clock(job['seconds'])}", export,
                                  code + "/scripts/slurm/scaling_readers19618.sbatch", stage,
                                  f"{S.checkout(where)}/{S.PACKAGE}/{relative}", sha(base / relative), job["out"]]
            tail = sbatch[9:]
            assert tail[:len(job["flags"])] == job["flags"]
            extra = tail[len(job["flags"]):]
            document = {"pose-ridge": ("pose_budget", "--pose-budget", "--pose-budget-sha256"),
                        "probe-fit": ("probe_resources", "--resource-plan", "--resource-sha256")}.get(stage)
            if document:
                assert extra == [document[1], f"{S.checkout('budgets')}/{S.PACKAGE}/{S.DOCUMENTS[document[0]][1]}", document[2],
                                 sha(base / S.DOCUMENTS[document[0]][1])]
            else:
                assert extra == []
    assert len(set(seen)) == len(seen) == 36
    gpu = [parse(line)[1]["--gpu-ore"] for line in S.stage_lines("extractions", CODE, tmp_path)]
    assert gpu == ["2.5", "2", "2.5"] and sum(map(float, gpu)) == 7.0
    ids = [j["id"] for j in S.jobs("probe-benchmark")]
    tranche = [parse(line)[1] for line in S.stage_lines("probe-benchmark", CODE, tmp_path, only=ids[3:6])]
    assert [(o["--id"], o.get("--dopo")) for o in tranche] == [(ids[3], None), (ids[4], ids[3]), (ids[5], ids[4])]
    assert S.stage_lines("probe-benchmark", CODE, tmp_path, only=[]) == []
    with pytest.raises(ValueError, match="own published lines"):
        S.stage_lines("training", CODE, ROOT)
    with pytest.raises(ValueError, match="full code SHA"):
        S.stage_lines("pose-validate", "abc123", tmp_path)
    with pytest.raises(FileNotFoundError):
        S.stage_lines("compare", CODE, tmp_path / "empty")


def test_checkout_lines_publish_a_deploy_reference_and_one_cpu_checkout():
    for name, campaign in S.CHECKOUTS.items():
        push, submit = S.checkout_lines(name, CODE)
        assert shlex.split(push) == ["git", "-C", S.MAC, "push", "leonardo", f"{CODE}:refs/heads/deploy/{campaign}"]
        program, options, sbatch = parse(submit)
        assert options["--id"] == f"SCALING-19618-{name}-checkout-20261106" and options["--gpu-ore"] == "0"
        assert options["--script-stdin"] == S.MAC + "/scripts/slurm/campaign_checkout.sbatch"
        assert sbatch == ["--cpus-per-task=1", "--mem=4G", "--time=00:05:00",
                          f"--export=ALL,CAMPAIGN={campaign},CAMPAIGN_SHA={CODE},EXPECTED_SHA={CODE}", "/dev/stdin"]
        assert "main" not in push and "--force" not in push and "+" not in push
    assert len(set(S.CHECKOUTS.values())) == 4 and all("19618" in c and c.endswith("_v1") for c in S.CHECKOUTS.values())
    assert not set(S.CHECKOUTS.values()) & {"scaling_300m_continuation19618_20261106_v1", "scaling_300m_evaluation_20261105_v1"}
    with pytest.raises(ValueError, match="full commit SHA"):
        S.checkout_lines("code", "deadbeef")


def test_documents_are_built_once_from_collected_evidence_on_disk(tmp_path, monkeypatch):
    """Command-line path of the first two builders, on a scratch root with collected evidence."""
    for relative in (S.INFRA + "/extraction_plan_5000_20261105.json", S.PACKAGE + "/training_plan.json",
                     S.INFRA + "/cpu_readers_20261105/pose_plan.json", S.INFRA + "/cpu_readers_20261105/probe_plan.json",
                     "results/infra/tokenizer_evaluation_20261103/pose_readout_plan_19618.json",
                     "results/infra/tokenizer_evaluation_20261103/probe_plan_19618.json",
                     *(f"{S.PACKAGE}/control19618_identity/{p.name}" for p in (PACKAGE / "control19618_identity").iterdir())):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    files, receipt, summary = training_files()
    texts = {files["receipt"]["path"]: body(receipt), files["summary"]["path"]: body(summary),
             files["config"]["path"]: b"{}\n", TRAINING["receipt_dir"] + "/launch.json": b"{}\n"}
    job = S.training_job(TRAINING)
    assert S.stage_table("training", tmp_path) == [job] and job["pinned"] == {"checkpoint_step0019618.pt": files["checkpoint"]["path"]}
    assert (job["gpus"], job["cpus"], job["seconds"], job["memory"]) == (4, 32, 53100, "480G")
    # The config hash of the receipt must be the one of the collected file.
    receipt["config_file_sha256"] = sha(b"{}\n")
    texts[files["receipt"]["path"]] = body(receipt)
    reading = reading_for([job], {job["id"]: "500"}, texts)
    for found in reading["files"]:
        if found["path"] == files["checkpoint"]["path"]:
            found.update(sha256=NEW_CHECKPOINT)
    manifest = S.collect("training", [job], {job["id"]: "500"}, reading, tmp_path / S.PACKAGE / "evidence" / "training")
    assert manifest["files"]["checkpoint_step0019618.pt"]["stored"] == "pin-only" and manifest["jobs"][job["id"]]["gpu_hours"] == 33 * 4 / 3600
    assert manifest["files"]["slurm_log.txt"]["path"] == S.LOGS + "/scaling_continue_500.out"
    S.build_extraction_plan(tmp_path)
    plan = tmp_path / S.PACKAGE / S.DOCUMENTS["extraction_plan"][1]
    assert json.loads(plan.read_text())["checkpoints"][ARM]["sha256"] == NEW_CHECKPOINT
    with pytest.raises(FileExistsError):
        S.build_extraction_plan(tmp_path)  # a published document is never rewritten

    extraction, (receipts, _, provenances), (pose_receipt, _, pose_provenances) = new_candidates(sha(plan))
    table = S.jobs("extractions")
    texts = {}
    for job in table:
        task = job["local"]
        for local, path in job["files"].items():
            key = local.replace(".npz.provenance.json", "")
            texts[path] = (body(pose_receipt if task == "pose" else receipts[task]) if local == "receipt.json" else
                           pose_provenances[{"train_subset": "train", "val": "val"}[key]] if task == "pose" else provenances[key])
    slurm = {job["id"]: str(600 + n) for n, job in enumerate(table)}
    S.collect("extractions", table, slurm, reading_for(table, slurm, texts), tmp_path / S.PACKAGE / "evidence" / "extractions")
    S.build_reader_plans(CODE, tmp_path)
    pose, probe = (json.loads((tmp_path / S.PACKAGE / S.DOCUMENTS[n][1]).read_text()) for n in ("pose_plan", "probe_plan"))
    assert pose == new_pose_plan() | {"caches": pose["caches"]} and probe["datasets"] == new_probe_plan()["datasets"]
    assert {p["receipt_path"] for p in pose["caches"][ARM].values()} == {S.EXTRACT_ROOT + "/pose/receipt.json"}
    assert {p["extraction_plan_sha256"] for p in probe["caches"][ARM].values()} == {sha(plan)}
    assert S.document("probe_plan", tmp_path)["path"].startswith(S.checkout("plans") + "/")
    lines = S.stage_lines("pose-validate", CODE, tmp_path)
    assert sha(tmp_path / S.PACKAGE / S.DOCUMENTS["pose_plan"][1]) in lines[0]
    with pytest.raises(ValueError, match="extraction receipt differs: code_sha"):
        S.build_reader_plans("d" * 40, tmp_path)
