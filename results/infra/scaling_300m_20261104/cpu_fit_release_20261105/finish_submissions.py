#!/usr/bin/env python3
"""Complete only the five remaining published fits; bounded eight hours, no retry.

Operational acceptance hashes metadata and large outputs without downloading OOF.
Scientific fold/score verification remains in the frozen aggregation/comparator.
--start launches one independent local process; no AI polling, service or git.
--run is its bounded foreground loop. --status reads only local process evidence.
"""
import base64
import csv
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
SUB = BASE / "submissions"
REGISTRY = Path.home() / "wearusfm_local/jobs"
PLAN_SHA = "0370cdf8a819d5e6e6c39731725b8754d1c884f3054f24875010f895bf35d505"
CODE = "aef665ba7ea66fc80291846487907bb3ef9107df"
NUMERIC = "1ac8481eb8b53fbdd423404a927cca02e908197f16ed6494268b56e5ddc24b1c"
INITIAL = [63626871, 63626931, 63626956, 63626966, 63626971, 63626975,
           63626978, 63627028, 63627047, 63627052]
ACTIVE = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING"}
MAX_WAIT = 8 * 3600
LOGROOT = "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def jhash(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def frozen():
    require(sha((BASE / "submission_plan.json").read_bytes()) == PLAN_SHA, "publication changed")
    p = read(BASE / "submission_plan.json")
    for key in ("dispatcher", "submitter", "probe_resources", "pose_budget", "accepted_benchmarks"):
        pin = p[key]
        data = (ROOT / pin["path"]).read_bytes()
        require(sha(data) == pin["sha256"] and len(data) == pin["bytes"], "changed pin: " + key)
    require(len(p["jobs"]) == 15 and p["runtime_code_sha"] == CODE, "unexpected campaign")
    require([j["id"] for j in p["jobs"][1:]] == ["SCALING-probe-fit-%02d-20261105" % i for i in range(1, 15)], "unexpected fit IDs")
    return p


def receipts(plan):
    ledger = list(csv.DictReader((REGISTRY / "ledger.tsv").open(), delimiter="\t"))
    found = {}
    for index, job in enumerate(plan["jobs"]):
        folder, local = REGISTRY / job["id"], SUB / (job["id"] + ".json")
        rows = [r for r in ledger if r["id"] == job["id"]]
        if not folder.exists():
            require(index >= 10 and not local.exists() and not rows, "missing/ambiguous receipt " + job["id"])
            continue
        require((folder / "state.json").is_file(), "reservation without final state " + job["id"])
        state = read(folder / "state.json")
        expected = ["sbatch", "--parsable", "--job-name=" + job["id"], "--comment=wearusfm:" + job["id"]]
        if job["depends_on"]:
            require(job["depends_on"] in found, "noncontiguous submissions")
            expected += ["--dependency=afterok:" + str(found[job["depends_on"]]["job_slurm"])]
        expected += job["argv"][job["argv"].index("--") + 1:]
        require(state.get("stato") == "sottomesso" and state.get("uscita") == 0 and state.get("sbatch") == expected and
                state.get("id") == job["id"] and state.get("codice_sha") == CODE and state.get("agente") == "gpt" and
                state.get("campagna") == "scaling-300m-20261104" and state.get("gpu_ore_stimate") == 0 and
                state.get("script_stdin_sha256") == plan["dispatcher"]["sha256"], "receipt differs " + job["id"])
        require(type(state.get("job_slurm")) is int and state["job_slurm"] > 0, "invalid job ID")
        require(index >= 10 or state["job_slurm"] == INITIAL[index], "initial job changed")
        require(rows and rows[-1]["stato"] == "sottomesso" and rows[-1]["job_slurm"] == str(state["job_slurm"]) and
                rows[-1]["codice_sha"] == CODE and shlex.split(rows[-1]["comando"]) == expected, "ledger mismatch")
        if local.exists():
            require(read(local) == state, "local receipt differs")
        else:
            require(index >= 10, "missing initial published receipt")
            write(local, state)
        found[job["id"]] = state
    require(len({s["job_slurm"] for s in found.values()}) == len(found), "duplicate job ID")
    return found


REMOTE = r'''
import base64,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
p=json.loads(sys.argv[1])
def command(argv):
 r=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=60)
 if r.returncode: raise RuntimeError(r.stderr)
 return r.stdout
def artifact(path,content):
 f=Path(path)
 if f.is_symlink() or not f.is_file(): raise RuntimeError('missing/symlink: '+path)
 before=f.stat(); h=hashlib.sha256(); chunks=[]
 if content and before.st_size>16*2**20: raise RuntimeError('oversized metadata: '+path)
 with f.open('rb') as stream:
  while True:
   block=stream.read(2**20)
   if not block: break
   h.update(block)
   if content: chunks.append(block)
 after=f.stat()
 if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns): raise RuntimeError('changed artifact: '+path)
 r={'path':path,'sha256':h.hexdigest(),'bytes':after.st_size}
 if content:r['base64']=base64.b64encode(b''.join(chunks)).decode()
 return r
if p['mode']=='status':
 result={'sacct':command(['sacct','-P','-S','2026-11-05','-u','sbenatti','--name='+','.join(p['names']), '--format=JobIDRaw,JobName%80,State%32,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,AllocTRES,NodeList,Partition,Start,End,Timelimit']),
 'squeue':command(['squeue','-r','-h','-u','sbenatti','-o','%i|%j|%T|%P|%C|%r|%q']),
 'existing_outputs':[x for x in p['unsubmitted_outputs'] if os.path.lexists(x)]}
else:
 if os.path.lexists(p['out']+'/failed.json'):raise RuntimeError('failed.json exists')
 result={'files':[artifact(x,True) for x in p['text']], 'outputs':[artifact(x,False) for x in p['outputs']]}
result['utc']=datetime.datetime.utcnow().isoformat()+'Z'
print(json.dumps(result,allow_nan=False))
'''


def remote(payload):
    command = "python3 -c " + shlex.quote(REMOTE) + " " + shlex.quote(json.dumps(payload))
    result = subprocess.run(["ssh", "leonardo", command], shell=False, capture_output=True, text=True, timeout=180)
    require(result.returncode == 0, "remote read failed: " + result.stderr[-2000:])
    return json.loads(result.stdout)


def status(plan, found, snapshot):
    require(not snapshot["existing_outputs"], "unsubmitted output already exists")
    rows = list(csv.DictReader(io.StringIO(snapshot["sacct"]), delimiter="|"))
    parents = {r["JobIDRaw"]: r for r in rows if r["JobIDRaw"].isdigit()}
    require(len(parents) == sum(r["JobIDRaw"].isdigit() for r in rows), "duplicate sacct parent")
    queue = [line.strip().split("|") for line in snapshot["squeue"].splitlines() if line.strip()]
    require(all(len(q) == 7 for q in queue), "malformed squeue")
    require(len({q[0] for q in queue}) == len(queue), "duplicate queue rows")
    serial = [q for q in queue if q[3] == "lrd_all_serial" or q[6] == "lrd_all_serial"]
    require(len(serial) <= 10 and sum(int(q[4]) for q in serial if q[2] in {"RUNNING", "COMPLETING", "CONFIGURING"}) <= 8, "QoS cap already violated")
    completed = []
    for job in plan["jobs"]:
        candidates = {r["JobIDRaw"] for r in parents.values() if r["JobName"] == job["id"]}
        candidates |= {q[0] for q in queue if q[1] == job["id"]}
        if job["id"] not in found:
            require(not candidates, "unreceipted remote job")
            continue
        jid = str(found[job["id"]]["job_slurm"])
        require(candidates == {jid}, "missing/duplicate remote identity " + job["id"])
        row = parents.get(jid)
        q = [q for q in queue if q[0] == jid]
        if row:
            state = row["State"].split()[0].rstrip("+")
            require(state in ACTIVE | {"COMPLETED"}, "terminal/unknown state " + jid + ": " + state)
            require(row["ExitCode"] in ("0:0", ""), "nonzero exit " + jid)
            if state == "COMPLETED":
                require(row["ExitCode"] == "0:0" and not q, "ambiguous completion " + jid)
                completed.append(job)
        if q:
            require(q[0][2] in ACTIVE and "DependencyNeverSatisfied" not in q[0][5], "bad queue state " + jid)
        else:
            require(row and row["State"] == "COMPLETED", "active job absent from queue " + jid)
    return completed, queue, rows


def summary(records):
    require(all(r.get("exception") is None for r in records), "failed fit in completed audit")
    for r in records:
        n = r["n_iter"]
        warnings = sum(w["category"] == "ConvergenceWarning" for w in r["warnings"])
        require(type(n) is int and 0 < n <= 12000 and r["reached_limit"] == (n >= 12000) and
                r["convergence_warnings"] == warnings and r["converged_by_criterion"] == (n < 12000 and warnings == 0), "inconsistent convergence audit")
    return {"fits": len(records), "convergence_warnings": sum(r["convergence_warnings"] for r in records),
            "reached_limit": sum(r["reached_limit"] for r in records),
            "converged_by_criterion": all(r["converged_by_criterion"] for r in records),
            "unresolved_fits": [{k: r[k] for k in ("arm", "dataset", "fold", "fit_phase", "C", "n_iter")} for r in records if not r["converged_by_criterion"]]}


def validate_report(job, report, files, outputs, plan, jid):
    stage_index = job["argv"].index("/dev/stdin")
    scientific = ROOT / "results/infra/scaling_300m_20261104/cpu_readers_20261105" / ("probe_plan.json" if job["stage"] == "probe-fit" else "pose_plan.json")
    data = scientific.read_bytes()
    require(sha(data) == job["argv"][stage_index + 3], "scientific plan changed")
    p = json.loads(data)
    require(report.get("complete") is True and report.get("step") == 5000 and report.get("recipe") == p["recipe"] and report.get("plan_sha256") == sha(data), "report identity differs")
    if job["stage"] == "pose-ridge":
        baseline = read(scientific.parent / "pose_benchmark/T100-G-30M-natural/report.json")
        require(report.get("protocol") == "scaling-pose-readout-20261104-v1" and report.get("status") == "completed" and
                report.get("runtime") == baseline["runtime"] and report.get("slurm_job_id") == str(jid), "pose runtime/status differs")
        for key, source in (("input_pins", "caches"), ("reference_pins", "references"), ("projection_pin", "projection"), ("training_source_contract", "training_source_contract")):
            require(report.get(key) == p[source], "pose provenance differs " + key)
        require(set(report["cells"]) == set(p["recipe"]["cells"]), "pose cells differ")
        for name, (arm, view) in p["recipe"]["cells"].items():
            cell = report["cells"][name]
            require(cell.get("complete") is True and cell["arm"] == arm and cell["view"] == view and cell["which"] == "teacher" and cell["checkpoint_step"] == 5000 and
                    cell["checkpoint_sha256"] == p["caches"][arm]["train"]["checkpoint_sha256"] and not cell["fit"]["selection"]["boundary_unresolved"], "pose cell differs")
            for key, suffix in (("coefficients", ".coefficients.npz"), ("predictions", ".predictions.npy")):
                path = job["out"] + "/" + name + suffix
                require(cell[key] == {"path": path, "sha256": outputs[path]["sha256"]}, "pose output hash differs")
        return {"scope": "pose metadata and artifact hashes; scientific comparator still required"}
    cell_recipe = p["recipe"]["cells"][job["cell"]]
    arm, view = cell_recipe["arm"], cell_recipe["readout"]
    require(report.get("protocol") == "scaling-probe-20261104-v1" and report.get("kind") == "shard" and report.get("cell") == job["cell"] and
            report.get("arm") == arm and report.get("readout") == view and report.get("task") == job["task"] and report.get("code_sha") == CODE and
            report.get("plan_content_sha256") == jhash(p) and report.get("projection_sha256") == p["projection"]["sha256"] and
            report.get("numeric_sha256") == NUMERIC == jhash(report["numeric_context"]), "probe runtime/provenance differs")
    budget = read(ROOT / plan["probe_resources"]["path"])
    allocation = [a for a in budget["shards"] if all(a[k] == job[k] for k in ("cell", "task", "datasets", "folds"))]
    rv = report["resource_verification"]
    require(len(allocation) == 1 and rv["allocation"] == allocation[0] and rv["resource_plan"]["sha256"] == plan["probe_resources"]["sha256"] and
            0 < rv["estimated_seconds_with_margin"] <= job["walltime_seconds"], "resource verification differs")
    auditpath, oofpath = job["out"] + "/audit.jsonl", job["out"] + "/shard.oof.npz"
    require(report["audit"] == auditpath and report["audit_sha256"] == sha(files[auditpath]) and report["oof"] == oofpath and report["oof_sha256"] == outputs[oofpath]["sha256"], "probe output hashes differ")
    records = [json.loads(line) for line in files[auditpath].decode().splitlines()]
    require(set(report["datasets"]) == set(job["datasets"]) and len(records) == 7 * len(job["datasets"]) * len(job["folds"]), "dataset/fit coverage differs")
    for ds in job["datasets"]:
        folds = report["datasets"][ds]["folds"]
        require(sorted(f["fold"] for f in folds) == job["folds"], "fold coverage differs")
        for fold in job["folds"]:
            fits = [r for r in records if r["dataset"] == ds and r["fold"] == fold]
            require(len(fits) == 7 and sorted(r["C"] for r in fits if r["fit_phase"] == "candidate") == [.001, .01, .1, 1., 10., 100.] and sum(r["fit_phase"] == "refit" for r in fits) == 1, "candidate/refit coverage differs")
            for r in fits:
                require(all(r[k] == v for k, v in {"cell": job["cell"], "arm": arm, "readout": view, "task": job["task"], "code_sha": CODE, "plan_sha256": sha(data), "projection_sha256": p["projection"]["sha256"], "cache_sha256": p["caches"][arm][ds]["sha256"], "checkpoint_sha256": p["caches"][arm][ds]["checkpoint_sha256"], "max_iter": 12000}.items()), "audit identity differs")
    audited = summary(records)
    require(audited == report["audit_summary"], "audit summary differs")
    return {"scope": "metadata/audit/hash only; full OOF fold and score checks remain in frozen aggregation", "solver_unresolved": not audited["converged_by_criterion"]}


def slurm_seconds(value):
    days, sep, value = value.partition("-") if "-" in value else ("0", "", value)
    fields = [int(v) for v in value.split(":")]
    require(len(fields) in (2, 3), "invalid scheduler duration")
    return int(days) * 86400 + sum(v * 60 ** i for i, v in enumerate(reversed(fields)))


def byte_count(value):
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]?)", value)
    require(match is not None, "invalid scheduler memory")
    return float(match[1]) * 1024 ** ("KMGT".index(match[2]) + 1 if match[2] else 0)


def account(job, jid, rows, raw):
    own = [r for r in rows if r["JobIDRaw"] == str(jid) or r["JobIDRaw"].startswith(str(jid) + ".")]
    parent = next(r for r in own if r["JobIDRaw"] == str(jid))
    tres = dict(pair.split("=", 1) for pair in parent["AllocTRES"].split(",") if "=" in pair)
    require(all(r["State"] == "COMPLETED" and r["ExitCode"] == "0:0" for r in own), "unsuccessful accounting step")
    require(int(parent["AllocCPUS"]) == 8 and parent["Partition"] == "lrd_all_serial" and parent["NodeList"] == "login08" and
            int(parent["ElapsedRaw"]) > 0 and slurm_seconds(parent["Timelimit"]) == job["walltime_seconds"] and
            tres.get("cpu") == "8" and byte_count(tres.get("mem", "")) == 30 * 2**30 and
            not any("gpu" in k and float(v) != 0 for k, v in tres.items()), "accounting allocation differs")
    batch = [r for r in own if r["JobIDRaw"] == str(jid) + ".batch"]
    require(len(batch) == 1 and 0 < byte_count(batch[0]["MaxRSS"]) <= 30 * 2**30, "batch memory missing/over cap")
    return {"job_slurm": jid, "state": parent["State"], "exit_code": parent["ExitCode"], "elapsed_seconds": int(parent["ElapsedRaw"]), "allocated_cpus": 8,
            "core_hours": int(parent["ElapsedRaw"]) * 8 / 3600, "requested_seconds": job["walltime_seconds"], "peak_rss_bytes": byte_count(batch[0]["MaxRSS"]),
            "sacct_raw_sha256": sha(raw.encode()), "rows": own}


def envelope(plan):
    # Conservative published allowance for all work preceding these 15 jobs:
    # 22.027778 actual at amendment + 18.133333 recovery cap (actual was lower).
    previous_allowance = 22.02777777777778 + 18.133333333333333
    remaining_operations = 2.6 + 4.4 + .5333333333333333
    jobs = 0.
    for job in plan["jobs"]:
        folder = SUB / "collected_jobs" / job["id"]
        jobs += read(folder / "accounting.json")["core_hours"] if (folder / "operational_acceptance.json").exists() else job["requested_core_hours"] + .4
    total = previous_allowance + remaining_operations + jobs
    require(total <= 310, "actual completed + outstanding caps exceed 310 core-hours")
    write(SUB / "finish_envelope.json", {"previous_conservative_allowance": previous_allowance, "remaining_operations_cap": remaining_operations,
          "completed_actual_plus_outstanding_job_caps": jobs, "total_core_hours": total, "ceiling_core_hours": 310})


def collect(job, jid, snapshot, rows, plan):
    folder = SUB / "collected_jobs" / job["id"]
    acceptance = folder / "operational_acceptance.json"
    if acceptance.exists():
        accepted = read(acceptance)
        require(accepted["plan_sha256"] == PLAN_SHA and accepted["job_slurm"] == jid, "acceptance identity changed")
        for path, digest in accepted["local_hashes"].items():
            require(sha((folder / path).read_bytes()) == digest, "collected file changed")
        return
    isprobe = job["stage"] == "probe-fit"
    textpaths = [job["out"] + ("/shard.json" if isprobe else "/report.json"), LOGROOT + "/scaling_readers_recovery_" + str(jid) + ".out"]
    if isprobe:
        textpaths += [job["out"] + "/audit.jsonl"]
        outputs = [job["out"] + "/shard.oof.npz"]
    else:
        outputs = [job["out"] + "/" + cell + ext for cell in ("T100-G-30M-natural", "T100-G-300M-natural", "T100-G-300M-matched") for ext in (".coefficients.npz", ".predictions.npy")]
    bundle = remote({"mode": "collect", "out": job["out"], "text": textpaths, "outputs": outputs})
    files = {r["path"]: base64.b64decode(r["base64"], validate=True) for r in bundle["files"]}
    outputpins = {r["path"]: r for r in bundle["outputs"]}
    require(set(files) == set(textpaths) and set(outputpins) == set(outputs), "collection paths differ")
    for item in bundle["files"]:
        require(sha(files[item["path"]]) == item["sha256"] and len(files[item["path"]]) == item["bytes"], "transport hash differs")
    # Preserve raw evidence even when the following operational gate rejects it.
    folder.mkdir(parents=True, exist_ok=True)
    for path, data in files.items():
        (folder / Path(path).name).write_bytes(data)
    (folder / "sacct.txt").write_text(snapshot["sacct"])
    write(folder / "remote_artifacts.json", {"utc": bundle["utc"], "outputs": bundle["outputs"], "text": [{k: v for k, v in r.items() if k != "base64"} for r in bundle["files"]]})
    report = json.loads(files[textpaths[0]])
    verdict = validate_report(job, report, files, outputpins, plan, jid)
    expectedlog = {"complete": True, "kind": "shard", "out": job["out"]} if isprobe else {"status": "completed", "out": job["out"]}
    logjson = []
    for line in files[textpaths[1]].decode().splitlines():
        try: logjson.append(json.loads(line))
        except ValueError: pass
    require(expectedlog in logjson, "completion absent from log")
    accounting = account(job, jid, rows, snapshot["sacct"])
    write(folder / "accounting.json", accounting)
    localhash = {f.name: sha(f.read_bytes()) for f in folder.iterdir() if f.is_file()}
    write(acceptance, {"plan_sha256": PLAN_SHA, "job_slurm": jid, "local_hashes": localhash, **verdict})
    print("Accepted operational evidence " + job["id"] + ("; solver_unresolved (no retry)" if verdict.get("solver_unresolved") else ""), flush=True)


def finish():
    started = time.monotonic()
    while time.monotonic() - started < MAX_WAIT:
        plan = frozen()
        found = receipts(plan)
        if len(found) == 15:
            return {"status": "all_15_submitted", "job_ids": {k: v["job_slurm"] for k, v in found.items()}}
        pending = [j for j in plan["jobs"] if j["id"] not in found]
        snapshot = remote({"mode": "status", "names": [j["id"] for j in plan["jobs"]], "unsubmitted_outputs": [j["out"] for j in pending]})
        completed, queue, rows = status(plan, found, snapshot)
        write(SUB / "finish_last_status.json", snapshot)
        for job in completed:
            collect(job, found[job["id"]]["job_slurm"], snapshot, rows, plan)
        serial = [q for q in queue if q[3] == "lrd_all_serial" or q[6] == "lrd_all_serial"]
        if len(serial) >= 10:
            time.sleep(min(60, max(0, MAX_WAIT - (time.monotonic() - started))))
            continue
        job = pending[0]
        require(job in plan["jobs"][10:], "attempt to submit outside remaining five")
        parent = str(found[job["depends_on"]]["job_slurm"])
        require(any(q[0] == parent and q[2] in ACTIVE for q in queue), "predecessor not queued; immutable dependency cannot be changed")
        require(frozen() == plan and receipts(plan) == found, "local state changed before submission")
        envelope(plan)
        require(time.monotonic() - started < MAX_WAIT, "eight-hour deadline reached before submission")
        if (SUB / "finish_attempt.json").exists():
            require(read(SUB / "finish_attempt.json")["id"] != job["id"], "previous attempt unresolved; no retry")
        write(SUB / "finish_attempt.json", {"id": job["id"], "argv": job["argv"], "plan_sha256": PLAN_SHA, "utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        # Exactly one attempt. Timeout/nonzero/ambiguous receipt aborts; never retry.
        result = subprocess.run(job["argv"], shell=False, capture_output=True, text=True, timeout=90, cwd=ROOT)
        write(SUB / (job["id"] + ".submit_output.json"), {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        require(result.returncode == 0, "submitter refused/failed " + job["id"] + ": " + result.stderr[-1500:])
        new = receipts(plan)
        require(set(new) == set(found) | {job["id"]}, "submission receipt ambiguous")
        print("Submitted " + job["id"] + " = " + str(new[job["id"]]["job_slurm"]), flush=True)
    raise RuntimeError("bounded eight-hour wait exhausted; no further submission")


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in ("--start", "--run", "--status"), "use --start, --run or --status; no overrides")
    SUB.mkdir(parents=True, exist_ok=True)
    startup = SUB / "finish_startup.json"
    if sys.argv[1] == "--status":
        state = {"startup": read(startup) if startup.exists() else None, "result": read(SUB / "finish_result.json") if (SUB / "finish_result.json").exists() else None}
        state["pid_present"] = False
        if state["startup"] and state["startup"].get("pid"):
            try:
                os.kill(state["startup"]["pid"], 0)
                state["pid_present"] = True
            except ProcessLookupError:
                pass
        print(json.dumps(state, indent=2), flush=True)
        return
    if sys.argv[1] == "--start":
        with (SUB / "finish_start.lock").open("a") as startlock:
            fcntl.flock(startlock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(not startup.exists(), "already started; inspect --status, no automatic restart")
            frozen()
            command = [sys.executable, str(Path(__file__).resolve()), "--run"]
            result = {"pid": None, "argv": command, "script_sha256": sha(Path(__file__).read_bytes()), "plan_sha256": PLAN_SHA,
                      "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "max_wait_seconds": MAX_WAIT, "no_automatic_restart": True}
            # Reserve before spawning: a crash cannot make a second start legal.
            write(startup, result)
            with (SUB / "finish_submissions.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(command, shell=False, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
            result["pid"] = child.pid
            write(startup, result)
            print(json.dumps(result), flush=True)
        return
    require(startup.exists() and read(startup)["pid"] in (None, os.getpid()), "--run is reserved to the single started child")
    with (SUB / "finish_submissions.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = finish()
        except BaseException as exc:
            write(SUB / "finish_result.json", {"status": "STOP", "error": type(exc).__name__ + ": " + str(exc), "no_automatic_retry": True})
            raise
        write(SUB / "finish_result.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
