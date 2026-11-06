#!/usr/bin/env python3
"""Stadi finali dello scaling 30M/300M a 5.000 passi: letture e piani operativi. Nessun calcolo scientifico, nessun invio.

    python -I final_stages.py state <etichetta>          stato reale su Leonardo -> live_state_<etichetta>.json
    python -I final_stages.py plan <live_state.json>     scrive aggregate_submission_plan.json e aggregate_submit_lines.txt
    python -I final_stages.py guard <piano.json>         prova su Leonardo i controlli sugli argomenti del dispatcher reale
    python -I final_stages.py submitter <piano.json>     passa gli argv dal submitter reale con ssh finto e registro temporaneo
    python -I final_stages.py check <piano.json> <live_state.json> [posti liberi richiesti]   cancello prima dell'invio

Le letture remote usano solo la libreria standard del login node (Python 3.6), non scrivono nulla e non leggono ne' stampano
score: dei report dei fit prendono identita', hash e riepilogo di convergenza. Gli invii veri non passano da questo file:
si fanno con `scripts/submit_job.py` e il ledger comune, una riga alla volta, fermandosi al primo esito diverso da `sottomesso`.
"""
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CAMPAIGN = "scaling-300m-20261104"
CODE_SHA = "aef665ba7ea66fc80291846487907bb3ef9107df"
WORK = "/leonardo_work/IscrB_WearUsFM"
CODE_DIR = WORK + "/wearusfm_campaigns/scaling_300m_evaluation_20261105_v1"
PROBE_PLAN = (WORK + "/wearusfm_campaigns/scaling_300m_cpu_plans_20261105_v1/results/infra/scaling_300m_20261104/"
              "cpu_readers_20261105/probe_plan.json")
PROBE_PLAN_SHA = "d2cc7da1cf08473067d96b52bff532463453ff5ce633814743b1a53dda5bbb63"
OUT_ROOT = WORK + "/wearusfm_runs/results/scaling_300m_20261105/cpu/probe_aggregates"
DISPATCHER = ROOT / "scripts/slurm/scaling_readers_recovery.sbatch"
DISPATCHER_SHA = "c2bd7979699b2d70150cdfad9d2a6eca657615551ba5073630417c8c46cf14d5"
SUBMITTER = ROOT / "scripts/submit_job.py"
SUBMITTER_SHA = "814da4bc66dbf18d69ee2ba332675fa0788ab25247d5f5460224cfd24d7e0d82"
FIT_PLAN = ROOT / "results/infra/scaling_300m_20261104/cpu_fit_release_20261105/submission_plan.json"
LEDGER = Path.home() / "wearusfm_local/jobs"
PYTHON = "/Users/simonebenatti/.venvs/wearusfm-tok/bin/python"
CELLS = ("T100-G-30M-natural", "T100-G-300M-natural", "T100-G-300M-matched")
TASKS = {"p1": ("ninapro_db2", "ninapro_db3", "ninapro_db6"), "p2": ("epn612", "uci_emg")}
CPUS, MEMORY_GIB, SECONDS, TIMEOUT_GRACE = 2, 30, 600, 180
SHARD_FILES = ["audit.jsonl", "audit_progress.jsonl", "shard.json", "shard.oof.npz"]
SSH = ["ssh", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
       "-o", "ProxyCommand=none", "-o", "ControlMaster=no", "-o", "ControlPath=none", "leonardo"]

REMOTE_STATE = r'''
import hashlib, json, os, subprocess, time
P = json.loads(__PAYLOAD__)
def pin(path):
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk); n += len(chunk)
    return {"path": path, "sha256": h.hexdigest(), "bytes": n, "canonical": os.path.realpath(path) == path}
def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                       env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
    return {"exit": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()[-300:]}
out = {"read_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": os.uname().nodename, "plan": pin(P["plan"]),
       "code": {"dir": P["code"], "canonical": os.path.realpath(P["code"]) == P["code"],
                "head": run(["git", "-C", P["code"], "rev-parse", "HEAD"]),
                "status": run(["git", "-C", P["code"], "status", "--porcelain", "--untracked-files=all"])}}
shards = []
for name, d in P["shards"]:
    s = json.load(open(d + "/shard.json"))
    a = s["audit_summary"]
    shards.append({"name": name, "dir": d, "files": sorted(os.listdir(d)),
                   "shard": pin(d + "/shard.json"), "oof": pin(d + "/shard.oof.npz"), "audit": pin(d + "/audit.jsonl"),
                   "report": dict((k, s.get(k)) for k in ("protocol", "kind", "complete", "cell", "arm", "readout", "task", "step",
                                  "code_sha", "plan_sha256", "numeric_sha256", "oof", "oof_sha256", "audit", "audit_sha256")),
                   "datasets": dict((ds, [fr["fold"] for fr in v["folds"]]) for ds, v in s["datasets"].items()),
                   "audit_summary": dict((k, a.get(k)) for k in ("fits", "convergence_warnings", "reached_limit", "converged_by_criterion")),
                   "unresolved_fits": len(a.get("unresolved_fits", []))})
out["shards"] = shards
out["outputs"] = [{"path": p, "exists": os.path.lexists(p)} for p in P["outputs"]]
out["sacct"] = run(["sacct", "-P", "-j", ",".join(P["jobs"]), "-o",
                    "JobIDRaw,JobName%40,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,MaxRSS,Start,End"])
out["squeue_serial"] = run(["squeue", "-u", os.environ.get("USER", ""), "-p", "lrd_all_serial", "-h", "-o", "%i|%j|%T|%M|%l|%C|%m|%r"])
print(json.dumps(out, indent=1, sort_keys=True))
'''

REMOTE_GUARD = r'''
import contextlib, io, json, os, pathlib, sys
P = json.loads(__PAYLOAD__)
shim = not hasattr(pathlib.PurePath, "is_relative_to")
if shim:  # Python 3.6 del login node: stessa definizione di pathlib 3.9
    def _is_relative_to(self, *other):
        try:
            self.relative_to(*other); return True
        except ValueError:
            return False
    pathlib.PurePath.is_relative_to = _is_relative_to
os.environ.update(P["env"])
guard = compile(P["guard"], "dispatcher_lines_%d_%d" % tuple(P["lines"]), "exec")
results = []
for case in P["cases"]:
    sys.argv = ["-"] + case["args"]
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            exec(guard, {"__name__": "dispatcher_guard"})
        got = "PASS"
    except SystemExit as e:
        got = "REFUSED: %s %s" % (e.code, err.getvalue().strip()[-200:])
    except Exception as e:
        got = "ERROR: %s: %s" % (type(e).__name__, e)
    results.append({"name": case["name"], "expect": case["expect"], "got": got})
print(json.dumps({"python": sys.version.split()[0], "is_relative_to_shim": shim, "host": os.uname().nodename, "results": results}, indent=1))
'''

FAKE_SSH = r'''
import hashlib, sys
from pathlib import Path
text, counter = sys.stdin.read(), Path(sys.argv[1])
if hashlib.sha256(text.encode()).hexdigest() != sys.argv[2] or not sys.argv[3].startswith("sbatch --parsable "):
    sys.exit("fake ssh: script o comando inattesi")
n = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(n))
print(90000000 + n)
'''


def require(ok, message):
    if not ok:
        raise SystemExit("RIFIUTO: " + message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as f:
        f.write(value if isinstance(value, str) else json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    print("scritto", Path(path).relative_to(ROOT), sha(path))


def fits():
    return [j for j in json.loads(FIT_PLAN.read_text())["jobs"] if j["stage"] == "probe-fit"]


def ledger(job_id):
    p = LEDGER / job_id / "state.json"
    return json.loads(p.read_text()) if p.exists() else None


def grid():
    return [(cell, task) for cell in CELLS for task in TASKS]


def out_of(cell, task):
    return "%s/%s-%s/aggregate.json" % (OUT_ROOT, cell, task)


def sidecars(out):
    return [out[:-len(".json")] + ".oof.npz", out[:-len(".json")] + ".audit.jsonl"]


def remote(template, payload):
    script = template.replace("__PAYLOAD__", repr(json.dumps(payload)))
    r = subprocess.run([*SSH, "python3 -"], input=script, capture_output=True, text=True, timeout=600)
    require(r.returncode == 0, "lettura remota fallita (%s): %s" % (r.returncode, r.stderr.strip()[-400:]))
    return json.loads(r.stdout), hashlib.sha256(script.encode()).hexdigest()


def cmd_state(label):
    released = ["SCALING-pose-validate-20261105", "SCALING-pose-ridge-20261105"] + [j["id"] for j in fits()]
    mine = ["SCALING-probe-aggregate-%02d-20261106" % n for n in range(1, 7)]
    states = {i: ledger(i) for i in released + mine}
    require(all(states[i] and states[i]["stato"] == "sottomesso" and states[i].get("job_slurm") for i in released), "ledger dei job rilasciati")
    outputs = [OUT_ROOT] + [p for cell, task in grid() for p in [out_of(cell, task)] + sidecars(out_of(cell, task))]
    payload = {"plan": PROBE_PLAN, "code": CODE_DIR, "shards": [[j["shard"], j["out"]] for j in fits()], "outputs": outputs,
               "jobs": [str(states[i]["job_slurm"]) for i in released]}
    value, script_sha = remote(REMOTE_STATE, payload)
    value["remote_script_sha256"] = script_sha
    value["local"] = {"read_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                      "repo_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                      "dispatcher_sha256": sha(DISPATCHER), "submitter_sha256": sha(SUBMITTER), "fit_plan_sha256": sha(FIT_PLAN)}
    value["ledger"] = {i: (None if s is None else {k: s.get(k) for k in ("stato", "job_slurm", "agente", "campagna")}) for i, s in states.items()}
    write_new(HERE / ("live_state_%s.json" % label), value)


def checks(state):
    """Controlli d'integrita' sullo stato letto: ogni voce e' (nome, esito). Nessuno score entra in questi controlli."""
    by_name = {j["shard"]: j for j in fits()}
    shards = state["shards"]
    rows = [r.split("|") for r in state["sacct"]["stdout"].splitlines()[1:]]
    main = [r for r in rows if "." not in r[0]]
    coverage = [(s["report"]["cell"], ds, f) for s in shards for ds, folds in s["datasets"].items() for f in folds]
    full = {(c, ds, f) for c in CELLS for t in TASKS for ds in TASKS[t] for f in range(4)}
    names = ["SCALING-probe-aggregate-%02d-20261106" % n for n in range(1, 7)]
    result = [
        ("piano delle sonde: hash e percorso canonico", state["plan"]["sha256"] == PROBE_PLAN_SHA and state["plan"]["canonical"]
         and state["plan"]["path"] == PROBE_PLAN),
        ("codice congelato: HEAD atteso, pulito, percorso canonico", state["code"]["head"] == {"exit": 0, "stdout": CODE_SHA, "stderr": ""}
         and state["code"]["status"] == {"exit": 0, "stdout": "", "stderr": ""} and state["code"]["canonical"]),
        ("quattordici shard, quelli del piano dei fit", sorted(s["name"] for s in shards) == sorted(by_name) and len(shards) == 14),
        ("file di ogni shard: i quattro attesi, nessun failed.json", all(s["files"] == SHARD_FILES for s in shards)),
        ("percorsi canonici di shard, OOF e audit", all(s[k]["canonical"] for s in shards for k in ("shard", "oof", "audit"))),
        ("report completi, stesso codice, piano e passo", all(
            s["report"]["complete"] is True and s["report"]["kind"] == "shard" and s["report"]["step"] == 5000
            and s["report"]["code_sha"] == CODE_SHA and s["report"]["plan_sha256"] == PROBE_PLAN_SHA for s in shards)),
        ("cella, compito, dataset e fold uguali al piano dei fit", all(
            s["report"]["cell"] == by_name[s["name"]]["cell"] and s["report"]["task"] == by_name[s["name"]]["task"]
            and s["dir"] == by_name[s["name"]]["out"] and sorted(s["datasets"]) == sorted(by_name[s["name"]]["datasets"])
            and all(folds == by_name[s["name"]]["folds"] for folds in s["datasets"].values()) for s in shards)),
        ("hash interni dei report uguali ai file", all(
            s["report"]["oof"] == s["oof"]["path"] and s["report"]["oof_sha256"] == s["oof"]["sha256"]
            and s["report"]["audit"] == s["audit"]["path"] and s["report"]["audit_sha256"] == s["audit"]["sha256"] for s in shards)),
        ("una sola firma numerica", len({s["report"]["numeric_sha256"] for s in shards}) == 1),
        ("copertura: ogni cella, dataset e fold una volta sola (60)", len(coverage) == 60 and set(coverage) == full),
        ("fit per shard: sette per dataset e fold; 420 in tutto", all(
            s["audit_summary"]["fits"] == 7 * sum(len(f) for f in s["datasets"].values()) for s in shards)
            and sum(s["audit_summary"]["fits"] for s in shards) == 420),
        ("contabilita': validate, ridge e quattordici fit COMPLETED 0:0", len(main) == 16 and all(r[2] == "COMPLETED" and r[3] == "0:0" for r in main)),
        ("uscite degli aggregati assenti", all(not o["exists"] for o in state["outputs"])),
        ("dispatcher e submitter locali con gli hash del rilascio dei fit", state["local"]["dispatcher_sha256"] == DISPATCHER_SHA
         and state["local"]["submitter_sha256"] == SUBMITTER_SHA),
        ("ID logici degli aggregati mai usati nel ledger", all(state["ledger"][n] is None for n in names)),
    ]
    facts = {"fits": sum(s["audit_summary"]["fits"] for s in shards),
             "convergence_warnings": sum(s["audit_summary"]["convergence_warnings"] for s in shards),
             "reached_limit": sum(s["audit_summary"]["reached_limit"] for s in shards),
             "unresolved_fits": sum(s["unresolved_fits"] for s in shards),
             "all_converged_by_criterion": all(s["audit_summary"]["converged_by_criterion"] is True for s in shards),
             "numeric_sha256": sorted({s["report"]["numeric_sha256"] for s in shards}),
             "fit_core_hours": sum(int(r[4]) * int(r[5]) for r in main if "probe-fit" in r[1]) / 3600,
             "serial_queue_jobs": len([x for x in state["squeue_serial"]["stdout"].splitlines() if x.strip()])}
    return result, facts


def cmd_plan(state_path):
    state_path = Path(state_path).resolve()
    state = json.loads(state_path.read_text())
    result, facts = checks(state)
    for name, ok in result:
        print("PASSA " if ok else "FALLISCE ", name)
    require(all(ok for _, ok in result), "stato non accettabile: nessun piano scritto")
    by_name = {j["shard"]: j for j in fits()}
    live = {s["name"]: s for s in state["shards"]}
    jobs = []
    for n, (cell, task) in enumerate(grid(), 1):
        names = sorted(k for k, j in by_name.items() if j["cell"] == cell and j["task"] == task)
        shards = [{"name": k, "fit_id": by_name[k]["id"], "datasets": by_name[k]["datasets"], "folds": by_name[k]["folds"],
                   **{f: live[k]["shard"][f] for f in ("path", "sha256", "bytes")}} for k in names]
        job_id, out = "SCALING-probe-aggregate-%02d-20261106" % n, out_of(cell, task)
        after = jobs[-1]["id"] if jobs else None
        argv = [PYTHON, str(SUBMITTER), "--agente", "claude", "--campagna", CAMPAIGN, "--timeout", "120", "--id", job_id, "--gpu-ore", "0",
                "--script-stdin", str(DISPATCHER), *(["--dopo", after] if after else []), "--",
                "--cpus-per-task=%d" % CPUS, "--mem=%dG" % MEMORY_GIB, "--time=00:10:00",
                "--export=ALL,CODE_DIR=%s,EXPECTED_SHA=%s" % (CODE_DIR, CODE_SHA), "/dev/stdin",
                "probe-aggregate", PROBE_PLAN, PROBE_PLAN_SHA, out, "--cell", cell, "--task", task, "--shards", *[s["path"] for s in shards]]
        jobs.append({"id": job_id, "stage": "probe-aggregate", "cell": cell, "task": task, "shards": shards, "out": out,
                     "sidecars": sidecars(out), "cpus": CPUS, "memory_gib": MEMORY_GIB, "walltime_seconds": SECONDS,
                     "requested_core_hours": CPUS * SECONDS / 3600, "depends_on": after, "argv": argv})
    plan = {"protocol": "scaling5000-aggregates-release-20261106-v1", "status": "PREPARED_NOT_SUBMITTED", "agente": "claude",
            "campagna": CAMPAIGN, "runtime_code_sha": CODE_SHA, "code_dir": CODE_DIR,
            "probe_plan": {f: state["plan"][f] for f in ("path", "sha256", "bytes")},
            "dispatcher": {"path": str(DISPATCHER.relative_to(ROOT)), "sha256": sha(DISPATCHER), "bytes": DISPATCHER.stat().st_size},
            "submitter": {"path": str(SUBMITTER.relative_to(ROOT)), "sha256": sha(SUBMITTER), "bytes": SUBMITTER.stat().st_size},
            "fit_layout": {"path": str(FIT_PLAN.relative_to(ROOT)), "sha256": sha(FIT_PLAN)},
            "live_state": {"path": str(state_path.relative_to(ROOT)), "sha256": sha(state_path), "read_at": state["read_at"]},
            "integrity_checks": [{"check": name, "passed": ok} for name, ok in result], "recorded_facts": facts,
            "resources": {"partition": "lrd_all_serial", "node": "login08", "cpus": CPUS, "memory_gib": MEMORY_GIB, "walltime_seconds": SECONDS},
            "requested_core_hours": sum(j["requested_core_hours"] for j in jobs),
            "maximum_charged_core_hours": len(jobs) * CPUS * (SECONDS + TIMEOUT_GRACE) / 3600,
            "order": "Cells in reader order, p1 then p2. One afterok chain: a refusal, failure or TIMEOUT stops the following jobs.",
            "submission_rule": ("Submit only after the operator handover is confirmed and this plan is reviewed. Before the first job: rerun "
                                "`state` and `check` (pins unchanged, outputs absent, code clean, logical IDs unused, six free places in "
                                "lrd_all_serial without taking places from jobs already waiting). Common submitter and ledger, one line at a "
                                "time, stop at the first outcome other than `sottomesso`. No retry, no background resubmission."),
            "after_aggregates": ("Collect and validate the six reports against the pins above before the comparison manifest exists. "
                                 "No afterok between aggregates and comparison."),
            "comparison_not_submitted": True, "jobs": jobs}
    write_new(HERE / "aggregate_submission_plan.json", plan)
    write_new(HERE / "aggregate_submit_lines.txt", "".join(" ".join(shlex.quote(x) for x in j["argv"]) + "\n" for j in jobs))


def dispatcher_args(job):
    return job["argv"][job["argv"].index("/dev/stdin") + 1:]


def cmd_guard(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    require(sha(DISPATCHER) == DISPATCHER_SHA, "dispatcher locale diverso")
    lines = DISPATCHER.read_text().splitlines()
    first, last = 37, 115
    require(lines[first - 2] == "python3 - \"$@\" <<'PY'" and lines[first - 1] == "import argparse"
            and lines[last].startswith("if a.stage == 'probe-benchmark':") and lines[last - 2].endswith("'canonical completed shard paths required')"),
            "le righe del controllo sugli argomenti non sono dove attese")
    cases = [{"name": j["id"], "expect": "PASS", "args": dispatcher_args(j)} for j in plan["jobs"]]
    base = dispatcher_args(plan["jobs"][2])  # 300M naturale p1: quattro shard
    def variant(name, expect, edit):
        args = list(base); edit(args); cases.append({"name": "controllo negativo: " + name, "expect": expect, "args": args})
    variant("hash del piano alterato", "input hash/size differs", lambda a: a.__setitem__(2, a[2][:-1] + ("0" if a[2][-1] != "0" else "1")))
    variant("uscita gia' esistente", "canonical new absolute output required", lambda a: a.__setitem__(3, os.path.dirname(base[-1])))
    variant("uscita dentro i checkout di campagna", "output inside code", lambda a: a.__setitem__(3, CODE_DIR + "/aggregate.json"))
    variant("shard ripetuto", "duplicate shard paths", lambda a: a.append(a[-1]))
    variant("shard inesistente", "No such file", lambda a: a.__setitem__(len(a) - 1, a[-1] + ".assente"))
    variant("opzione non ammessa per lo stadio", "stage requires exactly its whitelisted flags", lambda a: a.extend(["--folds", "0"]))
    variant("cella inesistente", "invalid choice", lambda a: a.__setitem__(a.index("--cell") + 1, "T100-G-30M-matched"))
    payload = {"guard": "\n".join(lines[first - 1:last]) + "\n", "lines": [first, last], "cases": cases,
               "env": {"CODE_DIR": CODE_DIR, "WORK": WORK, "EXPECTED_SHA": CODE_SHA}}
    value, script_sha = remote(REMOTE_GUARD, payload)
    for r in value["results"]:
        r["ok"] = r["got"] == "PASS" if r["expect"] == "PASS" else r["got"] != "PASS" and r["expect"] in r["got"]
        print("PASSA " if r["ok"] else "FALLISCE ", r["name"], "->", r["got"][:150])
    value.update({"dispatcher_sha256": DISPATCHER_SHA, "guard_lines": [first, last], "remote_script_sha256": script_sha,
                  "plan_sha256": sha(plan_path), "scope": ("Argument, pin, output and shard-path guards of the real dispatcher, executed read-only on "
                  "the login node. Scheduler allocation guards (CPUs, memory, TimeLimit, node) run only inside the job.")})
    require(all(r["ok"] for r in value["results"]), "il controllo sugli argomenti non ha dato gli esiti attesi")
    write_new(HERE / "aggregate_guard_dryrun.json", value)


def cmd_submitter(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    previews = []
    with tempfile.TemporaryDirectory() as tmp:
        fake, registry = Path(tmp) / "fake_ssh.py", Path(tmp) / "jobs"
        fake.write_text(FAKE_SSH)
        ssh = " ".join(shlex.quote(x) for x in [PYTHON, "-I", str(fake), str(Path(tmp) / "counter"), DISPATCHER_SHA])
        for n, job in enumerate(plan["jobs"], 1):
            argv = list(job["argv"])
            argv[2:2] = ["--registro", str(registry), "--ssh", ssh]
            r = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL)
            require(r.returncode == 0 and r.stdout.strip() == "%s: sottomesso, job %d" % (job["id"], 90000000 + n), "submitter: " + r.stdout + r.stderr)
            s = json.loads((registry / job["id"] / "state.json").read_text())
            dep = ["--dependency=afterok:%d" % (90000000 + n - 1)] if n > 1 else []
            expected = ["sbatch", "--parsable", "--job-name=" + job["id"], "--comment=wearusfm:" + job["id"], *dep,
                        *job["argv"][job["argv"].index("--") + 1:]]
            require(s["sbatch"] == expected and s["script_stdin_sha256"] == DISPATCHER_SHA and s["codice_sha"] == CODE_SHA
                    and s["agente"] == "claude" and s["campagna"] == CAMPAIGN and s["gpu_ore_stimate"] == 0, "comando di sbatch inatteso")
            previews.append({"id": job["id"], "sbatch": " ".join(shlex.quote(x) for x in s["sbatch"]).replace(
                "afterok:%d" % (90000000 + n - 1), "afterok:<job SLURM di %s>" % (job["depends_on"],))})
            print("PASSA ", job["id"])
        require(len((registry / "ledger.tsv").read_text().splitlines()) == 1 + 2 * len(plan["jobs"]), "ledger temporaneo inatteso")
    write_new(HERE / "aggregate_submitter_dryrun.json", {
        "scope": "Real scripts/submit_job.py on the six argv, with a fake ssh and a temporary registry: nothing reached Leonardo or the common ledger.",
        "submitter_sha256": sha(SUBMITTER), "plan_sha256": sha(plan_path), "sbatch_commands": previews})


def cmd_check(plan_path, state_path, places="6"):
    plan, state = json.loads(Path(plan_path).read_text()), json.loads(Path(state_path).read_text())
    result, facts = checks(state)
    live = {s["name"]: s["shard"] for s in state["shards"]}
    result += [("shard uguali a quelli fissati nel piano", all(
                   live[s["name"]]["sha256"] == s["sha256"] and live[s["name"]]["bytes"] == s["bytes"] and live[s["name"]]["path"] == s["path"]
                   for j in plan["jobs"] for s in j["shards"])),
               ("piano delle sonde uguale a quello del piano", state["plan"]["sha256"] == plan["probe_plan"]["sha256"]),
               ("posti liberi in lrd_all_serial: almeno %s su 10" % places, 10 - facts["serial_queue_jobs"] >= int(places))]
    for name, ok in result:
        print("PASSA " if ok else "FALLISCE ", name)
    print("job dell'utente in coda seriale:", facts["serial_queue_jobs"], "| letto alle", state["read_at"])
    require(all(ok for _, ok in result), "cancello prima dell'invio non superato")
    print("CANCELLO SUPERATO")


if __name__ == "__main__":
    command = {"state": cmd_state, "plan": cmd_plan, "guard": cmd_guard, "submitter": cmd_submitter, "check": cmd_check}.get(sys.argv[1] if len(sys.argv) > 1 else "")
    require(command is not None, __doc__)
    command(*sys.argv[2:])
