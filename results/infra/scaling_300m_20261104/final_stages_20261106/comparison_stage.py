#!/usr/bin/env python3
"""Scaling 30M/300M a 5.000 passi, dopo i sei aggregati: raccolta, validazione, manifest e piano del confronto. Nessun invio.

    python -I comparison_stage.py collect <etichetta>        legge i sei aggregati su Leonardo -> aggregates_state_<etichetta>.json
                                                             e copia report, audit e log in collected_aggregates_<etichetta>/<ID>/
    python -I comparison_stage.py validate <stato.json>      controlli B della Review 10 (d3976c3, sezione 5) sullo stato raccolto
    python -I comparison_stage.py manifest <stato.json>      scrive comparison_inputs.json e comparison_plan.json
    python -I -B comparison_stage.py frozen <manifest.json>  con l'ambiente del progetto: il manifest passa dal codice congelato
    python -I comparison_stage.py lines <piano.json> <commit>          scrive comparison_submit_lines.txt per il commit del manifest
    python -I comparison_stage.py submitter <piano.json> <righe.txt>   passa i due argv dal submitter reale con ssh finto
    python -I comparison_stage.py pre-deploy <piano.json> <righe.txt> <etichetta>      lettura prima del push del riferimento di deploy
    python -I comparison_stage.py gate-checkout <piano.json> <righe.txt> <etichetta>   cancello prima del checkout del manifest
    python -I comparison_stage.py gate-compare <piano.json> <righe.txt> <etichetta>    cancello prima del confronto

Nessun invio e nessuna scrittura su Leonardo: gli invii veri passano da `scripts/submit_job.py` e dal ledger comune. Le letture
remote usano solo la libreria standard del login node (Python 3.6) e non stampano score: dei report prendono identita', hash,
conteggi e stato del solutore. I report raccolti si copiano byte per byte e si verificano con lo sha256, senza aprirne i valori.
"""
import base64
import datetime
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_stages as F  # noqa: E402

ROOT, WORK, CODE_DIR, CODE_SHA, CAMPAIGN = F.ROOT, F.WORK, F.CODE_DIR, F.CODE_SHA, F.CAMPAIGN
require, sha, write_new = F.require, F.sha, F.write_new
PLAN = HERE / "aggregate_submission_plan.json"
PLAN_SHA = "abc489613dfbfddc5e33e2150854245bcd3b69292806953bafa622b91a22941c"
READERS = WORK + "/wearusfm_campaigns/%s/results/infra/scaling_300m_20261104/cpu_readers_20261105/%s"
POSE_PLAN = READERS % ("scaling_300m_cpu_plans_20261105_v1", "pose_plan.json")
POSE_PLAN_SHA = "a500cc3c56bde3938a2aa384bf6a942fd6eaf02be0ebd5935ac251f732cb2198"
POSE_BUDGET = READERS % ("scaling_300m_cpu_budgets_20261105_v1", "pose_budget.json")
POSE_BUDGET_SHA = "342d1576ef392ea0a69c9a9bc9361d89d7daff40cfb180cc4d33f0dd1ae691e0"
POSE_REPORT = WORK + "/wearusfm_runs/results/scaling_300m_20261105/cpu/pose_ridge/report.json"
POSE_REPORT_SHA = "8f70ff3b58c8f54413f6c8a89a861a4fdb7e6c9d934a65e73a11d4172a47e4ff"
NUMERIC_SHA = "1ac8481eb8b53fbdd423404a927cca02e908197f16ed6494268b56e5ddc24b1c"
PROBE_PROTOCOL, POSE_PROTOCOL = "scaling-probe-20261104-v1", "scaling-pose-readout-20261104-v1"
INPUT_PROTOCOL = "scaling-comparison-inputs-20261104-v1"
# Review 10 (d3976c3), sezione 5 C.2: i sei sorgenti che il comparatore congelato fissa nel manifest, nell'ordine di source_hashes().
SOURCES = {"scripts/scaling_probe.py": "0b3d40994ea7b6ee189a5831e22a5b298cc0cf2ceaa24908c80242642b5d26b0",
           "scripts/scaling_pose_readout.py": "238a5cd479e37310235c42dcb0b930d232230ff0d2a8d1d1309b05b7292900cf",
           "scripts/scaling_projection.py": "a8c0f7880095df02186e465a1ce86030b8b2fe5ee1beb8a965b349ff205ee88d",
           "scripts/scaling_train.py": "d641c375cbeeb80bf275afef185dfb4f695938b22d908626cb9b829ddc8440ac",
           "scripts/scaling_reading.py": "13231e82664cfa6a815a8e87fb6c85380b70838d8746109d8af627b5a4f056d5",
           "scripts/scaling_compare.py": "a3610519b453a3d37747f5a4ce926b2164a40dd503546b22b6c45829085eb938"}
# Review 10, sezioni 3 e 5: identita' degli OOF per dataset (impronta, finestre, utenti), uguale nelle tre celle.
ANCHORS = {"epn612": ("a828cda57ae0fef9c8145834a74235c379945f202f613d03cab687e16cfa8e35", 47100, 314),
           "ninapro_db2": ("b8d6e69fde0fa57dcc7623c7355c5b119f3526f418ba3f4a3b670822358b4367", 38544, 32),
           "ninapro_db3": ("056d02fd70736c75d54ec2610e541da808368f154f1ea9cf359ea27bc1cd41d6", 9834, 8),
           "ninapro_db6": ("3ee565080f079dfbecba717ef0c8238e2772efd41768c87548c973b4100b8f8e", 33961, 8),
           "uci_emg": ("132e4e8dbed63a9340585f235c8b68375364da0a0310827249446bddd85bd3fd", 364, 19)}
LOG_DIR = WORK + "/wearusfm_runs/logs"
EXCLUSIVE_FOLDERS = True  # ogni aggregato in una cartella sua, con i soli tre file
N_SHARDS = 14
CAMPAIGN_NAME = "scaling_300m_comparison_inputs_20261106_v1"
CAMPAIGN_DIR = WORK + "/wearusfm_campaigns/" + CAMPAIGN_NAME
MANIFEST = HERE / "comparison_inputs.json"
COMPARE_OUT = WORK + "/wearusfm_runs/results/scaling_300m_20261105/cpu/comparison/report.json"
CHECKOUT_SCRIPT = ROOT / "scripts/slurm/campaign_checkout.sbatch"
CHECKOUT_SCRIPT_SHA = "eaafc432b0047125e53fa8143db6a6366a7a160da74d6e8582c103a07e589fcc"
# L'ID del checkout non comincia con SCALING-compare-: la regola «Due fit per volta» cade solo quando e' sottomesso il confronto.
CHECKOUT_ID, COMPARE_ID = "SCALING-inputs-checkout-20261106", "SCALING-compare-20261106"
CHECKOUT = {"cpus": 2, "memory_gib": 4, "seconds": 300}
COMPARE = {"cpus": 8, "memory_gib": 30, "seconds": 1800}
CONSUMED_BEFORE_AGGREGATES, CAP = 124.993888889, 310.0  # conto pubblicato da GPT nella review 1e03fb3
COMMIT = "<COMMIT>"
EXEC_BRANCH = "claude/scaling-300m-exec-20261106"
CONTENT_LIMIT = 8 << 20
RUNTIME_KEYS = ("versions", "threads", "torch_threads", "threadpools")

REMOTE_COLLECT = r'''
import base64, hashlib, json, os, subprocess, time
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
def content(path):
    with open(path, "rb") as f:
        data = f.read(P["limit"] + 1)
    return base64.b64encode(data).decode() if len(data) <= P["limit"] else None
def load(path):
    with open(path) as f:
        return json.load(f)
def runtime(path):
    try:
        r = load(path).get("runtime") or {}
        return dict((k, r.get(k)) for k in ("code_sha", "sources") + tuple(P["runtime_keys"]))
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e)}
REPORT = ("protocol", "kind", "complete", "cell", "arm", "readout", "task", "step", "code_sha", "plan_sha256", "plan_content_sha256",
          "projection_sha256", "numeric_sha256", "reading_status", "oof", "oof_sha256", "audit", "audit_sha256", "shards")
SUMMARY = ("fits", "convergence_warnings", "reached_limit", "converged_by_criterion")
out = {"read_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": os.uname().nodename,
       "plan": pin(P["plan"]), "pose_plan": pin(P["pose_plan"]), "pose_report": pin(P["pose_report"]), "pose_budget": pin(P["pose_budget"]),
       "code": {"dir": P["code"], "canonical": os.path.realpath(P["code"]) == P["code"],
                "head": run(["git", "-C", P["code"], "rev-parse", "HEAD"]),
                "status": run(["git", "-C", P["code"], "status", "--porcelain", "--untracked-files=all"])},
       "sources": dict((rel, pin(P["code"] + "/" + rel)) for rel in P["sources"])}
ridge = load(P["pose_report"])
out["pose_fields"] = dict((k, ridge.get(k)) for k in ("protocol", "complete", "status", "step", "plan_sha256"))
out["pose_runtime"] = runtime(P["pose_report"])
out["benchmark_runtimes"] = dict((cell, runtime(p["path"])) for cell, p in sorted((load(P["pose_budget"]).get("benchmarks") or {}).items()))
aggregates = []
for job_id, path in P["aggregates"]:
    d, base = os.path.dirname(path), path[:-len(".json")]
    item = {"id": job_id, "out": path, "files": sorted(os.listdir(d)) if os.path.isdir(d) else None}
    if os.path.isfile(path):
        r = load(path)
        a = r.get("audit_summary") or {}
        item.update({"report": pin(path), "oof": pin(base + ".oof.npz"), "audit": pin(base + ".audit.jsonl"),
                     "audit_lines": sum(1 for _ in open(base + ".audit.jsonl", "rb")),
                     "fields": dict((k, r.get(k)) for k in REPORT),
                     "datasets": dict((ds, {"n_oof_windows": v.get("n_oof_windows"), "n_oof_subjects": v.get("n_oof_subjects"),
                                            "identity_sha256": v.get("identity_sha256"),
                                            "folds": [fr.get("fold") for fr in v.get("folds", [])],
                                            "scoring_warnings": len(v["scoring_warnings"]) if isinstance(v.get("scoring_warnings"), list) else None})
                                      for ds, v in sorted((r.get("datasets") or {}).items())),
                     "audit_summary": dict((k, a.get(k)) for k in SUMMARY), "unresolved_fits": len(a.get("unresolved_fits") or []),
                     "content": {"aggregate.json": content(path), "aggregate.audit.jsonl": content(base + ".audit.jsonl")}})
    aggregates.append(item)
out["aggregates"] = aggregates
out["shards"] = [{"name": name, "shard": pin(d + "/shard.json")} for name, d in P["shards"]]
logs = []
for job_id, slurm in P["logs"]:
    path = "%s/scaling_readers_recovery_%s.out" % (P["log_dir"], slurm)
    found = os.path.isfile(path)
    logs.append({"id": job_id, "pin": pin(path) if found else None, "content": content(path) if found else None})
out["logs"] = logs
out["absent"] = [{"path": p, "exists": os.path.lexists(p)} for p in P["absent"]]
out["sacct"] = run(["sacct", "-P", "-j", ",".join(P["jobs"]), "-o",
                    "JobIDRaw,JobName%40,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,MaxRSS,Start,End,Timelimit"])
out["squeue_serial"] = run(["squeue", "-u", os.environ.get("USER", ""), "-p", "lrd_all_serial", "-h", "-o", "%i|%j|%T|%M|%l|%C|%m|%r"])
print(json.dumps(out, indent=1, sort_keys=True))
'''

REMOTE_CHECKOUT = r'''
import hashlib, json, os, subprocess, time
P = json.loads(__PAYLOAD__)
def pin(path):
    with open(path, "rb") as f:
        data = f.read()
    return {"path": path, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "canonical": os.path.realpath(path) == path}
def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                       env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
    return {"exit": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()[-300:]}
d = P["campaign_dir"]
out = {"read_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": os.uname().nodename,
       "campaign_dir": {"path": d, "exists": os.path.lexists(d), "canonical": os.path.realpath(d) == d},
       "commit_in_canonical": run(["git", "-C", P["canonical"], "cat-file", "-t", P["commit"]]),
       "deploy_ref": run(["git", "-C", P["canonical"], "rev-parse", "--verify", "--quiet", "refs/heads/deploy/" + P["campaign"]]),
       "canonical_branch": run(["git", "-C", P["canonical"], "symbolic-ref", "--short", "HEAD"]),
       "hooks": sorted(x for x in os.listdir(P["canonical"] + "/.git/hooks") if not x.endswith(".sample"))
                if os.path.isdir(P["canonical"] + "/.git/hooks") else [],
       "receive_config": run(["git", "-C", P["canonical"], "config", "--get-regexp", "^receive\\."]),
       "worktrees": run(["git", "-C", P["canonical"], "worktree", "list", "--porcelain"]),
       "compare_out": {"path": P["compare_out"], "exists": os.path.lexists(P["compare_out"])},
       "pins": [dict(pin(q["path"]), expected=q["sha256"]) if os.path.isfile(q["path"]) else {"path": q["path"], "missing": True, "expected": q["sha256"]}
                for q in P["pins"]]}
if out["campaign_dir"]["exists"]:
    out["checkout"] = {"head": run(["git", "-C", d, "rev-parse", "HEAD"]),
                       "status": run(["git", "-C", d, "status", "--porcelain", "--untracked-files=all"]),
                       "manifest": pin(P["manifest"]) if os.path.isfile(P["manifest"]) else None}
out["sacct"] = run(["sacct", "-P", "-j", ",".join(P["jobs"]), "-o", "JobIDRaw,JobName%40,State,ExitCode,ElapsedRaw,AllocCPUS,Start,End"]) if P["jobs"] else None
out["squeue_serial"] = run(["squeue", "-u", os.environ.get("USER", ""), "-p", "lrd_all_serial", "-h", "-o", "%i|%j|%T|%M|%l|%C|%m|%r"])
print(json.dumps(out, indent=1, sort_keys=True))
'''


def aggregate_plan():
    require(sha(PLAN) == PLAN_SHA, "il piano degli aggregati non e' quello rivisto da GPT")
    return json.loads(PLAN.read_text())


def manifest_remote():
    return CAMPAIGN_DIR + "/" + MANIFEST.relative_to(ROOT).as_posix()


def read_ok(state, names):
    """Una lettura fallita non e' una risposta vuota: rifiuto prima di leggere o contare."""
    for name in names:
        code = (state.get(name) or {}).get("exit")
        require(type(code) is int and code == 0, "lettura di %s non riuscita (exit %s): nessun conteggio, nessun cancello" % (name, code))


def parents(state):
    return {r[0]: r for r in (x.split("|") for x in state["sacct"]["stdout"].splitlines()[1:]) if "." not in r[0]}


def cmd_collect(label):
    plan = aggregate_plan()
    ids = [j["id"] for j in plan["jobs"]]
    states = {i: F.ledger(i) for i in ids + [CHECKOUT_ID, COMPARE_ID]}
    require(all(states[i] and states[i]["stato"] == "sottomesso" and states[i].get("job_slurm") for i in ids),
            "ledger: i sei aggregati non risultano tutti sottomessi")
    payload = {"plan": F.PROBE_PLAN, "pose_plan": POSE_PLAN, "pose_report": POSE_REPORT, "pose_budget": POSE_BUDGET, "code": CODE_DIR,
               "sources": list(SOURCES), "runtime_keys": list(RUNTIME_KEYS), "limit": CONTENT_LIMIT, "log_dir": LOG_DIR,
               "aggregates": [[j["id"], j["out"]] for j in plan["jobs"]],
               "shards": [[s["name"], str(Path(s["path"]).parent)] for j in plan["jobs"] for s in j["shards"]],
               "logs": [[i, str(states[i]["job_slurm"])] for i in ids], "jobs": [str(states[i]["job_slurm"]) for i in ids],
               "absent": [COMPARE_OUT, CAMPAIGN_DIR]}
    value, script_sha = F.remote(REMOTE_COLLECT, payload)
    files = []  # (ID, nome locale, byte, pin remoto): prima si verifica tutto, poi si scrive
    for item, log in zip(value["aggregates"], value["logs"]):
        require(item["id"] == log["id"], "ordine inatteso nella lettura remota")
        for name, key in (("aggregate.json", "report"), ("aggregate.audit.jsonl", "audit")):
            if key in item:
                files.append((item["id"], name, item["content"].get(name), item[key]))
        if log["pin"] is not None:
            files.append((item["id"], Path(log["pin"]["path"]).name, log["content"], log["pin"]))
        item.pop("content", None)
        log.pop("content", None)
    folder, collected = HERE / ("collected_aggregates_%s" % label), {}
    for job_id, name, encoded, pin in files:
        require(encoded is not None, "%s/%s oltre il limite di raccolta: nulla scritto" % (job_id, name))
        data = base64.b64decode(encoded)
        require(hashlib.sha256(data).hexdigest() == pin["sha256"] and len(data) == pin["bytes"], "%s/%s: copia diversa dal file remoto" % (job_id, name))
    folder.mkdir(exist_ok=False)
    for job_id, name, encoded, pin in files:
        (folder / job_id).mkdir(exist_ok=True)
        with (folder / job_id / name).open("xb") as f:
            f.write(base64.b64decode(encoded))
        collected.setdefault(job_id, {})[name] = {"path": str((folder / job_id / name).relative_to(ROOT)), "sha256": pin["sha256"],
                                                  "bytes": pin["bytes"], "remote": pin["path"]}
    with (folder / "sacct.txt").open("x") as f:  # uscita originale di `sacct -P`, come letta
        f.write(value["sacct"]["stdout"] + "\n")
    value.update({"remote_script_sha256": script_sha, "collected": collected,
                  "accounting_file": {"path": str((folder / "sacct.txt").relative_to(ROOT)), "sha256": sha(folder / "sacct.txt")},
                  "local": {"read_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                            "repo_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                            "aggregate_plan_sha256": sha(PLAN), "dispatcher_sha256": sha(F.DISPATCHER), "submitter_sha256": sha(F.SUBMITTER),
                            "checkout_script_sha256": sha(CHECKOUT_SCRIPT), "sources": {rel: sha(ROOT / rel) for rel in SOURCES}},
                  "ledger": {i: (None if s is None else {k: s.get(k) for k in ("stato", "job_slurm", "agente", "campagna")}) for i, s in states.items()}})
    write_new(HERE / ("aggregates_state_%s.json" % label), value)
    print("raccolti %d file in %s" % (len(files), folder.relative_to(ROOT)))


def checks(state):
    """Controlli B della Review 10 sugli aggregati raccolti: ogni voce e' (nome, esito). Nessuno score entra in questi controlli."""
    read_ok(state, ("sacct", "squeue_serial"))
    plan = aggregate_plan()
    jobs, got = plan["jobs"], {a["id"]: a for a in state["aggregates"]}
    ledger, rows = state["ledger"], parents(state)
    done = all("report" in got.get(j["id"], {}) for j in jobs)
    row = {j["id"]: rows.get(str((ledger.get(j["id"]) or {}).get("job_slurm"))) for j in jobs}
    f = {j["id"]: got[j["id"]]["fields"] for j in jobs} if done else {}
    local = state["local"]
    expected_files = lambda j: sorted(Path(p).name for p in [j["out"]] + j["sidecars"])  # noqa: E731
    result = [
        ("letture di sacct e squeue riuscite (exit 0)", True),
        ("ledger: sei aggregati sottomessi da claude nella campagna", all(
            (ledger.get(j["id"]) or {}).get("stato") == "sottomesso" and ledger[j["id"]]["agente"] == "claude"
            and ledger[j["id"]]["campagna"] == CAMPAIGN for j in jobs)),
        ("contabilita': sei aggregati COMPLETED 0:0 con 2 CPU e il nome atteso", all(
            row[j["id"]] is not None and row[j["id"]][1] == j["id"] and row[j["id"]][2] == "COMPLETED" and row[j["id"]][3] == "0:0"
            and row[j["id"]][5] == "2" for j in jobs)),
        ("piano delle sonde, piano della posa, budget e report della ridge: hash attesi, percorsi canonici", all(
            state[k]["path"] == path and state[k]["sha256"] == digest and state[k]["canonical"] for k, path, digest in (
                ("plan", F.PROBE_PLAN, F.PROBE_PLAN_SHA), ("pose_plan", POSE_PLAN, POSE_PLAN_SHA),
                ("pose_budget", POSE_BUDGET, POSE_BUDGET_SHA), ("pose_report", POSE_REPORT, POSE_REPORT_SHA)))),
        ("report della ridge: protocollo, completo, passo 5000, piano della posa", state["pose_fields"] == {
            "protocol": POSE_PROTOCOL, "complete": True, "status": "completed", "step": 5000, "plan_sha256": POSE_PLAN_SHA}),
        ("runtime della ridge uguale a quello dei tre benchmark della posa (chiavi del controllo del dispatcher)",
         sorted(state["benchmark_runtimes"]) == sorted(F.CELLS) and state["pose_runtime"].get("code_sha") == CODE_SHA
         and all("error" not in r and all(r.get(k) == state["pose_runtime"].get(k) and r.get(k) is not None
                                          for k in ("code_sha", "sources") + RUNTIME_KEYS) for r in state["benchmark_runtimes"].values())),
        ("codice congelato: HEAD atteso, pulito, percorso canonico", state["code"]["head"] == {"exit": 0, "stdout": CODE_SHA, "stderr": ""}
         and state["code"]["status"] == {"exit": 0, "stdout": "", "stderr": ""} and state["code"]["canonical"]),
        ("sei sorgenti del comparatore: checkout congelato e copia locale uguali agli hash della Review 10", all(
            state["sources"].get(rel, {}).get("sha256") == digest and local["sources"].get(rel) == digest for rel, digest in SOURCES.items())
            and sorted(state["sources"]) == sorted(SOURCES)),
        ("shard dei fit (%d) invariati rispetto al piano degli aggregati" % N_SHARDS, sorted(
            (s["name"], s["shard"]["path"], s["shard"]["sha256"], s["shard"]["bytes"]) for s in state["shards"]) == sorted(
            (s["name"], s["path"], s["sha256"], s["bytes"]) for j in jobs for s in j["shards"]) and len(state["shards"]) == N_SHARDS),
        ("sei report presenti, alle uscite del piano", done and all(got[j["id"]]["report"]["path"] == j["out"] for j in jobs)),
    ]
    if done:
        a = {j["id"]: got[j["id"]] for j in jobs}
        result += [
            ("cartelle delle uscite: i tre file attesi" + (" e nient'altro" if EXCLUSIVE_FOLDERS else ""), all(
                (a[j["id"]]["files"] == expected_files(j)) if EXCLUSIVE_FOLDERS else set(expected_files(j)) <= set(a[j["id"]]["files"]) for j in jobs)),
            ("percorsi canonici di report, OOF e audit", all(a[j["id"]][k]["canonical"] for j in jobs for k in ("report", "oof", "audit"))),
            ("report completi: protocollo, tipo, passo, codice e piano", all(
                f[i]["protocol"] == PROBE_PROTOCOL and f[i]["kind"] == "aggregate" and f[i]["complete"] is True and f[i]["step"] == 5000
                and f[i]["code_sha"] == CODE_SHA and f[i]["plan_sha256"] == F.PROBE_PLAN_SHA for i in f)),
            ("cella, compito, braccio e lettura uguali al piano", all(
                f[j["id"]]["cell"] == j["cell"] and f[j["id"]]["task"] == j["task"]
                and [f[j["id"]]["arm"], f[j["id"]]["readout"]] == j["cell"].rsplit("-", 1) for j in jobs)),
            ("una sola firma numerica, quella dei quattordici fit", {f[i]["numeric_sha256"] for i in f} == {NUMERIC_SHA}),
            ("contenuto del piano e proiezione uguali nei sei report", len({f[i]["plan_content_sha256"] for i in f}) == 1
             and len({f[i]["projection_sha256"] for i in f}) == 1 and all(f[i]["plan_content_sha256"] and f[i]["projection_sha256"] for i in f)),
            ("shard dei report: percorsi e hash del piano, nello stesso ordine", all(
                f[j["id"]]["shards"] == [{"path": s["path"], "sha256": s["sha256"]} for s in j["shards"]] for j in jobs)),
            ("fit nell'audit: 28 per dataset (84 per p1, 56 per p2), righe dell'audit uguali", all(
                a[j["id"]]["audit_summary"]["fits"] == 28 * len(F.TASKS[j["task"]]) == a[j["id"]]["audit_lines"] for j in jobs)),
            ("hash interni dei report uguali ai file accanto (OOF e audit)", all(
                [f[j["id"]]["oof"], f[j["id"]]["audit"]] == j["sidecars"] and f[j["id"]]["oof"] == a[j["id"]]["oof"]["path"]
                and f[j["id"]]["oof_sha256"] == a[j["id"]]["oof"]["sha256"] and f[j["id"]]["audit"] == a[j["id"]]["audit"]["path"]
                and f[j["id"]]["audit_sha256"] == a[j["id"]]["audit"]["sha256"] for j in jobs)),
            ("dataset di ogni report: quelli del compito, quattro fold da 0 a 3", all(
                sorted(a[j["id"]]["datasets"]) == sorted(F.TASKS[j["task"]])
                and all(d["folds"] == [0, 1, 2, 3] for d in a[j["id"]]["datasets"].values()) for j in jobs)),
            ("identita' degli OOF: impronta, finestre e utenti uguali alle ancore della Review 10 in tutte le celle", all(
                (d["identity_sha256"], d["n_oof_windows"], d["n_oof_subjects"]) == ANCHORS.get(ds)
                for j in jobs for ds, d in a[j["id"]]["datasets"].items())),
            ("solutore: convergenza per criterio in tutti i report, nessun fit irrisolto", all(
                f[i]["reading_status"] == "solver_converged_by_criterion" and a[i]["audit_summary"]["converged_by_criterion"] is True
                and a[i]["unresolved_fits"] == 0 for i in f)),
            ("file raccolti sul Mac: report e audit di ogni aggregato, uguali ai file remoti", all(
                set(state["collected"].get(j["id"], {})) >= {"aggregate.json", "aggregate.audit.jsonl"}
                and all((ROOT / c["path"]).is_file() and sha(ROOT / c["path"]) == c["sha256"] and (ROOT / c["path"]).stat().st_size == c["bytes"]
                        for c in state["collected"][j["id"]].values())
                and state["collected"][j["id"]]["aggregate.json"]["sha256"] == a[j["id"]]["report"]["sha256"]
                and state["collected"][j["id"]]["aggregate.audit.jsonl"]["sha256"] == a[j["id"]]["audit"]["sha256"] for j in jobs)),
        ]
    result += [
        ("uscita del confronto e cartella del checkout del manifest assenti", [o["path"] for o in state["absent"]] == [COMPARE_OUT, CAMPAIGN_DIR]
         and not any(o["exists"] for o in state["absent"])),
        ("ID logici del checkout e del confronto mai usati nel ledger", ledger.get(CHECKOUT_ID) is None and ledger.get(COMPARE_ID) is None
         and CHECKOUT_ID in ledger and COMPARE_ID in ledger),
        ("dispatcher, submitter e script di checkout locali con gli hash attesi", local["dispatcher_sha256"] == F.DISPATCHER_SHA
         and local["submitter_sha256"] == F.SUBMITTER_SHA and local["checkout_script_sha256"] == CHECKOUT_SCRIPT_SHA),
    ]
    measured = [r for r in row.values() if r is not None]
    facts = {"aggregate_core_hours": sum(int(r[4]) * int(r[5]) for r in measured) / 3600,
             "aggregate_elapsed_seconds": {j["id"]: (int(row[j["id"]][4]) if row[j["id"]] else None) for j in jobs},
             "aggregate_max_rss": {i: next((x.split("|")[7] for x in state["sacct"]["stdout"].splitlines()[1:]
                                            if x.split("|")[0] == r[0] + ".batch"), None) for i, r in row.items() if r is not None},
             "reading_status": {i: f[i]["reading_status"] for i in f},
             "scoring_warnings": sum(d["scoring_warnings"] or 0 for i in f for d in got[i]["datasets"].values()),
             "serial_queue_jobs": len([x for x in state["squeue_serial"]["stdout"].splitlines() if x.strip()])}
    return result, facts


def cmd_validate(state_path):
    result, facts = checks(json.loads(Path(state_path).read_text()))
    for name, ok in result:
        print("PASSA " if ok else "FALLISCE ", name)
    print("fatti registrati:", json.dumps(facts, sort_keys=True))
    require(all(ok for _, ok in result), "aggregati non accettati: nessun manifest")
    print("AGGREGATI ACCETTATI")


def pin3(p):
    return {k: p[k] for k in ("path", "sha256", "bytes")}


def submitter_argv(job_id, script, sbatch):
    return [F.PYTHON, str(F.SUBMITTER), "--agente", "claude", "--campagna", CAMPAIGN, "--timeout", "120", "--id", job_id, "--gpu-ore", "0",
            "--script-stdin", str(script), "--", *sbatch]


def clock(seconds):
    return "--time=%02d:%02d:%02d" % (seconds // 3600, seconds % 3600 // 60, seconds % 60)


def cmd_manifest(state_path):
    state_path = Path(state_path).resolve()
    state = json.loads(state_path.read_text())
    result, facts = checks(state)
    for name, ok in result:
        print("PASSA " if ok else "FALLISCE ", name)
    require(all(ok for _, ok in result), "aggregati non accettati: nessun manifest scritto")
    plan = aggregate_plan()
    got = {a["id"]: a for a in state["aggregates"]}
    # Stesse chiavi e stesso ordine di scaling_compare.make_inputs; `frozen` lo verifica col codice congelato.
    inputs = {"protocol": INPUT_PROTOCOL, "checkpoint_step": 5000, "sources": {rel: sha(ROOT / rel) for rel in SOURCES},
              "probe_plan": pin3(state["plan"]), "pose_plan": pin3(state["pose_plan"]),
              "probe_reports": [pin3(got[j["id"]]["report"]) for j in plan["jobs"]], "pose_report": pin3(state["pose_report"])}
    require(inputs["sources"] == SOURCES and len({p["path"] for p in inputs["probe_reports"]}) == 6, "manifest inatteso")
    text = json.dumps(inputs, indent=1) + "\n"
    digest, remote = hashlib.sha256(text.encode()).hexdigest(), manifest_remote()
    checkout_sbatch = ["--cpus-per-task=%d" % CHECKOUT["cpus"], "--mem=%dG" % CHECKOUT["memory_gib"], clock(CHECKOUT["seconds"]),
                       "--export=ALL,CAMPAIGN=%s,CAMPAIGN_SHA=%s,EXPECTED_SHA=%s" % (CAMPAIGN_NAME, COMMIT, COMMIT), "/dev/stdin"]
    compare_sbatch = ["--cpus-per-task=%d" % COMPARE["cpus"], "--mem=%dG" % COMPARE["memory_gib"], clock(COMPARE["seconds"]),
                      "--export=ALL,CODE_DIR=%s,EXPECTED_SHA=%s" % (CODE_DIR, CODE_SHA), "/dev/stdin", "compare", remote, digest, COMPARE_OUT]
    hours = lambda r, extra=0: r["cpus"] * (r["seconds"] + extra) / 3600  # noqa: E731
    consumed = CONSUMED_BEFORE_AGGREGATES + facts["aggregate_core_hours"]
    worst = consumed + hours(CHECKOUT, F.TIMEOUT_GRACE) + hours(COMPARE, F.TIMEOUT_GRACE)
    require(worst <= CAP, "budget: il caso peggiore supera il tetto")
    jobs = [{"id": CHECKOUT_ID, "stage": "campaign-checkout", "campaign": CAMPAIGN_NAME, "campaign_dir": CAMPAIGN_DIR,
             "script": {"path": str(CHECKOUT_SCRIPT.relative_to(ROOT)), "sha256": sha(CHECKOUT_SCRIPT)}, **CHECKOUT,
             "requested_core_hours": hours(CHECKOUT), "depends_on": None,
             "argv_template": submitter_argv(CHECKOUT_ID, CHECKOUT_SCRIPT, checkout_sbatch),
             "commit_placeholder": COMMIT},
            {"id": COMPARE_ID, "stage": "compare", "manifest": {"path": remote, "sha256": digest, "bytes": len(text.encode())},
             "out": COMPARE_OUT, "script": {"path": str(F.DISPATCHER.relative_to(ROOT)), "sha256": sha(F.DISPATCHER)}, **COMPARE,
             "requested_core_hours": hours(COMPARE), "depends_on": None, "argv": submitter_argv(COMPARE_ID, F.DISPATCHER, compare_sbatch)}]
    release = {"protocol": "scaling5000-comparison-release-20261106-v1", "status": "PREPARED_NOT_SUBMITTED", "agente": "claude",
               "campagna": CAMPAIGN, "runtime_code_sha": CODE_SHA, "code_dir": CODE_DIR,
               "aggregate_plan": {"path": str(PLAN.relative_to(ROOT)), "sha256": sha(PLAN)},
               "aggregates_state": {"path": str(state_path.relative_to(ROOT)), "sha256": sha(state_path), "read_at": state["read_at"]},
               "manifest": {"path": str(MANIFEST.relative_to(ROOT)), "sha256": digest, "bytes": len(text.encode()),
                            "published_at": remote, "inputs_protocol": INPUT_PROTOCOL},
               "integrity_checks": [{"check": name, "passed": ok} for name, ok in result], "recorded_facts": facts,
               "budget_core_hours": {"consumed_before_aggregates": CONSUMED_BEFORE_AGGREGATES, "aggregates_measured": facts["aggregate_core_hours"],
                                     "consumed": consumed, "checkout_requested": hours(CHECKOUT), "checkout_maximum": hours(CHECKOUT, F.TIMEOUT_GRACE),
                                     "compare_requested": hours(COMPARE), "compare_maximum": hours(COMPARE, F.TIMEOUT_GRACE),
                                     "worst_case": worst, "cap": CAP},
               "order": ("Two jobs, never chained. First the campaign checkout that publishes this manifest on Leonardo, from the "
                         "commit that contains it (deploy reference, then the checkout job). The comparison is submitted only after "
                         "the checkout is verified: HEAD equal to the commit, clean tree, manifest with the pinned sha256 and size, "
                         "dispatcher argument guard passed on the real paths. No afterok, no retry, no background resubmission."),
               "submission_rule": ("Submit only after GPT's review of the collected aggregates, of this manifest and of the two commands. "
                                   "Before each job: fresh read with exit 0 of every command, pins unchanged, outputs absent, logical "
                                   "IDs unused, a free place in lrd_all_serial. Common submitter and ledger, one line at a time, stop "
                                   "at the first outcome other than `sottomesso`."),
               "lane_rule": ("The checkout ID does not begin with SCALING-compare-: the rule «Due fit per volta» of line 2 falls only "
                             "when the comparison job itself is `sottomesso` in the common ledger."),
               "comparison_reading": "The operator does not open or interpret scores: the frozen comparator applies the frozen reading S8.",
               "jobs": jobs}
    with MANIFEST.open("x") as f:
        f.write(text)
    print("scritto", MANIFEST.relative_to(ROOT), digest)
    write_new(HERE / "comparison_plan.json", release)


def cmd_frozen(manifest_path):
    """Con l'ambiente del progetto: il manifest e' quello che scriverebbe e accetterebbe il comparatore congelato."""
    require(all(sha(ROOT / rel) == digest for rel, digest in SOURCES.items()), "sorgenti locali diversi dal codice congelato")
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    import scaling_compare as M
    inputs = json.loads(Path(manifest_path).read_text())
    M.validate_inputs(inputs)
    print("PASSA  validate_inputs del comparatore congelato")
    same = M.make_inputs(inputs["probe_plan"], inputs["pose_plan"], inputs["probe_reports"], inputs["pose_report"]) == inputs
    print("PASSA " if same else "FALLISCE ", "make_inputs del comparatore congelato restituisce lo stesso manifest")
    require(same and M.source_hashes() == SOURCES and M.INPUT_PROTOCOL == INPUT_PROTOCOL, "manifest diverso da quello del codice congelato")
    print("manifest", sha(manifest_path), "| contenuto", M.C.jhash(inputs))


def release_jobs(plan_path, lines_path=None):
    release = json.loads(Path(plan_path).read_text())
    require(sha(MANIFEST) == release["manifest"]["sha256"], "manifest locale diverso da quello del piano")
    checkout, compare = release["jobs"]
    if lines_path is None:
        return release, checkout, compare, None
    lines = [shlex.split(x) for x in Path(lines_path).read_text().splitlines() if x.strip()]
    require(len(lines) == 2 and lines[1] == compare["argv"], "righe d'invio diverse dal piano")
    export = lines[0][-2]
    commit = export.rsplit("=", 1)[-1]
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
            and lines[0] == [x.replace(COMMIT, commit) for x in checkout["argv_template"]], "riga del checkout diversa dal piano")
    return release, dict(checkout, argv=lines[0], commit=commit), compare, commit


def cmd_lines(plan_path, commit):
    release, checkout, compare, _ = release_jobs(plan_path)
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "serve lo SHA completo del commit")
    def blob(path):  # sha256 del file in quel commit, None se non c'e'
        r = subprocess.run(["git", "-C", str(ROOT), "show", "%s:%s" % (commit, Path(path).resolve().relative_to(ROOT).as_posix())],
                           capture_output=True)
        return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None
    require(blob(MANIFEST) == release["manifest"]["sha256"], "il commit non contiene questo manifest")
    require(blob(plan_path) == sha(plan_path), "il commit non contiene questo piano")
    argv = [x.replace(COMMIT, commit) for x in checkout["argv_template"]]
    write_new(HERE / "comparison_submit_lines.txt", "".join(" ".join(shlex.quote(x) for x in a) + "\n" for a in (argv, compare["argv"])))


def cmd_submitter(plan_path, lines_path):
    release, checkout, compare, commit = release_jobs(plan_path, lines_path)
    previews = []
    with tempfile.TemporaryDirectory() as tmp:
        fake, registry = Path(tmp) / "fake_ssh.py", Path(tmp) / "jobs"
        fake.write_text(F.FAKE_SSH)
        for n, (job, code) in enumerate(((checkout, commit), (compare, CODE_SHA)), 1):
            ssh = " ".join(shlex.quote(x) for x in [F.PYTHON, "-I", str(fake), str(Path(tmp) / "counter"), job["script"]["sha256"]])
            argv = list(job["argv"])
            argv[2:2] = ["--registro", str(registry), "--ssh", ssh]
            r = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL)
            require(r.returncode == 0 and r.stdout.strip() == "%s: sottomesso, job %d" % (job["id"], 90000000 + n), "submitter: " + r.stdout + r.stderr)
            s = json.loads((registry / job["id"] / "state.json").read_text())
            expected = ["sbatch", "--parsable", "--job-name=" + job["id"], "--comment=wearusfm:" + job["id"],
                        *job["argv"][job["argv"].index("--") + 1:]]
            require(s["sbatch"] == expected and s["script_stdin_sha256"] == job["script"]["sha256"] == sha(ROOT / job["script"]["path"])
                    and s["codice_sha"] == code and s["agente"] == "claude" and s["campagna"] == CAMPAIGN and s["gpu_ore_stimate"] == 0
                    and "--dopo" not in job["argv"] and not any("dependency" in x for x in s["sbatch"]), "comando di sbatch inatteso")
            previews.append({"id": job["id"], "codice_sha": s["codice_sha"], "sbatch": " ".join(shlex.quote(x) for x in s["sbatch"])})
            print("PASSA ", job["id"])
        require(len((registry / "ledger.tsv").read_text().splitlines()) == 1 + 2 * 2, "ledger temporaneo inatteso")
    write_new(HERE / "comparison_submitter_dryrun.json", {
        "scope": "Real scripts/submit_job.py on the two argv, with a fake ssh and a temporary registry: nothing reached Leonardo or the common ledger.",
        "submitter_sha256": sha(F.SUBMITTER), "plan_sha256": sha(plan_path), "lines_sha256": sha(lines_path), "sbatch_commands": previews})


def checkout_state(compare, commit, label):
    """Sola lettura: repository canonico, checkout del manifest, i nove file fissati nel manifest, coda e ledger."""
    states = {i: F.ledger(i) for i in (CHECKOUT_ID, COMPARE_ID)}
    inputs = json.loads(MANIFEST.read_text())
    payload = {"campaign": CAMPAIGN_NAME, "campaign_dir": CAMPAIGN_DIR, "canonical": WORK + "/wearusfm", "commit": commit,
               "manifest": compare["manifest"]["path"], "compare_out": COMPARE_OUT,
               "pins": [inputs["probe_plan"], inputs["pose_plan"], inputs["pose_report"], *inputs["probe_reports"]],
               "jobs": [str(s["job_slurm"]) for s in states.values() if s and s.get("job_slurm")]}
    value, script_sha = F.remote(REMOTE_CHECKOUT, payload)
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)  # noqa: E731
    tip = git("ls-remote", "origin", "refs/heads/" + EXEC_BRANCH)
    tip_sha = tip.stdout.split("\t")[0] if tip.returncode == 0 else ""
    value.update({"remote_script_sha256": script_sha, "commit": commit,
                  "local": {"read_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                            "repo_head": git("rev-parse", "HEAD").stdout.strip(), "origin_branch": EXEC_BRANCH, "origin_tip": tip_sha,
                            "origin_has_commit": len(tip_sha) == 40 and git("merge-base", "--is-ancestor", commit, tip_sha).returncode == 0,
                            "dispatcher_sha256": sha(F.DISPATCHER), "submitter_sha256": sha(F.SUBMITTER),
                            "checkout_script_sha256": sha(CHECKOUT_SCRIPT), "plan_sha256": sha(HERE / "comparison_plan.json"),
                            "manifest_sha256": sha(MANIFEST)},
                  "ledger": {i: (None if s is None else {k: s.get(k) for k in ("stato", "job_slurm", "agente", "campagna", "codice_sha")})
                             for i, s in states.items()}})
    write_new(HERE / ("comparison_gate_%s.json" % label), value)
    return value


def gate(result, state, places=1):
    queue = len([x for x in state["squeue_serial"]["stdout"].splitlines() if x.strip()])
    result = result + [("posti liberi in lrd_all_serial: almeno %d su 10" % places, 10 - queue >= places)]
    for name, ok in result:
        print("PASSA " if ok else "FALLISCE ", name)
    print("job dell'utente in coda seriale:", queue, "| letto alle", state["read_at"])
    require(all(ok for _, ok in result), "cancello non superato")
    print("CANCELLO SUPERATO")


def common_gate(state, compare):
    read_ok(state, ("squeue_serial",))
    local, inputs = state["local"], json.loads(MANIFEST.read_text())
    pinned = [inputs["probe_plan"], inputs["pose_plan"], inputs["pose_report"], *inputs["probe_reports"]]
    return [
        ("lettura di squeue riuscita (exit 0)", True),
        ("commit del manifest pubblicato su origin, nel ramo di esecuzione", local["origin_has_commit"] is True),
        ("dispatcher, submitter e script di checkout locali con gli hash attesi; manifest uguale al piano",
         local["dispatcher_sha256"] == F.DISPATCHER_SHA and local["submitter_sha256"] == F.SUBMITTER_SHA
         and local["checkout_script_sha256"] == CHECKOUT_SCRIPT_SHA and local["manifest_sha256"] == compare["manifest"]["sha256"]),
        ("i nove file fissati nel manifest: presenti, canonici, hash e dimensioni invariati", len(state["pins"]) == 9 and all(
            not p.get("missing") and p["canonical"] for p in state["pins"])
            and [(p["path"], p.get("sha256"), p.get("bytes")) for p in state["pins"]] == [(q["path"], q["sha256"], q["bytes"]) for q in pinned]),
        ("uscita del confronto assente", state["compare_out"] == {"path": COMPARE_OUT, "exists": False}),
        ("ID logico del confronto mai usato nel ledger", state["ledger"][COMPARE_ID] is None),
    ]


def before_checkout(state):
    read_ok(state, ("worktrees",))
    return [
        ("cartella del checkout di campagna assente", state["campaign_dir"] == {"path": CAMPAIGN_DIR, "exists": False, "canonical": True}),
        ("repository canonico: nessun hook attivo", state["hooks"] == []),
        ("il riferimento di deploy non e' estratto in alcun worktree (un push non aggiorna checkout)",
         ("branch refs/heads/deploy/" + CAMPAIGN_NAME) not in state["worktrees"]["stdout"].splitlines()),
        ("ID logico del checkout mai usato nel ledger", state["ledger"][CHECKOUT_ID] is None),
    ]


def cmd_pre_deploy(plan_path, lines_path, label):
    release, checkout, compare, commit = release_jobs(plan_path, lines_path)
    state = checkout_state(compare, commit, label)
    print("configurazione di ricezione del canonico:", state["receive_config"]["stdout"].replace("\n", "; ") or "(nessuna)")
    gate(common_gate(state, compare) + before_checkout(state) + [
        ("riferimento di deploy non ancora presente su Leonardo", state["deploy_ref"]["exit"] == 1 and state["deploy_ref"]["stdout"] == ""),
    ], state)
    print("push ammesso: git push leonardo %s:refs/heads/deploy/%s" % (commit, CAMPAIGN_NAME))


def cmd_gate_checkout(plan_path, lines_path, label):
    release, checkout, compare, commit = release_jobs(plan_path, lines_path)
    state = checkout_state(compare, commit, label)
    gate(common_gate(state, compare) + before_checkout(state) + [
        ("riferimento di deploy su Leonardo uguale al commit del manifest", state["deploy_ref"] == {"exit": 0, "stdout": commit, "stderr": ""}),
        ("commit presente nel repository canonico di Leonardo", state["commit_in_canonical"]["exit"] == 0
         and state["commit_in_canonical"]["stdout"] == "commit"),
    ], state)


def cmd_gate_compare(plan_path, lines_path, label):
    release, checkout, compare, commit = release_jobs(plan_path, lines_path)
    state = checkout_state(compare, commit, label)
    require(state["sacct"] is not None, "il checkout non risulta sottomesso nel ledger")
    read_ok(state, ("sacct",))
    done, entry = state.get("checkout") or {}, state["ledger"][CHECKOUT_ID] or {}
    row = parents(state).get(str(entry.get("job_slurm")))
    args = compare["argv"][compare["argv"].index("/dev/stdin") + 1:]
    lines = F.DISPATCHER.read_text().splitlines()
    first, last = 37, 115
    require(sha(F.DISPATCHER) == F.DISPATCHER_SHA and lines[first - 1] == "import argparse"
            and lines[last].startswith("if a.stage == 'probe-benchmark':"), "le righe del controllo sugli argomenti non sono dove attese")
    cases = [{"name": COMPARE_ID, "expect": "PASS", "args": args},
             {"name": "controllo negativo: hash del manifest alterato", "expect": "input hash/size differs",
              "args": [args[0], args[1], args[2][:-1] + ("0" if args[2][-1] != "0" else "1"), args[3]]},
             {"name": "controllo negativo: uscita dentro i checkout di campagna", "expect": "output inside code",
              "args": [args[0], args[1], args[2], CODE_DIR + "/report.json"]},
             {"name": "controllo negativo: opzione non ammessa per lo stadio", "expect": "stage requires exactly its whitelisted flags",
              "args": args + ["--cell", F.CELLS[0]]}]
    guard, script_sha = F.remote(F.REMOTE_GUARD, {"guard": "\n".join(lines[first - 1:last]) + "\n", "lines": [first, last], "cases": cases,
                                                  "env": {"CODE_DIR": CODE_DIR, "WORK": WORK, "EXPECTED_SHA": CODE_SHA}})
    for r in guard["results"]:
        r["ok"] = r["got"] == "PASS" if r["expect"] == "PASS" else r["got"] != "PASS" and r["expect"] in r["got"]
        print("PASSA " if r["ok"] else "FALLISCE ", r["name"], "->", r["got"][:150])
    write_new(HERE / ("comparison_guard_dryrun_%s.json" % label), dict(
        guard, dispatcher_sha256=F.DISPATCHER_SHA, guard_lines=[first, last], remote_script_sha256=script_sha, plan_sha256=sha(plan_path),
        scope=("Argument, manifest pin and output guards of the real dispatcher, executed read-only on the login node. The protocol, "
               "pose report and numeric runtime checks of the compare stage and the allocation guards run only inside the job.")))
    gate(common_gate(state, compare) + [
        ("checkout sottomesso da claude per il commit del manifest e COMPLETED 0:0", entry.get("stato") == "sottomesso"
         and entry.get("agente") == "claude" and entry.get("codice_sha") == commit and row is not None and row[1] == CHECKOUT_ID
         and row[2] == "COMPLETED" and row[3] == "0:0"),
        ("checkout di campagna: presente, canonico, HEAD uguale al commit, pulito", state["campaign_dir"] == {
            "path": CAMPAIGN_DIR, "exists": True, "canonical": True} and done.get("head") == {"exit": 0, "stdout": commit, "stderr": ""}
            and done.get("status") == {"exit": 0, "stdout": "", "stderr": ""}),
        ("manifest nel checkout: percorso, hash e dimensione del piano, percorso canonico",
         done.get("manifest") == dict(compare["manifest"], canonical=True)),
        ("controlli sugli argomenti del dispatcher reale: confronto accettato, tre alterazioni rifiutate", all(r["ok"] for r in guard["results"])),
    ], state)


if __name__ == "__main__":
    command = {"collect": cmd_collect, "validate": cmd_validate, "manifest": cmd_manifest, "frozen": cmd_frozen, "lines": cmd_lines,
               "submitter": cmd_submitter, "pre-deploy": cmd_pre_deploy, "gate-checkout": cmd_gate_checkout,
               "gate-compare": cmd_gate_compare}.get(sys.argv[1] if len(sys.argv) > 1 else "")
    require(command is not None, __doc__)
    command(*sys.argv[2:])
