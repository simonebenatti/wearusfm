#!/usr/bin/env python3
"""Scaling 30M/300M a 5.000 passi, dopo il confronto: raccolta dell'esito e validazione tecnica. Nessun invio.

    python -I -B outcome_stage.py collect <piano.json> <righe.txt> <etichetta>   legge su Leonardo checkout del manifest, uscita del
                                                  confronto, log e contabilita' -> outcome_state_<etichetta>.json, collected_outcome_<etichetta>/
    python -I -B outcome_stage.py validate <stato.json>                          controlli D della Review 10 (d3976c3, sezione 5)

Nessun invio e nessuna scrittura su Leonardo: la lettura remota usa solo la libreria standard del login node (Python 3.6). Il
report del confronto, i due log e l'uscita originale di `sacct -P` si copiano byte per byte e si verificano con lo sha256.
`validate` e' la validazione tecnica, separata dall'analisi scientifica: apre il report ma stampa solo esiti vero/falso, conteggi,
tempi e stati di qualita'; mai score, contrasti, intervalli o raccomandazione. I confronti numerici (punti uguali agli aggregati
entro 1e-12, lettura S8 ricalcolata con `scripts/scaling_reading.py` congelato) producono solo vero o falso. Un report o uno stato
malformato fa fallire il controllo, non lo strumento. Si riesegue offline dal commit che contiene stato e file raccolti.
"""
import base64
import datetime
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparison_stage as S  # noqa: E402
import final_stages as F  # noqa: E402

require, sha, write_new = F.require, F.sha, F.write_new
PROTOCOL = "scaling-comparison-5000-20261104-v1"
PRIMARY = "natural_300m_minus_30m"
CONTROL, CANDIDATE = "T100-G-30M-natural", "T100-G-300M-natural"
BOOTSTRAP = {"unit": "subject", "replicates": 2000, "seed": 20261102}
QUALITY = {"artifact_integrity", "paired_support", "solver_converged"}
POSE_GROUPS = ("dynamic", "all", "static")
POSE_USERS = 15  # Review 10 (d3976c3), sezione 3: quindici utenti di validazione della posa
REPORT_KEYS = {"protocol", "complete", "checkpoint_step", "candidate", "control", "bootstrap", "quality", "pose", "secondary", "scope",
               "epn", "p1", "p2", "reading", "inputs", "inputs_content_sha256", "checkpoints", "source_contract", "projection",
               "runtime", "sources"}
READING = "scripts/scaling_reading.py"
LEDGER_KEYS = ("stato", "job_slurm", "agente", "campagna", "codice_sha", "gpu_ore_stimate", "sbatch", "script_stdin_sha256")
CONTENT_LIMIT = 32 << 20
TOLERANCE = 1e-12

REMOTE_OUTCOME = r'''
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
def found(path):
    if not os.path.isfile(path):
        return {"path": path, "pin": None, "content": None}
    with open(path, "rb") as f:
        data = f.read(P["limit"] + 1)
    return {"path": path, "pin": pin(path), "content": base64.b64encode(data).decode() if len(data) <= P["limit"] else None}
d, o = P["campaign_dir"], os.path.dirname(P["compare_out"])
out = {"read_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": os.uname().nodename,
       "code": {"dir": P["code"], "canonical": os.path.realpath(P["code"]) == P["code"],
                "head": run(["git", "-C", P["code"], "rev-parse", "HEAD"]),
                "status": run(["git", "-C", P["code"], "status", "--porcelain", "--untracked-files=all"])},
       "sources": dict((rel, pin(P["code"] + "/" + rel)) for rel in P["sources"]),
       "campaign": {"path": d, "exists": os.path.isdir(d), "canonical": os.path.realpath(d) == d},
       "pins": [dict(pin(q["path"]), expected=q["sha256"]) if os.path.isfile(q["path"]) else {"path": q["path"], "missing": True, "expected": q["sha256"]}
                for q in P["pins"]],
       "output": {"dir": o, "exists": os.path.isdir(o), "canonical": os.path.realpath(o) == o,
                  "files": sorted(os.listdir(o)) if os.path.isdir(o) else None, "report": found(P["compare_out"])},
       "logs": [dict(found(path), id=job_id) for job_id, path in P["logs"]]}
if out["campaign"]["exists"]:
    out["campaign"].update({"head": run(["git", "-C", d, "rev-parse", "HEAD"]),
                            "status": run(["git", "-C", d, "status", "--porcelain", "--untracked-files=all"]),
                            "manifest": pin(P["manifest"]) if os.path.isfile(P["manifest"]) else None})
out["sacct"] = run(["sacct", "-P", "-j", ",".join(P["jobs"]), "-o",
                    "JobIDRaw,JobName%40,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,MaxRSS,Start,End,Timelimit"])
out["squeue_serial"] = run(["squeue", "-u", os.environ.get("USER", ""), "-p", "lrd_all_serial", "-h", "-o", "%i|%j|%T|%M|%l|%C|%m|%r"])
print(json.dumps(out, indent=1, sort_keys=True))
'''


def rel(path):
    return str(Path(path).resolve().relative_to(S.ROOT))


def jhash(value):
    """Stesso hash canonico del codice congelato (scaling_probe.jhash)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def expected_sbatch(job):
    return ["sbatch", "--parsable", "--job-name=" + job["id"], "--comment=wearusfm:" + job["id"], *job["argv"][job["argv"].index("--") + 1:]]


def frozen_reading():
    """La regola di lettura S8 del codice congelato: solo libreria standard, caricata dal file con l'hash della Review 10."""
    path = S.ROOT / READING
    require(sha(path) == S.SOURCES[READING], "scaling_reading.py locale diverso dal codice congelato")
    spec = importlib.util.spec_from_file_location("scaling_reading_frozen", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cmd_collect(plan_path, lines_path, label):
    release, checkout, compare, commit = S.release_jobs(plan_path, lines_path)
    ids = (S.CHECKOUT_ID, S.COMPARE_ID)
    states = {i: F.ledger(i) for i in ids}
    require(all(s and s.get("stato") == "sottomesso" and s.get("job_slurm") for s in states.values()),
            "ledger: checkout e confronto non risultano entrambi sottomessi")
    slurm = {i: str(states[i]["job_slurm"]) for i in ids}
    inputs = json.loads(S.MANIFEST.read_text())
    payload = {"code": S.CODE_DIR, "sources": list(S.SOURCES), "campaign_dir": S.CAMPAIGN_DIR, "manifest": compare["manifest"]["path"],
               "compare_out": S.COMPARE_OUT, "limit": CONTENT_LIMIT,
               "pins": [inputs["probe_plan"], inputs["pose_plan"], inputs["pose_report"], *inputs["probe_reports"]],
               "logs": [[S.CHECKOUT_ID, "%s/campaign_checkout_%s.out" % (S.LOG_DIR, slurm[S.CHECKOUT_ID])],
                        [S.COMPARE_ID, "%s/scaling_readers_recovery_%s.out" % (S.LOG_DIR, slurm[S.COMPARE_ID])]],
               "jobs": [slurm[i] for i in ids]}
    value, script_sha = F.remote(REMOTE_OUTCOME, payload)
    files = []  # (nome locale, byte, pin remoto): prima si verifica tutto, poi si scrive
    for name, item in [("report.json", value["output"]["report"])] + [(Path(log["path"]).name, log) for log in value["logs"]]:
        encoded = item.pop("content")
        if item["pin"] is None:
            continue  # file assente: lo stato lo registra e `validate` lo rifiuta
        require(encoded is not None, "%s oltre il limite di raccolta: nulla scritto" % name)
        data = base64.b64decode(encoded)
        require(hashlib.sha256(data).hexdigest() == item["pin"]["sha256"] and len(data) == item["pin"]["bytes"],
                "%s: copia diversa dal file remoto" % name)
        files.append((name, data, item["pin"]))
    require(len({name for name, _, _ in files}) == len(files), "nomi dei file raccolti non distinti")
    folder, collected = S.HERE / ("collected_outcome_%s" % label), {}
    folder.mkdir(exist_ok=False)
    for name, data, pin in files:
        with (folder / name).open("xb") as f:
            f.write(data)
        collected[name] = {"path": rel(folder / name), "sha256": pin["sha256"], "bytes": pin["bytes"], "remote": pin["path"]}
    with (folder / "sacct.txt").open("x") as f:  # uscita originale di `sacct -P`, come letta
        f.write(value["sacct"]["stdout"] + "\n")
    value.update({"remote_script_sha256": script_sha, "commit": commit, "collected": collected,
                  "accounting_file": {"path": rel(folder / "sacct.txt"), "sha256": sha(folder / "sacct.txt")},
                  "local": {"read_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                            "repo_head": subprocess.check_output(["git", "-C", str(S.ROOT), "rev-parse", "HEAD"], text=True).strip(),
                            "plan": {"path": rel(plan_path), "sha256": sha(plan_path)},
                            "lines": {"path": rel(lines_path), "sha256": sha(lines_path)}, "manifest_sha256": sha(S.MANIFEST),
                            "dispatcher_sha256": sha(F.DISPATCHER), "submitter_sha256": sha(F.SUBMITTER),
                            "checkout_script_sha256": sha(S.CHECKOUT_SCRIPT), "sources": {r: sha(S.ROOT / r) for r in S.SOURCES}},
                  "ledger": {i: {k: s.get(k) for k in LEDGER_KEYS} for i, s in states.items()}})
    write_new(S.HERE / ("outcome_state_%s.json" % label), value)
    print("raccolti %d file in %s" % (len(files), rel(folder)))


def passed(test):
    try:
        return bool(test())
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, OSError, SystemExit):
        return False


def fact(read):
    try:
        return read()
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, OSError, SystemExit):
        return None


def checks(state, report=None):
    """Controlli D della Review 10 sull'esito raccolto: ogni voce e' (nome, esito). Nessun valore del report esce da qui."""
    S.read_ok(state, ("sacct", "squeue_serial"))
    local = state["local"]
    plan_path, lines_path = S.ROOT / local["plan"]["path"], S.ROOT / local["lines"]["path"]
    require(sha(plan_path) == local["plan"]["sha256"] and sha(lines_path) == local["lines"]["sha256"],
            "piano o righe d'invio diversi da quelli della raccolta")
    release, checkout, compare, commit = S.release_jobs(plan_path, lines_path)
    require(commit == state["commit"], "commit del manifest diverso da quello della raccolta")
    inputs = json.loads(S.MANIFEST.read_text())
    pinned = [inputs["probe_plan"], inputs["pose_plan"], inputs["pose_report"], *inputs["probe_reports"]]
    jobs = {S.CHECKOUT_ID: (checkout, commit, S.CHECKOUT), S.COMPARE_ID: (compare, S.CODE_SHA, S.COMPARE)}
    ledger, rows, budget = state["ledger"], S.parents(state), release["budget_core_hours"]
    row = {i: rows.get(str((ledger.get(i) or {}).get("job_slurm"))) for i in jobs}
    collected, campaign, output = state.get("collected") or {}, state["campaign"], state["output"]
    log_name = {log["id"]: Path(log["path"]).name for log in state["logs"]}
    clean = {"exit": 0, "stdout": "", "stderr": ""}

    def local_copy(name):
        c = collected[name]
        path = S.ROOT / c["path"]
        require(path.is_file() and sha(path) == c["sha256"] and path.stat().st_size == c["bytes"], "copia locale alterata: " + name)
        return path

    def accounted(i):
        (job, code, spec), r = jobs[i], row[i]
        return (r is not None and r[1] == i and r[2] == "COMPLETED" and r[3] == "0:0" and r[5] == str(spec["cpus"])
                and r[6] in ("%dG" % spec["memory_gib"], "%dM" % (1024 * spec["memory_gib"]))
                and 0 <= int(r[4]) <= spec["seconds"] and r[10] == S.clock(spec["seconds"]).split("=")[1])

    def submitted(i):
        (job, code, spec), entry = jobs[i], ledger[i]
        return (entry["stato"] == "sottomesso" and entry["agente"] == "claude" and entry["campagna"] == S.CAMPAIGN
                and entry["codice_sha"] == code and entry["gpu_ore_stimate"] == 0 and entry["sbatch"] == expected_sbatch(job)
                and entry["script_stdin_sha256"] == job["script"]["sha256"] == sha(S.ROOT / job["script"]["path"]))

    def receipt(i):
        saved = json.loads((S.HERE / "submissions" / (i + ".json")).read_text())
        return all(saved.get(k) == ledger[i].get(k) for k in LEDGER_KEYS)

    def remote_copies():
        expected = [("report.json", output["report"]["pin"])] + [(log_name[log["id"]], log["pin"]) for log in state["logs"]]
        accounting = S.ROOT / state["accounting_file"]["path"]
        return (len(log_name) == 2 and set(collected) == {name for name, _ in expected} and all(
            local_copy(name).is_file() and (collected[name]["sha256"], collected[name]["bytes"], collected[name]["remote"])
            == (pin["sha256"], pin["bytes"], pin["path"]) for name, pin in expected)
            and sha(accounting) == state["accounting_file"]["sha256"] and accounting.read_text() == state["sacct"]["stdout"] + "\n")

    def core_hours(i):
        return int(row[i][4]) * int(row[i][5]) / 3600

    def within_budget():
        return (core_hours(S.CHECKOUT_ID) <= budget["checkout_maximum"] and core_hours(S.COMPARE_ID) <= budget["compare_maximum"]
                and budget["consumed"] + sum(core_hours(i) for i in jobs) <= budget["cap"] == S.CAP)

    result = [
        ("letture di sacct e squeue riuscite (exit 0)", True),
        ("piano, righe d'invio e manifest locali uguali a quelli della raccolta", passed(
            lambda: local["manifest_sha256"] == sha(S.MANIFEST) == compare["manifest"]["sha256"])),
        ("ledger: checkout e confronto sottomessi da claude nella campagna, comando di sbatch e script uguali al piano", passed(
            lambda: all(submitted(i) for i in jobs))),
        ("ricevute d'invio nel repository uguali al ledger letto alla raccolta", passed(lambda: all(receipt(i) for i in jobs))),
        ("D.1 contabilita': checkout COMPLETED 0:0 (2 CPU, 4G, entro 5 minuti) e confronto COMPLETED 0:0 (8 CPU, 30G, entro 30 minuti), "
         "nomi uguali agli ID", passed(lambda: all(accounted(i) for i in jobs))),
        ("codice congelato: HEAD atteso, pulito, percorso canonico", passed(
            lambda: state["code"]["head"] == dict(clean, stdout=S.CODE_SHA) and state["code"]["status"] == clean
            and state["code"]["canonical"] is True and state["code"]["dir"] == S.CODE_DIR)),
        ("sei sorgenti del comparatore: checkout congelato e copia locale uguali agli hash della Review 10", passed(
            lambda: sorted(state["sources"]) == sorted(S.SOURCES) and all(
                state["sources"][r]["sha256"] == digest == local["sources"][r] for r, digest in S.SOURCES.items()))),
        ("checkout del manifest: presente, canonico, HEAD uguale al commit, pulito", passed(
            lambda: (campaign["path"], campaign["exists"], campaign["canonical"]) == (S.CAMPAIGN_DIR, True, True)
            and campaign["head"] == dict(clean, stdout=commit) and campaign["status"] == clean)),
        ("manifest nel checkout: percorso, hash e dimensione del piano, percorso canonico", passed(
            lambda: campaign["manifest"] == dict(compare["manifest"], canonical=True))),
        ("i nove file fissati nel manifest: presenti, canonici, hash e dimensioni invariati dopo il confronto", passed(
            lambda: len(state["pins"]) == 9 and all(not p.get("missing") and p["canonical"] is True for p in state["pins"])
            and [(p["path"], p["sha256"], p["bytes"]) for p in state["pins"]] == [(q["path"], q["sha256"], q["bytes"]) for q in pinned])),
        ("cartella d'uscita del confronto: canonica, con il solo report.json all'uscita del piano", passed(
            lambda: (output["dir"], output["exists"], output["canonical"], output["files"])
            == (str(Path(S.COMPARE_OUT).parent), True, True, [Path(S.COMPARE_OUT).name])
            and output["report"]["pin"]["path"] == S.COMPARE_OUT == compare["out"] and output["report"]["pin"]["canonical"] is True)),
        ("D.7 file raccolti sul Mac (report e due log) uguali ai file remoti; uscita originale di sacct conservata", passed(remote_copies)),
        ("log del checkout: «checkout di campagna pronto» per la cartella e il commit attesi", passed(
            lambda: "checkout di campagna pronto: %s @ %s" % (S.CAMPAIGN_DIR, commit)
            in local_copy(log_name[S.CHECKOUT_ID]).read_text().splitlines())),
        ("budget: core-ore misurate entro i massimi del piano, consumo cumulativo entro il tetto", passed(within_budget)),
    ]
    facts = {"elapsed_seconds": {i: fact(lambda: int(row[i][4])) for i in jobs},
             "core_hours": {i: fact(lambda: core_hours(i)) for i in jobs},
             "core_hours_consumed": fact(lambda: budget["consumed"] + sum(core_hours(i) for i in jobs)), "core_hours_cap": budget["cap"],
             "max_rss": {i: fact(lambda: next((x.split("|")[7] for x in state["sacct"]["stdout"].splitlines()[1:]
                                               if x.split("|")[0] == row[i][0] + ".batch"), None)) for i in jobs},
             "serial_queue_jobs": len([x for x in state["squeue_serial"]["stdout"].splitlines() if x.strip()]),
             "report": fact(lambda: {k: output["report"]["pin"][k] for k in ("sha256", "bytes")})}
    if report is None:
        report = fact(lambda: json.loads(local_copy("report.json").read_text()))
    if report is None:
        return result + [("report del confronto raccolto e integro (presente su Leonardo, copia locale invariata)", False)], facts

    def classification():
        return report["secondary"]["classification"]

    def pose():
        return report["secondary"]["pose"]

    def primary(block):
        contrast = block["contrasts"][PRIMARY]
        return {"delta_pp": contrast["delta_pp"], "ci95": contrast["ci95_pp"]}

    def aggregates_state():
        path = S.ROOT / release["aggregates_state"]["path"]
        require(sha(path) == release["aggregates_state"]["sha256"], "stato degli aggregati diverso da quello del piano")
        return json.loads(path.read_text())

    def same_points():
        """Punto di ogni cella, per dataset e per compito, uguale al `balanced_accuracy` dell'aggregato raccolto."""
        copies, fixed = aggregates_state()["collected"], {p["path"]: p["sha256"] for p in inputs["probe_reports"]}
        plan = S.aggregate_plan()["jobs"]
        for job in plan:
            c = copies[job["id"]]["aggregate.json"]
            path = S.ROOT / c["path"]
            require(c["remote"] == job["out"] and fixed[job["out"]] == c["sha256"] == sha(path), "aggregato raccolto diverso dal pin del manifest")
            aggregate = json.loads(path.read_text())
            pairs = [(classification()["datasets"][ds]["cells"][job["cell"]]["balanced_accuracy"], aggregate["datasets"][ds]["balanced_accuracy"])
                     for ds in F.TASKS[job["task"]]]
            pairs.append((classification()["tasks"][job["task"]]["cells"][job["cell"]]["balanced_accuracy"], aggregate["balanced_accuracy"]))
            if not all(type(a) is float and type(b) is float and abs(a - b) <= TOLERANCE for a, b in pairs):
                return False
        return sorted((job["cell"], job["task"]) for job in plan) == sorted(F.grid())

    def pose_block():
        dynamic = pose()["dynamic"]
        control, candidate = dynamic["cell_ratio"][CONTROL], dynamic["cell_ratio"][CANDIDATE]
        return {"control_E_dyn": control, "candidate_E_dyn": candidate, "delta_E_dyn": candidate - control,
                "ci95": dynamic["contrasts"][PRIMARY]["ci95"]}

    def compare_log():
        lines = local_copy(log_name[S.COMPARE_ID]).read_text().splitlines()
        final = [json.loads(x) for x in lines if x.startswith("{")]
        return (lines.count("codice: %s @ %s" % (S.CODE_DIR, S.CODE_SHA)) == 2
                and final == [{"complete": True, "reading_valid": report["reading"]["reading_valid"], "out": S.COMPARE_OUT}])

    replicates = (BOOTSTRAP["replicates"], BOOTSTRAP["seed"])
    counts = ("replicates_missing_classes", "single_class_replicates")
    result += [
        ("D.1 report: chiavi attese, protocollo, completo, passo 5000, candidato e controllo", passed(
            lambda: set(report) == REPORT_KEYS and report["protocol"] == PROTOCOL and report["complete"] is True
            and type(report["checkpoint_step"]) is int and report["checkpoint_step"] == 5000
            and (report["candidate"], report["control"]) == ("T100-G-300M", "T100-G-30M"))),
        ("D.1 report: `inputs` uguale al manifest inviato, `inputs_content_sha256` ricalcolato", passed(
            lambda: report["inputs"] == inputs and report["inputs_content_sha256"] == jhash(inputs))),
        ("D.1 report: sorgenti della Review 10, codice congelato e firma numerica dei fit", passed(
            lambda: report["sources"] == S.SOURCES and report["runtime"]["code_sha"] == S.CODE_SHA
            and report["runtime"]["probe_numeric_sha256"] == S.NUMERIC_SHA)),
        ("report: runtime della posa e proiezione uguali a quelli raccolti con gli aggregati", passed(
            lambda: all(report["runtime"]["pose_runtime"][k] == aggregates_state()["pose_runtime"][k] is not None
                        for k in ("code_sha", "sources") + tuple(S.RUNTIME_KEYS))
            and {a["fields"]["projection_sha256"] for a in aggregates_state()["aggregates"]} == {report["projection"]["sha256"]})),
        ("D.2 qualita': tre esiti booleani espliciti; lettura valida se e solo se tutti veri", passed(
            lambda: set(report["quality"]) == QUALITY and all(type(v) is bool for v in report["quality"].values())
            and report["reading"]["reading_valid"] is all(report["quality"].values())
            and report["reading"]["source_quality"] == report["quality"])),
        ("D.3 bootstrap: %d repliche e seme %d nella politica, nella classificazione e nei tre gruppi della posa" % replicates, passed(
            lambda: report["bootstrap"] == BOOTSTRAP == report["reading"]["policy"]["bootstrap"]
            and (classification()["n_bootstrap"], classification()["seed"]) == replicates and set(pose()) == set(POSE_GROUPS)
            and all((pose()[g]["n_bootstrap"], pose()[g]["seed"]) == replicates for g in POSE_GROUPS))),
        ("D.4 utenti per dataset uguali alle ancore della Review 10 (%s); conteggi delle repliche interi fra 0 e %d" % (
            ", ".join(str(S.ANCHORS[ds][2]) for ds in sorted(S.ANCHORS)), BOOTSTRAP["replicates"]), passed(
            lambda: set(classification()["datasets"]) == set(S.ANCHORS) and all(
                len(set(classification()["datasets"][ds]["subjects"])) == len(classification()["datasets"][ds]["subjects"]) == S.ANCHORS[ds][2]
                and all(type(classification()["datasets"][ds][k]) is int and 0 <= classification()["datasets"][ds][k] <= BOOTSTRAP["replicates"]
                        for k in counts) for ds in S.ANCHORS))),
        ("utenti della posa nel gruppo dinamico: %d distinti" % POSE_USERS, passed(
            lambda: len(set(pose()["dynamic"]["users"])) == len(pose()["dynamic"]["users"]) == POSE_USERS)),
        ("D.5 punti di ogni cella, per dataset e per compito, uguali ai `balanced_accuracy` dei sei aggregati entro 1e-12", passed(same_points)),
        ("D.5 `epn`, `p1` e `p2` uguali ai contrasti %s" % PRIMARY, passed(
            lambda: report["epn"] == primary(classification()["datasets"]["epn612"])
            and all(report[task] == primary(classification()["tasks"][task]) for task in ("p1", "p2")))),
        ("D.5 blocco `pose` uguale al gruppo dinamico: rapporti delle due celle, differenza e intervallo", passed(
            lambda: report["pose"] == pose_block())),
        ("lettura S8 uguale al ricalcolo con scripts/scaling_reading.py congelato", passed(
            lambda: json.loads(json.dumps(frozen_reading().read_comparison(report), allow_nan=False)) == report["reading"])),
        ("D.6 continuazione non autorizzata, nella lettura e nella politica", passed(
            lambda: report["reading"]["continuation_authorized"] is False
            and report["reading"]["policy"]["continuation_is_authorized"] is False)),
        ("log del confronto: codice congelato prima e dopo, una sola riga finale coerente col report", passed(compare_log)),
    ]
    facts.update({"quality": fact(lambda: {k: report["quality"][k] for k in sorted(QUALITY)}),
                  "reading_valid": fact(lambda: report["reading"]["reading_valid"]),
                  "failed_quality_checks": fact(lambda: report["reading"].get("failed_checks")),
                  "subjects": fact(lambda: {ds: len(v["subjects"]) for ds, v in sorted(classification()["datasets"].items())}),
                  "pose_users": fact(lambda: {g: len(pose()[g]["users"]) for g in POSE_GROUPS}),
                  **{k: fact(lambda: {ds: v[k] for ds, v in sorted(classification()["datasets"].items())}) for k in counts}})
    return result, facts


def cmd_validate(state_path):
    result, facts = checks(json.loads(Path(state_path).read_text()))
    for name, ok in result:
        print("PASSA " if ok else "FALLISCE ", name)
    print("fatti registrati:", json.dumps(facts, sort_keys=True))
    require(all(ok for _, ok in result), "esito del confronto non accettato")
    valid = facts.get("reading_valid") is True
    print("ESITO ACCETTATO: confronto completo e coerente con il manifest inviato; lettura S8 "
          + ("valida" if valid else "NON valida, nessuna lettura scientifica: i numeri restano descrittivi"))


if __name__ == "__main__":
    command = {"collect": cmd_collect, "validate": cmd_validate}.get(sys.argv[1] if len(sys.argv) > 1 else "")
    require(command is not None, __doc__)
    command(*sys.argv[2:])
