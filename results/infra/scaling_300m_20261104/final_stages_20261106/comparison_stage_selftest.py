"""Prova locale di comparison_stage.py sulla scena sintetica dei test del confronto (fixture `scene` di tests/cpu).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 <python del progetto> -c "import sys; sys.path.insert(0, 'src'); \
        import pytest; sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', \
        'results/infra/scaling_300m_20261104/final_stages_20261106/comparison_stage_selftest.py']))"

Catena provata, tutta in cartelle temporanee: collect (lo script remoto eseguito in locale, con sacct e squeue finti) ->
validate -> manifest -> frozen -> comparatore congelato sul manifest prodotto, con la stessa lettura del manifest della
fixture -> lines (in un repository git temporaneo) -> submitter reale con ssh finto. In piu': oltre venti stati alterati, ognuno
rifiutato dal controllo atteso. Nessuna connessione, nessun invio, nessuna scrittura nel repository, nessuno score stampato.
Limiti: l'identita' di codice della fixture e' finta ("e"*40), quindi HEAD e stato del checkout del codice sono sovrascritti
nella lettura; i cancelli che leggono Leonardo (pre-deploy, gate-checkout, gate-compare) non sono provati qui.
"""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

STAGE = Path(__file__).resolve().parent
WORKTREE = STAGE.parents[3]
sys.path.insert(0, str(WORKTREE / "tests/cpu"))
sys.path.insert(0, str(STAGE))
from test_scaling_compare import CODE, M, scene  # noqa: E402,F401
import comparison_stage as S  # noqa: E402
import final_stages as F  # noqa: E402


@pytest.fixture()
def staged(scene, tmp_path, monkeypatch):
    root, inputs, mp = Path(scene["root"]), scene["inputs"], monkeypatch
    repo = (tmp_path / "repo").resolve()
    here = repo / "results/infra/scaling_300m_20261104/final_stages_20261106"
    here.mkdir(parents=True)
    for name in ("scripts", "src"):
        (repo / name).symlink_to(WORKTREE / name)
    git = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.name=prova", "-c", "user.email=prova@example.invalid", *a],  # noqa: E731
                                    check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "base")
    jobs = []
    for n, (cell, task) in enumerate(F.grid(), 1):
        out, shard = root / (cell + task + "-aggregate.json"), root / (cell + task + "-shard") / "shard.json"
        jobs.append({"id": "SCALING-probe-aggregate-%02d-20261106" % n, "cell": cell, "task": task, "out": str(out),
                     "sidecars": F.sidecars(str(out)),
                     "shards": [{"name": cell + task, "path": str(shard), "sha256": F.sha(shard), "bytes": shard.stat().st_size}]})
    plan_path = here / "aggregate_submission_plan.json"
    plan_path.write_text(json.dumps({"jobs": jobs}))
    bench = {}
    for cell in F.CELLS:
        path = tmp_path / (cell + "-bench.json")
        path.write_text(json.dumps({"runtime": scene["pose_report"]["runtime"]}))
        bench[cell] = {"path": str(path), "sha256": F.sha(path)}
    budget = tmp_path / "pose_budget.json"
    budget.write_text(json.dumps({"benchmarks": bench}))
    logs, fakebin = tmp_path / "logs", tmp_path / "bin"
    logs.mkdir()
    fakebin.mkdir()
    ledger, rows = {}, ["JobIDRaw|JobName|State|ExitCode|ElapsedRaw|AllocCPUS|ReqMem|MaxRSS|Start|End|Timelimit"]
    for n, j in enumerate(jobs, 1):
        slurm = 91000000 + n
        ledger[j["id"]] = {"stato": "sottomesso", "job_slurm": slurm, "agente": "claude", "campagna": S.CAMPAIGN}
        (logs / ("scaling_readers_recovery_%d.out" % slurm)).write_text('{"complete": true, "kind": "aggregate"}\n')
        rows += ["%d|%s|COMPLETED|0:0|%d|2|30G||2026-11-06T13:00:00|2026-11-06T13:01:00|00:10:00" % (slurm, j["id"], 40 + n),
                 "%d.batch|batch|COMPLETED|0:0|%d|2||%dK|2026-11-06T13:00:00|2026-11-06T13:01:00|" % (slurm, 40 + n, 500000 + n),
                 "%d.extern|extern|COMPLETED|0:0|%d|2||0|2026-11-06T13:00:00|2026-11-06T13:01:00|" % (slurm, 40 + n)]
    (fakebin / "sacct").write_text("#!/bin/bash\ncat <<'EOF'\n" + "\n".join(rows) + "\nEOF\n")
    (fakebin / "squeue").write_text("#!/bin/bash\nexit 0\n")
    for name in ("sacct", "squeue"):
        (fakebin / name).chmod(0o755)
    first = [json.loads(Path(j["out"]).read_text()) for j in jobs[:2]]  # prima cella, p1 e p2: solo campi d'identita'
    anchors = {ds: (v["identity_sha256"], v["n_oof_windows"], v["n_oof_subjects"]) for r in first for ds, v in r["datasets"].items()}

    def local_remote(template, payload):
        script = template.replace("__PAYLOAD__", repr(json.dumps(payload)))
        r = subprocess.run([sys.executable, "-I", "-"], input=script, capture_output=True, text=True,
                           env=dict(os.environ, PATH=str(fakebin) + ":" + os.environ["PATH"], USER="prova"))
        assert r.returncode == 0, r.stderr[-2000:]
        value = json.loads(r.stdout)
        if "code" in value:  # limite della prova: la fixture usa un'identita' di codice finta
            value["code"]["head"] = {"exit": 0, "stdout": CODE, "stderr": ""}
            value["code"]["status"] = {"exit": 0, "stdout": "", "stderr": ""}
        return value, hashlib.sha256(script.encode()).hexdigest()

    for module in (S, F):
        mp.setattr(module, "ROOT", repo)
        mp.setattr(module, "HERE", here)
    for name, value in (("PLAN", plan_path), ("PLAN_SHA", F.sha(plan_path)), ("CODE_SHA", CODE), ("CODE_DIR", str(repo)),
                        ("POSE_PLAN", inputs["pose_plan"]["path"]), ("POSE_PLAN_SHA", inputs["pose_plan"]["sha256"]),
                        ("POSE_REPORT", inputs["pose_report"]["path"]), ("POSE_REPORT_SHA", inputs["pose_report"]["sha256"]),
                        ("POSE_BUDGET", str(budget)), ("POSE_BUDGET_SHA", F.sha(budget)), ("RUNTIME_KEYS", ("versions", "threadpools")),
                        ("NUMERIC_SHA", first[0]["numeric_sha256"]), ("ANCHORS", anchors), ("EXCLUSIVE_FOLDERS", False), ("N_SHARDS", 6),
                        ("LOG_DIR", str(logs)), ("COMPARE_OUT", str(tmp_path / "comparison" / "report.json")),
                        ("CAMPAIGN_DIR", str(tmp_path / "campaign")), ("MANIFEST", here / "comparison_inputs.json"),
                        ("CHECKOUT_SCRIPT", repo / "scripts/slurm/campaign_checkout.sbatch")):
        mp.setattr(S, name, value)
    for name, value in (("PROBE_PLAN", inputs["probe_plan"]["path"]), ("PROBE_PLAN_SHA", inputs["probe_plan"]["sha256"]),
                        ("DISPATCHER", repo / "scripts/slurm/scaling_readers_recovery.sbatch"), ("SUBMITTER", repo / "scripts/submit_job.py"),
                        ("ledger", lambda i: ledger.get(i)), ("remote", local_remote)):
        mp.setattr(F, name, value)
    return {"repo": repo, "here": here, "git": git, "jobs": jobs, "inputs": inputs, "tmp": tmp_path}


def test_chain_from_collection_to_frozen_comparison_and_commands(staged, capsys):
    here, inputs, tmp = staged["here"], staged["inputs"], staged["tmp"]
    S.cmd_collect("prova")
    state_path = here / "aggregates_state_prova.json"
    state = json.loads(state_path.read_text())
    assert '"content":' not in json.dumps(state) and len(state["collected"]) == 6
    assert all(len(files) == 3 for files in state["collected"].values())
    assert (here / "collected_aggregates_prova" / "sacct.txt").read_text().count("COMPLETED") == 18
    S.cmd_validate(str(state_path))
    assert "AGGREGATI ACCETTATI" in capsys.readouterr().out
    S.cmd_manifest(str(state_path))
    manifest = json.loads((here / "comparison_inputs.json").read_text())
    strip = lambda p: {k: p[k] for k in ("path", "sha256")}  # noqa: E731  la fixture non fissa le dimensioni
    assert list(manifest) == list(inputs) and manifest["sources"] == inputs["sources"] and manifest["protocol"] == inputs["protocol"]
    assert [strip(p) for p in manifest["probe_reports"]] == inputs["probe_reports"]
    assert all(strip(manifest[k]) == inputs[k] for k in ("probe_plan", "pose_plan", "pose_report"))
    assert all(set(p) == {"path", "sha256", "bytes"} for p in [manifest["probe_plan"], manifest["pose_plan"], manifest["pose_report"],
                                                               *manifest["probe_reports"]])
    S.cmd_frozen(str(here / "comparison_inputs.json"))
    result = M.compare(manifest, tmp / "summary.json", CODE)
    reference = M.compare(inputs, tmp / "reference.json", CODE)
    assert result["complete"] is True and result["quality"] == {"artifact_integrity": True, "paired_support": True, "solver_converged": True}
    assert result["inputs_content_sha256"] == M.C.jhash(manifest) and result["inputs"] == manifest
    keys = ("pose", "epn", "p1", "p2", "reading", "quality", "secondary", "checkpoints")
    assert {k: result[k] for k in keys} == {k: reference[k] for k in keys}  # stessa lettura, nessun valore stampato
    release = json.loads((here / "comparison_plan.json").read_text())
    assert release["status"] == "PREPARED_NOT_SUBMITTED" and all(c["passed"] for c in release["integrity_checks"])
    checkout, compare = release["jobs"]
    assert not checkout["id"].startswith("SCALING-compare-") and compare["id"].startswith("SCALING-compare-")
    assert compare["argv"][-4:] == ["compare", S.manifest_remote(), release["manifest"]["sha256"], S.COMPARE_OUT]
    assert release["manifest"]["sha256"] == F.sha(here / "comparison_inputs.json")
    budget = release["budget_core_hours"]
    assert abs(budget["worst_case"] - (budget["consumed"] + 2 * 480 / 3600 + 8 * 1980 / 3600)) < 1e-9 and budget["worst_case"] < 310
    git = staged["git"]
    git("add", "results")
    git("commit", "-q", "-m", "manifest")
    commit = git("rev-parse", "HEAD")
    S.cmd_lines(str(here / "comparison_plan.json"), commit)
    lines = (here / "comparison_submit_lines.txt").read_text().splitlines()
    assert len(lines) == 2 and ("CAMPAIGN_SHA=%s,EXPECTED_SHA=%s" % (commit, commit)) in lines[0] and "<COMMIT>" not in lines[0]
    assert lines[0].endswith("/dev/stdin") and " compare " in lines[1] and "--dopo" not in "".join(lines)
    S.cmd_submitter(str(here / "comparison_plan.json"), str(here / "comparison_submit_lines.txt"))
    dry = json.loads((here / "comparison_submitter_dryrun.json").read_text())
    assert [c["codice_sha"] for c in dry["sbatch_commands"]] == [commit, CODE]
    assert "--cpus-per-task=2 --mem=4G --time=00:05:00" in dry["sbatch_commands"][0]["sbatch"]
    assert "--cpus-per-task=8 --mem=30G --time=00:30:00" in dry["sbatch_commands"][1]["sbatch"]
    with pytest.raises(SystemExit):  # un commit che non contiene il manifest non genera righe
        S.cmd_lines(str(here / "comparison_plan.json"), git("rev-parse", "HEAD~1"))
    with pytest.raises(FileExistsError):  # nessuna sovrascrittura
        S.cmd_manifest(str(state_path))


def failing(state):
    return [name for name, ok in S.checks(state)[0] if not ok]


def test_validation_refuses_altered_states(staged):
    here = staged["here"]
    S.cmd_collect("prova")
    base = json.loads((here / "aggregates_state_prova.json").read_text())
    assert failing(base) == []

    def edit(change):
        state = copy.deepcopy(base)
        change(state)
        return state

    cases = [
        ("shard dei report", lambda s: s["aggregates"][2]["fields"]["shards"][0].__setitem__("sha256", "0" * 64)),
        ("shard dei fit", lambda s: s["shards"][0]["shard"].__setitem__("sha256", "0" * 64)),
        ("identita' degli OOF", lambda s: next(iter(s["aggregates"][4]["datasets"].values())).__setitem__("identity_sha256", "0" * 64)),
        ("identita' degli OOF", lambda s: next(iter(s["aggregates"][1]["datasets"].values())).__setitem__("n_oof_subjects", 1)),
        ("contabilita'", lambda s: s["sacct"].__setitem__("stdout", s["sacct"]["stdout"].replace("COMPLETED|0:0|43", "FAILED|1:0|43"))),
        ("solutore", lambda s: s["aggregates"][0]["fields"].__setitem__("reading_status", "solver_unresolved")),
        ("solutore", lambda s: s["aggregates"][3].__setitem__("unresolved_fits", 1)),
        ("una sola firma numerica", lambda s: s["aggregates"][5]["fields"].__setitem__("numeric_sha256", "0" * 64)),
        ("report completi", lambda s: s["aggregates"][1]["fields"].__setitem__("code_sha", "f" * 40)),
        ("hash interni dei report", lambda s: s["aggregates"][0]["oof"].__setitem__("sha256", "0" * 64)),
        ("fit nell'audit", lambda s: s["aggregates"][0].__setitem__("audit_lines", 83)),
        ("dataset di ogni report", lambda s: next(iter(s["aggregates"][0]["datasets"].values())).__setitem__("folds", [0, 1, 2])),
        ("uscita del confronto", lambda s: s["absent"][0].__setitem__("exists", True)),
        ("ID logici del checkout", lambda s: s["ledger"].__setitem__(S.COMPARE_ID, {"stato": "sottomesso"})),
        ("runtime della ridge", lambda s: s["pose_runtime"].__setitem__("versions", {"python": "0"})),
        ("sei sorgenti", lambda s: next(iter(s["sources"].values())).__setitem__("sha256", "0" * 64)),
        ("codice congelato", lambda s: s["code"]["status"].__setitem__("stdout", " M scripts/scaling_probe.py")),
        ("ledger: sei aggregati", lambda s: s["ledger"][staged["jobs"][0]["id"]].__setitem__("agente", "gpt")),
        ("cella, compito", lambda s: s["aggregates"][0]["fields"].__setitem__("cell", "T100-G-300M-natural")),
    ]
    for expected, change in cases:
        names = failing(edit(change))
        assert names and any(expected in n for n in names), (expected, names)
    missing = edit(lambda s: [s["aggregates"][5].pop(k) for k in ("report", "oof", "audit", "fields", "datasets", "audit_summary")])
    assert any("sei report presenti" in n for n in failing(missing))
    altered = here / base["collected"][staged["jobs"][0]["id"]]["aggregate.json"]["path"].split("final_stages_20261106/")[1]
    original = altered.read_bytes()
    altered.write_bytes(original + b" ")
    assert any("file raccolti sul Mac" in n for n in failing(base))
    altered.write_bytes(original)
    for name in ("squeue_serial", "sacct"):
        with pytest.raises(SystemExit) as refused:
            S.checks(edit(lambda s: s[name].update(exit=1, stdout="")))
        assert "non riuscita" in str(refused.value)
    with pytest.raises(SystemExit):
        S.cmd_manifest(str(_write(here / "alterato.json", edit(cases[0][1]))))
    assert not (here / "comparison_inputs.json").exists() and not (here / "comparison_plan.json").exists()


def _write(path, value):
    path.write_text(json.dumps(value))
    return path
