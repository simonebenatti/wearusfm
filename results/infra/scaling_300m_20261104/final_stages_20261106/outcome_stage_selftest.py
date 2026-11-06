"""Prova locale di outcome_stage.py sulla scena sintetica dei test del confronto (fixture `scene` di tests/cpu).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 <python del progetto> -c "import sys; sys.path.insert(0, 'src'); \
        import pytest; sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', \
        'results/infra/scaling_300m_20261104/final_stages_20261106/outcome_stage_selftest.py']))"

Catena provata, tutta in cartelle temporanee: secondo stadio fino alle righe d'invio (fixture `staged` di
comparison_stage_selftest.py) -> checkout del manifest come worktree git staccato -> comparatore congelato sul manifest del
checkout -> collect (lo script remoto eseguito in locale, con sacct e squeue finti, ledger e log finti) -> validate. In piu':
stati e report alterati, ognuno rifiutato dal controllo atteso, e un report con lettura non valida ma coerente. Nessuna
connessione, nessun invio, nessuna scrittura nel repository, nessuno score stampato.
Limiti: come nel secondo stadio l'identita' di codice della fixture e' finta ("e"*40) e HEAD e stato del codice congelato sono
sovrascritti nella lettura; la scena ha due utenti della posa, non quindici; i log dei job sono scritti dalla prova.
"""
import copy
import json
import sys
from pathlib import Path

import pytest

STAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE))
from comparison_stage_selftest import CODE, F, M, S, scene, staged  # noqa: E402,F401
import outcome_stage as O  # noqa: E402

ROWS = "JobIDRaw|JobName|State|ExitCode|ElapsedRaw|AllocCPUS|ReqMem|MaxRSS|Start|End|Timelimit"


@pytest.fixture()
def finished(staged, monkeypatch):
    here, tmp, git = staged["here"], staged["tmp"], staged["git"]
    S.cmd_collect("prova")
    S.cmd_manifest(str(here / "aggregates_state_prova.json"))
    git("add", "results")
    git("commit", "-q", "-m", "manifest")
    commit = git("rev-parse", "HEAD")
    plan, lines = here / "comparison_plan.json", here / "comparison_submit_lines.txt"
    S.cmd_lines(str(plan), commit)
    release, checkout, compare, _ = S.release_jobs(plan, lines)
    git("worktree", "add", "-q", "--detach", S.CAMPAIGN_DIR, commit)  # il checkout di campagna: stesso commit, albero pulito
    manifest = json.loads(Path(S.manifest_remote()).read_text())
    M.compare(manifest, Path(S.COMPARE_OUT), CODE)  # il comparatore congelato scrive il report all'uscita del piano
    jobs = {S.CHECKOUT_ID: (92000001, checkout, commit, S.CHECKOUT, 40), S.COMPARE_ID: (92000002, compare, CODE, S.COMPARE, 700)}
    logs, rows, entries = Path(S.LOG_DIR), [ROWS], {}
    (logs / "campaign_checkout_92000001.out").write_text(
        "Preparing worktree (detached HEAD)\ncheckout di campagna pronto: %s @ %s\nfiles: 20; dimensione: 1M\n" % (S.CAMPAIGN_DIR, commit))
    mark = "codice: %s @ %s\n" % (S.CODE_DIR, CODE)
    (logs / "scaling_readers_recovery_92000002.out").write_text(
        mark + "Loading cineca-ai/4.3.0\n" + json.dumps({"complete": True, "reading_valid": True, "out": S.COMPARE_OUT}) + "\n" + mark)
    (here / "submissions").mkdir()
    for job_id, (slurm, job, code, spec, elapsed) in jobs.items():
        limit = S.clock(spec["seconds"]).split("=")[1]
        rows += ["%d|%s|COMPLETED|0:0|%d|%d|%dG||2026-11-06T15:00:00|2026-11-06T15:10:00|%s" % (
                     slurm, job_id, elapsed, spec["cpus"], spec["memory_gib"], limit),
                 "%d.batch|batch|COMPLETED|0:0|%d|%d||%dK|2026-11-06T15:00:00|2026-11-06T15:10:00|" % (slurm, elapsed, spec["cpus"], 900000),
                 "%d.extern|extern|COMPLETED|0:0|%d|%d|||2026-11-06T15:00:00|2026-11-06T15:10:00|" % (slurm, elapsed, spec["cpus"])]
        entries[job_id] = {"stato": "sottomesso", "job_slurm": slurm, "agente": "claude", "campagna": S.CAMPAIGN, "codice_sha": code,
                           "gpu_ore_stimate": 0.0, "sbatch": O.expected_sbatch(job), "script_stdin_sha256": job["script"]["sha256"]}
        (here / "submissions" / (job_id + ".json")).write_text(json.dumps(dict(entries[job_id], creato="2026-11-06T15:00:00")))
    (tmp / "bin" / "sacct").write_text("#!/bin/bash\ncat <<'EOF'\n" + "\n".join(rows) + "\nEOF\n")
    previous = F.ledger
    monkeypatch.setattr(F, "ledger", lambda i: entries.get(i) or previous(i))
    monkeypatch.setattr(O, "POSE_USERS", 2)
    return dict(staged, plan=plan, lines=lines, commit=commit, entries=entries)


def failing(state, report=None):
    return [name for name, ok in O.checks(state, report)[0] if not ok]


def test_collect_and_validate_the_frozen_comparison(finished, capsys):
    here = finished["here"]
    O.cmd_collect(str(finished["plan"]), str(finished["lines"]), "prova")
    state_path, folder = here / "outcome_state_prova.json", here / "collected_outcome_prova"
    state = json.loads(state_path.read_text())
    assert '"content":' not in json.dumps(state) and state["commit"] == finished["commit"]
    assert sorted(p.name for p in folder.iterdir()) == ["campaign_checkout_92000001.out", "report.json", "sacct.txt",
                                                        "scaling_readers_recovery_92000002.out"]
    assert (folder / "report.json").read_bytes() == Path(S.COMPARE_OUT).read_bytes()
    assert (folder / "sacct.txt").read_text().count("COMPLETED") == 6
    capsys.readouterr()
    O.cmd_validate(str(state_path))
    out = capsys.readouterr().out
    assert "FALLISCE" not in out and out.count("PASSA ") == 28 and "ESITO ACCETTATO" in out and "lettura S8 valida" in out
    report = json.loads((folder / "report.json").read_text())
    shown = out.split("fatti registrati:")[1]
    hidden = [report["epn"]["delta_pp"], report["pose"]["delta_E_dyn"], report["pose"]["control_E_dyn"], report["p1"]["ci95"][0],
              report["reading"]["recommendation"], report["reading"]["pose_relative_point_reduction"]]
    hidden = [str(v) for v in hidden if len(str(v)) >= 8]  # valori corti come 0.0 comparirebbero per caso
    assert len(hidden) >= 3 and not any(v in out for v in hidden)  # nessuno score, contrasto o raccomandazione nell'uscita
    facts = json.loads(shown.splitlines()[0])
    assert facts["reading_valid"] is True and facts["quality"] == {"artifact_integrity": True, "paired_support": True, "solver_converged": True}
    assert facts["elapsed_seconds"] == {S.CHECKOUT_ID: 40, S.COMPARE_ID: 700} and facts["max_rss"][S.COMPARE_ID] == "900000K"
    assert abs(facts["core_hours"][S.COMPARE_ID] - 8 * 700 / 3600) < 1e-12 and facts["pose_users"] == {"dynamic": 2, "all": 2, "static": 2}
    assert O.jhash(report["inputs"]) == M.C.jhash(report["inputs"]) == report["inputs_content_sha256"]
    with pytest.raises(FileExistsError):  # nessuna sovrascrittura
        O.cmd_collect(str(finished["plan"]), str(finished["lines"]), "prova")


def test_validation_refuses_altered_states_and_reports(finished):
    here = finished["here"]
    O.cmd_collect(str(finished["plan"]), str(finished["lines"]), "prova")
    base = json.loads((here / "outcome_state_prova.json").read_text())
    report = json.loads((here / "collected_outcome_prova" / "report.json").read_text())
    assert failing(base) == [] and failing(base, copy.deepcopy(report)) == []

    def edit(value, change):
        value = copy.deepcopy(value)
        change(value)
        return value

    cls = lambda r: r["secondary"]["classification"]  # noqa: E731
    first = sorted(cls(report)["datasets"])[0]
    states = [
        ("contabilita'", lambda s: s["sacct"].__setitem__("stdout", s["sacct"]["stdout"].replace("COMPLETED|0:0|700|8", "FAILED|1:0|700|8"))),
        ("contabilita'", lambda s: s["sacct"].__setitem__("stdout", s["sacct"]["stdout"].replace("COMPLETED|0:0|700|8", "TIMEOUT|0:0|1801|8"))),
        ("contabilita'", lambda s: s["sacct"].__setitem__("stdout", s["sacct"]["stdout"].replace("|700|8|30G|", "|700|4|30G|"))),
        ("contabilita'", lambda s: s["sacct"].__setitem__("stdout", s["sacct"]["stdout"].replace("|40|2|4G|", "|40|2|8G|"))),
        ("ledger: checkout e confronto", lambda s: s["ledger"][S.COMPARE_ID].__setitem__("agente", "gpt")),
        ("ledger: checkout e confronto", lambda s: s["ledger"][S.CHECKOUT_ID].__setitem__("codice_sha", CODE)),
        ("ledger: checkout e confronto", lambda s: s["ledger"][S.COMPARE_ID]["sbatch"].append("--dependency=afterok:1")),
        ("ricevute d'invio", lambda s: s["ledger"][S.COMPARE_ID].__setitem__("job_slurm", 1)),
        ("codice congelato", lambda s: s["code"]["status"].__setitem__("stdout", " M scripts/scaling_compare.py")),
        ("sei sorgenti", lambda s: next(iter(s["sources"].values())).__setitem__("sha256", "0" * 64)),
        ("checkout del manifest", lambda s: s["campaign"]["head"].__setitem__("stdout", "f" * 40)),
        ("checkout del manifest", lambda s: s["campaign"]["status"].__setitem__("stdout", "?? nuovo")),
        ("manifest nel checkout", lambda s: s["campaign"]["manifest"].__setitem__("sha256", "0" * 64)),
        ("i nove file", lambda s: s["pins"][4].__setitem__("sha256", "0" * 64)),
        ("i nove file", lambda s: s["pins"].pop()),
        ("cartella d'uscita", lambda s: s["output"]["files"].append("report.json.tmp")),
        ("file raccolti sul Mac", lambda s: s["output"]["report"]["pin"].__setitem__("sha256", "0" * 64)),
        ("file raccolti sul Mac", lambda s: s["collected"].pop("campaign_checkout_92000001.out")),
        ("budget", lambda s: s["sacct"].__setitem__("stdout", s["sacct"]["stdout"].replace("|700|8|", "|1990|8|"))),
    ]
    for expected, change in states:
        names = failing(edit(base, change))
        assert names and any(expected in n for n in names), (expected, names)
    reports = [
        ("chiavi attese", lambda r: r.__setitem__("complete", False)),
        ("chiavi attese", lambda r: r.__setitem__("checkpoint_step", 19618)),
        ("chiavi attese", lambda r: r.__setitem__("extra", 1)),
        ("`inputs` uguale", lambda r: r["inputs"]["probe_reports"][0].__setitem__("sha256", "0" * 64)),
        ("`inputs` uguale", lambda r: r.__setitem__("inputs_content_sha256", "0" * 64)),
        ("sorgenti della Review 10", lambda r: r["sources"].__setitem__("scripts/scaling_compare.py", "0" * 64)),
        ("sorgenti della Review 10", lambda r: r["runtime"].__setitem__("probe_numeric_sha256", "0" * 64)),
        ("runtime della posa", lambda r: r["runtime"]["pose_runtime"].__setitem__("versions", {"python": "0"})),
        ("runtime della posa", lambda r: r["projection"].__setitem__("sha256", "0" * 64)),
        ("D.2 qualita'", lambda r: r["quality"].__setitem__("solver_converged", False)),
        ("D.2 qualita'", lambda r: r["quality"].__setitem__("paired_support", 1)),
        ("D.3 bootstrap", lambda r: cls(r).__setitem__("n_bootstrap", 1999)),
        ("D.3 bootstrap", lambda r: r["secondary"]["pose"]["static"].__setitem__("seed", 0)),
        ("D.3 bootstrap", lambda r: r["bootstrap"].__setitem__("replicates", 1000)),
        ("D.4 utenti per dataset", lambda r: cls(r)["datasets"][first]["subjects"].pop()),
        ("D.4 utenti per dataset", lambda r: cls(r)["datasets"][first].__setitem__("single_class_replicates", None)),
        ("utenti della posa", lambda r: r["secondary"]["pose"]["dynamic"]["users"].append("v9")),
        ("D.5 punti", lambda r: cls(r)["datasets"][first]["cells"][O.CONTROL].__setitem__(
            "balanced_accuracy", cls(r)["datasets"][first]["cells"][O.CONTROL]["balanced_accuracy"] + 1e-9)),
        ("D.5 punti", lambda r: cls(r)["tasks"]["p2"]["cells"][O.CANDIDATE].__setitem__(
            "balanced_accuracy", cls(r)["tasks"]["p2"]["cells"][O.CANDIDATE]["balanced_accuracy"] - 1e-9)),
        ("D.5 `epn`", lambda r: r["epn"].__setitem__("delta_pp", r["epn"]["delta_pp"] + 1e-9)),
        ("D.5 `epn`", lambda r: r["p1"]["ci95"].__setitem__(0, r["p1"]["ci95"][0] - 1e-9)),
        ("D.5 blocco `pose`", lambda r: r["pose"].__setitem__("candidate_E_dyn", r["pose"]["candidate_E_dyn"] + 1e-9)),
        ("lettura S8", lambda r: r["reading"].__setitem__("recommendation", "propose_19618_confirmation_now")),
        ("lettura S8", lambda r: r["reading"]["classification_noninferiority"].__setitem__("p1", "altro")),
        ("D.6 continuazione", lambda r: r["reading"].__setitem__("continuation_authorized", True)),
        ("D.6 continuazione", lambda r: r["reading"]["policy"].__setitem__("continuation_is_authorized", True)),
    ]
    for expected, change in reports:
        names = failing(base, edit(report, change))
        assert names and any(expected in n for n in names), (expected, names)
    assert failing(base, {"protocol": O.PROTOCOL}) and failing(base, [])  # un report malformato fa fallire i controlli, non lo strumento

    def unresolved(r):  # lettura non valida ma coerente: come la scriverebbe il comparatore congelato con un solutore irrisolto
        r["quality"]["solver_converged"] = False
        r["reading"] = json.loads(json.dumps(O.frozen_reading().read_comparison(r)))

    result, facts = O.checks(base, edit(report, unresolved))
    assert [name for name, ok in result if not ok] == ["log del confronto: codice congelato prima e dopo, una sola riga finale coerente col report"]
    assert facts["reading_valid"] is False and facts["failed_quality_checks"] == ["solver_converged"]

    log = here / "collected_outcome_prova" / "scaling_readers_recovery_92000002.out"
    original = log.read_bytes()
    log.write_bytes(original.replace(b'"reading_valid": true', b'"reading_valid": false'))
    assert any("file raccolti sul Mac" in n for n in failing(base)) and any("log del confronto" in n for n in failing(base))
    log.write_bytes(original)
    ready = here / "collected_outcome_prova" / "campaign_checkout_92000001.out"
    original = ready.read_bytes()
    ready.write_bytes(original.replace(b"pronto", b"NON pronto"))
    assert any("log del checkout" in n for n in failing(base))
    ready.write_bytes(original)
    copy_path = here / "collected_outcome_prova" / "report.json"
    original = copy_path.read_bytes()
    copy_path.write_bytes(original + b" ")
    assert any("file raccolti sul Mac" in n for n in failing(base)) and any("report del confronto raccolto" in n for n in failing(base))
    copy_path.write_bytes(original)
    receipt = here / "submissions" / (S.COMPARE_ID + ".json")
    original = receipt.read_text()
    receipt.write_text(original.replace('"agente": "claude"', '"agente": "gpt"'))
    assert any("ricevute d'invio" in n for n in failing(base))
    receipt.write_text(original)
    assert failing(base) == []
    for name in ("squeue_serial", "sacct"):
        with pytest.raises(SystemExit) as refused:
            O.checks(edit(base, lambda s: s[name].update(exit=1, stdout="")))
        assert "non riuscita" in str(refused.value)
    with pytest.raises(SystemExit):  # stato raccolto con un altro piano
        O.checks(edit(base, lambda s: s["local"]["plan"].__setitem__("sha256", "0" * 64)))
    missing = edit(base, lambda s: (s["collected"].pop("report.json"), s["output"]["report"].__setitem__("pin", None), s["output"].__setitem__("files", [])))
    names = failing(missing)
    assert any("report del confronto raccolto" in n for n in names) and any("cartella d'uscita" in n for n in names)
