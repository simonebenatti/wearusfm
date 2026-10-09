"""`scripts/submit_job.py` (decisione del 09/10/2026): prenotazione con lock prima di sbatch, stati persistenti, nessun rilancio alla cieca,
riconciliazione degli ambigui. Il login node e' simulato da un finto `ssh` che registra le chiamate."""

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("submit_job", Path(__file__).resolve().parents[2] / "scripts" / "submit_job.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

FAKE = r'''
import sys, time, pathlib
mode, calls = sys.argv[1], pathlib.Path(sys.argv[2])
remote = sys.argv[3]
with calls.open("a") as f:
    f.write(remote + "\n")
if mode == "ok":
    print("12345")
elif mode == "stdin":
    (calls.parent / "stdin.txt").write_text(sys.stdin.read()); print("777")
elif mode == "reject":
    print("sbatch: error: QOSMinCpuNotSatisfied", file=sys.stderr); sys.exit(1)
elif mode == "drop":
    print("Connection reset", file=sys.stderr); sys.exit(255)
elif mode == "hang":
    time.sleep(5)
elif mode == "garbage":
    print("Submitted batch job 12345")
else:  # riconciliazione: sacct e squeue arrivano in due chiamate separate
    import re
    name = re.search(r"--name=(\S+)", remote).group(1)
    acct, queue = remote.startswith("sacct"), remote.startswith("squeue")
    if mode == "found":
        print(f"12345|{name}|PENDING")
    elif mode == "queue_only" and queue:
        print(f"12345|{name}|PENDING")
    elif mode == "two" and acct:
        print(f"12345|{name}|PENDING"); print(f"12346|{name}|RUNNING")
    elif mode == "acct_fail" and acct:
        print("sacct: error: slurmdbd non raggiungibile", file=sys.stderr); sys.exit(1)
    elif mode == "acct_garbage" and acct:
        print("qualcosa di inatteso")
    elif mode == "other_name" and acct:
        print("12345|un_altro_job|PENDING")
'''


@pytest.fixture
def env(tmp_path):
    fake = tmp_path / "fake_ssh.py"
    fake.write_text(FAKE)
    calls = tmp_path / "calls.txt"

    def run(mode, *argv, timeout="10"):
        ssh = f"{sys.executable} {fake} {mode} {calls}"
        return S.main(["--registro", str(tmp_path / "jobs"), "--ssh", ssh, "--timeout", timeout, *argv])

    def state(job_id):
        return json.loads((tmp_path / "jobs" / job_id / "state.json").read_text())

    def n_calls():
        return len(calls.read_text().splitlines()) if calls.exists() else 0

    return run, state, n_calls, tmp_path


BASE = ["--id", "prova_01", "--agente", "claude", "--campagna", "infra", "--gpu-ore", "0.5"]
SB = ["--", "--export=ALL,CODE_DIR=/x,EXPECTED_SHA=" + "a" * 40, "scripts/slurm/sanity_jepa.sbatch", "m.json.gz"]


def test_submitted_once_and_never_again(env):
    run, state, n_calls, tmp = env
    assert run("ok", *BASE, *SB) == 0
    s = state("prova_01")
    assert s["stato"] == "sottomesso" and s["job_slurm"] == 12345 and s["codice_sha"] == "a" * 40
    assert "--job-name=prova_01" in s["sbatch"] and "--comment=wearusfm:prova_01" in s["sbatch"]
    assert [h["stato"] for h in s["storia"]] == ["prenotato", "in_sottomissione", "sottomesso"]
    with pytest.raises(SystemExit):  # stesso ID: mai una seconda sottomissione
        run("ok", *BASE, *SB)
    assert n_calls() == 1
    ledger = (tmp / "jobs" / "ledger.tsv").read_text().splitlines()
    assert ledger[0].startswith("quando\tid") and ledger[-1].split("\t")[4] == "sottomesso"
    assert not (tmp / "jobs" / "prova_01" / "operazione.lock").exists()


def test_rejected_needs_explicit_resubmission(env):
    run, state, n_calls, _ = env
    assert run("reject", *BASE, *SB) == 3
    assert state("prova_01")["stato"] == "rifiutato"
    with pytest.raises(SystemExit):
        run("ok", *BASE, *SB)
    assert run("ok", *BASE, "--dopo-rifiuto", *SB) == 0
    assert state("prova_01")["stato"] == "sottomesso" and n_calls() == 2


@pytest.mark.parametrize("mode,timeout", [("drop", "10"), ("hang", "1"), ("garbage", "10")])
def test_ambiguous_is_never_resubmitted(env, mode, timeout):
    run, state, n_calls, _ = env
    assert run(mode, *BASE, *SB, timeout=timeout) == 3
    assert state("prova_01")["stato"] == "ambiguo"
    for extra in ([], ["--dopo-rifiuto"]):
        with pytest.raises(SystemExit):
            run("ok", *BASE, *extra, *SB)
    assert n_calls() == 1


def test_reconcile_finds_the_job_in_accounting_or_queue(env):
    run, state, _, _ = env
    run("drop", *BASE, *SB)
    assert run("found", "--riconcilia", "prova_01") == 0
    s = state("prova_01")
    assert s["stato"] == "sottomesso" and s["job_slurm"] == 12345 and s["riconciliato"]
    second = ["--id", "prova_02", "--agente", "gpt", "--campagna", "infra", "--gpu-ore", "0"]
    run("drop", *second, *SB)
    assert run("queue_only", "--riconcilia", "prova_02") == 0 and state("prova_02")["job_slurm"] == 12345


def test_reconcile_duplicates_are_never_resubmitted(env):
    run, state, n_calls, _ = env
    run("drop", *BASE, *SB)
    assert run("two", "--riconcilia", "prova_01") == 3 and state("prova_01")["job_slurm_tutti"] == [12345, 12346]
    for extra in ([], ["--dopo-rifiuto"], ["--dopo-assenza-verificata"]):
        with pytest.raises(SystemExit):
            run("ok", *BASE, *extra, *SB)


@pytest.mark.parametrize("mode", ["acct_fail", "acct_garbage", "other_name"])
def test_unreliable_accounting_changes_nothing(env, mode):
    run, state, _, _ = env
    run("drop", *BASE, *SB)
    assert run(mode, "--riconcilia", "prova_01") == 3
    s = state("prova_01")
    assert s["stato"] == "ambiguo" and "controlli_assenza" not in s  # nessuna conclusione da una risposta non affidabile
    with pytest.raises(SystemExit):
        run("ok", *BASE, "--dopo-assenza-verificata", *SB)


def test_empty_answers_need_two_spaced_checks(env, monkeypatch):
    run, state, n_calls, _ = env
    t0 = dt.datetime(2026, 10, 9, 16, 0, tzinfo=dt.timezone.utc)
    clock = [t0]
    monkeypatch.setattr(S, "_clock", lambda: clock[0])
    run("drop", *BASE, *SB)  # tentativo a t0, risposta persa
    for minutes, expected in ((1, "ambiguo"), (11, "ambiguo"), (15, "ambiguo"), (22, "assente")):
        clock[0] = t0 + dt.timedelta(minutes=minutes)
        run("none", "--riconcilia", "prova_01")
        assert state("prova_01")["stato"] == expected, minutes
    # 1 min: troppo presto; 11: un solo controllo valido; 15: due, ma a 4 min; 22: due validi a 11 min l'uno dall'altro
    assert len(state("prova_01")["controlli_assenza"]) == 4
    with pytest.raises(SystemExit):  # --dopo-rifiuto non basta per un assente
        run("ok", *BASE, "--dopo-rifiuto", *SB)
    assert run("ok", *BASE, "--dopo-assenza-verificata", *SB) == 0
    assert state("prova_01")["stato"] == "sottomesso"


def test_dependency_and_input_checks(env):
    run, state, _, tmp = env
    run("ok", *BASE, *SB)
    nxt = ["--id", "prova_02", "--agente", "claude", "--campagna", "infra", "--gpu-ore", "0.5", "--dopo", "prova_01"]
    assert run("ok", *nxt, *SB) == 0
    assert "--dependency=afterok:12345" in state("prova_02")["sbatch"]
    with pytest.raises(SystemExit):  # dipendenza da un job non sottomesso
        run("ok", "--id", "prova_03", "--agente", "claude", "--campagna", "infra", "--gpu-ore", "0", "--dopo", "mai_visto", *SB)
    with pytest.raises(SystemExit):  # nome e commento li impone lo script
        run("ok", "--id", "prova_04", "--agente", "claude", "--campagna", "infra", "--gpu-ore", "0", "--", "-J", "x", "a.sbatch")
    with pytest.raises(SystemExit):  # stima del costo obbligatoria
        run("ok", "--id", "prova_05", "--agente", "claude", "--campagna", "infra", *SB)
    with pytest.raises(ValueError):
        run("ok", "--id", "../fuori", "--agente", "claude", "--campagna", "infra", "--gpu-ore", "0", *SB)
    assert not (tmp / "fuori").exists()


def test_interrupted_operation_lock_blocks(env):
    run, state, n_calls, tmp = env
    d = tmp / "jobs" / "prova_01"
    d.mkdir(parents=True)
    (d / "operazione.lock").write_text("123 interrotto\n")  # un'operazione interrotta a meta'
    with pytest.raises(SystemExit):
        run("ok", *BASE, *SB)
    assert n_calls() == 0


def test_script_passed_on_stdin(env, tmp_path):
    run, state, _, tmp = env
    script = tmp_path / "campaign_checkout.sbatch"
    script.write_text("#!/bin/bash\necho ciao\n")
    args = ["--id", "camp_01", "--agente", "claude", "--campagna", "infra", "--gpu-ore", "0", "--script-stdin", str(script),
            "--", "--export=ALL,CAMPAIGN=c1,CAMPAIGN_SHA=" + "b" * 40]
    assert run("stdin", *args) == 0
    assert (tmp / "stdin.txt").read_text() == "#!/bin/bash\necho ciao\n"
    s = state("camp_01")
    assert s["job_slurm"] == 777 and s["script_stdin_sha256"] and s["sbatch"][-1].startswith("--export=ALL,CAMPAIGN=c1")
    bad = tmp_path / "x.txt"
    bad.write_text("non uno script")
    with pytest.raises(SystemExit):
        run("stdin", "--id", "camp_02", "--agente", "claude", "--campagna", "infra", "--gpu-ore", "0", "--script-stdin", str(bad))
