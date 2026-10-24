"""Attesa dei job dal Mac (`scripts/leonardo_watch.py`, agente `leonardo-watch`): il login node e' un finto `ssh` che risponde a `sacct` con
una sequenza di stati e registra ogni comando ricevuto."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "leonardo_watch.py"
spec = importlib.util.spec_from_file_location("leonardo_watch", SCRIPT)
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)

FAKE = r'''
import json, sys
from pathlib import Path
seq, calls, remote = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
with calls.open("a") as f:
    f.write(remote + "\n")
if seq.name == "down":
    print("ssh: connect to host leonardo port 22: Connection timed out", file=sys.stderr); sys.exit(255)
if remote.startswith("sacct"):
    s = json.loads(seq.read_text()); out = s.pop(0) if len(s) > 1 else s[0]; seq.write_text(json.dumps(s))
    sys.stdout.write(out); sys.exit(0)
if remote.startswith("tail"):
    print("checkout di campagna pronto: /x @ " + "a" * 40); sys.exit(0)
print("comando non ammesso: " + remote, file=sys.stderr); sys.exit(1)
'''

ROW = "{j}|{n}|{s}|{e}|{x}|{r}\n"


def _env(tmp_path, responses, name="seq.json"):
    fake, seq, calls = tmp_path / "fake_ssh.py", tmp_path / name, tmp_path / "calls.txt"
    fake.write_text(FAKE)
    seq.write_text(json.dumps(responses))
    ssh = f"{sys.executable} {fake} {seq} {calls}"
    out = tmp_path / "logs" / "watch.txt"

    def run(*jobs, **kw):
        extra = [str(x) for k, v in kw.items() for x in (f"--{k.replace('_', '-')}", v)]
        return W.main(["--out", str(out), "--ssh", ssh, "--interval", "0.01", *extra, *jobs])

    return run, out, calls


def test_any_mode_stops_at_the_first_finished_job_and_appends_the_log_tail(tmp_path, capsys):
    pend = ROW.format(j=1, n="CHK", s="PENDING", e="00:00:00", x="0:0", r="Priority") + ROW.format(j=2, n="GATE", s="PENDING", e="00:00:00", x="0:0", r="Dependency")
    done = ROW.format(j=1, n="CHK", s="COMPLETED", e="00:00:13", x="0:0", r="None") + ROW.format(j=2, n="GATE", s="RUNNING", e="00:01:00", x="0:0", r="None")
    run, out, calls = _env(tmp_path, [pend, pend, done])
    assert run("1:/logs/campaign_checkout_%j.out", "2:/logs/gate_d8_%j.out") == 0
    text = out.read_text()
    assert "attesa avviata" in text and "1  COMPLETED" in text and "2  RUNNING" in text
    assert "### ultime 50 righe di /logs/campaign_checkout_1.out (job 1)" in text and "checkout di campagna pronto" in text
    assert "gate_d8_2.out" not in text.split("###", 1)[1]  # il log si legge solo per i job finiti
    remote = calls.read_text().splitlines()
    assert all(c.startswith(("sacct -X -n -P -j 1,2", "tail -n 50 ")) for c in remote) and remote.count("tail -n 50 /logs/campaign_checkout_1.out") == 1
    assert "finiti: 1" in capsys.readouterr().out
    # due righe di stato per job (PENDING, poi lo stato finale): i tick senza cambiamento non scrivono niente
    assert sum(1 for l in text.splitlines() if "  1  " in l) == 2


def test_all_mode_waits_for_every_job_and_a_dead_dependency_counts_as_finished(tmp_path):
    one = ROW.format(j=1, n="A", s="COMPLETED", e="00:10:00", x="0:0", r="None") + ROW.format(j=2, n="B", s="PENDING", e="00:00:00", x="0:0", r="Dependency")
    both = ROW.format(j=1, n="A", s="COMPLETED", e="00:10:00", x="0:0", r="None") + ROW.format(j=2, n="B", s="PENDING", e="00:00:00", x="0:0", r="DependencyNeverSatisfied")
    run, out, _ = _env(tmp_path, [one, one, both])
    assert run("1", "2", mode="all") == 0
    assert "condizione raggiunta: 1 COMPLETED, 2 PENDING" in out.read_text()


def test_a_call_ends_with_3_when_its_time_is_up_and_the_next_call_continues_the_same_wait(tmp_path, capsys):
    pend = ROW.format(j=7, n="X", s="PENDING", e="00:00:00", x="0:0", r="QOSMaxCpuPerUserLimit")
    run, out, calls = _env(tmp_path, [pend])
    assert run("7", max_seconds="0.05", interval="0.04") == 3
    assert "ancora in attesa (7 PENDING)" in capsys.readouterr().out
    n = len(calls.read_text().splitlines())
    assert run("7", max_seconds="0.05", interval="0.04") == 3
    assert len(calls.read_text().splitlines()) > n
    assert out.read_text().count("attesa avviata") == 1 and out.read_text().count("7  PENDING") == 1  # stato ripreso da <out>.state.json
    assert json.loads(out.with_name("watch.txt.state.json").read_text())["last"]["7"] == "PENDING|QOSMaxCpuPerUserLimit"


def test_total_time_limit_and_missing_jobs(tmp_path):
    run, out, _ = _env(tmp_path, [""])  # sacct senza righe: job assente (accounting in ritardo)
    assert run("9", max_seconds="0.05", interval="0.04") == 3
    assert "9  ASSENTE" in out.read_text()
    assert run("9", max_total_seconds="0") == 5


def test_five_ssh_failures_in_a_row_stop_the_watch_with_4(tmp_path):
    run, out, calls = _env(tmp_path, [""], name="down")
    assert run("3", max_seconds="60") == 4
    assert len(calls.read_text().splitlines()) == 5 and "sacct non risponde (5/5)" in out.read_text()


def test_job_specs_are_validated():
    with pytest.raises(SystemExit, match="job non valido"):
        W.parse_jobs(["abc"])
    assert W.parse_jobs(["12:/l/x_%j.out", "13"]) == {"12": "/l/x_12.out", "13": None}
