"""Alimentatore della coda seriale (`scripts/leonardo_feed.py`, agente `leonardo-watch`): un finto `ssh` risponde a `squeue` con un numero di
job letto da un file, un finto `submit_job.py` scrive un ledger e risponde «sottomesso» o «rifiutato» secondo un copione."""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "leonardo_feed.py"
spec = importlib.util.spec_from_file_location("leonardo_feed", SCRIPT)
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)

FAKE_SSH = r'''
import sys
from pathlib import Path
count, calls, remote = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
with calls.open("a") as f:
    f.write(remote + "\n")
if count.name == "down":
    print("ssh: connect to host leonardo port 22: Connection timed out", file=sys.stderr); sys.exit(255)
if remote.startswith("squeue -u $USER -p "):
    n = int(count.read_text().strip()); sys.stdout.write("".join(f"{600000 + i}\n" for i in range(n))); sys.exit(0)
print("comando non ammesso: " + remote, file=sys.stderr); sys.exit(1)
'''

FAKE_SUBMIT = r'''
import json, sys
from pathlib import Path
script, ledger, calls = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
words = sys.argv[4:]
ident = words[words.index("--id") + 1]
with calls.open("a") as f:
    f.write(ident + (" --dopo-rifiuto" if "--dopo-rifiuto" in words else "") + " <" + sys.stdin.read() + ">\n")
plan = json.loads(script.read_text()).get(ident, "ok")
rows = ledger.read_text().splitlines() if ledger.exists() else []
with ledger.open("a") as f:
    if plan == "ok":
        job = 700000 + len(rows)
        f.write(f"t\t{ident}\tclaude\tprova\tsottomesso\t{job}\n"); print(f"{ident}: sottomesso, job {job}"); sys.exit(0)
    f.write(f"t\t{ident}\tclaude\tprova\trifiutato\t\n"); print(f"{ident}: rifiutato"); print("sbatch: error: QOSMaxSubmitJobPerUserLimit", file=sys.stderr); sys.exit(3)
'''


def _env(tmp_path, count=0, plan=None, ledger_rows=(), flags=None, countname="count.txt"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "fake_ssh.py").write_text(FAKE_SSH)
    (tmp_path / "fake_submit.py").write_text(FAKE_SUBMIT)
    countfile = tmp_path / countname
    countfile.write_text(str(count))
    script = tmp_path / "plan.json"
    script.write_text(json.dumps(plan or {}))
    ledger = tmp_path / "ledger.tsv"
    ledger.write_text("".join(f"t\t{i}\tclaude\tprova\t{s}\t{j}\n" for i, s, j in ledger_rows))
    calls_ssh, calls_sub = tmp_path / "calls_ssh.txt", tmp_path / "calls_submit.txt"
    flags = flags or {}
    sub = f"{sys.executable} {tmp_path / 'fake_submit.py'} {script} {ledger} {calls_sub}"
    lines = ["# comandi di prova", "", f"{sub} --id ALTRO-1 -- x", "# ---- blocco scelto"]
    for i in ("PROVA-A", "PROVA-B", "PROVA-C"):
        lines.append(f"{sub} --id {i}{flags.get(i, '')} -- --export=ALL,CODE_DIR=/x /x/probe.sbatch /feat /rep.json --layer dec")
    commands = tmp_path / "comandi.txt"
    commands.write_text("\n".join(lines) + "\n")
    out = tmp_path / "logs" / "feed.txt"
    ssh = f"{sys.executable} {tmp_path / 'fake_ssh.py'} {countfile} {calls_ssh}"

    def run(max_seconds="0", **kw):
        extra = [str(x) for k, v in kw.items() for x in (f"--{k.replace('_', '-')}", v)]
        return F.main(["--commands", str(commands), "--select", "--id PROVA-", "--partition", "lrd_all_serial", "--max-queued", "10",
                       "--out", str(out), "--ledger", str(ledger), "--cwd", str(tmp_path), "--ssh", ssh, "--interval", "0.01",
                       "--max-seconds", max_seconds, *extra])

    return run, out, ledger, countfile, calls_ssh, calls_sub


def _submitted(calls_sub):
    return [l.split(" <")[0] for l in calls_sub.read_text().splitlines()] if calls_sub.exists() else []


def test_feeds_up_to_the_limit_then_asks_to_be_called_again(tmp_path, capsys):
    run, out, ledger, countfile, calls_ssh, calls_sub = _env(tmp_path, count=8)
    assert run() == 3
    assert _submitted(calls_sub) == ["PROVA-A", "PROVA-B"]  # 8 in coda + 2 = 10: la terza aspetta; ALTRO-1 non e' scelto
    assert all(l.endswith(" <>") for l in calls_sub.read_text().splitlines())  # stdin chiuso
    assert "richiamare identico" in capsys.readouterr().out
    countfile.write_text("9")  # una sonda e' finita
    assert run() == 0
    assert _submitted(calls_sub) == ["PROVA-A", "PROVA-B", "PROVA-C"]
    text = out.read_text()
    assert text.count("alimentazione avviata") == 1 and "in coda sulla partizione: 8/10" in text and "tutti i 3 comandi risultano sottomessi" in text
    assert all(c.startswith("squeue -u $USER -p lrd_all_serial -h -o %i") for c in calls_ssh.read_text().splitlines())
    assert F.ledger_states(ledger) == {"PROVA-A": "sottomesso", "PROVA-B": "sottomesso", "PROVA-C": "sottomesso"}


def test_skips_submitted_ids_and_stops_at_a_rejection_without_retrying(tmp_path):
    run, out, ledger, _, _, calls_sub = _env(tmp_path, count=0, plan={"PROVA-B": "rifiutato"}, ledger_rows=[("PROVA-A", "sottomesso", 1)])
    assert run() == 6
    assert _submitted(calls_sub) == ["PROVA-B"]  # A saltata, C mai tentata
    assert "PROVA-B: esito diverso da «sottomesso»" in out.read_text() and "QOSMaxSubmitJobPerUserLimit" in out.read_text()
    assert run() == 6 and _submitted(calls_sub) == ["PROVA-B"]  # richiamato: B e' «rifiutato» senza --dopo-rifiuto, nessun rilancio


def test_a_rejected_id_is_resubmitted_only_with_the_flag(tmp_path):
    run, out, ledger, _, _, calls_sub = _env(tmp_path, count=0, ledger_rows=[("PROVA-B", "rifiutato", "")])
    assert run() == 6 and _submitted(calls_sub) == []  # si ferma prima di sottomettere qualunque cosa
    assert "PROVA-B: «rifiutato» nel ledger e il comando non porta --dopo-rifiuto" in out.read_text()
    run, out, ledger, _, _, calls_sub = _env(tmp_path / "bis", count=0, ledger_rows=[("PROVA-B", "rifiutato", "")], flags={"PROVA-B": " --dopo-rifiuto"})
    assert run() == 0
    assert _submitted(calls_sub) == ["PROVA-A", "PROVA-B --dopo-rifiuto", "PROVA-C"]


def test_other_ledger_states_stop_the_feeder(tmp_path):
    run, out, *_ , calls_sub = _env(tmp_path, count=0, ledger_rows=[("PROVA-C", "ambiguo", "")])
    assert run() == 6 and _submitted(calls_sub) == [] and "stato 'ambiguo' nel ledger" in out.read_text()


def test_ssh_down_gives_4_after_five_failures(tmp_path):
    run, out, *_ = _env(tmp_path, count=0, countname="down")
    assert run(max_seconds="10") == 4
    assert out.read_text().count("squeue non risponde") == 5


def test_select_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("python3 s.py --id X -- a\npython3 s.py --id X -- b\n")
    try:
        F.select_commands(p, "--id X")
    except SystemExit as e:
        assert "ID ripetuti" in str(e)
    else:
        raise AssertionError("duplicati accettati")
