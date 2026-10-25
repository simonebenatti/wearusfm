#!/usr/bin/env python3
"""Alimentatore della coda seriale di Leonardo dal Mac, per l'agente leggero `leonardo-watch` (nato il 25/10/2026: la QoS della seriale
ammette 10 job in coda per utente, `QOSMaxSubmitJobPerUserLimit`, e l'undicesima sonda del pilot e' stata rifiutata da sbatch).

Legge da un file di comandi (es. `results/infra/<campagna>/comandi_valutazione.txt`) le righe `python3 scripts/submit_job.py ...` che
corrispondono a --select (espressione regolare, ordine del file), ne ricava l'ID logico (`--id`), salta quelli gia' `sottomesso` nel ledger e
ogni --interval secondi conta i job dell'utente in coda o in esecuzione sulla partizione (`squeue -u $USER -p <partizione> -h -o %i`, via
ssh): finche' il conteggio resta sotto --max-queued sottomette i comandi successivi, uno alla volta, attraverso `submit_job.py` (che fa la
propria ssh, con lo stdin chiuso). Scrive ogni passo nel file --out.

Si ferma: con 0 quando tutti i comandi scelti risultano sottomessi; con 3 quando la chiamata ha esaurito --max-seconds (default 540, sotto il
limite di 10 minuti del tool Bash: si richiama identico, lo stato sta in `<out>.state.json`); con 4 dopo cinque errori ssh di fila; con 5
oltre --max-total-seconds; con 6 davanti a un esito diverso da «sottomesso» (rifiuto, ambiguita'), a un ID del ledger in uno stato non
finale, o a un ID `rifiutato` il cui comando non porta `--dopo-rifiuto`: serve una decisione, nessun rilancio automatico. Sul login node
esegue solo `squeue`; non cancella e non risottomette mai un ID gia' sottomesso. Due alimentatori sulla stessa partizione (es. A2 e B2 del
pilot) non si pestano i piedi: conteggio e sottomissioni avvengono sotto un lock di file (`--lock`, default
`<cartella del ledger>/feed_<partizione>.lock`), cosi' il secondo conta dopo che il primo ha sottomesso.

  python3 scripts/leonardo_feed.py --commands results/infra/pilot_obiettivi_20261024/comandi_valutazione.txt \\
      --select='--id PILOT-EVAL-probe-R[0-9]-e2-p[12]-dec ' --partition lrd_all_serial --max-queued 10 --out ~/wearusfm_local/logs/feed_A2.txt

(`--select=...` con l'uguale: il valore comincia con `--`.)
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SSH_FAILURES_MAX = 5
DEFAULT_LEDGER = Path.home() / "wearusfm_local" / "jobs" / "ledger.tsv"
OK_MARK = ": sottomesso, job "


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def command_id(cmd: str) -> str:
    words = shlex.split(cmd)
    if "--id" not in words or words.index("--id") + 1 >= len(words):
        raise SystemExit(f"comando senza --id: {cmd[:80]!r}")
    return words[words.index("--id") + 1]


def select_commands(path: Path, pattern: str) -> list[tuple[str, str]]:
    """(ID logico, comando) per le righe non vuote e non di commento che contengono --id e corrispondono a `pattern`, nell'ordine del file."""
    rx = re.compile(pattern)
    chosen = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "--id" not in s or not rx.search(s):
            continue
        chosen.append((command_id(s), s))
    ids = [i for i, _ in chosen]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"ID ripetuti nei comandi scelti: {sorted({i for i in ids if ids.count(i) > 1})}")
    return chosen


def ledger_states(path: Path) -> dict[str, str]:
    """L'ultimo stato di ogni ID logico nel ledger (colonne: quando, id, agente, campagna, stato, job)."""
    states: dict[str, str] = {}
    if not path.exists():
        return states
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 5 and parts[1]:
            states[parts[1]] = parts[4]
    return states


def classify(chosen: list[tuple[str, str]], states: dict[str, str]) -> tuple[list[str], list[tuple[str, str]], str | None]:
    """(ID gia' sottomessi, comandi ancora da sottomettere, motivo di fermata o None)."""
    done, todo = [], []
    for i, cmd in chosen:
        st = states.get(i)
        if st == "sottomesso":
            done.append(i)
        elif st is None:
            todo.append((i, cmd))
        elif st == "rifiutato":
            if "--dopo-rifiuto" not in shlex.split(cmd):
                return done, todo, f"{i}: «rifiutato» nel ledger e il comando non porta --dopo-rifiuto: correggere la causa prima"
            todo.append((i, cmd))
        else:
            return done, todo, f"{i}: stato {st!r} nel ledger: serve una decisione (submit_job.py non lo risottomette)"
    return done, todo, None


def run_ssh(ssh: list[str], remote: str, timeout: float) -> tuple[int | None, str]:
    try:
        r = subprocess.run([*ssh, remote], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except (subprocess.TimeoutExpired, OSError) as e:
        return None, str(e)
    return r.returncode, r.stdout if r.returncode == 0 else (r.stderr or r.stdout)


def submit(cmd: str, cwd: Path, timeout: float) -> tuple[bool, str]:
    """Esegue un comando di sottomissione con lo stdin chiuso; vero solo con uscita 0 e la riga «ID: sottomesso, job N»."""
    try:
        r = subprocess.run(shlex.split(cmd), cwd=str(cwd), capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, str(e)
    text = (r.stdout + ("\n" + r.stderr if r.stderr else "")).strip()
    return r.returncode == 0 and OK_MARK in r.stdout, text + f"\n[rc={r.returncode}]"


@contextlib.contextmanager
def locked(path: Path):
    """Lock esclusivo (flock) sul file `path`, creato se manca: un solo alimentatore alla volta conta la coda e sottomette."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commands", type=Path, required=True, help="file dei comandi di sottomissione (righe `python3 scripts/submit_job.py ...`)")
    ap.add_argument("--select", required=True, help="espressione regolare: si prendono le righe che la contengono, nell'ordine del file")
    ap.add_argument("--partition", required=True, help="partizione di cui contare i job dell'utente (es. lrd_all_serial)")
    ap.add_argument("--max-queued", type=int, required=True, help="massimo di job dell'utente in coda o in esecuzione sulla partizione")
    ap.add_argument("--out", type=Path, required=True, help="file delle righe di stato (fuori dal repo, es. ~/wearusfm_local/logs/)")
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER, help="ledger di submit_job.py (ID gia' sottomessi)")
    ap.add_argument("--cwd", type=Path, default=ROOT, help="directory da cui eseguire i comandi (default: la radice del repo)")
    ap.add_argument("--lock", type=Path, help="file di lock fra alimentatori della stessa partizione (default: <cartella del ledger>/feed_<partizione>.lock)")
    ap.add_argument("--interval", type=float, default=180.0)
    ap.add_argument("--max-seconds", type=float, default=540.0, help="durata massima di questa chiamata")
    ap.add_argument("--max-total-seconds", type=float, default=48 * 3600.0, help="durata massima dell'alimentazione intera, dalla prima chiamata")
    ap.add_argument("--ssh", default="ssh leonardo", help="comando per il login node (le prove ne usano uno finto)")
    ap.add_argument("--ssh-timeout", type=float, default=60.0)
    ap.add_argument("--submit-timeout", type=float, default=300.0)
    args = ap.parse_args(argv)
    if args.max_queued < 1:
        ap.error("--max-queued >= 1")
    chosen = select_commands(args.commands, args.select)
    lock_path = args.lock or args.ledger.parent / f"feed_{args.partition}.lock"
    if not chosen:
        raise SystemExit(f"nessun comando corrisponde a {args.select!r} in {args.commands}")
    ssh = shlex.split(args.ssh)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.out.with_name(args.out.name + ".state.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"started": time.time(), "ssh_failures": 0, "calls": 0}
    state["calls"] += 1

    def note(line: str) -> None:
        with args.out.open("a") as f:
            f.write(f"{now()}  {line}\n")

    def stop(code: int, line: str) -> int:
        note(line + f" (uscita {code})")
        state_path.write_text(json.dumps(state))
        print(line)
        return code

    if state["calls"] == 1:
        note(f"alimentazione avviata: {len(chosen)} comandi ({args.select!r}); partizione {args.partition}, massimo {args.max_queued} in coda; "
             f"ogni {args.interval:g} s")
    t0 = time.time()
    while True:
        if time.time() - state["started"] > args.max_total_seconds:
            return stop(5, "tempo massimo dell'alimentazione superato: mi fermo")
        done, todo, why = classify(chosen, ledger_states(args.ledger))
        if why:
            return stop(6, why)
        if not todo:
            return stop(0, f"tutti i {len(chosen)} comandi risultano sottomessi: {', '.join(done)}")
        with locked(lock_path):  # un solo alimentatore alla volta fra conteggio e sottomissione
            done, todo, why = classify(chosen, ledger_states(args.ledger))  # riletto sotto lock: un altro alimentatore puo' aver sottomesso
            code, out = run_ssh(ssh, f"squeue -u $USER -p {shlex.quote(args.partition)} -h -o %i", args.ssh_timeout)
            if code != 0:
                state["ssh_failures"] += 1
                note(f"squeue non risponde ({state['ssh_failures']}/{SSH_FAILURES_MAX}): {out.strip()[-200:]}")
                if state["ssh_failures"] >= SSH_FAILURES_MAX:
                    state_path.write_text(json.dumps(state))
                    return 4
            else:
                state["ssh_failures"] = 0
                count = len([x for x in out.split() if x.strip()])
                note(f"in coda sulla partizione: {count}/{args.max_queued}; sottomessi {len(done)}/{len(chosen)}, da sottomettere {len(todo)}")
                while count < args.max_queued and todo:
                    i, cmd = todo.pop(0)
                    ok, text = submit(cmd, args.cwd, args.submit_timeout)
                    with args.out.open("a") as f:
                        f.write(f"{now()}  sottomissione di {i}:\n{text}\n")
                    if not ok:
                        return stop(6, f"{i}: esito diverso da «sottomesso», mi fermo senza rilanciare")
                    count += 1
                    done.append(i)
                if not todo:
                    return stop(0, f"tutti i {len(chosen)} comandi risultano sottomessi: {', '.join(done)}")
        state_path.write_text(json.dumps(state))
        if time.time() - t0 + args.interval > args.max_seconds:
            print(f"ancora da sottomettere {len(todo)} comandi su {len(chosen)} (sottomessi {len(done)}): richiamare identico")
            return 3
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
