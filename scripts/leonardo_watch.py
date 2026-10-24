#!/usr/bin/env python3
"""Attesa dei job SLURM di Leonardo dal Mac, per l'agente leggero `leonardo-watch` (Simone, 24/10/2026: «un agente ... solo per monitorare
le cose, cosi' usera' pochissimi token ... quando dal monitoring vede che il job e' finito te lo comunica e ti sveglia»).

Interroga `sacct` via ssh ogni --interval secondi (default 180: sul login node non costa niente), scrive una riga nel file --out a ogni
cambio di stato e si ferma quando la condizione e' raggiunta: --mode any (default) = almeno uno dei job e' finito; --mode all = tutti.
«Finito» = stato terminale in sacct (COMPLETED, FAILED, CANCELLED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, PREEMPTED, BOOT_FAIL, DEADLINE, REVOKED)
oppure PENDING col motivo DependencyNeverSatisfied (non partira' mai: una dipendenza afterok fallita). Alla fine, per i job finiti che hanno
un log (`JOB:/percorso/del/log_%j.out`), accoda al file --out le ultime 50 righe del log (`tail`) e stampa il riepilogo.

Una chiamata dura al massimo --max-seconds (default 540, sotto il limite di 10 minuti del tool Bash): se la condizione non e' raggiunta esce
con 3 («ancora in attesa») e si richiama identica; lo stato fra una chiamata e l'altra sta in `<out>.state.json`. Il tempo totale dalla prima
chiamata e' limitato da --max-total-seconds (default 24 h: uscita 5); cinque errori ssh di fila: uscita 4. Sul login node esegue solo `sacct`
e `tail`. Nessuna azione sui job: non cancella, non risottomette, non legge altro.

  python3 scripts/leonardo_watch.py --out ~/wearusfm_local/logs/watch_pilot.txt 62123456 62123457:/leonardo_work/.../logs/gate_d8_%j.out
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
NEVER = "DependencyNeverSatisfied"
FIELDS = "JobID,JobName%80,State,Elapsed,ExitCode,Reason"
SSH_FAILURES_MAX = 5


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_jobs(specs: list[str]) -> dict[str, str | None]:
    jobs: dict[str, str | None] = {}
    for s in specs:
        job, _, log = s.partition(":")
        if not job.isdigit():
            raise SystemExit(f"job non valido: {s!r} (atteso <id slurm> o <id slurm>:<log con %j>)")
        jobs[job] = log.replace("%j", job) if log else None
    return jobs


def run_ssh(ssh: list[str], remote: str, timeout: float) -> tuple[int | None, str]:
    try:
        r = subprocess.run([*ssh, remote], capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as e:
        return None, str(e)
    return r.returncode, r.stdout if r.returncode == 0 else (r.stderr or r.stdout)


def parse_sacct(out: str, jobs: dict) -> dict[str, dict]:
    """Una voce per job (le righe con -X sono le allocazioni); un job assente da sacct ha stato ASSENTE (accounting in ritardo o ID sbagliato)."""
    seen = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 6 or parts[0] not in jobs:
            continue
        state = parts[2].split()[0] if parts[2].strip() else "?"  # "CANCELLED by 123" -> CANCELLED
        seen[parts[0]] = {"name": parts[1], "state": state, "elapsed": parts[3], "exit": parts[4], "reason": parts[5].strip()}
    return {j: seen.get(j, {"name": "", "state": "ASSENTE", "elapsed": "", "exit": "", "reason": ""}) for j in jobs}


def finished(v: dict) -> bool:
    return v["state"] in TERMINAL or (v["state"] == "PENDING" and v["reason"] == NEVER)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="+", help="<id slurm> oppure <id slurm>:<percorso del log, %%j ammesso>")
    ap.add_argument("--out", type=Path, required=True, help="file delle righe di stato (fuori dal repo, es. ~/wearusfm_local/logs/)")
    ap.add_argument("--mode", choices=("any", "all"), default="any")
    ap.add_argument("--interval", type=float, default=180.0)
    ap.add_argument("--max-seconds", type=float, default=540.0, help="durata massima di questa chiamata")
    ap.add_argument("--max-total-seconds", type=float, default=24 * 3600.0, help="durata massima dell'attesa intera, dalla prima chiamata")
    ap.add_argument("--ssh", default="ssh leonardo", help="comando per il login node (le prove ne usano uno finto)")
    ap.add_argument("--ssh-timeout", type=float, default=60.0)
    ap.add_argument("--tail", type=int, default=50)
    args = ap.parse_args(argv)
    jobs = parse_jobs(args.jobs)
    ssh = shlex.split(args.ssh)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.out.with_name(args.out.name + ".state.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"started": time.time(), "last": {}, "ssh_failures": 0}

    def note(line: str) -> None:
        with args.out.open("a") as f:
            f.write(f"{now()}  {line}\n")

    if not state["last"]:
        note(f"attesa avviata: job {', '.join(jobs)}; modo {args.mode}; ogni {args.interval:g} s")
    t0 = time.time()
    while True:
        if time.time() - state["started"] > args.max_total_seconds:
            note("tempo massimo dell'attesa superato: mi fermo (uscita 5)")
            state_path.write_text(json.dumps(state))
            return 5
        code, out = run_ssh(ssh, f"sacct -X -n -P -j {','.join(jobs)} --format={FIELDS}", args.ssh_timeout)
        if code != 0:
            state["ssh_failures"] += 1
            note(f"sacct non risponde ({state['ssh_failures']}/{SSH_FAILURES_MAX}): {out.strip()[-200:]}")
            if state["ssh_failures"] >= SSH_FAILURES_MAX:
                state_path.write_text(json.dumps(state))
                return 4
        else:
            state["ssh_failures"] = 0
            cur = parse_sacct(out, jobs)
            for j, v in cur.items():
                key = f"{v['state']}|{v['reason']}"
                if state["last"].get(j) != key:
                    note(f"{j}  {v['state']:<12} {v['elapsed']:>10}  exit {v['exit'] or '-':<5} {v['reason'] or '-':<24} {v['name']}")
                    state["last"][j] = key
            done = [j for j, v in cur.items() if finished(v)]
            if done and (args.mode == "any" or len(done) == len(jobs)):
                for j in done:
                    if jobs[j]:
                        c, t = run_ssh(ssh, f"tail -n {args.tail} {shlex.quote(jobs[j])}", args.ssh_timeout)
                        with args.out.open("a") as f:
                            f.write(f"### ultime {args.tail} righe di {jobs[j]} (job {j})" + ("" if c == 0 else f" [tail fallito: {c}]") + f"\n{t}\n")
                note("condizione raggiunta: " + ", ".join(f"{j} {cur[j]['state']}" for j in done))
                state_path.write_text(json.dumps(state))
                for j in jobs:
                    v = cur[j]
                    print(f"{j}  {v['state']}  elapsed {v['elapsed'] or '-'}  exit {v['exit'] or '-'}  {v['reason'] or ''}  {v['name']}".rstrip())
                print(f"finiti: {', '.join(done)}; righe di stato e code dei log in {args.out}")
                return 0
        state_path.write_text(json.dumps(state))
        if time.time() - t0 + args.interval > args.max_seconds:
            last = ", ".join(f"{j} {state['last'].get(j, '?').split('|')[0]}" for j in jobs)
            print(f"ancora in attesa ({last}): richiamare identico")
            return 3
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
