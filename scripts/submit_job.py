#!/usr/bin/env python3
"""Sottomissione unica dei job SLURM per Claude e GPT (decisione del 09/10/2026, CLAUDE.md «Sottomissioni»): prenotazione sotto lock PRIMA di
`sbatch`, stato persistente, ID SLURM registrato, riconciliazione senza rilancio automatico se la risposta e' ambigua.

Ogni job ha un **ID logico** (es. `w1_ddp_prova_01`), unico per sempre. Lo stato vive sul Mac in `~/wearusfm_local/jobs/<id>/state.json`
(la cartella si crea con `mkdir` atomico: e' la prenotazione); `~/wearusfm_local/jobs/ledger.tsv` e' il riepilogo leggibile. Stati:

    prenotato -> in_sottomissione -> sottomesso (con l'ID SLURM)
                                  -> rifiutato  (sbatch ha risposto con un errore esplicito: nessun job creato)
                                  -> ambiguo    (connessione caduta, timeout, risposta illeggibile: il job PUO' esistere)

Un ID gia' usato non si risottomette: `sottomesso` e `ambiguo` mai; `rifiutato` solo con `--dopo-rifiuto` (per esempio dopo aver corretto
la QoS). Un `ambiguo` si chiarisce con `--riconcilia <id>`: cerca su SLURM i job col nome = ID logico (lo script lo impone con `--job-name`
e lo ripete in `--comment`). Uno trovato -> `sottomesso`; nessuno -> `assente` (si puo' risottomettere solo con `--dopo-rifiuto`, dopo aver
guardato); piu' d'uno -> `duplicato`, da guardare a mano. Nessuna cancellazione automatica di prenotazioni.

  python3 scripts/submit_job.py --id ID --agente claude --campagna NOME --gpu-ore 0.5 [--dopo ID_PRECEDENTE] -- <argomenti di sbatch>
  python3 scripts/submit_job.py --id ID ... --script-stdin scripts/slurm/campaign_checkout.sbatch -- --export=ALL,CAMPAIGN=...,CAMPAIGN_SHA=...
  python3 scripts/submit_job.py --riconcilia ID
  python3 scripts/submit_job.py --stato [ID]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
JOBID_RE = re.compile(r"^(\d+)(;[\w.-]+)?$")
DEFAULT_DIR = Path.home() / "wearusfm_local" / "jobs"
LEDGER_COLUMNS = ("quando", "id", "agente", "campagna", "stato", "job_slurm", "gpu_ore_stimate", "codice_sha", "comando")


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


class Registry:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, job_id: str) -> Path:
        if not ID_RE.match(job_id):
            raise ValueError(f"ID logico non valido: {job_id!r} (lettere, cifre, . _ -, al piu' 64)")
        return self.root / job_id

    def read(self, job_id: str) -> dict | None:
        p = self.dir(job_id) / "state.json"
        return json.loads(p.read_text()) if p.exists() else None

    def write(self, job_id: str, state: dict) -> None:
        p = self.dir(job_id) / "state.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(p)

    def transition(self, job_id: str, state: dict, new: str, **fields) -> dict:
        state = {**state, **fields, "stato": new}
        state.setdefault("storia", []).append({"quando": now(), "stato": new, **{k: v for k, v in fields.items() if k != "storia"}})
        self.write(job_id, state)
        if new not in ("in_sottomissione",):
            self.ledger(state)
        return state

    def ledger(self, s: dict) -> None:
        p = self.root / "ledger.tsv"
        new = not p.exists()
        with p.open("a") as f:
            if new:
                f.write("\t".join(LEDGER_COLUMNS) + "\n")
            row = [now(), s["id"], s.get("agente", ""), s.get("campagna", ""), s["stato"], str(s.get("job_slurm") or ""),
                   str(s.get("gpu_ore_stimate", "")), s.get("codice_sha") or "", " ".join(s.get("sbatch", []))]
            f.write("\t".join(x.replace("\t", " ").replace("\n", " ") for x in row) + "\n")

    def reserve(self, job_id: str, *, allow_after_reject: bool) -> dict | None:
        """La prenotazione: mkdir atomico. Se la cartella c'e' gia', si puo' proseguire solo da `rifiutato`/`assente` con il permesso esplicito;
        un lock d'operazione (O_EXCL) impedisce che due processi facciano la stessa transizione."""
        d = self.dir(job_id)
        try:
            d.mkdir()
            fresh = True
        except FileExistsError:
            fresh = False
        lock = d / "operazione.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {now()}\n".encode())
            os.close(fd)
        except FileExistsError:
            raise SystemExit(f"{job_id}: un'altra operazione e' in corso (o e' stata interrotta): guardare {lock}, non rilanciare alla cieca")
        if fresh:
            return None
        state = self.read(job_id)
        current = state["stato"] if state else "sconosciuto"
        if current in ("rifiutato", "assente") and allow_after_reject:
            return state
        self.release(job_id)
        raise SystemExit(f"{job_id}: gia' prenotato, stato {current!r}. "
                         + ("Si risottomette solo con --dopo-rifiuto." if current in ("rifiutato", "assente") else
                            "Non si risottomette: per un ambiguo usare --riconcilia." if current == "ambiguo" else "Usare un ID nuovo."))

    def release(self, job_id: str) -> None:
        (self.dir(job_id) / "operazione.lock").unlink(missing_ok=True)


def run_remote(ssh: list[str], remote: str, timeout: float, stdin: str | None = None) -> tuple[int | None, str, str]:
    """(codice d'uscita o None se timeout, stdout, stderr). `stdin`: testo passato al comando remoto (lo script di sbatch)."""
    try:
        r = subprocess.run([*ssh, remote], capture_output=True, text=True, timeout=timeout, input=stdin)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        return None, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), str(e)


def classify(code: int | None, out: str, err: str) -> tuple[str, int | None]:
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if code == 0 and len(lines) == 1 and JOBID_RE.match(lines[0]):
        return "sottomesso", int(JOBID_RE.match(lines[0]).group(1))
    if code not in (None, 0, 255) and "sbatch: error" in err and not any(JOBID_RE.match(ln) for ln in lines):
        return "rifiutato", None  # sbatch ha risposto e ha rifiutato: nessun job
    return "ambiguo", None  # 255 = errore di ssh (anche a meta'), timeout, uscita illeggibile


def code_sha_from(sbatch_args: list[str]) -> str | None:
    for a in sbatch_args:
        m = re.search(r"EXPECTED_SHA=([0-9a-f]{40})", a)
        if m:
            return m.group(1)
    return None


def submit(reg: Registry, args, ssh: list[str]) -> int:
    sbatch_args = list(args.sbatch)
    if sbatch_args and sbatch_args[0] == "--":
        sbatch_args = sbatch_args[1:]
    if not sbatch_args and not args.script_stdin:
        raise SystemExit("mancano gli argomenti di sbatch (dopo --)")
    if any(a.startswith(("--job-name", "-J", "--comment")) for a in sbatch_args):
        raise SystemExit("--job-name e --comment li impone lo script (= ID logico): toglierli")
    if args.gpu_ore is None or args.gpu_ore < 0:
        raise SystemExit("--gpu-ore (stima dichiarata) e' obbligatorio")
    dep = []
    if args.dopo:
        prev = reg.read(args.dopo)
        if not prev or prev.get("stato") != "sottomesso" or not prev.get("job_slurm"):
            raise SystemExit(f"--dopo {args.dopo}: non e' un job sottomesso con ID SLURM noto")
        dep = [f"--dependency=afterok:{prev['job_slurm']}"]
    script_text = None
    if args.script_stdin:  # lo script locale (versionato) va a sbatch su stdin: non serve che esista sul cluster
        script_text = args.script_stdin.read_text()
        if not script_text.startswith("#!"):
            raise SystemExit(f"{args.script_stdin}: non sembra uno script di sbatch")
    previous = reg.reserve(args.id, allow_after_reject=args.dopo_rifiuto)
    try:
        full = ["sbatch", "--parsable", f"--job-name={args.id}", f"--comment=wearusfm:{args.id}", *dep, *sbatch_args]
        base = previous or {"id": args.id, "creato": now()}
        state = {**base, "id": args.id, "agente": args.agente, "campagna": args.campagna, "gpu_ore_stimate": args.gpu_ore,
                 "codice_sha": code_sha_from(sbatch_args), "sbatch": full, "dopo": args.dopo,
                 "script_stdin": str(args.script_stdin) if args.script_stdin else None,
                 "script_stdin_sha256": hashlib.sha256(script_text.encode()).hexdigest() if script_text is not None else None}
        state = reg.transition(args.id, state, "prenotato" if previous is None else "prenotato_di_nuovo")
        state = reg.transition(args.id, state, "in_sottomissione")
        code, out, err = run_remote(ssh, " ".join(shlex.quote(x) for x in full), args.timeout, script_text)
        outcome, job = classify(code, out, err)
        state = reg.transition(args.id, state, outcome, job_slurm=job, uscita=code, stdout=out[-2000:], stderr=err[-2000:])
    finally:
        reg.release(args.id)
    print(f"{args.id}: {outcome}" + (f", job {job}" if job else "") + ("" if outcome == "sottomesso" else f"\n{err.strip()[-500:]}"))
    return 0 if outcome == "sottomesso" else 3


def reconcile(reg: Registry, job_id: str, ssh: list[str], timeout: float) -> int:
    state = reg.read(job_id)
    if state is None:
        raise SystemExit(f"{job_id}: nessuna prenotazione")
    if state["stato"] not in ("ambiguo", "assente"):
        print(f"{job_id}: stato {state['stato']}, niente da riconciliare")
        return 0
    reg.reserve(job_id, allow_after_reject=True) if state["stato"] == "assente" else _lock_only(reg, job_id)
    try:
        since = state.get("creato", now())[:10]
        remote = (f"sacct -X -n -P -S {shlex.quote(since)} --name={shlex.quote(job_id)} --format=JobID,State; "
                  f"squeue -h --name={shlex.quote(job_id)} -o '%i|%T'")
        code, out, err = run_remote(ssh, remote, timeout)
        if code != 0:
            print(f"{job_id}: riconciliazione non riuscita (uscita {code}), stato invariato\n{err.strip()[-500:]}")
            return 3
        ids = sorted({int(ln.split("|")[0]) for ln in out.splitlines() if ln.strip() and ln.split("|")[0].isdigit()})
        if len(ids) == 1:
            state = reg.transition(job_id, state, "sottomesso", job_slurm=ids[0], riconciliato=True)
        elif not ids:
            state = reg.transition(job_id, state, "assente", riconciliato=True)
        else:
            state = reg.transition(job_id, state, "duplicato", job_slurm_tutti=ids, riconciliato=True)
    finally:
        reg.release(job_id)
    print(f"{job_id}: {state['stato']}" + (f", job {state.get('job_slurm')}" if state.get("job_slurm") else ""))
    return 0 if state["stato"] == "sottomesso" else 3


def _lock_only(reg: Registry, job_id: str) -> None:
    lock = reg.dir(job_id) / "operazione.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        raise SystemExit(f"{job_id}: un'altra operazione e' in corso: guardare {lock}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id")
    ap.add_argument("--agente", choices=("claude", "gpt", "simone"))
    ap.add_argument("--campagna")
    ap.add_argument("--gpu-ore", type=float, default=None, help="stima dichiarata nel preventivo (0 per i job CPU)")
    ap.add_argument("--dopo", help="ID logico di un job gia' sottomesso: dipendenza afterok")
    ap.add_argument("--script-stdin", type=Path, help="script locale da passare a sbatch su stdin (es. campaign_checkout.sbatch)")
    ap.add_argument("--dopo-rifiuto", action="store_true", help="risottomette un ID rifiutato o assente (dopo aver guardato)")
    ap.add_argument("--riconcilia", metavar="ID")
    ap.add_argument("--stato", nargs="?", const="", metavar="ID")
    ap.add_argument("--registro", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--ssh", default="ssh leonardo", help="comando per raggiungere il login node (le prove ne usano uno finto)")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("sbatch", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    reg, ssh = Registry(args.registro), shlex.split(args.ssh)
    if args.stato is not None:
        ids = [args.stato] if args.stato else sorted(p.name for p in reg.root.iterdir() if p.is_dir())
        for i in ids:
            s = reg.read(i) or {}
            print(f"{i}\t{s.get('stato')}\t{s.get('job_slurm') or ''}\t{s.get('agente', '')}\t{s.get('campagna', '')}")
        return 0
    if args.riconcilia:
        return reconcile(reg, args.riconcilia, ssh, args.timeout)
    if not (args.id and args.agente and args.campagna):
        ap.error("servono --id, --agente e --campagna")
    return submit(reg, args, ssh)


if __name__ == "__main__":
    sys.exit(main())
