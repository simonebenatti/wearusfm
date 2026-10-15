#!/usr/bin/env python3
"""Sottomissione unica dei job SLURM per Claude e GPT (decisione del 09/10/2026, CLAUDE.md «Sottomissioni»): prenotazione sotto lock PRIMA di
`sbatch`, stato persistente, ID SLURM registrato, riconciliazione senza rilancio automatico se la risposta e' ambigua.

Ogni job ha un **ID logico** (es. `w1_ddp_prova_01`), unico per sempre. Lo stato vive sul Mac in `~/wearusfm_local/jobs/<id>/state.json`
(la cartella si crea con `mkdir` atomico: e' la prenotazione); `~/wearusfm_local/jobs/ledger.tsv` e' il riepilogo leggibile. Stati:

    prenotato -> in_sottomissione -> sottomesso (con l'ID SLURM)
                                  -> rifiutato  (sbatch ha risposto con un errore esplicito: nessun job creato)
                                  -> ambiguo    (connessione caduta, timeout, risposta illeggibile: il job PUO' esistere)

Un ID gia' usato non si risottomette: `sottomesso`, `ambiguo` e `duplicato` mai; `rifiutato` solo con `--dopo-rifiuto` (per esempio dopo
aver corretto la QoS). Un `ambiguo` si chiarisce con `--riconcilia <id>`: interroga `sacct` e `squeue` SEPARATAMENTE (ognuno deve uscire con
0 e ogni riga deve essere leggibile e avere come nome l'ID logico, che lo script impone con `--job-name` e ripete in `--comment`; altrimenti lo
stato resta `ambiguo`). Uno trovato -> `sottomesso`; piu' d'uno -> `duplicato`, da guardare a mano. **Una risposta vuota non basta**: l'accounting
puo' essere in ritardo. Si registra un controllo d'assenza; lo stato passa ad `assente` solo con almeno due controlli vuoti, il primo almeno
10 minuti dopo il tentativo di sottomissione e l'ultimo almeno 10 minuti dopo il primo. Da `assente` si risottomette solo con
`--dopo-assenza-verificata` (flag distinto da `--dopo-rifiuto`). Nessuna cancellazione automatica di prenotazioni.

Concorrenza (review GPT del 09/10/2026): ogni transizione avviene sotto il lock d'operazione del job e parte dallo stato riletto sotto quel
lock; `Registry.transition` lo verifica (stato e lunghezza della storia uguali a quelli su disco, transizione prevista da `TRANSIZIONI`) e
altrimenti non scrive. La riconciliazione prende il lock PRIMA di leggere lo stato. Il ledger, condiviso fra tutti i job, ha un lock suo
(`flock` esclusivo sul file aperto in append), distinto dai lock per job: intestazione solo se il file e' vuoto sotto quel lock.

  python3 scripts/submit_job.py --id ID --agente claude --campagna NOME --gpu-ore 0.5 [--dopo ID_PRECEDENTE ...] -- <argomenti di sbatch>
  python3 scripts/submit_job.py --id ID ... --script-stdin scripts/slurm/campaign_checkout.sbatch -- --export=ALL,CAMPAIGN=...,CAMPAIGN_SHA=...
  python3 scripts/submit_job.py --riconcilia ID
  python3 scripts/submit_job.py --stato [ID]
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
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
ABSENCE_DELAY = dt.timedelta(minutes=10)  # primo controllo vuoto non prima di 10 min dal tentativo; ultimo almeno 10 min dopo il primo
# stato su disco (None = nessuno stato ancora) -> stati ammessi dopo; `sottomesso` e `duplicato` sono finali
TRANSIZIONI = {
    None: {"prenotato"},
    "rifiutato": {"prenotato_di_nuovo"},
    "assente": {"prenotato_di_nuovo"},
    "prenotato": {"in_sottomissione"},
    "prenotato_di_nuovo": {"in_sottomissione"},
    "in_sottomissione": {"sottomesso", "rifiutato", "ambiguo"},
    "ambiguo": {"sottomesso", "duplicato", "assente", "ambiguo"},
}
_clock = dt.datetime.now  # sostituibile nelle prove


def now() -> str:
    return _clock().astimezone().isoformat(timespec="seconds")


def _parse_time(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


class Registry:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._held: set[str] = set()  # job di cui questo processo tiene il lock d'operazione

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
        """`state` -> `new`, solo col lock d'operazione del job in mano e solo se `state` e' ancora lo stato su disco (riletto qui: stesso
        stato, storia della stessa lunghezza) e la transizione e' in `TRANSIZIONI`. Altrimenti RuntimeError e nessuna scrittura: una copia
        vecchia non sovrascrive mai quello che un altro processo ha registrato."""
        if job_id not in self._held:
            raise RuntimeError(f"{job_id}: transizione a {new!r} senza il lock d'operazione")
        on_disk = self.read(job_id) or {}
        current = on_disk.get("stato")
        if current != state.get("stato") or len(on_disk.get("storia", [])) != len(state.get("storia", [])):
            raise RuntimeError(f"{job_id}: lo stato su disco ({current!r}) non e' quello da cui parte la transizione "
                               f"({state.get('stato')!r}): nessuna scrittura")
        if new not in TRANSIZIONI.get(current, ()):
            raise RuntimeError(f"{job_id}: transizione {current!r} -> {new!r} non prevista: nessuna scrittura")
        entry = {"quando": now(), "stato": new, **{k: v for k, v in fields.items() if k != "storia"}}
        state = {**state, **fields, "stato": new, "storia": [*state.get("storia", []), entry]}
        self.write(job_id, state)
        if new not in ("in_sottomissione",):
            self.ledger(state)
        return state

    def ledger(self, s: dict) -> None:
        """Una riga nel riepilogo condiviso da tutti i job. Lock proprio del ledger (`flock` esclusivo sul file aperto in append), distinto
        dal lock per job: intestazione solo se il file e' vuoto sotto il lock, intestazione e riga in una sola scrittura."""
        p = self.root / "ledger.tsv"
        with p.open("a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                row = [now(), s["id"], s.get("agente", ""), s.get("campagna", ""), s["stato"], str(s.get("job_slurm") or ""),
                       str(s.get("gpu_ore_stimate", "")), s.get("codice_sha") or "", " ".join(s.get("sbatch", []))]
                text = "\t".join(x.replace("\t", " ").replace("\n", " ") for x in row) + "\n"
                if os.fstat(f.fileno()).st_size == 0:
                    text = "\t".join(LEDGER_COLUMNS) + "\n" + text
                f.write(text)
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def lock(self, job_id: str) -> None:
        """Lock d'operazione (O_EXCL): due processi non fanno mai la stessa transizione; uno interrotto lascia il lock, da guardare a mano.
        Lo stato si legge DOPO averlo preso (`reserve`, `reconcile`), mai prima."""
        lock = self.dir(job_id) / "operazione.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {now()}\n".encode())
            os.close(fd)
        except FileExistsError:
            raise SystemExit(f"{job_id}: un'altra operazione e' in corso (o e' stata interrotta): guardare {lock}, non rilanciare alla cieca")
        self._held.add(job_id)

    def reserve(self, job_id: str, *, after_reject: bool = False, after_absence: bool = False) -> dict | None:
        """La prenotazione: mkdir atomico, poi il lock d'operazione. Se la cartella c'e' gia', lo stato letto sotto lock decide: si prosegue
        solo da `rifiutato` con `after_reject` o da `assente` con `after_absence`; tutto il resto si rifiuta."""
        d = self.dir(job_id)
        try:
            d.mkdir()
            fresh = True
        except FileExistsError:
            fresh = False
        self.lock(job_id)
        if fresh:
            return None
        state = self.read(job_id)
        current = state["stato"] if state else "sconosciuto"
        if (current == "rifiutato" and after_reject) or (current == "assente" and after_absence):
            return state
        self.release(job_id)
        hint = {"rifiutato": "Si risottomette solo con --dopo-rifiuto.",
                "assente": "Si risottomette solo con --dopo-assenza-verificata.",
                "ambiguo": "Non si risottomette: usare --riconcilia."}.get(current, "Usare un ID nuovo.")
        raise SystemExit(f"{job_id}: gia' prenotato, stato {current!r}. {hint}")

    def release(self, job_id: str) -> None:
        self._held.discard(job_id)
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
    if args.dopo:  # lettura senza lock di ALTRI job: basta perche' `sottomesso` e' finale (TRANSIZIONI) e qui non si scrive
        slurm_ids = []
        for d in args.dopo:  # piu' --dopo (catena di valutazione, 15/10/2026): afterok su tutti
            prev = reg.read(d)
            if not prev or prev.get("stato") != "sottomesso" or not prev.get("job_slurm"):
                raise SystemExit(f"--dopo {d}: non e' un job sottomesso con ID SLURM noto")
            slurm_ids.append(str(prev["job_slurm"]))
        dep = [f"--dependency=afterok:{':'.join(slurm_ids)}"]
    script_text = None
    if args.script_stdin:  # lo script locale (versionato) va a sbatch su stdin: non serve che esista sul cluster
        script_text = args.script_stdin.read_text()
        if not script_text.startswith("#!"):
            raise SystemExit(f"{args.script_stdin}: non sembra uno script di sbatch")
    previous = reg.reserve(args.id, after_reject=args.dopo_rifiuto, after_absence=args.dopo_assenza_verificata)  # letto sotto lock
    try:
        full = ["sbatch", "--parsable", f"--job-name={args.id}", f"--comment=wearusfm:{args.id}", *dep, *sbatch_args]
        base = previous or {"id": args.id, "creato": now()}
        state = {**base, "id": args.id, "agente": args.agente, "campagna": args.campagna, "gpu_ore_stimate": args.gpu_ore,
                 "codice_sha": code_sha_from(sbatch_args), "sbatch": full,
                 "dopo": args.dopo[0] if args.dopo and len(args.dopo) == 1 else args.dopo,
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


def _query_jobs(ssh: list[str], remote: str, job_id: str, timeout: float) -> tuple[set[int] | None, str]:
    """ID dei job col nome = `job_id`, da una riga `JobID|JobName|State` per job; None se la risposta non e' affidabile (uscita diversa da
    0, timeout, una riga illeggibile o con un altro nome): in quel caso nulla si deduce."""
    code, out, err = run_remote(ssh, remote, timeout)
    if code != 0:
        return None, f"uscita {code}: {err.strip()[-300:]}"
    ids = set()
    for ln in out.splitlines():
        if not ln.strip():
            continue
        parts = ln.strip().split("|")
        if len(parts) < 3 or not parts[0].isdigit() or parts[1] != job_id:
            return None, f"riga inattesa: {ln.strip()[:200]!r}"
        ids.add(int(parts[0]))
    return ids, ""


def reconcile(reg: Registry, job_id: str, ssh: list[str], timeout: float) -> int:
    """Lock PRIMA di leggere: stato, storia e controlli d'assenza si rileggono sotto il lock, cosi' una riconciliazione concorrente gia'
    chiusa (per esempio a `sottomesso`) non viene sovrascritta da una copia letta prima (review GPT del 09/10/2026)."""
    if not reg.dir(job_id).is_dir():
        raise SystemExit(f"{job_id}: nessuna prenotazione")
    reg.lock(job_id)
    try:
        state = reg.read(job_id)
        if state is None:
            raise SystemExit(f"{job_id}: nessuna prenotazione")
        if state["stato"] == "in_sottomissione":
            # il lock e' libero, quindi chi sottometteva non c'e' piu' (per esempio un Ctrl-C durante ssh): l'esito non si sa, come un ambiguo
            state = reg.transition(job_id, state, "ambiguo", nota="sottomissione interrotta senza esito: trattata come ambigua")
        if state["stato"] != "ambiguo":
            print(f"{job_id}: stato {state['stato']}, niente da riconciliare")
            return 0
        attempt = max(_parse_time(h["quando"]) for h in state.get("storia", []) if h["stato"] == "in_sottomissione")
        since = (attempt - dt.timedelta(days=1)).date().isoformat()  # margine: l'orologio del cluster puo' non coincidere col Mac
        q = shlex.quote(job_id)
        found_acct, why_acct = _query_jobs(ssh, f"sacct -X -n -P -S {since} --name={q} --format=JobID,JobName,State", job_id, timeout)
        found_queue, why_queue = _query_jobs(ssh, f"squeue -h --name={q} -o '%i|%j|%T'", job_id, timeout)
        if found_acct is None or found_queue is None:
            print(f"{job_id}: riconciliazione non affidabile, stato invariato (ambiguo). sacct: {why_acct or 'ok'}; squeue: {why_queue or 'ok'}")
            return 3
        ids = sorted(found_acct | found_queue)
        if len(ids) == 1:
            state = reg.transition(job_id, state, "sottomesso", job_slurm=ids[0], riconciliato=True)
        elif len(ids) > 1:
            state = reg.transition(job_id, state, "duplicato", job_slurm_tutti=ids, riconciliato=True)
        else:  # risposta vuota: un controllo d'assenza, non ancora una prova
            checks = [*state.get("controlli_assenza", []), now()]
            times = [_parse_time(c) for c in checks]
            valid = [t for t in times if t - attempt >= ABSENCE_DELAY]
            confirmed = len(valid) >= 2 and valid[-1] - valid[0] >= ABSENCE_DELAY
            state = reg.transition(job_id, state, "assente" if confirmed else "ambiguo", controlli_assenza=checks, riconciliato=True)
            if not confirmed:
                print(f"{job_id}: nessun job trovato, ma un controllo vuoto non basta (accounting in ritardo?). Controlli validi: {len(valid)}; "
                      f"servono 2, il primo >= 10 min dal tentativo, l'ultimo >= 10 min dopo il primo. Stato: ambiguo")
    finally:
        reg.release(job_id)
    print(f"{job_id}: {state['stato']}" + (f", job {state.get('job_slurm')}" if state.get("job_slurm") else ""))
    return 0 if state["stato"] == "sottomesso" else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id")
    ap.add_argument("--agente", choices=("claude", "gpt", "simone"))
    ap.add_argument("--campagna")
    ap.add_argument("--gpu-ore", type=float, default=None, help="stima dichiarata nel preventivo (0 per i job CPU)")
    ap.add_argument("--dopo", action="append", help="ID logico di un job gia' sottomesso: dipendenza afterok (ripetibile: afterok su tutti)")
    ap.add_argument("--script-stdin", type=Path, help="script locale da passare a sbatch su stdin (es. campaign_checkout.sbatch)")
    ap.add_argument("--dopo-rifiuto", action="store_true", help="risottomette un ID rifiutato da sbatch (dopo aver corretto la causa)")
    ap.add_argument("--dopo-assenza-verificata", action="store_true",
                    help="risottomette un ID in stato 'assente' (due controlli vuoti distanziati, vedi sopra)")
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
