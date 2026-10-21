"""Catena di valutazione della finestra 1 (accordo GPT-Claude del 15/10/2026, `docs/accordo_operativo_finestra1_20261015.md`, «Interfaccia
della catena di valutazione»): legge il piano di valutazione di GPT, risolve le dipendenze training -> estrazione -> sonde -> confronto e le
sottomette col submitter comune (`scripts/submit_job.py`: prenotazione, lock, ledger, nessun rilancio alla cieca). Default: dry-run.

GPT possiede harness, estrazioni, sonde, confronti e i loro launcher, e scrive il piano; questo script li concatena soltanto. Lo schema qui
sotto e' PROVVISORIO (`eval-plan-v0`) finche' GPT non pubblica `docs/evaluation_chain_contract.md`; si adatta a quello.

Piano (JSON):
  schema: "eval-plan-v0"; campaign; agent
  guard: {code_dir, expected_sha, report_root}   checkout e cartella dei rapporti delle guardie dei checkpoint
  limits: {coda: massimo di job in coda e in esecuzione}   facoltativo; la coda e' una partizione o una QoS (es. "normal", "lrd_all_serial"):
    un nodo conta per `queue`, per la partizione e per la QoS effettive (opzioni del nodo, intestazione del launcher, QoS di default)
  nodes: lista di
    {id, kind: extract | probe | compare, after: [ID logici di nodi o di job di training], queue, gpu_hours,
     code_dir, expected_sha (40 caratteri), launcher (percorso nel checkout), args: [...], exports: {CHIAVE: valore},
     sbatch: [...] (elenco chiuso di opzioni per intero, `SBATCH_OPTIONS`; mai nome, commento, dipendenze, export o array),
     inputs: [...], outputs: [...],
     checkpoint: {run_dir, name, training_job, min_steps, sha256}   facoltativo: il checkpoint che il nodo valuta}
  `training_job` e' l'ultimo job della catena di training (scorta compresa): la guardia controlla che il run sia completo.

Regole:
- Un nodo con `checkpoint` dipende da una guardia (`scripts/slurm/check_checkpoint.sbatch`) che dipende dal job di training e verifica run
  finito, niente STOP, checkpoint presente, passi minimi e, se dichiarato, SHA-256; se fallisce la valutazione non parte mai. Una guardia per
  checkpoint E requisiti (passi minimi, SHA): nodi con requisiti diversi hanno guardie diverse.
- Un nodo che legge l'uscita di un altro nodo deve dipenderne direttamente (il confronto parte solo con tutti gli output richiesti).
- Stato dal ledger del submitter: mai prenotato -> sottomettibile; `sottomesso` -> si salta; qualunque altro stato, o una prenotazione senza
  stato, blocca il nodo e cio' che ne dipende (si chiarisce col submitter, a mano; mai rilancio automatico).
- Con `--submit` lo stato SLURM dei predecessori gia' sottomessi si legge con `sacct`: finito bene -> nessun afterok (SLURM puo' aver gia'
  dimenticato il job e rifiuterebbe la dipendenza); in coda o in esecuzione -> afterok; finito male -> nodo bloccato; sconosciuto -> il nodo
  aspetta. Nel dry-run lo stato SLURM non si legge (afterok su tutti, indicato).
- Con `--submit` i nodi si sottomettono uno alla volta e prima di ciascuno lo stato si rilegge con `sacct`, compresi i predecessori
  sottomessi nella stessa chiamata: una guardia che finisce in pochi secondi non riceve afterok dal nodo che la segue (C1, 19/10/2026: sbatch
  aveva rifiutato l'estrazione con «Job dependency problem», 75 s dopo la fine della guardia); se fallisce, cio' che ne dipende si blocca.
  Un predecessore della stessa chiamata che sacct non vede ancora e' appena stato accettato da sbatch: afterok.
- `limits` conta i job gia' in coda: con `--submit` i conteggi si leggono da `squeue --me` (o da `--queue-counts`); senza conteggi non si
  sottomette.
- Rieseguibile: le dipendenze non ancora sottomesse fanno aspettare il nodo fino alla prossima chiamata. Uscite assolute e fuori dal codice.

Uso: python scripts/evaluation_chain.py piano.json                       dry-run: stato e comandi nell'ordine
     python scripts/evaluation_chain.py piano.json --submit [--ssh "ssh leonardo"] [--registro DIR]   sottomette (via leonardo-ops)
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "eval-plan-v0"
KINDS = ("extract", "probe", "compare", "guard")
# nome, commento, dipendenze ed export li impongono catena e submitter; un array moltiplicherebbe i job contati come uno
FORBIDDEN_SBATCH = ("--job-name", "-J", "--comment", "--dependency", "-d", "--export", "--export-file", "--array", "-a", "--wrap")
# opzioni ammesse in `sbatch` di un nodo, scritte per intero: sbatch accetta le abbreviazioni delle opzioni lunghe (`--exp=`, `--dep=`),
# quindi un elenco chiuso. Il valore va dopo "=" (opzioni lunghe) o nell'elemento successivo.
SBATCH_OPTIONS = {"--time": "--time", "-t": "--time", "--partition": "--partition", "-p": "--partition", "--qos": "--qos", "-q": "--qos",
                  "--nodes": "--nodes", "-N": "--nodes", "--ntasks": "--ntasks", "-n": "--ntasks", "--ntasks-per-node": "--ntasks-per-node",
                  "--cpus-per-task": "--cpus-per-task", "-c": "--cpus-per-task", "--mem": "--mem", "--mem-per-cpu": "--mem-per-cpu",
                  "--gres": "--gres", "--gpus": "--gpus", "--gpus-per-node": "--gpus-per-node", "--output": "--output", "-o": "--output",
                  "--error": "--error", "-e": "--error", "--constraint": "--constraint", "-C": "--constraint", "--nice": "--nice"}
SBATCH_FLAGS = ("--exclusive",)
# QoS che SLURM assegna a un job senza --qos, per partizione: nei limiti conta come quella esplicita (squeue la riporta). sacct del
# 15/10/2026 su job con launcher senza QoS: lrd_all_serial 60664574/84/86 -> normal; boost_usr_prod 59306259 (rvq_codes), 59412516
# (probe_p2) -> normal. Quindi su Leonardo un limite "normal" conta anche i job seriali; per i soli nodi GPU si usa "boost_usr_prod".
DEFAULT_QOS = {"lrd_all_serial": "normal", "boost_usr_prod": "normal"}
RUNNING = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "CONFIGURING", "COMPLETING", "REQUEUE_HOLD", "REQUEUE_FED", "SIGNALING",
           "STAGE_OUT"}


def _submitter():
    spec = importlib.util.spec_from_file_location("submit_job", ROOT / "scripts" / "submit_job.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sbatch_options(opts) -> dict[str, str]:
    """Le opzioni sbatch di un nodo coi nomi per intero ({"--partition": "boost_usr_prod", "--exclusive": ""}); errore su opzioni vietate,
    fuori elenco, ripetute o senza valore."""
    out, opts, i = {}, [str(o) for o in opts], 0
    while i < len(opts):
        a = opts[i]
        name, eq, val = a.partition("=") if a.startswith("--") else (a, "", "")
        if name in SBATCH_FLAGS and not eq:
            key, val = name, ""
        elif name in SBATCH_OPTIONS:
            key = SBATCH_OPTIONS[name]
            if not eq:
                if i + 1 >= len(opts) or opts[i + 1].startswith("-"):
                    raise ValueError(f"opzione sbatch {a}: manca il valore")
                i, val = i + 1, opts[i + 1]
        elif (name.startswith("--") and len(name) > 2 and any(f.startswith(name) for f in FORBIDDEN_SBATCH if f.startswith("--"))) or \
                (not name.startswith("--") and name[:2] in FORBIDDEN_SBATCH):
            raise ValueError(f"opzione sbatch {a}: nome, commento, dipendenze, export e array li impongono catena e submitter")
        else:
            raise ValueError(f"opzione sbatch {a} non ammessa (ammesse, per intero: {sorted(set(SBATCH_OPTIONS.values()) | set(SBATCH_FLAGS))})")
        if key in out:
            raise ValueError(f"opzione sbatch {key} ripetuta")
        if key == "--nice" and not (val.isdigit() and int(val) > 0):  # un nice negativo alzerebbe la priorita' (e richiede l'amministratore)
            raise ValueError(f"opzione sbatch --nice={val}: solo valori interi positivi")
        out[key] = val
        i += 1
    return out


@functools.lru_cache(maxsize=None)
def launcher_header(sha: str, launcher: str) -> tuple[tuple[str, str], ...]:
    """Partizione e QoS dell'intestazione #SBATCH del launcher allo SHA atteso, dal repository locale (vuota se non leggibile: SHA o file non
    presenti qui). sbatch legge le direttive fino al primo comando."""
    try:
        text = subprocess.run(["git", "-C", str(ROOT), "show", f"{sha}:{launcher}"], capture_output=True, text=True, check=True,
                              timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    found = {}
    for line in text.splitlines()[1:]:
        line = line.strip()
        if line and not line.startswith("#"):
            break
        if not line.startswith("#SBATCH"):
            continue
        toks = shlex.split(line[len("#SBATCH"):], comments=True)
        for k, t in enumerate(toks):
            name, eq, val = t.partition("=") if t.startswith("--") else (t, "", "")
            if name in ("-p", "--partition", "-q", "--qos"):
                found[SBATCH_OPTIONS[name]] = val if eq else (toks[k + 1] if k + 1 < len(toks) else "")
    return tuple(sorted(found.items()))


def node_labels(n: dict) -> set[str]:
    """Le etichette di coda del nodo per i limiti: `queue`, la partizione e la QoS effettive (opzioni del nodo, altrimenti l'intestazione del
    launcher allo SHA atteso; senza QoS, quella di default della partizione). Un job conta per tutte: squeue riporta partizione e QoS."""
    opts = sbatch_options(n.get("sbatch", []))
    head = dict(launcher_header(n["expected_sha"], n["launcher"]))
    part = opts.get("--partition") or head.get("--partition") or (n.get("queue") if n.get("queue") in DEFAULT_QOS else None)
    qos = opts.get("--qos") or head.get("--qos") or DEFAULT_QOS.get(part)
    return {x for x in (n.get("queue"), part, qos) if x}


def _inside_code(path: str) -> bool:
    return bool(re.search(r"/wearusfm(_campaigns)?(/|$)", path))


def expand(plan: dict) -> list[dict]:
    """I nodi del piano piu' una guardia per ogni (checkpoint, requisiti); i nodi con checkpoint dipendono dalla guardia al posto del training."""
    nodes, guards = [], {}
    g = plan.get("guard") or {}
    for n in plan["nodes"]:
        n = dict(n)
        ck = n.get("checkpoint")
        if ck:
            key = json.dumps([ck["run_dir"], ck["name"], ck.get("min_steps"), ck.get("sha256")])
            gid = f"{ck['training_job']}-guard-{hashlib.sha1(key.encode()).hexdigest()[:8]}"
            if gid not in guards:
                args = [ck["run_dir"], ck["name"], f"{g['report_root']}/{gid}.json"]
                if ck.get("min_steps") is not None:
                    args += ["--min-steps", str(ck["min_steps"])]
                if ck.get("sha256"):
                    args += ["--sha256", ck["sha256"]]
                guards[gid] = {"id": gid, "kind": "guard", "after": [ck["training_job"]], "queue": "lrd_all_serial", "gpu_hours": 0,
                               "code_dir": g["code_dir"], "expected_sha": g["expected_sha"], "launcher": "scripts/slurm/check_checkpoint.sbatch",
                               "args": args, "exports": {}, "sbatch": [], "inputs": [f"{ck['run_dir']}/{ck['name']}"], "outputs": [args[2]]}
            n["after"] = [a for a in n.get("after", []) if a != ck["training_job"]] + [gid]
        nodes.append(n)
    return list(guards.values()) + nodes


def validate(plan: dict) -> list[dict]:
    """Errori espliciti su piani malformati; ritorna i nodi espansi in ordine topologico."""
    if plan.get("schema") != SCHEMA:
        raise ValueError(f"schema {plan.get('schema')!r}: atteso {SCHEMA}")
    if any(n.get("checkpoint") for n in plan["nodes"]) and not {"code_dir", "expected_sha", "report_root"} <= set(plan.get("guard") or {}):
        raise ValueError("guard: servono code_dir, expected_sha e report_root per le guardie dei checkpoint")
    nodes = expand(plan)
    ids = [n["id"] for n in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError(f"ID logici ripetuti: {sorted({i for i in ids if ids.count(i) > 1})}")
    by_id = {n["id"]: n for n in nodes}
    producer = {}
    for n in nodes:
        if n["kind"] not in KINDS:
            raise ValueError(f"{n['id']}: tipo {n['kind']!r} non fra {KINDS[:3]}")
        if not re.fullmatch(r"[0-9a-f]{40}", str(n.get("expected_sha", ""))):
            raise ValueError(f"{n['id']}: expected_sha deve essere uno SHA completo (40 caratteri)")
        if not n.get("code_dir") or Path(n["launcher"]).is_absolute() or ".." in Path(n["launcher"]).parts:
            raise ValueError(f"{n['id']}: code_dir obbligatorio e launcher relativo al checkout")
        for k, v in n.get("exports", {}).items():
            if k in ("CODE_DIR", "EXPECTED_SHA") or "," in f"{k}={v}":
                raise ValueError(f"{n['id']}: export {k} non ammesso (CODE_DIR/EXPECTED_SHA li mette la catena; niente virgole)")
        try:
            sbatch_options(n.get("sbatch", []))
        except ValueError as e:
            raise ValueError(f"{n['id']}: {e}") from None
        for o in n.get("outputs", []):
            if not o.startswith("/") or _inside_code(o):
                raise ValueError(f"{n['id']}: uscita {o} relativa o dentro il codice")
            if o in producer:
                raise ValueError(f"{n['id']}: uscita {o} gia' prodotta da {producer[o]}")
            producer[o] = n["id"]
        if n["kind"] == "compare" and not n.get("after"):
            raise ValueError(f"{n['id']}: un confronto dipende dalle sonde che confronta")
    for n in nodes:
        for i in n.get("inputs", []):
            if i in producer and producer[i] not in n.get("after", []):
                raise ValueError(f"{n['id']}: legge {i} di {producer[i]} senza dipenderne")
    order, seen, visiting = [], set(), set()

    def visit(i):
        if i in seen or i not in by_id:
            return
        if i in visiting:
            raise ValueError(f"dipendenze circolari attraverso {i}")
        visiting.add(i)
        for d in by_id[i].get("after", []):
            visit(d)
        visiting.discard(i)
        seen.add(i)
        order.append(by_id[i])

    for i in ids:
        visit(i)
    return order


def command(plan: dict, n: dict, after: list[str] | None = None) -> list[str]:
    """Gli argomenti di submit_job.py per il nodo (dopo `python3 scripts/submit_job.py`); `after` = le dipendenze da mettere in afterok."""
    exports = [f"CODE_DIR={n['code_dir']}", f"EXPECTED_SHA={n['expected_sha']}"] + [f"{k}={v}" for k, v in n.get("exports", {}).items()]
    argv = ["--id", n["id"], "--agente", plan.get("agent", "claude"), "--campagna", plan["campaign"], "--gpu-ore", f"{n.get('gpu_hours', 0):g}"]
    for d in (n.get("after", []) if after is None else after):
        argv += ["--dopo", d]
    return argv + ["--", "--export=ALL," + ",".join(exports), *n.get("sbatch", []), f"{n['code_dir']}/{n['launcher']}", *n.get("args", [])]


def schedule(plan: dict, read_state, queue_counts: dict | None = None, slurm_state=None) -> list[dict]:
    """Per ogni nodo, nell'ordine: gia' sottomesso / bloccato / in attesa / da sottomettere (entro i limiti di coda), col comando.
    `read_state(id)`: lo stato del ledger (None = mai prenotato). `slurm_state(id)`: per un job gia' sottomesso, "finito" / "in corso" /
    "fallito" / "sconosciuto"; None nel dry-run (afterok su tutto)."""
    order = validate(plan)
    node_ids = {n["id"] for n in order}
    limits = plan.get("limits", {})
    used = dict(queue_counts or {})
    status: dict[str, str] = {}
    out = []
    for n in order:
        st = (read_state(n["id"]) or {}).get("stato")
        why, after = "", list(n.get("after", []))
        if st == "sottomesso":
            s = "gia' sottomesso"
        elif st is not None:
            s, why = "bloccato", f"stato {st} nel ledger: si chiarisce col submitter (--riconcilia, --stato), mai rilancio automatico"
        else:
            blocked, pending, done = [], [], []
            for d in n.get("after", []):
                if d in node_ids and status[d] == "sottomesso ora":
                    continue  # sottomesso in questa stessa chiamata: in coda, afterok
                dst = status[d] if d in node_ids else ("gia' sottomesso" if (read_state(d) or {}).get("stato") == "sottomesso"
                                                       else (read_state(d) or {}).get("stato") or "non sottomesso")
                if dst == "bloccato" or dst not in ("gia' sottomesso", "non sottomesso", "in attesa"):
                    blocked.append(d)
                elif dst != "gia' sottomesso":
                    pending.append(d)
                elif slurm_state is not None:
                    js = slurm_state(d)
                    if js == "finito":
                        done.append(d)
                    elif js == "fallito":
                        blocked.append(d)
                    elif js != "in corso":
                        pending.append(d)
            after = [d for d in n.get("after", []) if d not in done]
            if blocked:
                s, why = "bloccato", f"dipende da {blocked}, bloccati o falliti"
            elif pending:
                s, why = "in attesa", f"dipendenze non ancora sottomesse o di stato SLURM sconosciuto: {pending}"
            elif full := sorted(q for q in node_labels(n) if q in limits and used.get(q, 0) >= limits[q]):
                s, why = "in attesa", "; ".join(f"coda {q} al limite di {limits[q]} job" for q in full)
            else:
                s = "sottomesso ora"
                for q in node_labels(n):
                    used[q] = used.get(q, 0) + 1
                if done:
                    why = f"senza afterok su {done}: gia' finiti bene secondo sacct"
                elif slurm_state is None and after:
                    why = "afterok su tutte le dipendenze (dry-run: stato SLURM non letto)"
        status[n["id"]] = s
        out.append({"id": n["id"], "kind": n["kind"], "status": s, "why": why, "after": after, "command": command(plan, n, after)})
    return out


def _ledger_reader(reg):
    """Stato del ledger; una prenotazione senza stato (cartella senza state.json) e' uno stato a se', bloccante."""
    def read(i):
        s = reg.read(i)
        if s is None and (reg.root / i).exists():
            return {"stato": "prenotazione senza stato"}
        return s
    return read


def slurm_states(S, ssh: list[str], read_state, ids: list[str], timeout: float = 120.0) -> dict[str, str]:
    """Stato SLURM dei job gia' sottomessi `ids` (ID logici) da `sacct`. Errore se sacct non risponde in modo leggibile."""
    slurm = {i: str((read_state(i) or {}).get("job_slurm")) for i in ids if (read_state(i) or {}).get("job_slurm")}
    if not slurm:
        return {}
    code, out, err = S.run_remote(ssh, f"sacct -X -n -P --format=JobID,State,ExitCode -j {','.join(sorted(set(slurm.values())))}", timeout)
    if code != 0:
        raise RuntimeError(f"sacct non leggibile (uscita {code}): {err.strip()[-300:]}")
    rows = {}
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            raise RuntimeError(f"riga di sacct illeggibile: {line!r}")
        rows[parts[0]] = (parts[1].split()[0], parts[2])
    res = {}
    for i, j in slurm.items():
        if j not in rows:
            res[i] = "sconosciuto"
        else:
            state, exit_code = rows[j]
            res[i] = "finito" if state == "COMPLETED" and exit_code == "0:0" else "in corso" if state in RUNNING else "fallito"
    return res


def live_queue_counts(S, ssh: list[str], labels: list[str], timeout: float = 120.0) -> dict[str, int]:
    """Job dell'utente in coda e in esecuzione per etichetta (partizione o QoS), da `squeue --me`."""
    code, out, err = S.run_remote(ssh, "squeue --me -h -o '%P|%q'", timeout)
    if code != 0:
        raise RuntimeError(f"squeue non leggibile (uscita {code}): {err.strip()[-300:]}")
    counts = {lab: 0 for lab in labels}
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split("|")
        if len(parts) != 2:
            raise RuntimeError(f"riga di squeue illeggibile: {line!r}")
        for lab in labels:
            counts[lab] += lab in parts
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plan", type=Path)
    ap.add_argument("--submit", action="store_true", help="sottomette i nodi pronti (senza: dry-run)")
    ap.add_argument("--registro", type=Path, default=None, help="cartella del ledger del submitter (default quella di submit_job.py)")
    ap.add_argument("--ssh", default="ssh leonardo")
    ap.add_argument("--queue-counts", type=Path, help="JSON {coda: job gia' in coda e in esecuzione}; con --submit, senza, si legge squeue --me")
    args = ap.parse_args(argv)
    S = _submitter()
    reg = S.Registry(args.registro or S.DEFAULT_DIR)
    read_state = _ledger_reader(reg)
    plan = json.loads(args.plan.read_text())
    order = validate(plan)
    ssh = shlex.split(args.ssh)
    counts = json.loads(args.queue_counts.read_text()) if args.queue_counts else None
    if not args.submit:
        for r in schedule(plan, read_state, counts, None):
            _print_row(r)
        return 0
    try:
        if plan.get("limits") and counts is None:
            counts = live_queue_counts(S, ssh, sorted(plan["limits"]))
    except RuntimeError as e:
        print(f"stato del cluster non leggibile, nessuna sottomissione: {e}")
        return 1
    nodes = {n["id"]: n for n in order}
    used = dict(counts or {})
    now: list[str] = []  # sottomessi in questa chiamata, nell'ordine
    while True:  # un nodo per giro, con lo stato SLURM riletto prima di ciascuno (anche dei predecessori appena sottomessi)
        try:
            deps = sorted({d for n in order for d in n.get("after", []) if (read_state(d) or {}).get("stato") == "sottomesso"})
            known = slurm_states(S, ssh, read_state, deps)
        except RuntimeError as e:
            print(f"stato del cluster non leggibile, " + (f"mi fermo dopo {now}: {e}" if now else f"nessuna sottomissione: {e}"))
            return 1
        def slurm_state(i):  # sconosciuto a sacct ma appena accettato da sbatch in questa chiamata: in coda, afterok
            js = known.get(i, "sconosciuto")
            return "in corso" if js == "sconosciuto" and i in now else js

        rows = schedule(plan, read_state, used, slurm_state)
        r = next((r for r in rows if r["status"] == "sottomesso ora"), None)
        if r is None:
            break
        _print_row(r)
        extra = ["--registro", str(args.registro)] if args.registro else []
        try:
            code = S.main([*extra, "--ssh", args.ssh, *r["command"]])
        except SystemExit as e:  # il submitter rifiuta (prenotazione esistente, dipendenza non valida): ci si ferma con un messaggio
            code, msg = 3, str(e)
            print(f"    {r['id']}: rifiutato dal submitter: {msg}")
        if code != 0:  # un esito diverso da `sottomesso` ferma la chiamata: il resto aspetta il chiarimento
            print(f"    {r['id']}: non sottomesso (uscita {code}); mi fermo")
            return 1
        now.append(r["id"])
        for q in node_labels(nodes[r["id"]]):
            used[q] = used.get(q, 0) + 1
    for r in rows:
        if r["id"] not in now:
            _print_row(r)
    return 0


def _print_row(r: dict) -> None:
    print(f"{r['status']:16s} {r['kind']:8s} {r['id']}" + (f"  ({r['why']})" if r["why"] else ""))
    if r["status"] == "sottomesso ora":
        print("    python3 scripts/submit_job.py " + " ".join(shlex.quote(x) for x in r["command"]))


if __name__ == "__main__":
    sys.exit(main())
