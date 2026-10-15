"""Catena di valutazione della finestra 1 (accordo GPT-Claude del 15/10/2026, `docs/accordo_operativo_finestra1_20261015.md`, «Interfaccia
della catena di valutazione»): legge il piano di valutazione di GPT, risolve le dipendenze training -> estrazione -> sonde -> confronto e le
sottomette col submitter comune (`scripts/submit_job.py`: prenotazione, lock, ledger, nessun rilancio alla cieca). Default: dry-run.

GPT possiede harness, estrazioni, sonde, confronti e i loro launcher, e scrive il piano; questo script li concatena soltanto. Lo schema qui
sotto e' PROVVISORIO (`eval-plan-v0`) finche' GPT non pubblica `docs/evaluation_chain_contract.md`; si adatta a quello.

Piano (JSON):
  schema: "eval-plan-v0"; campaign; agent
  guard: {code_dir, expected_sha, report_root}   checkout e cartella dei rapporti delle guardie dei checkpoint
  limits: {coda: massimo di job in coda e in esecuzione}   facoltativo; la coda e' una partizione o una QoS (es. "normal", "lrd_all_serial")
  nodes: lista di
    {id, kind: extract | probe | compare, after: [ID logici di nodi o di job di training], queue, gpu_hours,
     code_dir, expected_sha (40 caratteri), launcher (percorso nel checkout), args: [...], exports: {CHIAVE: valore}, sbatch: [...],
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
- `limits` conta i job gia' in coda: con `--submit` i conteggi si leggono da `squeue --me` (o da `--queue-counts`); senza conteggi non si
  sottomette.
- Rieseguibile: le dipendenze non ancora sottomesse fanno aspettare il nodo fino alla prossima chiamata. Uscite assolute e fuori dal codice.

Uso: python scripts/evaluation_chain.py piano.json                       dry-run: stato e comandi nell'ordine
     python scripts/evaluation_chain.py piano.json --submit [--ssh "ssh leonardo"] [--registro DIR]   sottomette (via leonardo-ops)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "eval-plan-v0"
KINDS = ("extract", "probe", "compare", "guard")
FORBIDDEN_SBATCH = ("--job-name", "-J", "--comment", "--dependency", "-d")
RUNNING = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "CONFIGURING", "COMPLETING", "REQUEUE_HOLD", "REQUEUE_FED", "SIGNALING",
           "STAGE_OUT"}


def _submitter():
    spec = importlib.util.spec_from_file_location("submit_job", ROOT / "scripts" / "submit_job.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
        if any(a.split("=")[0] in FORBIDDEN_SBATCH for a in n.get("sbatch", [])):
            raise ValueError(f"{n['id']}: nome, commento e dipendenze li impongono catena e submitter")
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
            elif n.get("queue") in limits and used.get(n["queue"], 0) >= limits[n["queue"]]:
                s, why = "in attesa", f"coda {n['queue']} al limite di {limits[n['queue']]} job"
            else:
                s = "sottomesso ora"
                used[n.get("queue")] = used.get(n.get("queue"), 0) + 1
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
    slurm_state = None
    if args.submit:
        try:
            if plan.get("limits") and counts is None:
                counts = live_queue_counts(S, ssh, sorted(plan["limits"]))
            deps = sorted({d for n in order for d in n.get("after", []) if (read_state(d) or {}).get("stato") == "sottomesso"})
            known = slurm_states(S, ssh, read_state, deps)
            slurm_state = lambda i: known.get(i, "sconosciuto")  # noqa: E731
        except RuntimeError as e:
            print(f"stato del cluster non leggibile, nessuna sottomissione: {e}")
            return 1
    rows = schedule(plan, read_state, counts, slurm_state)
    failed = 0
    for r in rows:
        print(f"{r['status']:16s} {r['kind']:8s} {r['id']}" + (f"  ({r['why']})" if r["why"] else ""))
        if r["status"] != "sottomesso ora":
            continue
        print("    python3 scripts/submit_job.py " + " ".join(shlex.quote(x) for x in r["command"]))
        if args.submit:
            extra = ["--registro", str(args.registro)] if args.registro else []
            try:
                code = S.main([*extra, "--ssh", args.ssh, *r["command"]])
            except SystemExit as e:  # il submitter rifiuta (prenotazione esistente, dipendenza non valida): ci si ferma con un messaggio
                code, msg = 3, str(e)
                print(f"    {r['id']}: rifiutato dal submitter: {msg}")
            if code != 0:  # un esito diverso da `sottomesso` ferma la chiamata: il resto aspetta il chiarimento
                print(f"    {r['id']}: non sottomesso (uscita {code}); mi fermo")
                failed = 1
                break
    return failed


if __name__ == "__main__":
    sys.exit(main())
