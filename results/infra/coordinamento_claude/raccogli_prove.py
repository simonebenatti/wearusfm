#!/usr/bin/env python3
"""Raccoglie nel repo le letture dei job di Leonardo fatte dal coordinamento di Claude e una copia del ledger comune.

Uso, dalla radice del worktree di Claude, prima di ogni commit di coordinamento:

    ~/.venvs/wearusfm-tok/bin/python -I results/infra/coordinamento_claude/raccogli_prove.py

Legge soltanto i sorgenti (`~/wearusfm_local`): fuori da questa cartella non scrive e non cancella nulla.
Decisione di Simone del 06/11/2026: alla domanda «Metto nei commit anche le letture dei job e una copia del ledger?»
ha risposto «Sì, aggiungili». Che cosa entra e che cosa resta fuori: README.md.
"""
import hashlib
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC_LOGS = pathlib.Path.home() / "wearusfm_local" / "logs"
SRC_LEDGER = pathlib.Path.home() / "wearusfm_local" / "jobs" / "ledger.tsv"
OUT_LOGS = HERE / "letture_leonardo"
MANIFEST = HERE / "MANIFEST.tsv"
ACCOUNT = "iscrb_wearusfm"

# Letture del coordinamento che entrano nel repo: stato dei job, code, sacct, priorita' dei nostri job, quota, riferimenti git.
INCLUDE = ("leonardo_stato_", "leonardo_seriale_", "leonardo_squeue_", "leonardo_sacct_", "leonardo_priorita_",
           "leonardo_quota_", "leonardo_lsremote_", "leonardo_audit_sacct_")
# Resta solo sul Mac la coda intera della partizione: sono job di altri utenti.
EXCLUDE_SUBSTR = ("_coda",)
# Altro progetto dello stesso utente: le sue righe si tolgono dalle copie.
# Simone, 05/11/2026: «che c'entrano, sono di un a ltro progetto».
OTHER = re.compile(r"(^|[|\s/=:,;\"'])(mona_|t1_|t2_)|MONALISA", re.IGNORECASE)
# Riga della tabella di `saldo` (account, inizio, fine, ore): resta solo quella del nostro account.
SALDO_ROW = re.compile(r"^\s*([A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+)\s+20\d{6}\s+20\d{6}\s+\d+")
# Righe scritte dal client ssh del Mac, non da Leonardo: i tre avvisi e la riga col certificato e l'indirizzo di posta.
SSH_NOISE = re.compile(r"^\*\* (WARNING: connection is not using a post-quantum|This session may be vulnerable to"
                       r"|The server may need to be upgraded)|^The key \S+ \S+ .*is already present in the SSH agent")
# Se una riga da copiare somiglia a un segreto, a una chiave o a un indirizzo di posta, ci si ferma e si guarda.
SECRET = re.compile(r"(?i:passw|secret|api[_-]?key|token\s*=)|BEGIN [A-Z ]*PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}"
                    r"|ssh-(?:rsa|ed25519|dss) AAAA|-cert-v01@openssh\.com|AAAA[0-9A-Za-z+/]{60,}"
                    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
HEADER = ("file", "sha256_originale", "righe_originale", "righe_altro_progetto", "righe_ssh", "sha256_copia")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def other_project(line: str) -> bool:
    if OTHER.search(line):
        return True
    m = SALDO_ROW.match(line)
    return bool(m) and m.group(1).lower() != ACCOUNT


def write_if_changed(dest: pathlib.Path, data: bytes) -> int:
    if dest.exists() and dest.read_bytes() == data:
        return 0
    dest.write_bytes(data)
    return 1


def main() -> int:
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    old = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").split("\n"):
            fields = line.split("\t")
            if len(fields) == len(HEADER) and fields[0] != "file":
                old[fields[0]] = tuple(fields)
    rows, written, n_other, n_ssh, blank = {}, 0, 0, 0, []
    for src in sorted(SRC_LOGS.glob("leonardo_*")):
        name = src.name
        if not src.is_file() or not name.startswith(INCLUDE) or any(s in name for s in EXCLUDE_SUBSTR):
            continue
        raw = src.read_bytes()
        lines = raw.decode("utf-8", errors="replace").split("\n")
        ssh = [line for line in lines if SSH_NOISE.search(line)]
        other = [line for line in lines if not SSH_NOISE.search(line) and other_project(line)]
        kept = [line for line in lines if not SSH_NOISE.search(line) and not other_project(line)]
        if not "".join(kept).strip():
            blank.append(name)
            continue
        for line in kept:
            if SECRET.search(line):
                print(f"FERMO: possibile segreto, chiave o indirizzo in {name}: {line[:120]}")
                return 2
        data = "\n".join(kept).encode("utf-8")
        written += write_if_changed(OUT_LOGS / name, data)
        n_other += len(other)
        n_ssh += len(ssh)
        key = f"letture_leonardo/{name}"
        rows[key] = (key, sha(raw), len(lines), len(other), len(ssh), sha(data))

    # Copie gia' nel repo il cui sorgente non c'e' piu' sul Mac: restano, con la riga di prima.
    orphans = []
    for copy in sorted(OUT_LOGS.iterdir()):
        key = f"letture_leonardo/{copy.name}"
        if key in rows:
            continue
        if key in old and old[key][5] == sha(copy.read_bytes()):
            rows[key] = old[key]
        else:
            orphans.append(copy.name)

    ledger = SRC_LEDGER.read_bytes()
    if not ledger.endswith(b"\n"):
        print("FERMO: il ledger non finisce con un a capo, forse e' in scrittura: riprovare")
        return 2
    text = ledger.decode("utf-8")
    if SECRET.search(text):
        print("FERMO: possibile segreto, chiave o indirizzo nel ledger")
        return 2
    suspicious = sum(1 for line in text.split("\n") if other_project(line))
    if suspicious:
        print(f"FERMO: {suspicious} righe del ledger sembrano dell'altro progetto: guardarle prima di copiare")
        return 2
    n_fields = {len(line.split("\t")) for line in text.split("\n")[:-1]}
    if n_fields != {9}:
        print(f"FERMO: il ledger ha righe con un numero di campi diverso da 9: {sorted(n_fields)}")
        return 2
    written += write_if_changed(HERE / "ledger_copia.tsv", ledger)
    n_ledger = len(text.split("\n")) - 1
    rows["ledger_copia.tsv"] = ("ledger_copia.tsv", sha(ledger), n_ledger, 0, 0, sha(ledger))

    out = ["# sorgenti in ~/wearusfm_local; righe_altro_progetto e righe_ssh = righe tolte dalla copia (vedi README.md)",
           "\t".join(HEADER)]
    out += ["\t".join(str(v) for v in rows[key]) for key in sorted(rows)]
    written += write_if_changed(MANIFEST, ("\n".join(out) + "\n").encode("utf-8"))
    size = sum((HERE / key).stat().st_size for key in rows)
    print(f"letture: {len(rows) - 1} file; righe tolte: {n_other} dell'altro progetto, {n_ssh} del client ssh; "
          f"ledger: {n_ledger} righe; file scritti o aggiornati: {written}; totale {size / 1e6:.2f} MB")
    if blank:
        print("letture senza contenuto utile, non copiate:", ", ".join(blank))
    if orphans:
        print("ATTENZIONE: copie senza sorgente e senza riga nel manifest, da guardare:", ", ".join(orphans))
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
