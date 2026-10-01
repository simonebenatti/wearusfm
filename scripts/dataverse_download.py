#!/usr/bin/env python3
"""Scarica da un Dataverse (es. Harvard) i file di uno o piu' dataset filtrati per estensione, con verifica di dimensione e MD5 contro l'API.

Nato per NinaPro DB10 = MeganePro (Simone, 01/10/2026: solo i `.mat` di MDS1, MDS2, MDS4; niente video, niente MDSInfo). Solo libreria standard.
- elenco dei file: `/api/datasets/:persistentId/?persistentId=<doi>` (ultima versione pubblicata);
- un file: `/api/access/datafile/<id>` (per i file tabulari si chiede il formato originale), scritto in `<nome>.part` e rinominato solo dopo che
  dimensione e MD5 coincidono con l'API;
- ripresa: un file gia' presente con dimensione e MD5 giusti si salta; uno sbagliato si riscarica;
- `--time-budget-s`: non comincia un nuovo file oltre il budget (il job si risottomette e riprende); `--dry-run`: elenca e basta.

  python3 scripts/dataverse_download.py --server https://dataverse.harvard.edu \\
      --dataset doi:10.7910/DVN/1Z3IOM=MDS1 --dataset doi:10.7910/DVN/78QFZH=MDS2 --dataset doi:10.7910/DVN/F9R33N=MDS4 \\
      --suffix .mat --dest $SCRATCH/data/raw/ninapro/DB10 --report report.json
Esce con 1 se almeno un file e' fallito.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

CHUNK = 8 * 1024 * 1024
RETRIES = 3
TIMEOUT_S = 120
USER_AGENT = "wearusfm-dataverse-download/1.0"


def _get(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=TIMEOUT_S)


def list_files(server: str, doi: str) -> list[dict]:
    """[{id, filename, size, md5, tabular}] dell'ultima versione del dataset."""
    url = f"{server}/api/datasets/:persistentId/?persistentId={urllib.parse.quote(doi, safe=':/')}"
    with _get(url) as r:
        d = json.load(r)
    if d.get("status") != "OK":
        raise RuntimeError(f"{doi}: risposta {d.get('status')}: {str(d)[:200]}")
    out = []
    for f in d["data"]["latestVersion"]["files"]:
        df = f["dataFile"]
        md5 = df.get("md5") or ((df.get("checksum") or {}).get("value") if (df.get("checksum") or {}).get("type") == "MD5" else None)
        tabular = bool(df.get("originalFileFormat"))
        # per un file tabulare si scarica l'ORIGINALE (?format=original): nome e dimensione sono quelli dell'originale, non del .tab d'archivio
        # (review del codice, 02/10/2026); se l'API non li da', si usa il .tab e la verifica di dimensione salta
        name = df.get("originalFileName") if tabular and df.get("originalFileName") else df["filename"]
        size = df.get("originalFileSize") if tabular else df.get("filesize")
        out.append({"id": df["id"], "filename": name, "directory": f.get("directoryLabel"), "size": size,
                    "md5": md5, "tabular": tabular, "restricted": bool(f.get("restricted"))})
    return out


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def is_complete(path: Path, size: int | None, md5: str | None) -> bool:
    return path.is_file() and (size is None or path.stat().st_size == size) and (md5 is None or md5_of(path) == md5)


def download(server: str, f: dict, dest: Path) -> int:
    """Scarica un file in `dest` passando per `.part`; ritorna i byte scaricati. Solleva se dimensione o MD5 non coincidono."""
    url = f"{server}/api/access/datafile/{f['id']}" + ("?format=original" if f["tabular"] else "")
    part = dest.with_name(dest.name + ".part")
    h, n = hashlib.md5(), 0
    with _get(url) as r, open(part, "wb") as out:
        for b in iter(lambda: r.read(CHUNK), b""):
            out.write(b)
            h.update(b)
            n += len(b)
    if f["size"] is not None and n != f["size"]:
        raise IOError(f"{f['filename']}: {n} byte, attesi {f['size']}")
    if f["md5"] is not None and h.hexdigest() != f["md5"]:
        raise IOError(f"{f['filename']}: MD5 {h.hexdigest()}, atteso {f['md5']}")
    part.replace(dest)
    return n


def run(server: str, datasets: list[tuple[str, str]], suffixes: list[str], dest_root: Path, time_budget_s: float | None, dry_run: bool) -> dict:
    t0 = time.time()
    report: dict = {"server": server, "suffixes": suffixes, "datasets": {}, "stopped": None}
    for doi, label in datasets:
        files = [f for f in list_files(server, doi) if f["filename"].lower().endswith(tuple(s.lower() for s in suffixes))]
        entry = {"doi": doi, "n_files": len(files), "bytes": sum(f["size"] or 0 for f in files), "downloaded": [], "skipped_complete": [],
                 "failed": {}, "restricted": [f["filename"] for f in files if f["restricted"]]}
        report["datasets"][label] = entry
        print(f"=== {label} ({doi}): {len(files)} file, {entry['bytes'] / 1e9:.2f} GB", flush=True)
        if dry_run:
            continue
        out_dir = dest_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            dest = out_dir / (f"{f['directory']}/" if f["directory"] else "") / f["filename"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if is_complete(dest, f["size"], f["md5"]):
                entry["skipped_complete"].append(f["filename"])
                continue
            if time_budget_s is not None and time.time() - t0 > time_budget_s:
                report["stopped"] = "time-budget"
                print(f"budget di tempo esaurito prima di {f['filename']}: si riprende al prossimo lancio", flush=True)
                break
            for attempt in range(1, RETRIES + 1):
                try:
                    t = time.time()
                    n = download(server, f, dest)
                    entry["downloaded"].append(f["filename"])
                    entry["failed"].pop(f["filename"], None)
                    print(f"  ok {f['filename']} {n / 1e6:.1f} MB in {time.time() - t:.0f} s", flush=True)
                    break
                except Exception as e:  # rete, dimensione o MD5: si riprova, poi si registra e si passa al file dopo
                    entry["failed"][f["filename"]] = f"{type(e).__name__}: {e}"
                    print(f"  tentativo {attempt} fallito per {f['filename']}: {entry['failed'][f['filename']]}", flush=True)
        if report["stopped"]:
            break
    report["elapsed_s"] = time.time() - t0
    return report


def _dataset_arg(s: str) -> tuple[str, str]:
    doi, sep, label = s.partition("=")
    if not sep or not doi.startswith("doi:") or not label or "/" in label:
        raise argparse.ArgumentTypeError(f"atteso doi:<...>=<ETICHETTA>, non {s!r}")
    return doi, label


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True)
    ap.add_argument("--dataset", type=_dataset_arg, action="append", required=True, help="doi:<...>=<ETICHETTA> (sottocartella)")
    ap.add_argument("--suffix", action="append", required=True, help="estensione da tenere, es. .mat (ripetibile)")
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--time-budget-s", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    report = run(args.server.rstrip("/"), args.dataset, args.suffix, args.dest, args.time_budget_s, args.dry_run)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    summary = {k: {kk: (len(vv) if isinstance(vv, (list, dict)) else vv) for kk, vv in v.items()} for k, v in report["datasets"].items()}
    print(json.dumps({"stopped": report["stopped"], "elapsed_s": round(report["elapsed_s"]), "datasets": summary}, indent=1))
    return 1 if any(v["failed"] for v in report["datasets"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
