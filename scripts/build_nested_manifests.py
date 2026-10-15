"""Manifest annidati per soggetto (D16) come file congelati, uno per frazione, dal manifest congelato (GPT e Claude, 15/10/2026): stesse
sessioni dei soggetti del sottoinsieme firmato, pesi rifatti con le regole di D9 sulle ore del sottoinsieme e residuo di classe gestito
esplicitamente (`pretraining_manifest.nested_manifest`), provenienza nei parametri.

Prima di scrivere: (1) sulle righe del 100% `assign_weights_nested` riproduce bit per bit i pesi congelati (stesso codice del
costruttore); (2) per ogni frazione la distribuzione campionata (pesi normalizzati) resta entro il tetto di passaggi; (3) i sottoinsiemi sono
annidati. Una verifica fallita: nessun file scritto, uscita 1. L'hash dichiarato dal manifest d'origine e quello ricalcolato dalle sue righe
si registrano entrambi (informativo: in v1.1 il campo `hash` e' rimasto quello di v1).

Uso: python scripts/build_nested_manifests.py manifest.json.gz cartella_di_uscita
     -> <cartella>/<nome>_nested_0.125.json.gz, _0.25, _0.5 e <cartella>/nested_manifests.json (riepilogo con SHA dei file)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wearusfm.data import pretraining_manifest as M  # noqa: E402


def _commit() -> str:
    if os.environ.get("WEARUSFM_CODE_SHA"):
        return os.environ["WEARUSFM_CODE_SHA"]
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def build(manifest: Path) -> tuple[dict, dict[str, dict]]:
    raw = manifest.read_bytes()
    doc = json.loads(gzip.decompress(raw) if manifest.suffix == ".gz" else raw)
    rows = [M.SessionRow(**dict(zip(doc["columns"], r))) for r in doc["rows"]]
    # informativo: il manifest v1.1 (correzione di CapgMyo del 07/10) ha conservato il campo `hash` di v1 senza ricalcolarlo; l'identita' del
    # manifest per trainer e firme e' lo SHA del file. Nei manifest annidati si registrano entrambi
    source_hash = {"declared": doc["hash"], "recomputed": M.manifest_hash(doc["params"], rows)}
    source_hash["equal"] = source_hash["declared"] == source_hash["recomputed"]
    checks = {}
    full = M.sort_rows([M.SessionRow(**dict(zip(doc["columns"], r))) for r in doc["rows"]])
    frozen = {(r.dataset, r.subject, r.session): r.weight for r in full}
    p = doc["params"]
    M.assign_weights_nested(full, p["quota"], p["alpha"], p["max_passes"], p.get("epochs", 4.0))
    checks["full_weights_reproduced_bitwise"] = all(r.weight == frozen[(r.dataset, r.subject, r.session)] for r in full)
    sha = hashlib.sha256(raw).hexdigest()
    commit = _commit()
    out = {f: M.nested_manifest(doc, f, sha, commit, source_hash["recomputed"]) for f in M.NESTED_FRACTIONS}
    subjects = {f: {(r[0], r[1]) for r in d["rows"] if r[d["columns"].index("split")] == "pretraining"} for f, d in out.items()}
    full_subjects = {(r.dataset, r.subject) for r in full if r.split == "pretraining"}
    chain = [subjects[f] for f in M.NESTED_FRACTIONS] + [full_subjects]
    checks["nested_subjects"] = all(a <= b for a, b in zip(chain, chain[1:]))
    checks["within_cap"] = all(d["summary"]["sampled"]["max_passes"] <= p["max_passes"] * (1 + 1e-9) for d in out.values())
    report = {"source": {"path": str(manifest), "file_sha256": sha, "hash": source_hash}, "code_commit": commit, "checks": checks,
              "fractions": {f: {"hash": d["hash"], **d["summary"]["sampled"], "residual": d["params"]["nested_residual"]} for f, d in out.items()}}
    return report, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args(argv)
    report, out = build(args.manifest)
    failed = [k for k, v in report["checks"].items() if not v]
    print(json.dumps({"checks": report["checks"], "fractions": {f: {k: v for k, v in r.items() if k != "residual"} | {"residual": r["residual"]}
                                                                 for f, r in report["fractions"].items()}}, indent=1, default=str))
    if failed:
        print(f"VERIFICHE FALLITE: {failed}: nessun file scritto")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = args.manifest.name.removesuffix(".gz").removesuffix(".json")
    for f, d in out.items():
        path = args.out_dir / f"{base}_nested_{f}.json.gz"
        data = gzip.compress(json.dumps(d, separators=(",", ":"), default=list).encode(), mtime=0)  # mtime fisso: stessi byte a ogni costruzione
        if path.exists() and path.read_bytes() != data:
            print(f"{path} esiste con un contenuto diverso: non lo sovrascrivo")
            return 1
        path.write_bytes(data)
        report["fractions"][f]["file"] = path.name
        report["fractions"][f]["file_sha256"] = hashlib.sha256(data).hexdigest()
    (args.out_dir / "nested_manifests.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"scritti {len(out)} manifest annidati in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
