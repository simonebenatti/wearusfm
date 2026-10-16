"""Rapporto sui manifest dell'asse dei soggetti (D16): il genitore (v1.1) e i derivati come file (`scripts/build_nested_manifests.py`).
Per ogni file, sulla distribuzione che campiona davvero il loader (`pretraining_manifest.sampled_distribution`: normalizzazione dei pesi,
probabilita' peso / durata delle finestre, quote nel tempo): massa, quote per classe e per unita', passaggi attesi per unita' a E epoche, ore
campionabili ed escluse, endpoint a E epoche col floor sul totale e dose effettiva. Verifiche sui derivati: frazione dichiarata, pesi finiti e
non negativi, massa 1, quota di A, tetto; annidamento per soggetti e per sessioni (12,5 ⊂ 25 ⊂ 50 ⊂ 100); il derivato al 100% con le stesse
sessioni di pretraining del genitore; righe di test e benchmark identiche al genitore (test per soggetto o per sessione, come emg2pose);
ogni soggetto con tutte le sue sessioni di pretraining o con nessuna; provenienza (SHA del genitore e di window_s).

Uso: python scripts/nested_manifest_report.py window_seconds.json genitore.json.gz derivati... [--batch 32] [--epochs 4] [--json uscita.json]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wearusfm.data import pretraining_manifest as M  # noqa: E402

ORDER = M.DERIVED_FRACTIONS


def _load(path: Path) -> dict:
    raw = path.read_bytes()
    doc = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    rows = [M.SessionRow(**dict(zip(doc["columns"], r))) for r in doc["rows"]]
    return {"doc": doc, "rows": rows, "sha256": hashlib.sha256(raw).hexdigest(), "path": str(path)}


def _keys(rows) -> set:
    return {M.session_key(r.dataset, r.subject, r.session) for r in rows if r.split == "pretraining"}


def report(window_seconds: Path, parent: Path, derived: list[Path], batch: int = 32, epochs: float = 4.0) -> dict:
    wraw = window_seconds.read_bytes()
    ws = json.loads(wraw)["window_s"]
    ws_sha = hashlib.sha256(wraw).hexdigest()
    par = _load(parent)
    problems = []
    if par["doc"]["params"].get("nested"):
        problems.append(f"{parent}: il genitore dichiara la frazione {par['doc']['params']['nested']}")
    out = {"genitore": None}
    files = {}
    for path in derived:
        x = _load(path)
        f = x["doc"]["params"].get("nested")
        if f not in ORDER:
            problems.append(f"{path}: frazione dichiarata {f!r} non fra {ORDER}")
            continue
        if f in files:
            problems.append(f"due derivati per la frazione {f}")
        files[f] = x
    order = [f for f in ORDER if f in files]
    for label, x in [("genitore", par)] + [(f, files[f]) for f in order]:
        d = M.sampled_distribution(x["rows"], ws, epochs, batch)
        info = {"file": x["path"], "file_sha256": x["sha256"], "hash": x["doc"].get("hash"), "mass": d["mass"], "classes": d["classes"],
                "max_passes": d["max_passes"], "hours_samplable": d["hours_samplable"], "mean_window_s": d["mean_window_s"],
                "windows_per_epoch": d["windows_per_epoch"], "steps_at_epochs": d["steps_at_epochs"], "effective_epochs": d["effective_epochs"],
                "sessions": d["sessions"], "subjects": d["subjects"], "per_unit": {u: {"share": d["shares"][u], "passes": d["passes"][u]}
                                                                                  for u in d["shares"]}}
        if label != "genitore":
            p = x["doc"]["params"]
            info.update({"derived_hours": p.get("derived_hours"), "residual_c_to_b": p.get("residual_c_to_b")})
            # le stesse verifiche del costruttore, sul file: durate (chiave mancante = problema, non rinormalizzazione silenziosa), pesi,
            # somma 1, quote A / B + residuo / C - residuo e tetto dopo il sampler (review del critico, 16/10/2026)
            problems += [f"frazione {label}: {e}" for e in M.derived_checks(x["rows"], ws, p, p.get("residual_c_to_b") or 0.0)]
            inp = p.get("derived_inputs", {})
            if inp.get("parent_file_sha256") != par["sha256"]:
                problems.append(f"frazione {label}: costruita da {inp.get('parent_file_sha256')}, non dal genitore dato ({par['sha256']})")
            if inp.get("window_seconds_sha256") != ws_sha:
                problems.append(f"frazione {label}: window_s {inp.get('window_seconds_sha256')}, non quello dato ({ws_sha})")
            others = sorted(json.dumps(r.__dict__, sort_keys=True, default=list) for r in x["rows"] if r.split != "pretraining")
            if others != sorted(json.dumps(r.__dict__, sort_keys=True, default=list) for r in par["rows"] if r.split != "pretraining"):
                problems.append(f"frazione {label}: righe di test/benchmark diverse dal genitore")
        out[label] = info
    keys = {f: _keys(files[f]["rows"]) for f in order}
    subjects = {f: {(r.dataset, r.subject) for r in files[f]["rows"] if r.split == "pretraining"} for f in order}
    for a, b in zip(order, order[1:]):
        if not subjects[a] <= subjects[b] or not keys[a] <= keys[b]:
            problems.append(f"il sottoinsieme {a} non e' contenuto in {b}")
    pkeys = _keys(par["rows"])
    if "1" in keys and keys["1"] != pkeys:
        problems.append("il derivato al 100% non ha le stesse sessioni di pretraining del genitore")
    by_subject: dict[tuple, set] = {}
    for r in par["rows"]:
        if r.split == "pretraining":
            by_subject.setdefault((r.dataset, r.subject), set()).add(M.session_key(r.dataset, r.subject, r.session))
    for f in order:
        partial = sorted(f"{d}/{s}" for d, s in subjects[f] if not by_subject.get((d, s), set()) <= keys[f])
        if partial:
            problems.append(f"frazione {f}: soggetti con solo una parte delle sessioni: {partial[:3]}")
        if not keys[f] <= pkeys:
            problems.append(f"frazione {f}: sessioni di pretraining che nel genitore non lo sono")
    return {"window_seconds": {"path": str(window_seconds), "sha256": ws_sha}, "params": {"batch": batch, "epochs": epochs},
            "manifests": out, "problems": problems}


def table(rep: dict) -> str:
    lines = ["| Manifest | Soggetti | Sessioni | Ore campionabili | Classi A / B / C | Passaggi max | Passi a E (floor) | Epoche effettive |",
             "|---|---|---|---|---|---|---|---|"]
    for f, r in rep["manifests"].items():
        cls = " / ".join(f"{100 * r['classes'][k]:.4f}" for k in ("A", "B", "C") if k in r["classes"])
        lines.append(f"| {f} | {r['subjects']} | {r['sessions']} | {r['hours_samplable']:.2f} | {cls} | {r['max_passes']:.6f} | "
                     f"{r['steps_at_epochs']} | {r['effective_epochs']:.6f} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("window_seconds", type=Path)
    ap.add_argument("parent", type=Path)
    ap.add_argument("derived", type=Path, nargs="+")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    rep = report(args.window_seconds, args.parent, args.derived, args.batch, args.epochs)
    print(table(rep))
    for p in rep["problems"]:
        print("PROBLEMA:", p)
    if args.json:
        args.json.write_text(json.dumps(rep, indent=1, default=str))
    return 1 if rep["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
