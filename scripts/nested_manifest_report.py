"""Rapporto sui manifest dell'asse dei soggetti (D16): il manifest completo e i manifest annidati come file (`scripts/build_nested_manifests.py`).
Per ogni file, sulla distribuzione che campiona davvero il loader (pesi normalizzati sulle righe di pretraining, `ManifestIndex.load`):
soggetti, sessioni e ore; quote per classe e per unita'; passaggi per unita' a E epoche; passi per epoca dal corpus. Verifiche: frazione
dichiarata, sottoinsiemi annidati per soggetti e per sessioni (12,5 ⊂ 25 ⊂ 50 ⊂ 100: nessuna sessione che nel completo e' di test o
benchmark), righe di test e benchmark identiche al completo, ogni soggetto con tutte le sue sessioni di pretraining o con nessuna, passaggi
entro il tetto, provenienza (SHA dei file e del manifest d'origine).

Passi per epoca: un'epoca consuma le ore di pretraining del manifest (D_t, v10 §10.3); il loader estrae le sessioni con probabilita'
proporzionale a peso / durata media delle finestre della sessione (`window_seconds`), quindi il tempo consumato da ogni sessione e'
proporzionale al peso e le finestre per epoca sono ore x 3600 x somma_i(peso_i / durata_i). Accanto, la vecchia stima con finestre da 4 s.

Uso: python scripts/nested_manifest_report.py window_seconds.json manifest_completo.json.gz [manifest_annidati...] [--batch 32] [--epochs 4]
     [--json uscita.json]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path


def _key(r: dict) -> str:
    return f"{r['dataset']}/{r['subject']}/{r['session']}"


def _load(path: Path) -> dict:
    raw = path.read_bytes()
    return {"doc": json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw), "sha256": hashlib.sha256(raw).hexdigest()}


def stats(doc: dict, ws: dict, batch: int, epochs: float) -> dict:
    rows = [dict(zip(doc["columns"], r)) for r in doc["rows"]]
    pre = [r for r in rows if r["split"] == "pretraining"]
    for r in pre:
        r["hours"] = r["n_samples"] / r["fs_hz"] / 3600.0
    w = sum(r["weight"] for r in pre)
    hours = sum(r["hours"] for r in pre)
    units: dict[str, dict] = {}
    for r in pre:
        u = units.setdefault(r["unit"], {"class": r["quota_class"], "subjects": set(), "sessions": 0, "hours": 0.0, "share": 0.0})
        u["subjects"].add(r["subject"])
        u["sessions"] += 1
        u["hours"] += r["hours"]
        u["share"] += r["weight"] / w
    classes: dict[str, float] = {}
    per_unit = {}
    for name, u in sorted(units.items()):
        classes[u["class"]] = classes.get(u["class"], 0.0) + u["share"]
        per_unit[name] = {"class": u["class"], "subjects": len(u["subjects"]), "sessions": u["sessions"], "hours": round(u["hours"], 3),
                          "share": u["share"], "passes_at_E": u["share"] * epochs * hours / u["hours"]}
    weighted = [r for r in pre if r["weight"] > 0]
    missing = sum(1 for r in weighted if not ws.get(_key(r)))
    windows_per_hour = sum(r["weight"] / w / ws[_key(r)] for r in weighted if ws.get(_key(r))) * 3600.0
    steps = hours * windows_per_hour / batch
    return {"nested": doc.get("params", {}).get("nested"), "subjects": len({(r["dataset"], r["subject"]) for r in pre}), "sessions": len(pre),
            "hours": hours, "classes": dict(sorted(classes.items())), "max_passes_at_E": max(u["passes_at_E"] for u in per_unit.values()),
            "steps_per_epoch": round(steps), "steps_at_E": round(epochs * steps), "steps_per_epoch_4s_windows": round(hours * 3600 / 4 / batch),
            "mean_window_s": 3600 / windows_per_hour, "sessions_without_window_seconds": missing, "units": per_unit,
            "_subjects": {(r["dataset"], r["subject"]) for r in pre}, "_keys": {_key(r) for r in pre},
            "_other_rows": sorted(json.dumps(r, sort_keys=True) for r in rows if r["split"] != "pretraining")}


def report(window_seconds: Path, manifests: list[Path], batch: int = 32, epochs: float = 4.0) -> dict:
    ws = json.loads(window_seconds.read_text())["window_s"]
    loaded = {str(p): _load(p) for p in manifests}
    problems, out = [], {}
    full_rows = None
    for path, x in loaded.items():
        doc = x["doc"]
        label = doc.get("params", {}).get("nested") or "1"
        if label in out:
            problems.append(f"due manifest per la frazione {label}")
        s = stats(doc, ws, batch, epochs)
        max_passes = doc["params"].get("max_passes", 8.0)
        if s["max_passes_at_E"] > max_passes * (1 + 1e-9):
            problems.append(f"frazione {label}: {s['max_passes_at_E']:.6f} passaggi nella distribuzione campionata, oltre il tetto {max_passes}")
        if label == "1":
            full_rows = [dict(zip(doc["columns"], r)) for r in doc["rows"] if r[doc["columns"].index("split")] == "pretraining"]
        s["file"] = path
        s["file_sha256"] = x["sha256"]
        s["derived_from"] = doc["params"].get("derived_from")
        out[label] = s
    order = [f for f in ("0.125", "0.25", "0.5", "1") if f in out]
    for a, b in zip(order, order[1:]):
        if not out[a]["_subjects"] <= out[b]["_subjects"] or not out[a]["_keys"] <= out[b]["_keys"]:
            problems.append(f"il sottoinsieme {a} non e' contenuto in {b}")
    if full_rows is not None:  # test e benchmark sono per soggetto o per sessione (emg2pose: test ufficiale per sessione, 02/10/2026): il
        # controllo e' per sessione, cioe' il contenimento delle chiavi di pretraining in quelle del manifest completo (sopra) e righe di
        # test/benchmark identiche a quelle del completo
        sha_full = next(x["sha256"] for x in loaded.values() if not x["doc"]["params"].get("nested"))
        by_subject: dict[tuple, set] = {}
        for r in full_rows:
            by_subject.setdefault((r["dataset"], r["subject"]), set()).add(_key(r))
        for f in order[:-1]:
            partial = sorted(f"{d}/{s}" for d, s in out[f]["_subjects"] if not by_subject.get((d, s), set()) <= out[f]["_keys"])
            if partial:
                problems.append(f"frazione {f}: soggetti con solo una parte delle sessioni: {partial[:3]}")
            if out[f]["_other_rows"] != out["1"]["_other_rows"]:
                problems.append(f"frazione {f}: righe di test/benchmark diverse da quelle del manifest completo")
            origin = (out[f]["derived_from"] or {}).get("file_sha256")
            if origin != sha_full:
                problems.append(f"frazione {f}: costruita da {origin}, non dal manifest completo dato ({sha_full})")
    for s in out.values():
        for k in [k for k in s if k.startswith("_")]:
            del s[k]
    return {"window_seconds": {"path": str(window_seconds), "sha256": hashlib.sha256(window_seconds.read_bytes()).hexdigest()},
            "params": {"batch": batch, "epochs": epochs}, "fractions": {f: out[f] for f in order}, "problems": problems}


def table(rep: dict) -> str:
    lines = ["| Frazione | Soggetti | Sessioni | Ore | Classi A / B / C | Passaggi max a E | Passi per epoca | (finestre da 4 s) |",
             "|---|---|---|---|---|---|---|---|"]
    for f, r in rep["fractions"].items():
        cls = " / ".join(f"{100 * r['classes'][k]:.2f}" for k in ("A", "B", "C") if k in r["classes"])
        lines.append(f"| {f} | {r['subjects']} | {r['sessions']} | {r['hours']:.2f} | {cls} | {r['max_passes_at_E']:.6f} | {r['steps_per_epoch']} | "
                     f"{r['steps_per_epoch_4s_windows']} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("window_seconds", type=Path)
    ap.add_argument("manifests", type=Path, nargs="+")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    rep = report(args.window_seconds, args.manifests, args.batch, args.epochs)
    print(table(rep))
    for p in rep["problems"]:
        print("PROBLEMA:", p)
    if args.json:
        args.json.write_text(json.dumps(rep, indent=1))
    return 1 if rep["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
