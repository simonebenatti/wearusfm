"""Manifest annidati per soggetto (D16, firmata il 03/10/2026) dal manifest congelato: per ogni frazione dei soggetti (12,5 / 25 / 50 / 100%)
soggetti, sessioni e ore per unita' di allocazione e per classe; quote effettive col peso che usa il loader (`ManifestIndex.load(nested=...)`:
pesi del manifest completo rinormalizzati sul sottoinsieme) accanto a quelle del manifest completo; passaggi per unita' a E epoche; annidamento
(12,5 ⊂ 25 ⊂ 50 ⊂ 100) ed esclusione di test e benchmark verificati; passi per epoca dal corpus; provenienza (SHA degli ingressi).

Passi per epoca: un'epoca consuma le ore di pretraining del (sotto)manifest (D_t, v10 §10.3; fogli della finestra 1); il loader estrae le
sessioni con probabilita' proporzionale a peso / durata media delle finestre della sessione (`window_seconds`), quindi il tempo consumato da
ogni sessione e' proporzionale al peso e le finestre per epoca sono ore x 3600 x somma_i(peso_i / durata_i). Accanto, la vecchia stima con
finestre da 4 s.

Non decide nulla: i numeri servono a scegliere (GPT, Simone) se nei sottoinsiemi le quote restano quelle del manifest completo.

Uso: python scripts/nested_manifest_report.py manifest.json.gz window_seconds.json [--batch 32] [--epochs 4] [--json uscita.json]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

FRACTIONS = ("0.125", "0.25", "0.5", "1")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(r: dict) -> str:
    return f"{r['dataset']}/{r['subject']}/{r['session']}"


def report(manifest: Path, window_seconds: Path, batch: int = 32, epochs: float = 4.0) -> dict:
    doc = json.loads(gzip.decompress(manifest.read_bytes()) if str(manifest).endswith(".gz") else manifest.read_text())
    rows = [dict(zip(doc["columns"], r)) for r in doc["rows"]]
    wdoc = json.loads(window_seconds.read_text())
    ws = wdoc["window_s"]
    problems = []
    tagged_outside = [_key(r) for r in rows if r["split"] != "pretraining" and r["nested"]]
    if tagged_outside:
        problems.append(f"{len(tagged_outside)} righe di test/benchmark con un sottoinsieme annidato (es. {tagged_outside[:3]})")
    pre = [r for r in rows if r["split"] == "pretraining"]
    for r in pre:
        r["hours"] = r["n_samples"] / r["fs_hz"] / 3600.0
    total_w = sum(r["weight"] for r in pre)
    subsets = {f: [r for r in pre if f == "1" or f in r["nested"]] for f in FRACTIONS}
    keys = {f: {_key(r) for r in subsets[f]} for f in FRACTIONS}
    subj = {f: {(r["dataset"], r["subject"]) for r in subsets[f]} for f in FRACTIONS}
    for a, b in zip(FRACTIONS, FRACTIONS[1:]):
        if not keys[a] <= keys[b] or not subj[a] <= subj[b]:
            problems.append(f"il sottoinsieme {a} non e' contenuto in {b}")
    for f in FRACTIONS[:-1]:  # un soggetto o c'e' con tutte le sue sessioni di pretraining o non c'e'
        partial = sorted({f"{r['dataset']}/{r['subject']}" for r in pre if (r["dataset"], r["subject"]) in subj[f] and _key(r) not in keys[f]})
        if partial:
            problems.append(f"frazione {f}: soggetti con solo una parte delle sessioni: {partial[:3]}")
    full_units = {}
    for r in pre:
        u = full_units.setdefault(r["unit"], {"class": r["quota_class"], "hours": 0.0, "weight": 0.0})
        u["hours"] += r["hours"]
        u["weight"] += r["weight"] / total_w
    out = {"inputs": {"manifest": str(manifest), "manifest_file_sha256": _sha(manifest), "manifest_hash": doc.get("hash"),
                      "manifest_version": doc["params"].get("version"), "revision": doc["params"].get("revision", {}).get("version"),
                      "splits_sha256": doc["params"].get("splits_sha256"), "window_seconds": str(window_seconds),
                      "window_seconds_sha256": _sha(window_seconds), "window_seconds_manifest": wdoc.get("manifest")},
           "params": {"batch": batch, "epochs": epochs, "quota": doc["params"]["quota"], "max_passes": doc["params"]["max_passes"]},
           "fractions": {}, "problems": problems}
    for f in FRACTIONS:
        sub = subsets[f]
        w = sum(r["weight"] for r in sub)
        hours = sum(r["hours"] for r in sub)
        missing = [_key(r) for r in sub if r["weight"] > 0 and not ws.get(_key(r))]
        windows_per_hour = sum(r["weight"] / w / ws[_key(r)] for r in sub if r["weight"] > 0 and ws.get(_key(r))) * 3600.0
        units = {}
        for r in sub:
            u = units.setdefault(r["unit"], {"class": r["quota_class"], "subjects": set(), "sessions": 0, "hours": 0.0, "weight": 0.0})
            u["subjects"].add(r["subject"])
            u["sessions"] += 1
            u["hours"] += r["hours"]
            u["weight"] += r["weight"] / w
        per_unit = {}
        for name, u in sorted(units.items()):
            share = u["weight"]
            per_unit[name] = {"class": u["class"], "subjects": len(u["subjects"]), "sessions": u["sessions"], "hours": round(u["hours"], 2),
                              "hours_fraction_of_full": round(u["hours"] / full_units[name]["hours"], 4),
                              "share": round(share, 5), "share_full": round(full_units[name]["weight"], 5),
                              "passes_at_E": round(share * epochs * hours / u["hours"], 3) if u["hours"] else None}
        classes = {}
        for name, u in per_unit.items():
            c = classes.setdefault(u["class"], {"share": 0.0, "share_full": 0.0, "hours": 0.0})
            c["share"] += u["share"]
            c["share_full"] += u["share_full"]
            c["hours"] += u["hours"]
        steps = hours * windows_per_hour / batch
        out["fractions"][f] = {
            "subjects": len(subj[f]), "sessions": len(sub), "hours": round(hours, 2), "time_patches_per_epoch": round(hours * 3600 / 0.025),
            "steps_per_epoch": round(steps), "steps_at_E": round(epochs * steps), "steps_per_epoch_4s_windows": round(hours * 3600 / 4 / batch),
            "mean_window_s": round(3600 / windows_per_hour, 4), "sessions_without_window_seconds": len(missing),
            "classes": {k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in sorted(classes.items())},
            "max_share_shift": round(max(abs(u["share"] - u["share_full"]) for u in per_unit.values()), 5),
            "max_passes_at_E": max((u["passes_at_E"] or 0) for u in per_unit.values()), "units": per_unit}
    return out


def table(rep: dict) -> str:
    lines = ["| Frazione | Soggetti | Sessioni | Ore | Classi A / B / C (completo 20 / 75 / 5) | Massimo spostamento di quota | Passaggi max a E | "
             "Passi per epoca | (finestre da 4 s) |", "|---|---|---|---|---|---|---|---|---|"]
    for f, r in rep["fractions"].items():
        c = r["classes"]
        cls = " / ".join(f"{100 * c[k]['share']:.1f}" for k in ("A", "B", "C") if k in c)
        lines.append(f"| {f} | {r['subjects']} | {r['sessions']} | {r['hours']} | {cls} | {100 * r['max_share_shift']:.2f} punti | "
                     f"{r['max_passes_at_E']} | {r['steps_per_epoch']} | {r['steps_per_epoch_4s_windows']} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("window_seconds", type=Path)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    rep = report(args.manifest, args.window_seconds, args.batch, args.epochs)
    print(table(rep))
    for p in rep["problems"]:
        print("PROBLEMA:", p)
    if args.json:
        args.json.write_text(json.dumps(rep, indent=1))
    return 1 if rep["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
