"""Manifest derivati per l'asse dei soggetti (D16) come file congelati, uno per frazione (12,5 / 25 / 50% e il derivato al 100%), dal manifest
congelato v1.1 e dalle durate delle finestre congelate (accordo GPT-Claude del 15/10/2026, `docs/accordo_operativo_finestra1_20261015.md`):
sessioni dei soggetti del sottoinsieme firmato, base temporale sulle ore campionabili (window_s positivo; null -> peso 0), D9 su quelle ore,
residuo di C solo verso B con A alla sua quota (`pretraining_manifest.DERIVED_RULE`), provenienza degli ingressi nei parametri. Il v1.1 non si
tocca.

Prima di scrivere, sui derivati: pesi finiti e non negativi, massa 1, quota di A e tetto di passaggi dopo le trasformazioni del sampler
(`sampled_distribution`, che replica `ManifestIndex.load` + `time_weighted_probs`: la coerenza col loader vero e' verificata qui sul derivato al
100%), sottoinsiemi annidati, configurazione delle finestre del training uguale a quella con cui `window_s` e' stato calcolato. Confronto del
derivato al 100% col v1.1: sessioni escluse, ore, quote per unita' e passaggi del v1.1 dopo il sampler; spiega le differenze. Una verifica
fallita: nessun file scritto, uscita 1.

Uso: python scripts/build_nested_manifests.py manifest.json.gz window_seconds.json cartella_di_uscita [--batch 32]
     -> <cartella>/<nome>_nested_<f>.json.gz per f in 0.125, 0.25, 0.5, 1 e <cartella>/nested_manifests.json (riepilogo)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.data import pretraining_manifest as M  # noqa: E402


def _commit() -> dict:
    """SHA del codice e pulizia del checkout: i rapporti finali si generano da un commit pulito (accordo del 15/10/2026)."""
    if os.environ.get("WEARUSFM_CODE_SHA"):
        return {"sha": os.environ["WEARUSFM_CODE_SHA"], "clean": True, "source": "launcher (lib_code_dir.sh: checkout di campagna pulito)"}
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                               check=True).stdout.strip()
        return {"sha": sha, "clean": not dirty, "source": "git"}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": "", "clean": False, "source": "sconosciuta"}


def window_config() -> dict:
    """La configurazione delle finestre con cui `scripts/window_seconds.py` calcola window_s (`signed_config`) e quella del training della
    finestra 1 (`with_window1_rules` sul preset del sanity): devono coincidere nei campi che determinano le durate."""
    from wearusfm.training import run as R

    signed = asdict(L.signed_config())
    train = asdict(R.with_window1_rules(R.sanity_config()).loader)
    fields = {k: signed[k] for k in M.WINDOW_CONFIG_FIELDS}
    return {"fields": fields, "same_as_training": all(signed[k] == train[k] for k in M.WINDOW_CONFIG_FIELDS),
            "training": {k: train[k] for k in M.WINDOW_CONFIG_FIELDS}}


def _loader_shares(rows: list[M.SessionRow], window_s: dict) -> list[float]:
    """Le quote nel tempo calcolate col codice vero del loader (`ManifestIndex` normalizzato + `time_weighted_probs`)."""
    pre = [asdict(r) for r in rows if r.split == "pretraining"]
    w = np.asarray([r["weight"] for r in pre], dtype=np.float64)
    probs = L.time_weighted_probs(L.ManifestIndex(pre, w / w.sum(), []), window_s)
    t = probs * np.asarray([window_s[M.session_key(r["dataset"], r["subject"], r["session"])] or 0.0 for r in pre])
    return list(t / t.sum())


def build(manifest: Path, window_seconds: Path, batch: int = 32) -> tuple[dict, dict[str, dict]]:
    raw = manifest.read_bytes()
    doc = json.loads(gzip.decompress(raw) if manifest.suffix == ".gz" else raw)
    wraw = window_seconds.read_bytes()
    wdoc = json.loads(wraw)
    ws = wdoc["window_s"]
    parent = [M.SessionRow(**dict(zip(doc["columns"], r))) for r in doc["rows"]]
    # l'hash dichiarato dal genitore e quello del suo contenuto (in v1.1 il campo `hash` e' rimasto quello di v1): si registrano entrambi
    source_hash = {"declared": doc.get("hash"), "recomputed": M.manifest_hash(doc["params"], parent)}
    wcfg = window_config()
    commit = _commit()
    inputs = {"parent_file_sha256": hashlib.sha256(raw).hexdigest(), "parent_hash_declared": source_hash["declared"],
              "parent_hash_content": source_hash["recomputed"], "parent_version": doc["params"].get("version"),
              "parent_revision": doc["params"].get("revision", {}).get("version"), "splits_sha256": doc["params"].get("splits_sha256"),
              "window_seconds_sha256": hashlib.sha256(wraw).hexdigest(), "window_seconds_manifest": wdoc.get("manifest"),
              "window_config": wcfg["fields"]}
    out = {f: M.derived_manifest(doc, f, ws, inputs, commit["sha"]) for f in M.DERIVED_FRACTIONS}
    rows_of = {f: [M.SessionRow(**dict(zip(d["columns"], r))) for r in d["rows"]] for f, d in out.items()}
    dists = {f: M.sampled_distribution(rows_of[f], ws, doc["params"].get("epochs", 4.0), batch) for f in out}
    subjects = {f: {(r.dataset, r.subject) for r in rows_of[f] if r.split == "pretraining"} for f in out}
    keys = {f: {M.session_key(r.dataset, r.subject, r.session) for r in rows_of[f] if r.split == "pretraining"} for f in out}
    order = list(M.DERIVED_FRACTIONS)
    shares_loader = _loader_shares(rows_of["1"], ws)
    shares_ours = M.time_shares(rows_of["1"], ws)
    checks = {"window_config_same_as_training": wcfg["same_as_training"],
              "nested_subjects_and_sessions": all(subjects[a] <= subjects[b] and keys[a] <= keys[b] for a, b in zip(order, order[1:])),
              "within_cap_after_sampler": all(d["max_passes"] <= doc["params"]["max_passes"] * (1 + 1e-9) for d in dists.values()),
              # pesi, somma 1, quote A / B + residuo / C - residuo dopo il sampler (prima era la massa, vera per costruzione; critico 16/10)
              "derived_checks": all(not M.derived_checks(rows_of[f], ws, doc["params"], out[f]["params"]["residual_c_to_b"]) for f in out),
              "sampler_matches_loader": bool(max(abs(a - b) for a, b in zip(shares_loader, shares_ours)) < 1e-12)}
    # confronto col genitore: il v1.1 pesa anche le sessioni senza finestre, e il sampler ne ridistribuisce la massa su tutte le altre
    pdist = M.sampled_distribution(parent, ws, doc["params"].get("epochs", 4.0), batch)
    d100 = dists["1"]
    excluded = [M.session_key(r.dataset, r.subject, r.session) for r in parent if r.split == "pretraining" and ws.get(
        M.session_key(r.dataset, r.subject, r.session)) is None]
    pw = {M.session_key(r.dataset, r.subject, r.session): r.weight for r in parent if r.split == "pretraining"}
    nw = {M.session_key(r.dataset, r.subject, r.session): r.weight for r in rows_of["1"] if r.split == "pretraining"}
    diff = {"sessions_excluded": len(excluded), "hours_excluded": out["1"]["params"]["derived_hours"]["hours_excluded"],
            "parent_weight_of_excluded": sum(pw[k] for k in excluded),
            "max_abs_weight_change": max(abs(pw[k] - nw[k]) for k in pw),
            "parent_after_sampler": {"classes": pdist["classes"], "max_passes": pdist["max_passes"], "steps_at_epochs": pdist["steps_at_epochs"]},
            "derived_after_sampler": {"classes": d100["classes"], "max_passes": d100["max_passes"], "steps_at_epochs": d100["steps_at_epochs"]},
            "unit_share_change": {u: d100["shares"][u] - pdist["shares"].get(u, 0.0) for u in d100["shares"]},
            "explanation": (
                f"il v1.1 assegna peso ({sum(pw[k] for k in excluded):.6g}) anche alle {len(excluded)} sessioni con window_s null; il sampler "
                "da' loro probabilita' 0 e ridistribuisce quella massa su tutte le altre con un fattore comune, quindi dopo il sampler le "
                "classi del v1.1 sono " + " / ".join(f"{k} {100 * v:.4f}%" for k, v in pdist["classes"].items()) +
                f" (passaggi massimi {pdist['max_passes']:.4f}). Il derivato porta quelle sessioni a peso 0 e rifa D9 sulle ore "
                "campionabili: dopo il sampler " + " / ".join(f"{k} {100 * v:.4f}%" for k, v in d100["classes"].items()) +
                f" (passaggi massimi {d100['max_passes']:.4f}); i pesi delle altre sessioni cambiano al massimo di "
                f"{max(abs(pw[k] - nw[k]) for k in pw):.3g}.")}
    report = {"inputs": inputs, "parent_hash": source_hash, "code": commit, "window_config": wcfg, "checks": checks, "batch": batch,
              "fractions": {f: {"hash": out[f]["hash"], **out[f]["params"]["derived_hours"], "residual_c_to_b": out[f]["params"]["residual_c_to_b"],
                                "classes": dists[f]["classes"], "max_passes": dists[f]["max_passes"], "mean_window_s": dists[f]["mean_window_s"],
                                "windows_per_epoch": dists[f]["windows_per_epoch"], "steps_at_4_epochs": dists[f]["steps_at_epochs"],
                                "effective_epochs": dists[f]["effective_epochs"], "sessions": dists[f]["sessions"], "subjects": dists[f]["subjects"]}
                            for f in order},
              "derived_100_vs_parent": diff}
    return report, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("window_seconds", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args(argv)
    try:
        report, out = build(args.manifest, args.window_seconds, args.batch)
    except ValueError as e:
        print(f"COSTRUZIONE RIFIUTATA: {e}")
        return 1
    failed = [k for k, v in report["checks"].items() if not v]
    print(json.dumps({"checks": report["checks"], "fractions": report["fractions"], "derived_100_vs_parent":
                      {k: v for k, v in report["derived_100_vs_parent"].items() if k != "unit_share_change"}}, indent=1, default=str))
    if failed:
        print(f"VERIFICHE FALLITE: {failed}: nessun file scritto")
        return 1
    base = args.manifest.name.removesuffix(".gz").removesuffix(".json")
    # mtime fisso e allocazione in ordine fisso (`pretraining_manifest.allocate`, corretta il 16/10: prima il 25% dipendeva da
    # PYTHONHASHSEED): stessi byte a ogni costruzione con lo stesso codice. Tutti i conflitti si controllano PRIMA di scrivere: niente insiemi misti (ultrareview 15/10)
    files = {f: (args.out_dir / f"{base}_nested_{f}.json.gz", gzip.compress(json.dumps(d, separators=(",", ":"), default=list).encode(), mtime=0))
             for f, d in out.items()}
    conflicts = [str(p) for p, data in files.values() if p.exists() and p.read_bytes() != data]
    if conflicts:
        print(f"{conflicts} esistono con un contenuto diverso: non sovrascrivo e non scrivo nulla")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for f, (path, data) in files.items():
        path.write_bytes(data)
        report["fractions"][f]["file"] = path.name
        report["fractions"][f]["file_sha256"] = hashlib.sha256(data).hexdigest()
    (args.out_dir / "nested_manifests.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"scritti {len(out)} manifest derivati in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
