#!/usr/bin/env python3
"""Misura (e, solo con --apply, corregge) il QC dei canali con la regola di «canale piatto» RELATIVA sui dati processati.

Contesto (docs/decisioni.md, «Misura del QC dei canali»): 7 ingest (CapgMyo, GRABMyo, putEMG, CSL-hdemg, Camargo, NinaPro DB5, Kaifosh) usano una
soglia assoluta, che con unita' diverse puo' lasciar passare come validi canali morti. Qui si applica a ogni sessione la stessa regola degli ingest
piu' recenti: un canale e' piatto se std < max(1e-12, 1e-3 x mediana delle std dei canali), std calcolata su TUTTA la sessione, nell'unita' del
dato (codice / scala). Si riportano i canali che la regola relativa scarterebbe e che oggi risultano validi (`new_flat`).

SOLA LETTURA di default. Con `--apply` (autorizzato da Simone il 30/09/2026: «se sono morti non li usiamo, ma va tutto documentato») si correggono
SOLO i flag `qc_valid` nei `metadata.json`; i dati (`data_int16.npy`) non si toccano mai e le colonne non si tolgono mai. Ogni correzione e':
  - reversibile: si salva una volta `metadata.pre_qc_revision.json` (mai sovrascritto);
  - documentata nel sidecar stesso: `qc_revisions` (data, regola, colonne aggiunte) e `n_channels_discarded_by_qc` aggiornato se presente;
  - verificata: dopo la scrittura la sessione deve passare `validate_session`, altrimenti si ripristina il sidecar;
  - idempotente: rilanciando, i canali gia' corretti non risultano piu' `new_flat`.
Freno di prudenza scritto PRIMA di guardare i dati: per un dataset con una sessione con piu' del 25% dei canali `new_flat`, o con `new_flat`
in piu' del 5% delle sessioni, `--apply` NON scrive nulla e lo segnala: la revisione e' di Simone (poi `--force-dataset NOME`).

  python scripts/qc_relative_check.py --scan $WORK/data/processed --scan $SCRATCH/data/processed --report out.json
  python scripts/qc_relative_check.py --scan ... --only kaifosh --apply --report out.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from wearusfm.data.processed import CONSTANT_RUN_MIN_S, constant_runs, discover_sessions, qc_valid_from_metadata, read_scale, validate_session
from wearusfm.ingest.common import relative_min_std_from_std

MAX_FRACTION_CHANNELS_IN_SESSION = 0.25  # freno di prudenza (scritto prima di guardare i dati)
MAX_FRACTION_SESSIONS_WITH_NEW = 0.05
MAX_FRACTION_RUNS = 0.01  # buchi oltre l'1% del tempo dei canali validi di un dataset: non si scrive, si riporta (decisioni.md, 01/10/2026)
RUNS_RULE = "buco: canale QC-valido con valore int16 identico per almeno 1 s alla frequenza nativa"
BACKUP_NAME = "metadata.pre_qc_revision.json"
RULE = "piatto se std < max(1e-12, 1e-3 x mediana delle std dei canali), std sull'intera sessione nell'unita' del dato"
DECISION = "Simone 30/09/2026, docs/decisioni.md «Misura del QC dei canali»"


def channel_std(arr, scale=None, chunk_elements: int = 4_000_000) -> np.ndarray:
    """Deviazione standard per canale (colonna) di un array (T, C) o (n, T, C), a blocchi (fusione di Chan) in float64, nell'unita' del dato
    (codice / scala). Non carica l'array: `arr` puo' essere un memmap."""
    n_ch = arr.shape[-1]
    flat = arr.reshape(-1, n_ch)
    rows = max(1, chunk_elements // n_ch)
    n, mean, m2 = 0, np.zeros(n_ch), np.zeros(n_ch)
    for i in range(0, flat.shape[0], rows):
        x = np.asarray(flat[i : i + rows], dtype=np.float64)
        if scale is not None:
            x = x / scale
        k = x.shape[0]
        bm, bm2 = x.mean(axis=0), ((x - x.mean(axis=0)) ** 2).sum(axis=0)
        delta = bm - mean
        tot = n + k
        mean = mean + delta * k / tot
        m2 = m2 + bm2 + delta**2 * n * k / tot
        n = tot
    return np.sqrt(m2 / n) if n else np.zeros(n_ch)


def analyze_session(session_dir: Path) -> dict:
    meta = json.loads((session_dir / "metadata.json").read_text())
    arr = np.load(session_dir / "data_int16.npy", mmap_mode="r")
    scale = read_scale(meta)
    if isinstance(scale, (int, float)) and not scale:
        scale = None
    std = channel_std(arr, scale)
    valid = qc_valid_from_metadata(meta, arr.shape[-1])
    flat = std < relative_min_std_from_std(std)
    med = float(np.median(std))
    # buchi (decisioni.md, 01/10/2026): solo sui canali che restano validi anche dopo la regola relativa
    fs = float(meta["native_fs_hz"])
    keep = valid & ~flat
    existing = meta.get("constant_runs") or []
    runs = [r for r in constant_runs(arr, int(round(CONSTANT_RUN_MIN_S * fs))) if keep[r["channel"]] and r not in existing]
    samples_per_channel = int(np.prod(arr.shape[:-1]))
    return {
        "n_channels": int(arr.shape[-1]), "currently_invalid": np.flatnonzero(~valid).tolist(),
        "relative_flat": np.flatnonzero(flat).tolist(), "new_flat": np.flatnonzero(flat & valid).tolist(),
        "ratio_min": float(std.min() / med) if med > 0 else 0.0,
        "new_constant_runs": runs, "constant_runs_s": sum(r["n_samples"] for r in runs) / fs,
        "valid_channel_s": int(keep.sum()) * samples_per_channel / fs,
    }


def summarize(name: str, root: Path, sessions: dict[str, dict]) -> dict:
    n = len(sessions)
    with_new = {k: v for k, v in sessions.items() if v["new_flat"]}
    worst = max((len(v["new_flat"]) / v["n_channels"] for v in sessions.values()), default=0.0)
    frac_sessions = len(with_new) / n if n else 0.0
    ratios = np.array([v["ratio_min"] for v in sessions.values()]) if n else np.array([0.0])
    with_runs = {k: v["new_constant_runs"] for k, v in sessions.items() if v.get("new_constant_runs")}
    runs_s = sum(v.get("constant_runs_s", 0.0) for v in sessions.values())
    valid_s = sum(v.get("valid_channel_s", 0.0) for v in sessions.values())
    runs_fraction = runs_s / valid_s if valid_s else 0.0
    return {
        "n_sessions_with_constant_runs": len(with_runs), "n_constant_runs": sum(len(v) for v in with_runs.values()),
        "constant_runs_s": runs_s, "constant_runs_fraction_of_valid_time": runs_fraction,
        "guard_runs_blocks_apply": bool(runs_fraction > MAX_FRACTION_RUNS), "sessions_with_constant_runs": with_runs,
        "root": str(root), "n_sessions": n, "n_sessions_with_new_flat": len(with_new), "n_new_flat_channels": sum(len(v["new_flat"]) for v in with_new.values()),
        "fraction_sessions_with_new_flat": frac_sessions, "worst_session_fraction_new_flat": worst,
        "n_sessions_currently_invalid_channels": sum(1 for v in sessions.values() if v["currently_invalid"]),
        "ratio_min_quantiles_0_1_5_50": [float(np.quantile(ratios, q)) for q in (0.0, 0.01, 0.05, 0.5)],
        "guard_blocks_apply": bool(worst > MAX_FRACTION_CHANNELS_IN_SESSION or frac_sessions > MAX_FRACTION_SESSIONS_WITH_NEW),
        "sessions_with_new_flat": with_new,
    }


def apply_revision(session_dir: Path, new_flat: list[int], now: str) -> None:
    """Corregge i flag nel sidecar (con backup, documentazione e verifica). Solleva se la sessione risultasse non conforme."""
    meta_path, backup = session_dir / "metadata.json", session_dir / BACKUP_NAME
    original = meta_path.read_text()
    if not backup.exists():
        backup.write_text(original)
    meta = json.loads(original)
    col = 0
    for g in meta["montage"]["groups"]:
        for ch in g["channels"]:
            if col in new_flat:
                ch["qc_valid"] = False
            col += 1
    if "n_channels_discarded_by_qc" in meta:
        meta["n_channels_discarded_by_qc"] = sum(1 for g in meta["montage"]["groups"] for ch in g["channels"] if not ch["qc_valid"])
    meta.setdefault("qc_revisions", []).append(
        {"date": now, "rule": RULE, "added_invalid_columns": sorted(new_flat), "tool": "scripts/qc_relative_check.py", "decision": DECISION}
    )
    tmp = session_dir / "metadata.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, meta_path)
    problems = validate_session(session_dir)
    if problems:
        shutil.copyfile(backup, meta_path)
        raise RuntimeError(f"{session_dir}: dopo la correzione la sessione non e' conforme ({problems}); sidecar ripristinato")


def apply_constant_runs(session_dir: Path, runs: list[dict], now: str) -> None:
    """Aggiunge i buchi al sidecar (`constant_runs`), con backup, voce in `qc_revisions` e verifica; i dati non si toccano."""
    meta_path, backup = session_dir / "metadata.json", session_dir / BACKUP_NAME
    original = meta_path.read_text()
    if not backup.exists():
        backup.write_text(original)
    meta = json.loads(original)
    old = meta.get("constant_runs") or []
    meta["constant_runs"] = old + [r for r in runs if r not in old]
    fs = float(meta["native_fs_hz"])
    meta.setdefault("qc_revisions", []).append({
        "date": now, "rule": RUNS_RULE, "added_constant_runs": len(meta["constant_runs"]) - len(old),
        "added_seconds": round(sum(r["n_samples"] for r in runs if r not in old) / fs, 3), "tool": "scripts/qc_relative_check.py",
        "decision": "Simone 30/09-01/10/2026, docs/decisioni.md «Buchi di un canale»",
    })
    tmp = session_dir / "metadata.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, meta_path)
    problems = validate_session(session_dir)
    if problems:
        meta_path.write_text(original)
        raise RuntimeError(f"{session_dir}: dopo l'aggiunta dei buchi la sessione non e' conforme ({problems}); sidecar ripristinato")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=[], metavar="NOME=DIR")
    ap.add_argument("--scan", action="append", default=[], metavar="DIR", help="ogni sottocartella e' un dataset")
    ap.add_argument("--only", action="append", default=[], metavar="NOME", help="solo questi dataset (nome della cartella)")
    ap.add_argument("--apply", action="store_true", help="corregge i flag (default: sola lettura)")
    ap.add_argument("--force-dataset", action="append", default=[], metavar="NOME", help="applica anche se il freno di prudenza scatta (dopo revisione di Simone)")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)

    targets: list[tuple[str, Path]] = []
    for item in args.root:
        if "=" not in item:
            ap.error(f"--root vuole NOME=DIR, ricevuto {item!r}")
        name, _, path = item.partition("=")
        targets.append((name, Path(path)))
    for d in args.scan:
        base = Path(d)
        if not base.is_dir():
            ap.error(f"--scan: {base} non e' una cartella")
        targets.extend((sub.name, sub) for sub in sorted(base.iterdir()) if sub.is_dir())
    if args.only:
        targets = [t for t in targets if t[0] in args.only]
    if not targets:
        ap.error("niente da controllare (--root/--scan, eventualmente --only)")

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report: dict = {"mode": "apply" if args.apply else "sola lettura", "rule": RULE, "date": now, "datasets": {}}
    rc = 0

    def flush_report() -> None:  # dopo ogni dataset: un job interrotto non butta via il lavoro fatto
        if args.report:
            args.report.write_text(json.dumps(report, indent=2))

    for name, root in targets:
        sessions, errors = {}, {}
        for subject, session, path in discover_sessions(root, name):
            key = f"{subject}/{session}"
            try:
                sessions[key] = analyze_session(path)
                sessions[key]["_path"] = str(path)
            except Exception as e:  # una sessione illeggibile non ferma le altre
                errors[key] = f"{type(e).__name__}: {e}"
        summ = summarize(name, root, {k: {kk: vv for kk, vv in v.items() if kk != "_path"} for k, v in sessions.items()})
        summ["errors"] = errors
        key = name if name not in report["datasets"] else f"{name}@{root}"
        report["datasets"][key] = summ
        print(
            f"{key}: {summ['n_sessions']} sessioni, nuovi canali piatti in {summ['n_sessions_with_new_flat']} sessioni "
            f"({summ['n_new_flat_channels']} canali), peggiore sessione {summ['worst_session_fraction_new_flat']:.0%}, "
            f"freno={'SCATTA' if summ['guard_blocks_apply'] else 'ok'}; buchi >= 1 s: {summ['n_constant_runs']} in "
            f"{summ['n_sessions_with_constant_runs']} sessioni, {summ['constant_runs_s']:.1f} s "
            f"({summ['constant_runs_fraction_of_valid_time']:.4%} del tempo valido), freno={'SCATTA' if summ['guard_runs_blocks_apply'] else 'ok'}; "
            f"errori di lettura={len(errors)}",
            flush=True,
        )
        if errors:
            rc = 1
        if args.apply and summ["n_sessions_with_new_flat"]:
            if summ["guard_blocks_apply"] and name not in args.force_dataset:
                summ["apply"] = "NON APPLICATO: il freno di prudenza e' scattato, serve la revisione di Simone"
                print(f"  {summ['apply']}", flush=True)
                rc = max(rc, 2)
                flush_report()
                continue
            done = 0
            for k, v in summ["sessions_with_new_flat"].items():
                try:
                    apply_revision(Path(sessions[k]["_path"]), v["new_flat"], now)
                    done += 1
                except Exception as e:
                    errors[k] = f"apply: {e}"
                    rc = 1
            summ["apply"] = f"applicato a {done} sessioni"
            print(f"  {summ['apply']}", flush=True)
        if args.apply and summ["n_constant_runs"]:
            if summ["guard_runs_blocks_apply"] and name not in args.force_dataset:
                summ["apply_runs"] = "NON APPLICATO: i buchi superano l'1% del tempo valido, serve la revisione di Simone"
                rc = max(rc, 2)
            else:
                done = 0
                for k, runs in summ["sessions_with_constant_runs"].items():
                    try:
                        apply_constant_runs(Path(sessions[k]["_path"]), runs, now)
                        done += 1
                    except Exception as e:
                        errors[k] = f"apply buchi: {e}"
                        rc = 1
                summ["apply_runs"] = f"buchi scritti in {done} sessioni"
            print(f"  {summ['apply_runs']}", flush=True)
        flush_report()
    flush_report()
    return rc


if __name__ == "__main__":
    sys.exit(main())
