#!/usr/bin/env python3
"""Costruisce il manifest del pretraining (D9) dalle sessioni processate e dagli split dei soggetti: una riga per sessione (split, durata, canali validi,
tempo-canale escluso dai buchi, classe di quota, peso di campionamento, ancora RVQ, sha256 del sidecar), il riepilogo per unita' (ore, D_t, D_c, quota,
passaggi) e un hash del contenuto. Legge solo i sidecar e l'intestazione degli array (np.load in mmap): nessun dato EMG viene caricato.

Finche' D9 non e' firmato l'uscita e' una BOZZA (`"version": "draft"`); il manifest congelato si scrive con `--version manifest-v1` dopo la firma.

  python3 scripts/build_manifest.py --root $WORK/data/processed --root $SCRATCH/data/processed --splits splits/draft/splits_draft.json \\
      --out $WORK/wearusfm_runs/results/passo4/manifest_draft.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.data import pretraining_manifest as M  # noqa: E402
from wearusfm.data.processed import discover_sessions, qc_valid_from_metadata  # noqa: E402


def session_row(path: Path, dataset: str, subject: str, session: str, splits: dict) -> M.SessionRow:
    raw = (path / "metadata.json").read_bytes()
    meta = json.loads(raw)
    shape = np.load(path / "data_int16.npy", mmap_mode="r").shape
    if len(shape) == 3:  # (prove, T, C)
        n_samples, n_channels, n_segments = shape[0] * shape[1], shape[2], shape[0]
    else:
        n_samples, n_channels = shape
        trials = meta.get("trials") or []
        n_segments = len(trials) if trials and all("offset" in t for t in trials) else 1
    valid = qc_valid_from_metadata(meta, n_channels)
    excluded = sum(int(r["n_samples"]) for r in meta.get("constant_runs") or [] if valid[int(r["channel"])])
    split, nested = M.split_of(splits, dataset, subject)
    unit = M.unit_of(dataset, session)
    if unit not in M.CLASS_BY_UNIT:
        raise KeyError(f"{dataset}/{session}: unita' {unit} senza classe di quota")
    return M.SessionRow(dataset, subject, session, split, nested, int(n_samples), float(meta["native_fs_hz"]), int(n_channels), int(valid.sum()),
                        excluded, int(n_segments), unit, M.CLASS_BY_UNIT[unit], M.rvq_status(dataset), hashlib.sha256(raw).hexdigest())


def summarize(rows: list[M.SessionRow], alloc: dict, patch_s: float) -> dict:
    units: dict[str, dict] = {}
    for r in rows:
        u = units.setdefault(r.unit, {"class": r.quota_class, "rvq": r.rvq, "sessions": 0, "subjects": set(), "hours": {}, "d_t": 0.0, "d_c": 0.0})
        u["sessions"] += 1
        u["subjects"].add(r.subject)
        u["hours"][r.split] = u["hours"].get(r.split, 0.0) + r.hours
        if r.split == "pretraining":
            u["d_t"] += r.d_t(patch_s)
            u["d_c"] += r.d_c(patch_s)
    for k, u in units.items():
        u["subjects"] = len(u["subjects"])
        share, passes = alloc["per_unit"].get(k, (0.0, 0.0))
        u.update({"share": share, "passes": passes})
    pre = [r for r in rows if r.split == "pretraining"]
    return {"units": units, "totals": {"sessions": len(rows), "pretraining_sessions": len(pre), "pretraining_hours": sum(r.hours for r in pre),
                                       "d_t": sum(r.d_t(patch_s) for r in pre), "d_c": sum(r.d_c(patch_s) for r in pre),
                                       "weight_sum": sum(r.weight for r in rows), "unused_quota": alloc["unused_quota"]}}


def build(roots: list[Path], splits: dict, quota: dict, alpha: float, max_passes: float, patch_ms: float) -> tuple[list[M.SessionRow], dict]:
    rows = []
    for dataset in sorted(splits["datasets"]):
        where = [r / dataset for r in roots if (r / dataset).is_dir()]
        if len(where) != 1:
            raise FileNotFoundError(f"{dataset}: trovato in {len(where)} radici ({where}), atteso in una")
        for subject, session, path in discover_sessions(where[0], dataset):
            rows.append(session_row(path, dataset, subject, session, splits))
    alloc = M.assign_weights(rows, quota, alpha, max_passes)
    return rows, summarize(rows, alloc, patch_ms / 1000)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, action="append", required=True, help="cartella con le sottocartelle dei dataset processati (ripetibile)")
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help=".json o .json.gz")
    ap.add_argument("--quota", default="A=0.2,B=0.75,C=0.05")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max-passes", type=float, default=8.0)
    ap.add_argument("--patch-ms", type=float, default=25.0)
    ap.add_argument("--version", default="draft")
    args = ap.parse_args(argv)
    t0 = time.time()
    splits = json.loads(args.splits.read_text())
    quota = {k: float(v) for k, v in (x.split("=") for x in args.quota.split(","))}
    rows, summary = build(args.root, splits, quota, args.alpha, args.max_passes, args.patch_ms)
    params = {"version": args.version, "quota": quota, "alpha": args.alpha, "max_passes": args.max_passes, "epochs": 4.0, "patch_ms": args.patch_ms,
              "splits_file": str(args.splits), "splits_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
              "classes": M.CLASS_BY_UNIT, "rvq_on": sorted(M.RVQ_ON), "rvq_off": sorted(M.RVQ_OFF)}
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    doc = {"params": params, "hash": M.manifest_hash(params, rows), "code_commit": commit, "created_s": time.time(), "summary": summary,
           "columns": list(asdict(rows[0]).keys()) if rows else [], "rows": [list(asdict(r).values()) for r in rows]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(doc, separators=(",", ":"), default=list).encode()
    (gzip.open if args.out.suffix == ".gz" else open)(args.out, "wb").write(data)
    print(f"manifest {args.version}: {len(rows)} sessioni, hash {doc['hash'][:16]}..., in {time.time() - t0:.0f} s -> {args.out}")
    print("| Unita' | Classe | RVQ | Sessioni | Soggetti | Ore pretraining | Ore test | Ore benchmark | D_t (M) | D_c (G) | Quota | Passaggi |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, u in sorted(summary["units"].items(), key=lambda t: (t[1]["class"], t[0])):
        h = u["hours"]
        print(f"| {k} | {u['class']} | {u['rvq']} | {u['sessions']} | {u['subjects']} | {h.get('pretraining', 0):.1f} | {h.get('test', 0):.1f} | "
              f"{h.get('benchmark', 0):.1f} | {u['d_t'] / 1e6:.1f} | {u['d_c'] / 1e9:.2f} | {100 * u['share']:.1f}% | {u['passes']:.1f} |")
    t = summary["totals"]
    print(f"\nPretraining: {t['pretraining_sessions']} sessioni, {t['pretraining_hours']:.1f} h, D_t {t['d_t'] / 1e6:.1f} M, D_c {t['d_c'] / 1e9:.2f} G; "
          f"somma dei pesi {t['weight_sum']:.4f}; quota non assegnabile {t['unused_quota']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
