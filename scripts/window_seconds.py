#!/usr/bin/env python3
"""Lunghezza media delle finestre di ogni sessione di pretraining (quote nel tempo; Simone, 04/10/2026, decisione 1), con la configurazione del
dataloader del sanity e della finestra 1 (finestre 1-4 s, salti spezzati, margine di 200 ms attorno ai tratti costanti). Legge solo i sidecar:
nessun dato. Uscita: un JSON {dataset/soggetto/sessione: secondi, o null se la sessione non ha finestre} con un riepilogo per dataset.

  python3 scripts/window_seconds.py --manifest manifest_59253155.json.gz --root $WORK/data/processed --root $SCRATCH/data/processed --out FILE
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.data import pretraining_loader as L  # noqa: E402


def _worker(args):
    rows, roots = args
    cfg = L.signed_config()
    index = L.ManifestIndex(rows, np.ones(len(rows)) / len(rows), [Path(r) for r in roots])
    out, errors = {}, {}
    for row in rows:
        k = f"{row['dataset']}/{row['subject']}/{row['session']}"
        try:
            view = L.SessionView.open(index.path_of(row), cfg.split_at_gaps, row.get("sidecar_sha256"))
            out[k] = L.mean_window_s(view, cfg)
        except Exception as e:  # registrata, non ferma il calcolo
            errors[k] = f"{type(e).__name__}: {e}"
    return out, errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    t0 = time.time()
    index = L.ManifestIndex.load(args.manifest, [Path(r) for r in args.root])
    rows = index.rows
    chunks = [rows[i::args.workers * 8] for i in range(args.workers * 8)]
    window_s, errors = {}, {}
    with get_context("spawn").Pool(args.workers) as pool:
        for w, e in pool.imap_unordered(_worker, [(c, args.root) for c in chunks if c]):
            window_s.update(w)
            errors.update(e)
    by = defaultdict(list)
    for k, v in window_s.items():
        by[k.split("/")[0]].append(v)
    summary = {d: {"sessions": len(v), "without_windows": sum(x is None for x in v),
                   "mean_window_s": float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None}
               for d, v in sorted(by.items())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"manifest": str(args.manifest), "window_s": window_s, "errors": errors, "summary": summary,
                                    "elapsed_s": time.time() - t0}, indent=1))
    for d, s in summary.items():
        print(d, s, flush=True)
    print(f"{len(window_s)} sessioni, {len(errors)} errori, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
