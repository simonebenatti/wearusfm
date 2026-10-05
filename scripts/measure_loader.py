#!/usr/bin/env python3
"""Scale di sessione e misura del ritmo del dataloader del pretraining (passo 6; autorizzato da Simone il 03/10/2026: «sì, via alla misura con le
scale di sessione»; conferma della decisione 13, filtro nel dataloader).

1. **Scale di sessione** (v10 §4.3: mediana dei MAD dei canali validi, un numero per sessione) per tutte le righe di pretraining del manifest, su
   piu' processi, con il filtro firmato (passa-banda 20-450 Hz, notch 50 e 60 Hz). Seme per sessione dalla sua chiave: rilanciando si ottengono gli
   stessi numeri. Errori per sessione registrati, non fatali. Uscita: `session_scales.json`.
2. **Ritmo**: `--workers` processi indipendenti producono `--batches` batch da `--batch-size` con la configurazione firmata (finestre 1-4 s, salti
   spezzati, maschere D10 a «tubi», frazione 0,5) e le scale appena calcolate, con il filtro e senza; finestre al secondo per processo e secondi
   per fase. Il passo 0 stimava ~68 finestre/s per GPU a 30M con 8 processi per GPU: servono ~9 finestre/s per processo. Uscita: `throughput.json`.

  python3 scripts/measure_loader.py --manifest manifest.json.gz --root $WORK/data/processed --root $SCRATCH/data/processed --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.data import virtual_montage as VM  # noqa: E402

FILTER = (20.0, 450.0)
REQUIRED_PER_PROCESS = 68.0 / 8  # passo 0: ~68 finestre/s per GPU, 8 processi per GPU


def signed_config(filter_band) -> L.LoaderConfig:
    """La configurazione firmata il 03/10/2026 (`pretraining_loader.signed_config`)."""
    return L.signed_config(filter_band)


def key_of(row: dict) -> str:
    return f"{row['dataset']}/{row['subject']}/{row['session']}"


def scale_keys(row: dict, bipolar: bool) -> list[tuple[str, tuple[str, int] | None]]:
    """Le chiavi delle scale di una sessione: la sua e, con `bipolar`, quelle delle derivazioni bipolari dei montaggi virtuali (D6a) per le
    classi a cui si applicano, una per asse e passo (`PretrainLoader.scale`: chiave `sessione|bip:asse:passo`)."""
    k = key_of(row)
    out: list[tuple[str, tuple[str, int] | None]] = [(k, None)]
    spec = VM.VirtualSpec()
    if bipolar and row.get("quota_class") in spec.classes:
        out += [(f"{k}|{VM.bipolar_scale_key(a, st)}", (a, st)) for a in VM.AXES for st in spec.strides]
    return out


def _scale_worker(args) -> tuple[dict, dict, int]:
    manifest, roots, tasks = args
    rows = [r for r, _ in tasks]
    index = L.ManifestIndex(rows, np.ones(len(rows)) / len(rows), [Path(r) for r in roots])
    cfg = signed_config(FILTER)
    scales, errors = {}, {}
    for row, keys in tasks:
        try:
            view = L.SessionView.open(index.path_of(row), cfg.split_at_gaps, row.get("sidecar_sha256"))
        except Exception as e:
            errors[key_of(row)] = f"{type(e).__name__}: {e}"
            continue
        for k, deriv in keys:
            try:  # lo stesso seme del dataloader; la bipolare sulle coppie di tutte le griglie della sessione
                pairs = VM.all_pairs(view.montage, *deriv) if deriv else None
                scales[k] = L.estimate_session_scale(view, cfg, np.random.default_rng(L.scale_seed(k)), pairs)
            except Exception as e:  # una sessione difettosa si registra, non ferma il calcolo
                errors[k] = f"{type(e).__name__}: {e}"
    return scales, errors, len(tasks)


def compute_scales(index: L.ManifestIndex, workers: int, max_sessions: int | None, known: dict | None = None,
                   checkpoint: Path | None = None, bipolar: bool = False) -> dict:
    """`known`: scale gia' calcolate (es. da un run precedente): si ricalcolano solo le sessioni che mancano. `checkpoint`: file riscritto ogni
    ~500 sessioni con le scale gia' calcolate, cosi' un TIMEOUT non perde la fase (si riprende con `--scales-from`)."""
    known = dict(known or {})
    rows = index.rows[:max_sessions] if max_sessions else index.rows
    tasks = [(r, [kd for kd in scale_keys(r, bipolar) if kd[0] not in known]) for r in rows]
    reused = sum(1 for _, keys in tasks if not keys)
    rows = [t for t in tasks if t[1]]  # sessioni con almeno una scala da calcolare
    chunks = [rows[i::workers * 8] for i in range(workers * 8)]
    t0 = time.time()
    scales, errors, done, saved = dict(known), {}, 0, 0
    with get_context("spawn").Pool(max(1, workers)) as pool:
        for s, e, n in pool.imap_unordered(_scale_worker, [(None, [str(r) for r in index.roots], c) for c in chunks if c]):
            scales.update(s)
            errors.update(e)
            done += n
            print(f"[{time.strftime('%H:%M:%S')}] scale: {done}/{len(rows)} sessioni, {len(errors)} errori, {time.time() - t0:.0f} s", flush=True)
            if checkpoint is not None and done - saved >= 500:
                tmp = checkpoint.with_suffix(".tmp")
                tmp.write_text(json.dumps({"partial": True, "scales": scales, "errors": errors}))
                tmp.replace(checkpoint)
                saved = done
    return {"n_sessions": len(rows) + reused, "reused": reused, "scales": scales, "errors": errors, "elapsed_s": time.time() - t0,
            "method": "mediana dei MAD dei canali validi su 32 tratti da 2 s filtrati (20-450 Hz, notch 50/60), seme dalla chiave della sessione"}


def _rate_worker(args) -> dict:
    manifest, roots, scales, filter_band, batches, batch_size, seed = args
    loader = L.PretrainLoader(L.ManifestIndex.load(Path(manifest), [Path(r) for r in roots]), signed_config(filter_band), dict(scales))
    rng = np.random.default_rng(seed)
    loader.batch(batch_size, rng)  # apertura delle prime sessioni fuori dalla misura
    loader.timings.clear()
    t0 = time.perf_counter()
    n_windows = 0
    for _ in range(batches):
        b = loader.batch(batch_size, rng)
        n_windows += len(b.signals)
    elapsed = time.perf_counter() - t0
    return {"windows": n_windows, "elapsed_s": elapsed, "windows_per_s": n_windows / elapsed, "timings_s": loader.timings}


def measure_rate(manifest: Path, roots: list[str], scales: dict, workers: int, batches: int, batch_size: int, filter_band) -> dict:
    with get_context("spawn").Pool(workers) as pool:
        res = pool.map(_rate_worker, [(str(manifest), roots, scales, filter_band, batches, batch_size, 1000 + i) for i in range(workers)])
    rates = [r["windows_per_s"] for r in res]
    stages: dict[str, float] = {}
    for r in res:
        for k, v in r["timings_s"].items():
            stages[k] = stages.get(k, 0.0) + v
    total = sum(stages.values()) or 1.0
    return {"filter": filter_band, "workers": workers, "per_process_windows_per_s": rates, "min_per_process": min(rates),
            "required_per_process": REQUIRED_PER_PROCESS, "passes": min(rates) >= REQUIRED_PER_PROCESS,
            "stage_fraction": {k: v / total for k, v in sorted(stages.items())}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-sessions", type=int, default=None, help="solo per le prove: limita le sessioni delle scale")
    ap.add_argument("--scales-from", type=Path, default=None, help="session_scales.json di un run precedente: ricalcola solo le mancanti")
    ap.add_argument("--skip-rate", action="store_true", help="solo le scale, senza la misura del ritmo")
    ap.add_argument("--bipolar-scales", action="store_true", help="anche le scale delle derivazioni bipolari dei montaggi virtuali (D6a)")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = L.ManifestIndex.load(args.manifest, [Path(r) for r in args.root])
    print(f"[{time.strftime('%H:%M:%S')}] manifest {args.manifest}: {len(index.rows)} sessioni di pretraining, {args.workers} processi", flush=True)
    known = json.loads(args.scales_from.read_text())["scales"] if args.scales_from else None
    sc = compute_scales(index, args.workers, args.max_sessions, known, checkpoint=args.out_dir / "session_scales.json",
                        bipolar=args.bipolar_scales)
    sc["manifest"] = str(args.manifest)
    (args.out_dir / "session_scales.json").write_text(json.dumps(sc, indent=1))
    vals = np.array(list(sc["scales"].values()))
    print(f"scale: {len(vals)} calcolate, {len(sc['errors'])} errori, mediana {np.median(vals):.4g}, min {vals.min():.4g}, max {vals.max():.4g}"
          if len(vals) else "scale: nessuna calcolata", flush=True)
    if args.skip_rate:
        return 0
    out = {"manifest": str(args.manifest), "batch_size": args.batch_size, "batches_per_process": args.batches, "runs": []}
    for band in (None, FILTER):
        r = measure_rate(args.manifest, args.root, sc["scales"], args.workers, args.batches, args.batch_size, band)
        out["runs"].append(r)
        print(f"ritmo, filtro {band}: finestre/s per processo {[round(v, 1) for v in r['per_process_windows_per_s']]} "
              f"(servono {REQUIRED_PER_PROCESS:.1f}: {'OK' if r['passes'] else 'NON BASTA'}); fasi "
              + ", ".join(f"{k} {100 * v:.0f}%" for k, v in r["stage_fraction"].items()), flush=True)
    (args.out_dir / "throughput.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
