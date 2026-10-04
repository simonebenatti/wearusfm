#!/usr/bin/env python3
"""Codici dell'ancora RVQ (D5b: livello 0 del ramo 0 del tokenizer NeuroRVQ-EMG congelato) per le sessioni di pretraining di `manifest-v1` con
l'ancora accesa, nei dataset scelti. Stesso ingresso di V2 (`wearusfm.data.rvq_codes`). Gira su Leonardo con una GPU: la vista canonica la
preparano `--workers` processi, il tokenizer gira nel processo principale. Riprendibile: le sessioni gia' scritte si saltano.

  python3 scripts/precompute_rvq_codes.py --manifest manifest_59253155.json.gz --root $WORK/data/processed --root $SCRATCH/data/processed \\
      --datasets emg2qwerty --repo-dir $SCRATCH/external/NeuroRVQ_926e770 --checkpoint $WORK/models/NeuroRVQ_EMG_tokenizer_v1.pt \\
      --out-dir $WORK/wearusfm_runs/results/passo6/rvq_codes

Uscita fuori dal repo: un `.npz` per sessione e `index_<job>.json` (sessioni scritte, saltate, errori, sha256 del checkpoint, fattore).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.data import rvq_codes as RC  # noqa: E402
from wearusfm.data.processed import load_session  # noqa: E402


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _prepare(args):
    """Nel processo di lavoro: sessione -> prove canoniche. Ritorna (chiave, prove, qc, scala, errore)."""
    row, path = args
    key = (row["dataset"], row["subject"], row["session"])
    try:
        sd = Path(path)
        if row.get("sidecar_sha256") and hashlib.sha256((sd / "metadata.json").read_bytes()).hexdigest() != row["sidecar_sha256"]:
            raise ValueError("metadata.json diverso da quello del manifest (sha256)")
        s = load_session(sd, *key)
        trials, scale = RC.canonical_trials(s.segments, s.fs, s.qc_valid)
        return key, trials, s.qc_valid, scale, None
    except Exception as e:  # una sessione difettosa si registra, non ferma il calcolo
        return key, None, None, None, f"{type(e).__name__}: {e}"


def bounded_map(pool, fn, items, max_pending: int):
    """Come `pool.imap`, ma con al piu' `max_pending` risultati in volo: i processi di lavoro non corrono avanti al consumatore. Con `imap_unordered`
    le viste canoniche (~140 MB a sessione di emg2qwerty) si accumulavano nel processo principale piu' in fretta della GPU: job 59303528, memoria
    esaurita (120 GB) dopo 175 sessioni."""
    pending, it = [], iter(items)
    for item in it:
        pending.append(pool.apply_async(fn, (item,)))
        if len(pending) >= max_pending:
            break
    while pending:
        res = pending.pop(0).get()
        nxt = next(it, None)
        if nxt is not None:
            pending.append(pool.apply_async(fn, (nxt,)))
        yield res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, action="append", required=True)
    ap.add_argument("--datasets", required=True, help="elenco separato da virgole")
    ap.add_argument("--repo-dir", type=Path, required=True, help="clone di NeuroRVQ (commit 926e770), fuori da questo repo")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--factor", type=float, default=RC.V2_SCALE_FACTOR)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=3, help="processi per la vista canonica (la GPU e' il collo di bottiglia: ~8,5 s a sessione)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-sessions", type=int, default=None, help="solo per il collaudo")
    ap.add_argument("--time-limit-s", type=float, default=None, help="smette di prendere sessioni nuove dopo questo tempo (riprendibile)")
    args = ap.parse_args(argv)
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sha = sha256_of(args.checkpoint)
    if sha != RC.V2_CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint {args.checkpoint}: sha256 {sha} diverso da quello di V2 ({RC.V2_CHECKPOINT_SHA256})")
    datasets = set(args.datasets.split(","))
    index = L.ManifestIndex.load(args.manifest, args.root)
    rows = [r for r in index.rows if r["dataset"] in datasets and r["rvq"] == "on"]
    todo = [r for r in rows if not RC.code_path(args.out_dir, r["dataset"], r["subject"], r["session"]).exists()]
    if args.max_sessions:
        todo = todo[: args.max_sessions]
    print(f"[{time.strftime('%H:%M:%S')}] {len(rows)} sessioni di pretraining con l'ancora accesa in {sorted(datasets)}; da calcolare {len(todo)}",
          flush=True)

    from wearusfm.tokenizer_checks.neurorvq import SPATIAL_INDEX_FIXED, NeuroRVQRunner

    runner = NeuroRVQRunner(args.repo_dir, args.checkpoint, device=args.device)

    def encode(x: np.ndarray) -> np.ndarray:  # (B, n_time*200) di un canale -> (B, n_time): livello 0 del ramo 0
        return runner.run(x[:, None, :], [SPATIAL_INDEX_FIXED], want_recon=False)["codes"][0, 0].reshape(len(x), -1)

    written, errors, n_tok, stopped = 0, {}, 0, False
    with get_context("spawn").Pool(max(1, args.workers)) as pool:
        for key, trials, qc, scale, err in bounded_map(pool, _prepare, [(r, str(index.path_of(r))) for r in todo], args.workers + 2):
            if err is not None:
                errors["/".join(key)] = err
                continue
            out = RC.session_codes(trials, qc, encode, factor=args.factor, batch=args.batch)
            p = RC.code_path(args.out_dir, *key)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.stem + ".tmp.npz")
            np.savez(tmp, scale=np.float64(scale), factor=np.float64(args.factor), **out)
            os.replace(tmp, p)
            written += 1
            n_tok += int((out["codes"] >= 0).sum())
            if written % 25 == 0 or written == len(todo):
                print(f"[{time.strftime('%H:%M:%S')}] {written}/{len(todo)} sessioni, {len(errors)} errori, {n_tok / 1e6:.1f} M token, "
                      f"{time.time() - t0:.0f} s", flush=True)
            if args.time_limit_s is not None and time.time() - t0 > args.time_limit_s:
                stopped = True
                pool.terminate()
                break
    report = {"manifest": str(args.manifest), "datasets": sorted(datasets), "checkpoint_sha256": sha, "factor": args.factor,
              "block_patches": RC.BLOCK_PATCHES, "sessions_with_rvq": len(rows), "todo": len(todo), "written": written, "errors": errors,
              "tokens": n_tok, "stopped_by_time_limit": stopped, "elapsed_s": time.time() - t0}
    (args.out_dir / f"index_{os.environ.get('SLURM_JOB_ID', 'locale')}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "errors"} | {"n_errors": len(errors)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
