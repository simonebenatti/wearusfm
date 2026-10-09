#!/usr/bin/env python3
"""Metrica P1 di D12 sul nostro FM: cross-soggetto sul ramo sparso (NinaPro DB2, DB3, DB6), sonda lineare sull'encoder congelato, accuratezza
bilanciata sui soggetti di test, errore standard per bootstrap sui soggetti (protocollo in `docs/foglio_p1_cross_soggetto.md`, da firmare;
`harness/p1_cross_subject.py`). Richiede torch; su Leonardo in un job con GPU (nessuna rete).

  python3 scripts/probe_p1.py --checkpoint CKPT --root $SCRATCH/data/processed --splits splits/v1/splits_v1.json --out p1.json

Due fasi separabili, come il confronto dei target candidati: `--stage extract --features DIR` (GPU: feature per dataset in DIR/<dataset>.npz),
poi `--stage probe --features DIR` (CPU: sonde ed errori standard). `--stage all` (default) fa tutte e due nello stesso processo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.harness import p1_cross_subject as P1  # noqa: E402
from wearusfm.harness.downstream_ab import fingerprint, subject_counts  # noqa: E402


def _root_of(roots: list[Path], dataset: str, subject: str) -> Path:
    for r in roots:
        if (r / dataset / subject).is_dir():
            return r
    raise FileNotFoundError(f"{dataset}/{subject} non trovato in {roots}")


def dataset_features(model, roots: list[Path], dataset: str, roles: dict, args, log) -> dict:
    """{ruolo: (feature (N, 2d), etichette (N,), soggetti (N,))}. Una sessione alla volta, col suo montaggio; la scala e' per sessione."""
    from wearusfm.harness import fm_features as FF

    out = {}
    digest = hashlib.sha256()
    for role, subjects in roles.items():
        xs, ys, ss = [], [], []
        for s in subjects:
            for sess in P1.sessions_of(_root_of(roots, dataset, s), dataset, s):
                sw = P1.session_windows(sess, dataset, window_s=args.window_s)
                if not len(sw.windows):
                    log(f"  {dataset}/{s}/{sess.name}: nessuna finestra")
                    continue
                key = np.full(len(sw.windows), f"{s}/{sw.session}")  # chiave della scala: la sessione, come nel pretraining
                if args.audit_inputs:
                    digest.update(fingerprint(role, s, sw.session, sw.windows, sw.labels, sw.fs, sw.montage, key).encode())
                xs.append(FF.extract_features(model, sw.windows, sw.fs, sw.montage, key, batch=args.batch, device=args.device))
                ys.append(sw.labels)
                ss.append(np.full(len(sw.windows), s))
        out[role] = (np.concatenate(xs), np.concatenate(ys), np.concatenate(ss))
        log(f"  {dataset} {role}: {len(subjects)} soggetti, {len(out[role][1])} finestre")
    args.input_sha256 = digest.hexdigest() if args.audit_inputs else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", type=Path, action="append", required=True)
    ap.add_argument("--splits", type=Path, default=ROOT / "splits" / "v1" / "splits_v1.json")
    ap.add_argument("--datasets", nargs="+", default=list(P1.P1_DATASETS))
    ap.add_argument("--window-s", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--which", default="teacher", choices=["teacher", "student"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weighting", choices=["subjects", "datasets"], default="subjects",
                    help="pesi dei dataset nella media P1 (foglio): proporzionali ai soggetti di test, oppure uguali")
    ap.add_argument("--out", type=Path, default=None, help="report JSON (fasi probe e all)")
    ap.add_argument("--stage", choices=["all", "extract", "probe"], default="all")
    ap.add_argument("--features", type=Path, default=None, help="cartella delle feature per dataset (necessaria con extract e probe)")
    ap.add_argument("--audit-inputs", action="store_true", help="fingerprint actual ordered extraction inputs")
    args = ap.parse_args(argv)
    if args.stage != "all" and args.features is None:
        ap.error("--stage extract/probe richiede --features")
    if args.stage != "extract" and args.out is None:
        ap.error("serve --out")
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    splits = json.loads(args.splits.read_text())
    model = None
    if args.stage != "probe":
        from wearusfm.harness import fm_features as FF

        model = FF.load_model(args.checkpoint, args.device, args.which)
        log(f"modello {args.checkpoint} ({args.which}), dataset {args.datasets}, finestre da {args.window_s} s")
    # sklearn DOPO torch: nell'ordine opposto, sul Mac, le operazioni di torch a piu' thread vanno in segmentation fault (due runtime OpenMP;
    # verificato il 05/10, venv del tokenizer)
    from wearusfm.harness.probe import linear_probe
    report = {"checkpoint": str(args.checkpoint), "which": args.which, "window_s": args.window_s, "val_frac": args.val_frac, "seed": args.seed,
              "splits": str(args.splits), "datasets": {}}
    parts = []
    for ds in args.datasets:
        roles = P1.subject_roles(splits, ds, args.val_frac, args.seed)
        npz = args.features / f"{ds}.npz" if args.features is not None else None
        if args.stage == "probe":
            with np.load(npz) as z:
                f = {r: (z[f"{r}_x"], z[f"{r}_y"], z[f"{r}_s"]) for r in ("train", "val", "test")}
        else:
            f = dataset_features(model, args.root, ds, roles, args, log)
            if npz is not None:
                npz.parent.mkdir(parents=True, exist_ok=True)
                extra = {"input_sha256": np.asarray(args.input_sha256)} if args.audit_inputs else {}
                np.savez(npz, **{f"{r}_{k}": v for r, t in f.items() for k, v in zip("xys", t)}, **extra)
                log(f"  feature salvate in {npz}")
            if args.stage == "extract":
                continue
        (xt, yt, _), (xv, yv, _), (xs, ys, ss) = f["train"], f["val"], f["test"]
        mv, ms = P1.keep_train_classes(yt, yv, ys)
        res = linear_probe(xt, yt, xv[mv], yv[mv], xs[ms], ys[ms], ss[ms], seed=args.seed)
        parts.append((ys[ms], res.test_pred, ss[ms]))
        report["datasets"][ds] = {
            "subjects": roles, "test_bacc": res.test_bacc, "test_bacc_se": res.test_bacc_se, "val_bacc": res.val_bacc, "train_bacc": res.train_bacc,
            "c": res.c, "n_train": res.n_train, "n_val": res.n_val, "n_test": res.n_test, "n_classes": int(len(np.unique(yt))),
            "dropped_val_windows": int((~mv).sum()), "dropped_test_windows": int((~ms).sum()),
            "paired_counts": subject_counts(ys[ms], res.test_pred, ss[ms]),
        }
        log(f"{ds}: accuratezza bilanciata di test {100 * res.test_bacc:.2f} ± {100 * res.test_bacc_se:.2f} (validazione "
            f"{100 * res.val_bacc:.2f}, C {res.c}, {len(np.unique(yt))} classi)")
    if args.stage == "extract":
        log(f"estrazione finita in {time.time() - t0:.0f} s")
        return 0
    w = P1.dataset_weights([len(report["datasets"][d]["subjects"]["test"]) for d in args.datasets], args.weighting)
    report["weighting"], report["weights"] = args.weighting, dict(zip(args.datasets, w.tolist()))
    report["p1"] = float(np.dot(w, [report["datasets"][d]["test_bacc"] for d in args.datasets]))
    report["p1_se"] = P1.mean_bacc_bootstrap_se(parts, seed=args.seed, weights=w)
    report["p1_equal_weights"] = float(np.mean([report["datasets"][d]["test_bacc"] for d in args.datasets]))
    report["elapsed_s"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    log(f"P1 = {100 * report['p1']:.2f} ± {100 * report['p1_se']:.2f} (media pesata per {args.weighting} su {len(args.datasets)} dataset); "
        f"{report['elapsed_s']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
