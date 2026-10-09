#!/usr/bin/env python3
"""Metrica P2 di D12 sul nostro FM: transfer su dataset mai visti, sonda lineare sull'encoder congelato su EPN-612 e UCI-EMG, accuratezza
bilanciata, media dei due (D12, firmata il 04/10/2026). **Stesse finestre e stessi split della replica di NeuroRVQ** (foglio firmato il 04/10:
«stesse finestre e stessi split, preprocessing suo, encoder congelato e sonda lineare»): `harness.benchmarks`, split per soggetto 7:1:2 coi semi
0, 1, 2 (`harness.splits.split_subjects`). Ingresso e feature come P1 (`harness.fm_features`: bracciale Myo a 8 canali, scala per soggetto).

**Aggregazione (scelta di AG, da dichiarare):** per dataset, media dell'accuratezza bilanciata di test sui 3 split; errore standard = radice della
media dei quadrati degli SE di bootstrap dei singoli split (i test dei 3 split si sovrappongono: la media non riduce l'errore). P2 = media dei due
dataset, SE = radice della somma dei quadrati / 2 (dataset indipendenti).

  python3 scripts/probe_p2.py --checkpoint CKPT --epn-root DIR1 --epn-root DIR2 --uci-root DIR --out p2.json
Due fasi separabili come P1: `--stage extract --features DIR` (GPU), poi `--stage probe --features DIR` (CPU).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.harness import benchmarks as B  # noqa: E402
from wearusfm.harness.splits import masks_from_split, split_subjects  # noqa: E402
from wearusfm.harness.downstream_ab import fingerprint, subject_counts  # noqa: E402

DATASETS = ("epn612", "uci_emg")
SPLIT_SEEDS = (0, 1, 2)


def load_windows(dataset: str, args) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if dataset == "epn612":
        return B.load_epn612(args.epn_root, n_users=args.max_users)
    return B.load_uci_emg(args.uci_root, n_subjects=args.max_users)


def probe_dataset(x: np.ndarray, labels: np.ndarray, subjects: np.ndarray, seeds, seed: int = 0) -> dict:
    """Sonda lineare su ciascuno split; media e SE come nel docstring del modulo."""
    from wearusfm.harness.probe import linear_probe

    classes = sorted(set(labels.tolist()))
    y = np.array([classes.index(v) for v in labels.tolist()], dtype=np.int64)
    runs = []
    for s in seeds:
        m = masks_from_split(subjects.tolist(), split_subjects(subjects.tolist(), (0.7, 0.1, 0.2), seed=s))
        r = linear_probe(x[m["train"]], y[m["train"]], x[m["val"]], y[m["val"]], x[m["test"]], y[m["test"]], subjects[m["test"]], seed=seed)
        runs.append({"split_seed": s, "test_bacc": r.test_bacc, "test_bacc_se": r.test_bacc_se, "val_bacc": r.val_bacc, "train_bacc": r.train_bacc,
                     "c": r.c, "n_train": r.n_train, "n_val": r.n_val, "n_test": r.n_test, "n_test_subjects": r.n_test_subjects,
                     "paired_counts": subject_counts(y[m["test"]], r.test_pred, subjects[m["test"]])})
    b = np.array([r["test_bacc"] for r in runs])
    se = np.array([r["test_bacc_se"] for r in runs])
    return {"classes": [str(c) for c in classes], "test_bacc": float(b.mean()), "test_bacc_se": float(np.sqrt((se ** 2).mean())), "splits": runs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--epn-root", type=Path, action="append", default=None, help="trainingJSON e testingJSON di EPN-612")
    ap.add_argument("--uci-root", type=Path, default=None)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--split-seeds", type=int, nargs="+", default=list(SPLIT_SEEDS))
    ap.add_argument("--which", default="teacher", choices=["teacher", "student"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-users", type=int, default=None, help="solo per le prove")
    ap.add_argument("--stage", choices=["all", "extract", "probe"], default="all")
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--audit-inputs", action="store_true", help="fingerprint actual ordered extraction inputs")
    args = ap.parse_args(argv)
    if args.stage != "all" and args.features is None:
        ap.error("--stage extract/probe richiede --features")
    if args.stage != "extract" and args.out is None:
        ap.error("serve --out")
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    model = None
    if args.stage != "probe":
        from wearusfm.harness import fm_features as FF  # torch prima di sklearn (sul Mac l'ordine opposto va in segmentation fault)

        model = FF.load_model(args.checkpoint, args.device, args.which)
        log(f"modello {args.checkpoint} ({args.which})")
    report = {"checkpoint": str(args.checkpoint), "which": args.which, "split_seeds": args.split_seeds, "datasets": {}}
    for ds in args.datasets:
        npz = args.features / f"{ds}.npz" if args.features is not None else None
        if args.stage == "probe":
            with np.load(npz) as z:
                x, labels, subjects = z["x"], z["y"], z["s"]
        else:
            w, labels, subjects, fs = load_windows(ds, args)
            log(f"{ds}: {len(w)} finestre, {len(set(subjects.tolist()))} soggetti, {w.shape[1] / fs:.1f} s a {fs} Hz")
            input_hash = fingerprint(w, labels, subjects, fs, FF.myo8_montage(ds, fs)) if args.audit_inputs else None
            x = FF.extract_features(model, w, fs, FF.myo8_montage(ds, fs), subjects, batch=args.batch, device=args.device)
            del w
            if npz is not None:
                npz.parent.mkdir(parents=True, exist_ok=True)
                extra = {"input_sha256": np.asarray(input_hash)} if args.audit_inputs else {}
                np.savez(npz, x=x, y=labels, s=subjects, **extra)
                log(f"  feature salvate in {npz}")
            if args.stage == "extract":
                continue
        report["datasets"][ds] = probe_dataset(x, labels, subjects, args.split_seeds, args.seed)
        d = report["datasets"][ds]
        log(f"{ds}: accuratezza bilanciata di test {100 * d['test_bacc']:.2f} ± {100 * d['test_bacc_se']:.2f} (split "
            + ", ".join(f"{100 * r['test_bacc']:.2f}" for r in d["splits"]) + ")")
    if args.stage == "extract":
        log(f"estrazione finita in {time.time() - t0:.0f} s")
        return 0
    vals = [report["datasets"][d] for d in args.datasets]
    report["p2"] = float(np.mean([v["test_bacc"] for v in vals]))
    report["p2_se"] = float(np.sqrt(sum(v["test_bacc_se"] ** 2 for v in vals)) / len(vals))
    report["elapsed_s"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    log(f"P2 = {100 * report['p2']:.2f} ± {100 * report['p2_se']:.2f} (media su {len(vals)} dataset); {report['elapsed_s']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
