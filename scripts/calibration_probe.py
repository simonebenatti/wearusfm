#!/usr/bin/env python3
"""Sonda della calibrazione C1-C4 della finestra 1 (protocollo congelato in `docs/calibrazione_c1_c4_selezione.md`): validation INTERNA alle
partizioni train delle sonde P1, a fold di soggetti. Si leggono solo i soggetti di pretraining di NinaPro DB2, DB3 e DB6 negli split firmati
(quelli su cui la P1 addestra la sua sonda); i soggetti di test non si aprono mai (controllato) e nessun numero di test entra nella selezione.

Stessa estrazione della P1 (`scripts/probe_p1.py`, `dataset_features`: encoder congelato, teacher, feature "model", finestre da 1 s, scala per
sessione) e stessa sonda (`harness.probe.linear_probe`: logistica bilanciata, C scelto sui soggetti di validazione, normalizzazione sul train).
Fold: per dataset i soggetti di pretraining (ordinati) si permutano col seme `FOLD_SEED` e si distribuiscono a turno in `K_FOLDS` fold; per il
fold k la sonda si valuta sui suoi soggetti, sceglie C su `val_frac` dei rimanenti (per eccesso, almeno 1, seme [FOLD_SEED, k]) e si addestra
sugli altri. Ogni soggetto e' predetto una volta, da una sonda che non l'ha visto. Le classi assenti dal train del fold si tolgono e si contano.

Metrica S: per dataset l'accuratezza bilanciata delle predizioni fuori fold; S = media pesata coi pesi della P1 ("subjects": proporzionali ai
soggetti di TEST degli split, solo i conteggi), errore standard per bootstrap sui soggetti stratificato per dataset (come la P1). Le predizioni
fuori fold vanno in un npz per i confronti accoppiati della selezione (`scripts/calibration_select.py`). Descrittivi, non criteri: rango
effettivo e colonne costanti delle feature.

  python3 scripts/calibration_probe.py --stage extract --checkpoint CKPT --root R1 --root R2 --features DIR       (GPU)
  python3 scripts/calibration_probe.py --stage probe --features DIR --out rapporto.json                         (CPU)

`--task p2` (bracci dell'obiettivo, decisione del 16/10/2026: il default si sceglie su S, non sui test): la validation interna della P2 su
EPN-612 e UCI-EMG, stesse finestre, stessa estrazione e stessa sonda della P2 (`scripts/probe_p2.py`). I soggetti interni sono quelli fuori
dai test di TUTTI e tre gli split 7:1:2 della P2 (semi 0, 1, 2): le finestre degli altri si scartano subito dopo la lettura, prima
dell'estrazione, e la sonda ricontrolla l'esclusione. Stessi fold di soggetti; S della P2 = media dei due dataset a pesi uguali, come la P2.

  python3 scripts/calibration_probe.py --task p2 --stage extract --checkpoint CKPT --epn-root D1 --epn-root D2 --uci-root D3 --features DIR
  python3 scripts/calibration_probe.py --task p2 --stage probe --features DIR --out rapporto.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.harness import p1_cross_subject as P1  # noqa: E402

PROTOCOL = "w1-calibrazione-selezione-v1"
K_FOLDS = 4
FOLD_SEED = 20261015
VAL_FRAC = 0.1
FEATURE_SET = "model"
WINDOW_S = 1.0
P2_DATASETS = ("epn612", "uci_emg")
P2_SPLIT_SEEDS = (0, 1, 2)  # gli split ufficiali della P2 (scripts/probe_p2.py: SPLIT_SEEDS)
P2_RATIOS = (0.7, 0.1, 0.2)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def pretraining_subjects(splits: dict, dataset: str) -> list[str]:
    """I soggetti di pretraining del dataset; errore se uno e' anche di test (gli split firmati li tengono separati)."""
    d = splits["datasets"][dataset]
    pre, test = sorted(d["pretraining"]), set(d["test"])
    if set(pre) & test:
        raise ValueError(f"{dataset}: soggetti sia di pretraining sia di test {sorted(set(pre) & test)}")
    return pre


def p2_internal_subjects(subjects) -> list[str]:
    """I soggetti della P2 fuori dai test dei tre split ufficiali (`split_subjects` 7:1:2 coi semi 0, 1, 2, come `probe_p2.py`)."""
    from wearusfm.harness.splits import split_subjects

    ids = sorted({str(x) for x in subjects})
    test = set()
    for seed in P2_SPLIT_SEEDS:
        test |= set(split_subjects(ids, P2_RATIOS, seed=seed).test)
    return [x for x in ids if x not in test]


def fold_roles(subjects: list[str], k: int = K_FOLDS, seed: int = FOLD_SEED, val_frac: float = VAL_FRAC) -> list[dict]:
    """[{train, val, held}] per fold: held = i soggetti del fold, val = `val_frac` dei rimanenti (scelta di C), train = gli altri."""
    subs = sorted(subjects)
    if len(subs) < k:
        raise ValueError(f"{len(subs)} soggetti per {k} fold")
    order = [subs[i] for i in np.random.default_rng(seed).permutation(len(subs))]
    folds = [sorted(order[i::k]) for i in range(k)]
    out = []
    for i, held in enumerate(folds):
        rest = [s for s in subs if s not in held]
        n_val = max(1, math.ceil(val_frac * len(rest)))
        if n_val >= len(rest):
            raise ValueError(f"fold {i}: {len(rest)} soggetti non bastano per train e validazione")
        val = sorted(np.random.default_rng([seed, i]).choice(rest, size=n_val, replace=False).tolist())
        out.append({"train": [s for s in rest if s not in val], "val": val, "held": held})
    return out


def feature_diagnostics(x: np.ndarray) -> dict:
    """Descrittivi del non-collasso delle feature (nessuna soglia): rango effettivo (exp dell'entropia dei valori singolari normalizzati,
    come la diagnostica del training) delle feature centrate e colonne quasi costanti."""
    xc = x.astype(np.float64) - x.mean(0, keepdims=True)
    sv = np.linalg.svd(xc, compute_uv=False)
    p = sv / sv.sum() if sv.sum() > 0 else np.ones_like(sv) / len(sv)
    p = p[p > 0]
    std = x.std(0)
    return {"effective_rank": float(np.exp(-(p * np.log(p)).sum())), "columns": int(x.shape[1]),
            "constant_columns": int((std <= 1e-6 * max(float(std.max()), 1e-12)).sum()), "finite": bool(np.isfinite(x).all())}


def _probe_p1_module():
    spec = importlib.util.spec_from_file_location("probe_p1_extraction", ROOT / "scripts" / "probe_p1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extract(args, splits: dict, log) -> None:
    from wearusfm.harness import fm_features as FF
    from wearusfm.harness.feature_sets import cache_record

    PP = _probe_p1_module()
    ckpt_sha = _sha256(Path(args.checkpoint))
    model = FF.load_model(args.checkpoint, args.device, "teacher")
    log(f"modello {args.checkpoint} (teacher, sha256 {ckpt_sha[:12]}), dataset {args.datasets}")
    ns = argparse.Namespace(window_s=WINDOW_S, audit_inputs=True, batch=args.batch, device=args.device, feature_set=FEATURE_SET)
    for ds in args.datasets:
        pre = pretraining_subjects(splits, ds)
        x, y, s = PP.dataset_features(model, args.root, ds, {"pretraining": pre}, ns, log)["pretraining"]
        if sorted(set(s.tolist())) != pre:
            raise RuntimeError(f"{ds}: soggetti estratti {sorted(set(s.tolist()))} diversi da quelli di pretraining")
        npz = args.features / f"{ds}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, x=x, y=y, s=s, subjects=np.asarray(pre), input_sha256=np.asarray(ns.input_sha256), checkpoint=np.asarray(str(args.checkpoint)),
                 checkpoint_sha256=np.asarray(ckpt_sha), encoder_dim=np.asarray(model.cfg.dim), protocol=np.asarray(PROTOCOL),
                 **cache_record(FEATURE_SET, FF.feature_width(model, FEATURE_SET)))
        log(f"  {ds}: {len(pre)} soggetti, {len(y)} finestre -> {npz}")


def extract_p2(args, log) -> None:
    """Feature della P2 interna: come `probe_p2.py`, sui soli soggetti fuori dai test dei tre split (le altre finestre si scartano prima)."""
    from wearusfm.harness import benchmarks as B
    from wearusfm.harness import fm_features as FF
    from wearusfm.harness.downstream_ab import fingerprint
    from wearusfm.harness.feature_sets import cache_record

    ckpt_sha = _sha256(Path(args.checkpoint))
    model = FF.load_model(args.checkpoint, args.device, "teacher")
    log(f"modello {args.checkpoint} (teacher, sha256 {ckpt_sha[:12]}), P2 interna {args.datasets}")
    for ds in args.datasets:
        if ds == "epn612":
            w, y, s, fs = B.load_epn612(args.epn_root)
        else:
            w, y, s, fs = B.load_uci_emg(args.uci_root)
        s = np.asarray([str(v) for v in s])
        all_subjects = sorted(set(s.tolist()))
        internal = p2_internal_subjects(all_subjects)
        keep = np.isin(s, internal)
        w, y, s = w[keep], y[keep], s[keep]
        montage = FF.myo8_montage(ds, fs)
        input_sha = fingerprint(w, y, s, fs, montage)
        x = FF.extract_features(model, w, fs, montage, s, batch=args.batch, device=args.device, feature_set=FEATURE_SET)
        npz = args.features / f"{ds}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, x=x, y=y, s=s, subjects=np.asarray(internal), all_subjects=np.asarray(all_subjects), input_sha256=np.asarray(input_sha),
                 checkpoint=np.asarray(str(args.checkpoint)), checkpoint_sha256=np.asarray(ckpt_sha), encoder_dim=np.asarray(model.cfg.dim),
                 protocol=np.asarray(PROTOCOL), task=np.asarray("p2"), **cache_record(FEATURE_SET, FF.feature_width(model, FEATURE_SET)))
        log(f"  {ds}: {len(internal)} soggetti interni su {len(all_subjects)} ({len(all_subjects) - len(internal)} nei test esclusi), "
            f"{len(y)} finestre -> {npz}")


def oof_predictions(x, y, s, roles: list[dict], seed: int) -> dict:
    """Predizioni fuori fold: (y, pred, soggetti) delle finestre tenute, piu' i dettagli per fold."""
    from wearusfm.harness.probe import linear_probe

    ys, ps, ss, per_fold = [], [], [], []
    for r in roles:
        mt, mv, mh = (np.isin(s, r[k]) for k in ("train", "val", "held"))
        keep_v, keep_h = P1.keep_train_classes(y[mt], y[mv], y[mh])
        res = linear_probe(x[mt], y[mt], x[mv][keep_v], y[mv][keep_v], x[mh][keep_h], y[mh][keep_h], s[mh][keep_h], seed=seed)
        ys.append(y[mh][keep_h])
        ps.append(res.test_pred)
        ss.append(s[mh][keep_h])
        per_fold.append({**r, "held_bacc": res.test_bacc, "val_bacc": res.val_bacc, "train_bacc": res.train_bacc, "c": res.c,
                         "n_train": res.n_train, "n_val": res.n_val, "n_held": res.n_test,
                         "dropped_val_windows": int((~keep_v).sum()), "dropped_held_windows": int((~keep_h).sum())})
    return {"y": np.concatenate(ys), "pred": np.concatenate(ps), "s": np.concatenate(ss), "folds": per_fold}


def probe(args, splits: dict, log) -> dict:
    from sklearn.metrics import balanced_accuracy_score

    from wearusfm.harness.feature_sets import check_cache

    report = {"protocol": PROTOCOL, "task": args.task, "k_folds": K_FOLDS, "fold_seed": FOLD_SEED, "val_frac": VAL_FRAC,
              "probe_seed": args.seed, "feature_set": FEATURE_SET, "window_s": WINDOW_S if args.task == "p1" else None,
              "splits": str(args.splits) if args.task == "p1" else f"P2 7:1:2, test esclusi dei semi {list(P2_SPLIT_SEEDS)}",
              "splits_sha256": _sha256(args.splits) if args.task == "p1" else None, "features": str(args.features), "datasets": {}}
    parts, oof, idents = [], {}, set()
    for ds in args.datasets:
        with np.load(args.features / f"{ds}.npz") as z:
            check_cache(z, FEATURE_SET, z["x"].shape[1])
            task = str(z["task"]) if "task" in z.files else "p1"
            if args.task == "p1":
                pre = pretraining_subjects(splits, ds)
            else:  # la sonda ricontrolla l'esclusione dei soggetti di test dei tre split
                pre = p2_internal_subjects(z["all_subjects"].tolist()) if "all_subjects" in z.files else None
            if (str(z["protocol"]) != PROTOCOL or task != args.task or pre is None or sorted(z["subjects"].tolist()) != pre
                    or sorted(set(z["s"].tolist())) != pre):
                raise ValueError(f"{ds}: feature di un altro protocollo, di un altro compito o di altri soggetti")
            x, y, s = z["x"], z["y"], z["s"]
            idents.add((str(z["checkpoint"]), str(z["checkpoint_sha256"]), int(z["encoder_dim"])))  # model_dim: larghezza delle feature
            report["datasets"][ds] = {"input_sha256": str(z["input_sha256"])}
        res = oof_predictions(x, y, s, fold_roles(pre), args.seed)
        bacc = float(balanced_accuracy_score(res["y"], res["pred"]))
        parts.append((res["y"], res["pred"], res["s"]))
        oof.update({f"{ds}_y": res["y"], f"{ds}_pred": res["pred"], f"{ds}_s": res["s"]})
        report["datasets"][ds].update({"subjects": pre, "s": bacc, "folds": res["folds"], "n_classes": int(len(np.unique(y))),
                                       "n_windows": int(len(y)), "n_oof_windows": int(len(res["y"])), "features": feature_diagnostics(x)})
        log(f"{ds}: S {100 * bacc:.2f} (fuori fold, {len(pre)} soggetti, {K_FOLDS} fold)")
    if len(idents) != 1:
        raise ValueError(f"feature di checkpoint diversi: {sorted(idents)}")
    (ckpt, ckpt_sha, dim), = idents
    if args.task == "p1":
        n_test = [len(splits["datasets"][d]["test"]) for d in args.datasets]  # solo i conteggi, per i pesi della P1
        w = P1.dataset_weights(n_test, "subjects")
    else:
        w = P1.dataset_weights([1] * len(args.datasets), "datasets")  # la P2 e' la media dei due dataset
    report.update({"checkpoint": ckpt, "checkpoint_sha256": ckpt_sha, "model_dim": dim, "weights": dict(zip(args.datasets, w.tolist())),
                   "S": float(np.dot(w, [report["datasets"][d]["s"] for d in args.datasets])),
                   "S_se": P1.mean_bacc_bootstrap_se(parts, seed=args.seed, weights=w)})
    oof_path = args.out.with_suffix(".oof.npz")
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(oof_path, datasets=np.asarray(args.datasets), **oof)
    report["oof"], report["oof_sha256"] = str(oof_path), _sha256(oof_path)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["extract", "probe"], required=True)
    ap.add_argument("--task", choices=["p1", "p2"], default="p1", help="p1 (calibrazione, default) o p2 (validation interna della P2)")
    ap.add_argument("--epn-root", type=Path, action="append", default=None, help="P2: trainingJSON e testingJSON di EPN-612")
    ap.add_argument("--uci-root", type=Path, default=None, help="P2: UCI-EMG")
    ap.add_argument("--features", type=Path, required=True, help="cartella delle feature per dataset (fuori dal codice)")
    ap.add_argument("--checkpoint", help="checkpoint del training (extract)")
    ap.add_argument("--root", type=Path, action="append", default=[], help="radici dei dati processati (extract)")
    ap.add_argument("--splits", type=Path, default=ROOT / "splits" / "v1" / "splits_v1.json")
    ap.add_argument("--datasets", nargs="+", default=None, help="default: quelli del compito (P1: DB2, DB3, DB6; P2: EPN-612, UCI-EMG)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="seme della sonda e del bootstrap (come la P1)")
    ap.add_argument("--out", type=Path, help="rapporto JSON (probe); le predizioni fuori fold accanto, .oof.npz")
    args = ap.parse_args(argv)
    args.datasets = args.datasets or list(P1.P1_DATASETS if args.task == "p1" else P2_DATASETS)
    if args.task == "p2" and set(args.datasets) - set(P2_DATASETS):
        ap.error(f"--task p2: dataset fra {P2_DATASETS}")
    if args.stage == "extract" and args.task == "p1" and (not args.checkpoint or not args.root):
        ap.error("extract: servono --checkpoint e --root")
    if args.stage == "extract" and args.task == "p2" and (not args.checkpoint or ("epn612" in args.datasets and not args.epn_root)
                                                          or ("uci_emg" in args.datasets and not args.uci_root)):
        ap.error("extract della P2: servono --checkpoint, --epn-root e --uci-root")
    if args.stage == "probe" and args.out is None:
        ap.error("probe: serve --out")
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    splits = json.loads(args.splits.read_text())
    if args.stage == "extract":
        (extract(args, splits, log) if args.task == "p1" else extract_p2(args, log))
        log(f"estrazione finita in {time.time() - t0:.0f} s")
        return 0
    report = probe(args, splits, log)
    report["elapsed_s"] = time.time() - t0
    args.out.write_text(json.dumps(report, indent=1))
    log(f"S = {100 * report['S']:.2f} ± {100 * report['S_se']:.2f}; {report['elapsed_s']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
