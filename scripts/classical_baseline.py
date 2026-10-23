#!/usr/bin/env python3
"""Baseline classica sulla validazione interna (23/10/2026, Simone: «parti con 1 e 2»): le feature di Hudgins del progetto
(`fm_features.td_features` -> `harness.features_hudgins`: MAV, ZC, SSC, WL per canale, soglia 0,05 in unita' della scala), sullo stesso
segnale filtrato e scalato delle feature del modello, con la STESSA sonda e la STESSA validazione interna di `calibration_probe.py` (fold di
soggetti, C scelto sui soggetti di validazione, S fuori fold coi pesi della P1; P2 sui soggetti fuori dai test dei tre split). Nessun
modello, nessun numero di test. Differenze dalla pipeline del modello (dichiarate anche nel rapporto): finestra intera, canali scartati dal
QC a 0, sonda logistica del modello e non la LDA di v10 §8. I test ufficiali passano da `probe_p1.py` / `probe_p2.py --feature-set td
--checkpoint none`. Le predizioni fuori fold vanno in `<out>.oof.npz` nel formato di `calibration_probe` (confronti accoppiati).

  python3 scripts/classical_baseline.py --task p1 --root R1 --root R2 --out td_p1.json
  python3 scripts/classical_baseline.py --task p2 --epn-root E1 --epn-root E2 --uci-root U --out td_p2.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch  # noqa: F401  prima di sklearn sul Mac

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from wearusfm.harness import fm_features as FF  # noqa: E402
from wearusfm.harness import p1_cross_subject as P1  # noqa: E402
from wearusfm.harness.downstream_ab import fingerprint  # noqa: E402

PROTOCOL = "baseline_classica_td_v1"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_td", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features_p1(roots, ds: str, subjects: list[str], log) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Come `probe_p1.dataset_features` (stesse sessioni, finestre e chiave di scala per sessione), con le feature classiche."""
    PP = _script("probe_p1")
    ns = argparse.Namespace(window_s=_script("calibration_probe").WINDOW_S, audit_inputs=True, batch=64, device="cpu", feature_set="td")
    x, y, s = PP.dataset_features(None, roots, ds, {"pretraining": subjects}, ns, log)["pretraining"]
    return x, y, s, ns.input_sha256


def features_p2(args, ds: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, list[str]]:
    """Come `calibration_probe.extract_p2`: tutti i soggetti letti, poi solo quelli fuori dai test dei tre split ufficiali."""
    from wearusfm.harness import benchmarks as B

    CP = _script("calibration_probe")
    w, y, s, fs = B.load_epn612(args.epn_root) if ds == "epn612" else B.load_uci_emg(args.uci_root)
    s = np.asarray([str(v) for v in s])
    internal = CP.p2_internal_subjects(sorted(set(s.tolist())))
    keep = np.isin(s, internal)
    w, y, s = w[keep], y[keep], s[keep]
    montage = FF.myo8_montage(ds, fs)
    return FF.extract_features(None, w, fs, montage, s, feature_set="td"), y, s, fingerprint(w, y, s, fs, montage), internal


def run(args, log) -> dict:
    from sklearn.metrics import balanced_accuracy_score

    CP = _script("calibration_probe")
    p1 = args.task == "p1"
    splits = json.loads(args.splits.read_text()) if p1 else None
    datasets = list(P1.P1_DATASETS) if p1 else ["epn612", "uci_emg"]
    report = {"protocol": PROTOCOL, "task": args.task, "features": list(FF.TD_FEATURES), "deadzone": FF.TD_DEADZONE,
              "differences_from_model": {"window": "intera, nessun ritaglio da 4 s", "qc": "canali scartati dal QC con feature a 0",
                                         "probe": "sonda logistica del modello (calibration_probe), non la LDA di v10 §8"},
              "probe": {"k_folds": CP.K_FOLDS, "fold_seed": CP.FOLD_SEED, "val_frac": CP.VAL_FRAC, "probe_seed": args.seed,
                        "window_s": CP.WINDOW_S if p1 else None, "same_as": "calibration_probe.oof_predictions"},
              "splits_sha256": _sha256(args.splits) if p1 else CP.P2_SPLITS_SHA256, "datasets": {}}
    parts, oof = [], {}
    for ds in datasets:
        if p1:
            pre = CP.pretraining_subjects(splits, ds)
            x, y, s, input_sha = features_p1(args.root, ds, pre, log)
        else:
            x, y, s, input_sha, pre = features_p2(args, ds)
        if sorted(set(s.tolist())) != pre:
            raise ValueError(f"{ds}: soggetti delle feature diversi da quelli della validazione interna")
        res = CP.oof_predictions(x, y, s, CP.fold_roles(pre), args.seed)
        bacc = float(balanced_accuracy_score(res["y"], res["pred"]))
        parts.append((res["y"], res["pred"], res["s"]))
        oof.update({f"{ds}_y": res["y"], f"{ds}_pred": res["pred"], f"{ds}_s": res["s"], f"{ds}_row": res["row"]})
        report["datasets"][ds] = {"input_sha256": input_sha, "subjects": pre, "s": bacc, "columns": int(x.shape[1]), "n_windows": int(len(y)),
                                  "n_oof_windows": int(len(res["y"])), "oof_rows_sha256": fingerprint(res["row"]), "folds": res["folds"]}
        log(f"{ds}: S {100 * bacc:.2f} ({len(pre)} soggetti, {x.shape[1]} colonne)")
    w = (P1.dataset_weights([len(splits["datasets"][d]["test"]) for d in datasets], "subjects") if p1
         else P1.dataset_weights([1] * len(datasets), "datasets"))
    report.update({"weights": dict(zip(datasets, w.tolist())), "S": float(np.dot(w, [report["datasets"][d]["s"] for d in datasets])),
                   "S_se": P1.mean_bacc_bootstrap_se(parts, seed=args.seed, weights=w)})
    oof_path = args.out.with_suffix(".oof.npz")
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(oof_path, datasets=np.asarray(datasets), **oof)
    report["oof"], report["oof_sha256"] = str(oof_path), _sha256(oof_path)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["p1", "p2"], required=True)
    ap.add_argument("--root", type=Path, action="append", help="P1: radici dei dati processati")
    ap.add_argument("--epn-root", type=Path, action="append")
    ap.add_argument("--uci-root", type=Path)
    ap.add_argument("--splits", type=Path, default=ROOT / "splits" / "v1" / "splits_v1.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.task == "p1" and not args.root:
        ap.error("P1: servono le radici dei dati (--root)")
    if args.task == "p2" and (not args.epn_root or not args.uci_root):
        ap.error("P2: servono --epn-root e --uci-root")
    t0 = time.time()

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    report = run(args, log)
    report["elapsed_s"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, allow_nan=False))
    log(f"{args.task}: S {100 * report['S']:.2f} ± {100 * report['S_se']:.2f}, {report['elapsed_s']:.0f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
