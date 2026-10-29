#!/usr/bin/env python3
"""Readout per canale sui checkpoint esistenti (23/10/2026, Simone: «Parti dal readout per canale sui checkpoint esistenti (3.1), insieme
alla baseline classica»; protocollo `docs/readout_per_canale_v1.md`, modulo `harness.channel_readout`).

Domanda: quanta informazione dei compiti P1/P2 l'encoder ha e il readout medio (`pool_p1p2`: media sui canali e sui latenti) butta via?
Due strati della stessa passata in avanti tenuti per canale (`local`, `dec`), piu' il readout di sempre (`pool`) come controllo
d'uguaglianza, valutati con la STESSA validazione interna della calibrazione (`calibration_probe`: fold di soggetti, C sui soggetti di
validazione, S coi pesi della P1; P2 sui soggetti fuori dai test dei tre split) e la STESSA sonda, a larghezza pari (PCA sul train del fold
a k = 2d componenti, default) oppure intera (`--k 0`). Nessun training, nessun numero di test.

  python3 scripts/channel_readout.py --stage extract --checkpoint CKPT --root R1 --root R2 --features DIR                (GPU)
  python3 scripts/channel_readout.py --stage extract --task p2 --checkpoint CKPT --epn-root E1 --epn-root E2 --uci-root U --features DIR
  python3 scripts/channel_readout.py --stage probe --features DIR --layer local --out rapporto.json [--k 768] [--task p2]   (CPU)
  python3 scripts/channel_readout.py --stage compare --reference RIF_s0.json RIF_s1.json --candidate CAND_s0.json CAND_s1.json \\
      [--oof-map /leonardo_work/...=/Users/.../copia] --out confronto.json                                                 (Mac)
  python3 scripts/channel_readout.py --stage equality --reference STORICO_s_p1.json --candidate POOL_s0_p1.json [--oof-map ...] --out eq.json

I rapporti della fase probe hanno il formato dei rapporti di `calibration_probe` (S, S_se, pesi, fold, predizioni fuori fold accanto):
`compare` li accoppia per finestra (stessi `input_sha256` e `oof_rows_sha256`) e per modello (stesso checkpoint per seme) con i rapporti
storici di S dei bracci e con quelli della baseline classica, e applica la regola dei bracci con lo stesso bootstrap congiunto dei soggetti
(`arms_compare.s_comparison`). Le cache hanno `feature_set` "channel_readout": le sonde di sempre le rifiutano.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.harness import channel_readout as CR  # noqa: E402
from wearusfm.harness import p1_cross_subject as P1  # noqa: E402
from wearusfm.harness.downstream_ab import fingerprint  # noqa: E402
from wearusfm.harness.probe import MAX_ITER, TIE_BREAK  # noqa: E402

PROTOCOL = "readout_per_canale_v1"
DIAG_ROWS = 8192  # righe (seme fisso) per il rango effettivo descrittivo: la SVD intera di 60.000 x 4.608 non serve a un descrittivo


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_channel_readout", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CP = _script("calibration_probe")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def recipe(task: str) -> dict:
    """Ricetta dell'estrazione, registrata nella cache e verificata dalla sonda: quella della calibrazione piu' gli strati per canale."""
    return {"base": CP.RECIPES[task], "extractor": "harness.channel_readout.extract_channel_readouts", "layers": list(CR.LAYERS),
            "channel_mean": "media sulle patch valide per canale; canali scartati dal QC o senza patch a 0", "dtype": "float16 per canale, float32 pool",
            "crop_s": 4.0, "k_neighbors": 8, "edge_guard_s": 0.1, "encoder": "teacher"}


def _root_of(roots: list[Path], dataset: str, subject: str) -> Path:
    for r in roots:
        if (r / dataset / subject).is_dir():
            return r
    raise FileNotFoundError(f"{dataset}/{subject} non trovato in {roots}")


def _save(npz: Path, x: dict, y, s, *, subjects, input_sha256, task, channels, model, ckpt_sha, args, splits_sha256, extra=None):
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, **{f"x_{k}": x[k] for k in CR.LAYERS}, y=y, s=s, subjects=np.asarray(subjects), input_sha256=np.asarray(input_sha256),
             checkpoint=np.asarray(str(args.checkpoint)), checkpoint_sha256=np.asarray(ckpt_sha), encoder_dim=np.asarray(model.cfg.dim),
             channels=np.asarray(int(channels)), protocol=np.asarray(PROTOCOL), calibration_protocol=np.asarray(CP.PROTOCOL),
             task=np.asarray(task), splits_sha256=np.asarray(splits_sha256), extraction_recipe=np.asarray(json.dumps(recipe(task), sort_keys=True)),
             feature_set=np.asarray(CR.FEATURE_SET), code_sha=np.asarray(os.environ.get("WEARUSFM_CODE_SHA", "")), **(extra or {}))


def montage_signature(montage: dict) -> list:
    """L'ordine dei canali da cui dipende il significato delle colonne: (gruppo, indice del canale) nell'ordine del montaggio. Non entra il
    QC per sessione (quei canali valgono 0, colonne al posto giusto)."""
    return [[gi, int(ch["sensor_coords"]["channel_index"])] for gi, g in enumerate(montage["groups"]) for ch in g["channels"]]


def _same_montage(seen, montage: dict, where: str) -> list:
    sig = montage_signature(montage)
    if seen is not None and sig != seen:
        raise ValueError(f"{where}: ordine dei canali del montaggio diverso dalle altre sessioni")
    return sig


def extract_p1(args, splits: dict, log) -> None:
    """Come `calibration_probe.extract` (stesse sessioni, finestre da 1 s, chiave di scala per sessione, stessa impronta degli ingressi), con
    gli strati per canale al posto del solo readout medio."""
    from wearusfm.harness import fm_features as FF

    ckpt_sha, splits_sha = _sha256(Path(args.checkpoint)), _sha256(args.splits)
    model = FF.load_model(args.checkpoint, args.device, "teacher")
    log(f"modello {args.checkpoint} (teacher, sha256 {ckpt_sha[:12]}), dataset {args.datasets}")
    for ds in args.datasets:
        pre = CP.pretraining_subjects(splits, ds)
        digest, xs, ys, ss, signature = hashlib.sha256(), {k: [] for k in CR.LAYERS}, [], [], None
        for s in pre:
            for sess in P1.sessions_of(_root_of(args.root, ds, s), ds, s):
                sw = P1.session_windows(sess, ds, window_s=CP.WINDOW_S)
                if not len(sw.windows):
                    log(f"  {ds}/{s}/{sess.name}: nessuna finestra")
                    continue
                key = np.full(len(sw.windows), f"{s}/{sw.session}")  # chiave della scala: la sessione, come nel pretraining e nella P1
                digest.update(fingerprint("pretraining", s, sw.session, sw.windows, sw.labels, sw.fs, sw.montage, key).encode())
                signature = _same_montage(signature, sw.montage, f"{ds}/{s}/{sw.session}")
                out = CR.extract_channel_readouts(model, sw.windows, sw.fs, sw.montage, key, batch=args.batch, device=args.device)
                for k in CR.LAYERS:
                    xs[k].append(out[k])
                ys.append(sw.labels)
                ss.append(np.full(len(sw.windows), s))
        x, y, s_all = {k: np.concatenate(v) for k, v in xs.items()}, np.concatenate(ys), np.concatenate(ss)
        if sorted(set(s_all.tolist())) != pre:
            raise RuntimeError(f"{ds}: soggetti estratti {sorted(set(s_all.tolist()))} diversi da quelli di pretraining")
        npz = args.features / f"{ds}.npz"
        _save(npz, x, y, s_all, subjects=pre, input_sha256=digest.hexdigest(), task="p1", channels=len(signature), model=model, ckpt_sha=ckpt_sha,
              args=args, splits_sha256=splits_sha, extra={"montage_signature": np.asarray(json.dumps(signature))})
        log(f"  {ds}: {len(pre)} soggetti, {len(y)} finestre, {len(signature)} canali x {model.cfg.dim} -> {npz}")


def extract_p2(args, log) -> None:
    """Come `calibration_probe.extract_p2`: soli soggetti fuori dai test dei tre split, finestre dei caricatori della P2, montaggio myo8."""
    from wearusfm.harness import benchmarks as B
    from wearusfm.harness import fm_features as FF

    ckpt_sha = _sha256(Path(args.checkpoint))
    model = FF.load_model(args.checkpoint, args.device, "teacher")
    log(f"modello {args.checkpoint} (teacher, sha256 {ckpt_sha[:12]}), P2 interna {args.datasets}")
    for ds in args.datasets:
        w, y, s, fs = B.load_epn612(args.epn_root) if ds == "epn612" else B.load_uci_emg(args.uci_root)
        s = np.asarray([str(v) for v in s])
        all_subjects = sorted(set(s.tolist()))
        internal = CP.p2_internal_subjects(all_subjects)
        keep = np.isin(s, internal)
        w, y, s = w[keep], y[keep], s[keep]
        montage = FF.myo8_montage(ds, fs)
        input_sha = fingerprint(w, y, s, fs, montage)
        out = CR.extract_channel_readouts(model, w, fs, montage, s, batch=args.batch, device=args.device)
        npz = args.features / f"{ds}.npz"
        _save(npz, out, y, s, subjects=internal, input_sha256=input_sha, task="p2", channels=out["channels"], model=model, ckpt_sha=ckpt_sha,
              args=args, splits_sha256=CP.P2_SPLITS_SHA256,
              extra={"all_subjects": np.asarray(all_subjects), "montage_signature": np.asarray(json.dumps(montage_signature(montage)))})
        log(f"  {ds}: {len(internal)} soggetti interni su {len(all_subjects)}, {len(y)} finestre, {out['channels']} canali x {model.cfg.dim} -> {npz}")


def feature_diagnostics(x: np.ndarray, seed: int = 0) -> dict:
    """Come `calibration_probe.feature_diagnostics` (rango effettivo delle feature centrate, colonne costanti), su al piu' `DIAG_ROWS` righe
    scelte col seme: descrittivo, nessuna soglia."""
    rows = np.sort(np.random.default_rng(seed).choice(len(x), size=min(DIAG_ROWS, len(x)), replace=False))
    d = CP.feature_diagnostics(x[rows])
    return {**d, "rows": int(len(rows)), "constant_columns": int((x.std(0) <= 1e-6 * max(float(x.std(0).max()), 1e-12)).sum())}


def oof_predictions(x, y, s, roles: list[dict], seed: int, k: int | None, *, max_iter: int = MAX_ITER, audit: list | None = None,
                    folds: list[int] | None = None, context: dict | None = None) -> dict:
    """`calibration_probe.oof_predictions` con la riduzione a larghezza pari dentro ogni fold: con `k` sotto la larghezza, standardizzazione e
    PCA stimate sul solo train del fold e poi `linear_probe(scale=False)` sulle componenti (nessuna nuova scala: la penalita' L2 resta nello
    spazio delle colonne standardizzate); altrimenti le colonne cosi' come sono a `linear_probe`, lo stesso percorso della calibrazione.
    Stesse righe, stesse classi tolte, stesso formato.

    Controllo della convergenza delle sonde (29/10/2026, `docs/sonde_convergenza_20261029.md`; default invariati): `max_iter` per ogni fit;
    `audit` = lista dei record per fit di `harness.probe.linear_probe` (con `context` e il numero del fold in ciascuno) e, nei record per fold,
    i punteggi di validazione di ogni candidato e i conteggi dei fit al limite; `folds` = indici dei fold da calcolare (uno shard), nell'ordine
    dei ruoli: le righe OOF restano quelle dei fold scelti, in quell'ordine."""
    from wearusfm.harness.probe import linear_probe

    ys, ps, ss, rows, per_fold = [], [], [], [], []
    reduce = k is not None and 0 < k < x.shape[1]
    for i, r in enumerate(roles):
        if folds is not None and i not in folds:
            continue
        mt, mv, mh = (np.isin(s, r[kk]) for kk in ("train", "val", "held"))
        keep_v, keep_h = P1.keep_train_classes(y[mt], y[mv], y[mh])
        if reduce:
            scaler, pca = CR.fit_reduction(x[mt], k, seed)  # k effettivo = min(k, righe di train - 1): `channel_readout.effective_k`
            xt, xv, xh = (CR.apply_reduction(scaler, pca, v) for v in (x[mt], x[mv][keep_v], x[mh][keep_h]))
        else:
            pca, (xt, xv, xh) = None, (x[mt], x[mv][keep_v], x[mh][keep_h])
        ctx, n_before = None, 0
        if audit is not None:
            ctx = {**(context or {}), "fold": i, "roles_sha256": fingerprint(list(r["train"]), list(r["val"]), list(r["held"])),
                   "reduced": bool(reduce)}
            n_before = len(audit)
        res = linear_probe(xt, y[mt], xv, y[mv][keep_v], xh, y[mh][keep_h], s[mh][keep_h], seed=seed, scale=not reduce, max_iter=max_iter,
                           audit=audit, context=ctx)
        ys.append(y[mh][keep_h])
        ps.append(res.test_pred)
        ss.append(s[mh][keep_h])
        rows.append(np.flatnonzero(mh)[keep_h].astype(np.int64))
        rec = {**r, "held_bacc": res.test_bacc, "val_bacc": res.val_bacc, "train_bacc": res.train_bacc, "c": res.c,
               "n_train": res.n_train, "n_val": res.n_val, "n_held": res.n_test,
               "dropped_val_windows": int((~keep_v).sum()), "dropped_held_windows": int((~keep_h).sum()),
               "k": int(xt.shape[1]), "k_capped": bool(reduce and xt.shape[1] < k),
               "explained_variance": float(pca.explained_variance_ratio_.sum()) if pca is not None else 1.0}
        if audit is not None:
            new = audit[n_before:]
            rec.update(fold=i, max_iter=int(max_iter), tie_break=TIE_BREAK, val_scores={str(c): v for c, v in res.val_scores.items()},
                       n_fits=len(new), fits_reached_limit=int(sum(bool(a.get("reached_limit")) for a in new)),
                       fits_with_convergence_warning=int(sum(int(a.get("convergence_warnings", 0)) > 0 for a in new)),
                       fits_anomalous_stop=int(sum(bool(a.get("anomalous_stop")) for a in new)), bootstrap_warnings=res.bootstrap_warnings,
                       roles_sha256=ctx["roles_sha256"], scaler_sha256=new[-1].get("scaler_sha256"))
        per_fold.append(rec)
    if not rows:
        raise ValueError("nessun fold selezionato")
    row = np.concatenate(rows)
    if len(np.unique(row)) != len(row):
        raise ValueError("finestre ripetute nei fold di valutazione")
    return {"y": np.concatenate(ys), "pred": np.concatenate(ps), "s": np.concatenate(ss), "row": row, "folds": per_fold}


def load_cache(features: Path, ds: str, task: str, splits: dict | None, splits_sha256: str, layer: str) -> dict:
    """La cache di un dataset del readout per canale, controllata come la sonda l'ha sempre controllata (insieme di feature e protocollo,
    compito e soggetti, SHA-256 degli split, ricetta di estrazione, strato richiesto, feature finite): x (float32), y, s, soggetti, dimensione
    dell'encoder, canali, identita' del checkpoint, impronta degli ingressi e SHA del codice che ha prodotto la cache."""
    p1 = task == "p1"
    with np.load(features / f"{ds}.npz") as z:
        if "feature_set" not in z.files or str(z["feature_set"]) != CR.FEATURE_SET or str(z["protocol"]) != PROTOCOL:
            raise ValueError(f"{ds}: non e' una cache del readout per canale")
        cache_task = str(z["task"])
        if p1:
            pre = CP.pretraining_subjects(splits, ds)
        else:
            pre = CP.p2_internal_subjects(z["all_subjects"].tolist()) if "all_subjects" in z.files else None
        if cache_task != task or pre is None or sorted(z["subjects"].tolist()) != pre or sorted(set(z["s"].tolist())) != pre:
            raise ValueError(f"{ds}: feature di un altro compito o di altri soggetti")
        if str(z["splits_sha256"]) != splits_sha256:
            raise ValueError(f"{ds}: SHA-256 degli split diverso nella cache; ripetere l'estrazione")
        if json.loads(str(z["extraction_recipe"])) != recipe(task):
            raise ValueError(f"{ds}: ricetta di estrazione diversa nella cache; ripetere l'estrazione")
        if f"x_{layer}" not in z.files:
            raise ValueError(f"{ds}: strato {layer!r} assente dalla cache")
        out = {"x": z[f"x_{layer}"].astype(np.float32), "y": z["y"], "s": z["s"], "subjects": pre, "dim": int(z["encoder_dim"]),
               "channels": int(z["channels"]), "checkpoint": str(z["checkpoint"]), "checkpoint_sha256": str(z["checkpoint_sha256"]),
               "input_sha256": str(z["input_sha256"]), "code_sha": str(z["code_sha"]) if "code_sha" in z.files else ""}
    if not np.isfinite(out["x"]).all():
        raise ValueError(f"{ds}: feature non finite")
    return out


def probe(args, splits: dict | None, log) -> dict:
    from sklearn.metrics import balanced_accuracy_score

    p1 = args.task == "p1"
    report = {"protocol": PROTOCOL, "calibration_protocol": CP.PROTOCOL, "task": args.task, "k_folds": CP.K_FOLDS, "fold_seed": CP.FOLD_SEED,
              "val_frac": CP.VAL_FRAC, "probe_seed": args.seed, "feature_set": f"{CR.FEATURE_SET}/{args.layer}",
              "window_s": CP.WINDOW_S if p1 else None,
              "splits": str(args.splits) if p1 else f"P2 7:1:2, test esclusi dei semi {list(CP.P2_SPLIT_SEEDS)}",
              "splits_sha256": _sha256(args.splits) if p1 else CP.P2_SPLITS_SHA256, "extraction_recipe": recipe(args.task),
              "features": str(args.features), "code_sha": os.environ.get("WEARUSFM_CODE_SHA", ""),
              "readout": {"layer": args.layer, "k_requested": args.k,
                          "reduction": "se k < larghezza: standardizzazione e PCA (randomized, seme della sonda) a k componenti stimate sul solo "
                                       "train del fold, poi `linear_probe(scale=False)` sulle componenti (nessuna nuova scala); altrimenti "
                                       "`linear_probe` sulle colonne, come la calibrazione"},
              "datasets": {}}
    parts, oof, idents = [], {}, set()
    for ds in args.datasets:
        cache = load_cache(args.features, ds, args.task, splits, report["splits_sha256"], args.layer)
        x, y, s, pre, dim, channels = cache["x"], cache["y"], cache["s"], cache["subjects"], cache["dim"], cache["channels"]
        idents.add((cache["checkpoint"], cache["checkpoint_sha256"], dim))
        report["datasets"][ds] = {"input_sha256": cache["input_sha256"]}
        k = 2 * dim if args.k is None else args.k  # default: la larghezza del readout medio
        k_eff = k if 0 < k < x.shape[1] else None
        res = oof_predictions(x, y, s, CP.fold_roles(pre), args.seed, k_eff)
        bacc = float(balanced_accuracy_score(res["y"], res["pred"]))
        parts.append((res["y"], res["pred"], res["s"]))
        oof.update({f"{ds}_y": res["y"], f"{ds}_pred": res["pred"], f"{ds}_s": res["s"], f"{ds}_row": res["row"]})
        report["datasets"][ds].update({"subjects": pre, "s": bacc, "folds": res["folds"], "n_classes": int(len(np.unique(y))),
                                       "n_windows": int(len(y)), "n_oof_windows": int(len(res["y"])), "oof_rows_sha256": fingerprint(res["row"]),
                                       "readout": {"layer": args.layer, "width": int(x.shape[1]), "channels": channels, "k": k_eff or int(x.shape[1]),
                                                   "reduced": k_eff is not None, "k_capped_folds": int(sum(f["k_capped"] for f in res["folds"])),
                                                   "k_by_fold": [f["k"] for f in res["folds"]]},
                                       "features": feature_diagnostics(x, args.seed)})
        log(f"{ds}: S {100 * bacc:.2f} ({args.layer}, larghezza {x.shape[1]} -> k {k_eff or x.shape[1]}; {len(pre)} soggetti, {CP.K_FOLDS} fold)")
    if len(idents) != 1:
        raise ValueError(f"feature di checkpoint diversi: {sorted(idents)}")
    (ckpt, ckpt_sha, dim), = idents
    if p1:
        w = P1.dataset_weights([len(splits["datasets"][d]["test"]) for d in args.datasets], "subjects")  # solo i conteggi, per i pesi della P1
    else:
        w = P1.dataset_weights([1] * len(args.datasets), "datasets")
    report.update({"checkpoint": ckpt, "checkpoint_sha256": ckpt_sha, "model_dim": dim, "weights": dict(zip(args.datasets, w.tolist())),
                   "S": float(np.dot(w, [report["datasets"][d]["s"] for d in args.datasets])),
                   "S_se": P1.mean_bacc_bootstrap_se(parts, seed=args.seed, weights=w)})
    oof_path = args.out.with_suffix(".oof.npz")
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(oof_path, datasets=np.asarray(args.datasets), **oof)
    report["oof"], report["oof_sha256"] = str(oof_path), _sha256(oof_path)
    return report


PROBE_KEYS = ("k_folds", "fold_seed", "val_frac", "probe_seed", "window_s")
MODEL_FREE = ("none", "")  # rapporti senza modello (baseline classica): nessuna identita' di checkpoint da accoppiare


def _probe_params(report: dict) -> dict:
    """Parametri della sonda: in cima nei rapporti di `calibration_probe` e di questo script, sotto "probe" in quelli della baseline classica.
    Una chiave assente e' un errore; un valore None (la P2 non ha `window_s`) e' un valore come gli altri, da confrontare."""
    src = report.get("probe", report)
    missing = [k for k in PROBE_KEYS if k not in src]
    if missing:
        raise ValueError(f"parametri della sonda mancanti nel rapporto: {missing}")
    return {k: src[k] for k in PROBE_KEYS}


def check_readout_pairing(a: dict, b: dict) -> None:
    """Stesso compito, stessi split, stessa sonda, stessi pesi e stesse finestre (impronta degli ingressi e righe fuori fold) per dataset."""
    if a.get("task", "p1") != b.get("task", "p1") or a.get("splits_sha256") != b.get("splits_sha256") or not a.get("splits_sha256"):
        raise ValueError("compito o split diversi: nessun confronto")
    if _probe_params(a) != _probe_params(b):
        raise ValueError("sonda non accoppiata (fold, seme, validazione o finestra diversi)")
    if "solver_recipe_sha256" in a or "solver_recipe_sha256" in b:  # rapporti del controllo di convergenza (29/10/2026): stessa ricetta del
        # solver, `max_iter` compreso; un rapporto senza la firma (legacy) non si accoppia con uno che ce l'ha: i suoi parametri non sono verificati
        if not a.get("solver_recipe_sha256") or a.get("solver_recipe_sha256") != b.get("solver_recipe_sha256"):
            raise ValueError("sonda non accoppiata: ricetta del solver diversa, o assente in uno dei due rapporti")
    if sorted(a["datasets"]) != sorted(b["datasets"]) or a.get("weights") != b.get("weights") or not a.get("weights"):
        raise ValueError("dataset o pesi diversi fra i due rapporti")
    for ds in a["datasets"]:
        for key in ("input_sha256", "oof_rows_sha256"):
            if not a["datasets"][ds].get(key) or a["datasets"][ds].get(key) != b["datasets"][ds].get(key):
                raise ValueError(f"{ds}: {key} diverso o mancante: il confronto non e' accoppiato")


def _model_free(report: dict) -> bool:
    return str(report.get("checkpoint", "none")) in MODEL_FREE


def check_models(refs: list[dict], cands: list[dict], single_seed: bool = False) -> None:
    """Identita' dei modelli: i due candidati sono due modelli diversi (i due semi); per ogni seme il riferimento e' lo stesso checkpoint del
    candidato, salvo un riferimento senza modello (la baseline classica), che puo' essere lo stesso rapporto per tutti e due i semi. (Che i due
    riferimenti con modello siano diversi segue: ciascuno ha lo SHA del suo candidato.) `single_seed`: un solo candidato, ripetuto."""
    shas = [str(c.get("checkpoint_sha256", "")) for c in cands]
    if any(not s or _model_free(c) for s, c in zip(shas, cands)):
        raise ValueError("candidato senza checkpoint: il readout per canale si confronta per modello")
    if shas[0] == shas[1] and not single_seed:
        raise ValueError("i due candidati sono lo stesso checkpoint: servono i due semi")
    for ref, cand in zip(refs, cands):
        if not _model_free(ref) and str(ref.get("checkpoint_sha256", "")) != str(cand["checkpoint_sha256"]):
            raise ValueError("riferimento e candidato di uno stesso seme vengono da checkpoint diversi")


def check_two_models_same_readout(refs: list[dict], cands: list[dict], single_seed: bool = False) -> None:
    """`--models` (opzione ancore-only del protocollo: riferimento ancore-only, candidato A, stesso readout): tutti e quattro i rapporti hanno
    un modello, i due riferimenti sono due checkpoint diversi come i due candidati (salvo `single_seed`: un rapporto per lato, ripetuto), per
    ogni seme riferimento e candidato sono checkpoint diversi, e i due lati sono lo stesso tipo di rapporto (protocollo, insieme di feature,
    ricetta, strato, k richiesto, k/riduzione/larghezza per dataset): si confrontano due modelli, non due readout."""
    for r in refs + cands:
        if _model_free(r) or not r.get("checkpoint_sha256"):
            raise ValueError("--models: rapporto senza modello: il confronto fra modelli vuole un checkpoint per ogni rapporto")
    shas_r, shas_c = [str(r["checkpoint_sha256"]) for r in refs], [str(c["checkpoint_sha256"]) for c in cands]
    if shas_r[0] == shas_r[1] and not single_seed:
        raise ValueError("--models: i due riferimenti sono lo stesso checkpoint: servono i due semi")
    if shas_c[0] == shas_c[1] and not single_seed:
        raise ValueError("--models: i due candidati sono lo stesso checkpoint: servono i due semi")
    for ref, cand in zip(refs, cands):
        if str(ref["checkpoint_sha256"]) == str(cand["checkpoint_sha256"]):
            raise ValueError("--models: riferimento e candidato di uno stesso seme sono lo stesso checkpoint: nessun confronto fra modelli")
        if any(ref.get(kk) != cand.get(kk) for kk in SIDE_KEYS):
            raise ValueError("--models: riferimento e candidato di tipo diverso (protocollo, insieme di feature o ricetta): stesso readout richiesto")
        if _readout_identity(ref) != _readout_identity(cand):
            raise ValueError("--models: riferimento e candidato con strato, k o larghezza diversi: stesso readout richiesto")


SIDE_KEYS = ("protocol", "feature_set", "extraction_recipe")


def _readout_identity(r: dict) -> dict:
    ro = r.get("readout") or {}
    return {"layer": ro.get("layer"), "k_requested": ro.get("k_requested"),
            "datasets": {d: {kk: (v.get("readout") or {}).get(kk) for kk in ("k", "reduced", "width")} for d, v in sorted(r["datasets"].items())}}


def check_sides_alike(refs: list[dict], cands: list[dict]) -> None:
    """I due semi di uno stesso lato sono lo stesso tipo di rapporto (la normalizzazione per `s_comparison` cancella queste chiavi, quindi
    vanno confrontate prima): per i candidati protocollo, insieme di feature, ricetta, strato, k richiesto e k/riduzione/larghezza per dataset;
    per i riferimenti protocollo, insieme di feature e ricetta, e tutti e due con modello o tutti e due senza."""
    for side, reps in (("candidati", cands), ("riferimenti", refs)):
        if any(reps[0].get(kk) != reps[1].get(kk) for kk in SIDE_KEYS):
            raise ValueError(f"{side} dei due semi di tipo diverso (protocollo, insieme di feature o ricetta)")
    if _readout_identity(cands[0]) != _readout_identity(cands[1]):
        raise ValueError("candidati dei due semi con strato, k o larghezza diversi")
    if _model_free(refs[0]) != _model_free(refs[1]):
        raise ValueError("riferimenti dei due semi: uno con modello e uno senza")


def remap_oof(report: dict, maps: list[tuple[str, str]]) -> dict:
    """Il rapporto con il percorso delle predizioni fuori fold riscritto da un prefisso del cluster a uno locale (i rapporti copiati sul Mac):
    lo SHA-256 del file resta quello del rapporto e `load_oof` lo verifica."""
    oof = str(report["oof"])
    for old, new in maps:
        if oof.startswith(old):
            oof = new + oof[len(old):]
            break
    return {**report, "oof": oof}


def _normalized(r: dict, like: dict) -> dict:
    """Per `arms_compare.s_comparison` (che accoppia con `calibration_select.check_pairing`: protocollo, insieme di feature e ricetta uguali):
    dopo i controlli di `check_readout_pairing` contano split, fold, sonda e righe, che restano quelli del rapporto; i parametri della sonda
    della baseline classica salgono in cima."""
    return {**r, **_probe_params(r), "protocol": like["protocol"], "feature_set": like.get("feature_set"),
            "extraction_recipe": like.get("extraction_recipe")}


def reading(row: dict, single_seed: bool = False) -> dict:
    """La regola dei bracci su S (`arms_compare.verdict`: delta medio dei due semi >= +0,01, limite inferiore dell'intervallo dei percentili al
    95% > 0, delta > 0 in tutti e due i semi) e la sua speculare; altrimenti «regola non superata» (non esclude una differenza, neanche oltre
    1 punto). `single_seed` (pilot degli obiettivi, un rapporto per lato): stessa stima e stesso intervallo, senza il criterio dell'accordo
    fra i semi, che con un seme non esiste (dichiarato nel rapporto: `seeds` 1)."""
    AC = _script("arms_compare")
    v = AC.verdict(row)
    if single_seed:
        v = {**v, "beats": bool(row["delta"] >= AC.MIN_DELTA and row["ci95"][0] > 0)}
        worse = row["delta"] <= -AC.MIN_DELTA and row["ci95"][1] < 0
    else:
        worse = row["delta"] <= -AC.MIN_DELTA and row["ci95"][1] < 0 and all(d < 0 for d in row["delta_by_model_seed"])
    verdict = "migliora" if v["beats"] else "peggiora" if worse else "regola non superata"
    return {**v, "verdict": verdict, "min_delta": AC.MIN_DELTA, "seeds": 1 if single_seed else 2}


def compare(args) -> dict:
    """Candidato contro riferimento su S coi due semi, come i bracci: soggetti ricampionati insieme per i quattro rapporti
    (`arms_compare.s_comparison`, `downstream_ab.paired_component`, 1.000 ricampionamenti, seme 20261009), delta medio dei due semi e
    intervallo dei percentili. `--reference` e `--candidate` prendono i rapporti del seme 0 e del seme 1 (la baseline classica, senza modello,
    puo' essere lo stesso rapporto due volte)."""
    AC = _script("arms_compare")
    single = bool(getattr(args, "single_seed", False))
    refs = [remap_oof(json.loads(Path(p).read_text()), args.oof_maps) for p in args.reference]
    cands = [remap_oof(json.loads(Path(p).read_text()), args.oof_maps) for p in args.candidate]
    if single:  # un rapporto per lato: lo stesso rapporto due volte per `s_comparison` (soggetti ricampionati insieme, stessa stima)
        if str(args.reference[0]) == str(args.candidate[0]):
            raise ValueError("--single-seed: riferimento e candidato sono lo stesso rapporto")
        refs, cands = refs * 2, cands * 2
    elif len({str(p) for p in args.candidate}) != 2:
        raise ValueError("servono due rapporti candidati diversi (seme 0 e seme 1)")
    for ref, cand in zip(refs, cands):
        check_readout_pairing(ref, cand)
    check_sides_alike(refs, cands)
    if args.models:
        check_two_models_same_readout(refs, cands, single)
    else:
        check_models(refs, cands, single)
    row = AC.s_comparison([_normalized(r, refs[0]) for r in refs], [_normalized(c, refs[0]) for c in cands])

    def describe(paths, reps):
        return [{"path": str(p), "S": r["S"], "S_se": r.get("S_se"), "feature_set": r.get("feature_set"), "checkpoint": r.get("checkpoint"),
                 "checkpoint_sha256": r.get("checkpoint_sha256"), "readout": r.get("readout")} for p, r in zip(paths, reps)]

    mode = "due modelli con lo stesso readout (--models)" if args.models else "due readout dello stesso modello, o readout contro baseline senza modello"
    if single:
        mode += "; un solo seme per lato (--single-seed): nessun accordo fra semi"
    return {"protocol": PROTOCOL, "task": refs[0].get("task", "p1"), "mode": mode, "seeds": 1 if single else 2,
            "reference": describe(args.reference, refs[:len(args.reference)]), "candidate": describe(args.candidate, cands[:len(args.candidate)]),
            "comparison": row, "bootstrap": {"n_boot": AC.N_BOOT, "seed": AC.BOOT_SEED,
            "method": "soggetti ricampionati insieme per i quattro rapporti (downstream_ab.paired_component), intervallo dei percentili"},
            "reading": reading(row, single), "oof_maps": [list(m) for m in args.oof_maps]}


EQUALITY_TOL = 1e-3  # protocollo, punto 4: |S(pool nuovo) - S(storico)| oltre questa soglia = nessuna lettura


def equality(args) -> dict:
    """Controllo d'uguaglianza del protocollo (punto 4): il `pool` riestratto e risondato contro il rapporto storico di S dello stesso checkpoint
    (stesse finestre e righe), |delta S| <= EQUALITY_TOL e conteggio delle predizioni diverse (bf16 su GPU: qualche ulp puo' spostarne alcune)."""
    CS = _script("calibration_select")
    ref, cand = (remap_oof(json.loads(Path(p).read_text()), args.oof_maps) for p in (args.reference[0], args.candidate[0]))
    check_readout_pairing(ref, cand)
    if str(ref.get("checkpoint_sha256", "")) != str(cand.get("checkpoint_sha256", "")) or not ref.get("checkpoint_sha256"):
        raise ValueError("il controllo d'uguaglianza vuole lo stesso checkpoint nei due rapporti")
    oof_ref, oof_cand = CS.load_oof(ref), CS.load_oof(cand)
    per_ds = {}
    for d in sorted(ref["datasets"]):
        (y_r, p_r, s_r), (y_c, p_c, s_c) = oof_ref[d], oof_cand[d]
        if not (np.array_equal(y_r, y_c) and np.array_equal(s_r, s_c)):
            raise ValueError(f"{d}: finestre o soggetti diversi")
        per_ds[d] = {"s_reference": ref["datasets"][d]["s"], "s_candidate": cand["datasets"][d]["s"],
                     "delta_s": cand["datasets"][d]["s"] - ref["datasets"][d]["s"], "n_windows": int(len(y_r)),
                     "predictions_differing": int((p_r != p_c).sum())}
    delta = float(cand["S"] - ref["S"])
    return {"protocol": PROTOCOL, "task": ref.get("task", "p1"), "reference": str(args.reference[0]), "candidate": str(args.candidate[0]),
            "S_reference": ref["S"], "S_candidate": cand["S"], "delta_S": delta, "tolerance": EQUALITY_TOL, "equal": bool(abs(delta) <= EQUALITY_TOL),
            "datasets": per_ds, "oof_maps": [list(m) for m in args.oof_maps]}


def _map(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--oof-map vuole PREFISSO_CLUSTER=PREFISSO_LOCALE: {text!r}")
    old, new = text.split("=", 1)
    return old, new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["extract", "probe", "compare", "equality"], required=True)
    ap.add_argument("--task", choices=["p1", "p2"], default="p1")
    ap.add_argument("--epn-root", type=Path, action="append", default=None, help="P2: trainingJSON e testingJSON di EPN-612")
    ap.add_argument("--uci-root", type=Path, default=None, help="P2: UCI-EMG")
    ap.add_argument("--features", type=Path, help="cartella delle cache per dataset (fuori dal codice)")
    ap.add_argument("--checkpoint", help="checkpoint del training (extract)")
    ap.add_argument("--root", type=Path, action="append", default=[], help="radici dei dati processati (extract, P1)")
    ap.add_argument("--splits", type=Path, default=ROOT / "splits" / "v1" / "splits_v1.json")
    ap.add_argument("--datasets", nargs="+", default=None, help="default: quelli del compito (P1: DB2, DB3, DB6; P2: EPN-612, UCI-EMG)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="seme della sonda, della PCA e del bootstrap (come la calibrazione)")
    ap.add_argument("--layer", choices=CR.LAYERS, help="probe: strato da leggere")
    ap.add_argument("--k", type=int, default=None, help="probe: componenti della PCA sul train del fold; default 2d (larghezza del readout medio), 0 = intera")
    ap.add_argument("--reference", nargs="+", type=Path, metavar="S.json", help="compare: rapporti del riferimento, seme 0 e seme 1; equality: uno")
    ap.add_argument("--candidate", nargs="+", type=Path, metavar="S.json", help="compare: rapporti del candidato, seme 0 e seme 1; equality: uno")
    ap.add_argument("--oof-map", dest="oof_maps", type=_map, action="append", default=[],
                    help="compare: PREFISSO_CLUSTER=PREFISSO_LOCALE per le predizioni fuori fold copiate sul Mac (SHA-256 verificato)")
    ap.add_argument("--out", type=Path, help="rapporto JSON (probe, compare); le predizioni fuori fold accanto, .oof.npz")
    ap.add_argument("--models", action="store_true",
                    help="compare: due modelli con lo stesso readout (opzione ancore-only: riferimento ancore-only s0 s1, candidato A s0 s1), "
                         "invece di due readout dello stesso modello; stessa regola e stesso bootstrap")
    ap.add_argument("--single-seed", action="store_true",
                    help="compare: un rapporto per lato (pilot degli obiettivi, seme 0): stessa stima e stesso bootstrap, lettura senza il "
                         "criterio dell'accordo fra i semi (dichiarato nel rapporto)")
    args = ap.parse_args(argv)
    if (args.models or args.single_seed) and args.stage != "compare":
        ap.error("--models e --single-seed: solo con --stage compare")
    args.datasets = args.datasets or list(P1.P1_DATASETS if args.task == "p1" else CP.P2_DATASETS)
    if args.task == "p2" and set(args.datasets) - set(CP.P2_DATASETS):
        ap.error(f"--task p2: dataset fra {CP.P2_DATASETS}")
    if args.stage in ("extract", "probe") and args.features is None:
        ap.error(f"{args.stage}: serve --features")
    if args.stage == "extract" and args.task == "p1" and (not args.checkpoint or not args.root):
        ap.error("extract: servono --checkpoint e --root")
    if args.stage == "extract" and args.task == "p2" and (not args.checkpoint or ("epn612" in args.datasets and not args.epn_root)
                                                          or ("uci_emg" in args.datasets and not args.uci_root)):
        ap.error("extract della P2: servono --checkpoint, --epn-root e --uci-root")
    if args.stage == "probe" and (args.out is None or args.layer is None or (args.k is not None and args.k < 0)):
        ap.error("probe: servono --layer e --out; --k >= 0")
    per_side = 1 if args.single_seed else 2
    if args.stage == "compare" and (args.out is None or len(args.reference or []) != per_side or len(args.candidate or []) != per_side):
        ap.error(f"compare: servono --reference, --candidate ({'un rapporto' if per_side == 1 else 'due rapporti'} ciascuno) e --out")
    if args.stage == "equality" and (args.out is None or len(args.reference or []) != 1 or len(args.candidate or []) != 1):
        ap.error("equality: servono --reference, --candidate (un rapporto ciascuno) e --out")
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    if args.stage == "equality":
        result = equality(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        log(f"S {100 * result['S_candidate']:.3f} contro {100 * result['S_reference']:.3f} (delta {100 * result['delta_S']:+.3f} punti, soglia "
            f"{100 * EQUALITY_TOL:.1f}): {'uguale' if result['equal'] else 'DIVERSO: nessuna lettura'}")
        return 0 if result["equal"] else 1
    if args.stage == "compare":
        result = compare(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        row, r = result["comparison"], result["reading"]
        log(f"delta medio {100 * row['delta']:+.2f} punti, IC95 [{100 * row['ci95'][0]:+.2f}; {100 * row['ci95'][1]:+.2f}], per seme "
            f"{[round(100 * d, 2) for d in row['delta_by_model_seed']]}: {r['verdict']}")
        return 0
    splits = json.loads(args.splits.read_text()) if args.task == "p1" else None
    if args.stage == "extract":
        (extract_p1(args, splits, log) if args.task == "p1" else extract_p2(args, log))
        log(f"estrazione finita in {time.time() - t0:.0f} s")
        return 0
    report = probe(args, splits, log)
    report["elapsed_s"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    log(f"S = {100 * report['S']:.2f} ± {100 * report['S_se']:.2f} ({args.layer}); {report['elapsed_s']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
