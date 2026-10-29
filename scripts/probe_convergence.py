#!/usr/bin/env python3
"""Controllo della convergenza delle sonde P1/P2 di B19618 contro R3c a 19.618 (29/10/2026; prompt di GPT
`docs/prompt_claude_convergenza_sonde_20261029.md`, protocollo `docs/sonde_convergenza_20261029.md`).

Domanda: la conclusione sul trasferimento (P1 B−R3c = −0,556 punti, P2 = −1,233; 18 ConvergenceWarning di lbfgs nei log, senza C, fold o tipo di
fit) e' sensibile all'arresto del solver? Non si cerca una ricetta che faccia vincere B. Due fasi fissate prima dell'esecuzione:
  A = riproduzione strumentata a 3.000 iterazioni (i parametri di sempre), warning attribuiti ai 280 fit, uguaglianza con gli originali;
  B = la sola variante `max_iter` 12.000 (stessa tolleranza, stessa griglia, refit rifatto da zero) per entrambi gli encoder e tutti i dataset.
Le cache dei teacher (codice 24139ab) sono in sola lettura; ogni uscita e' nuova, assoluta e fuori dal codice; il codice di analisi gira da uno SHA
pubblicato e congelato (`WEARUSFM_CODE_SHA`, `lib_code_dir.sh`) e si registra separatamente da quello che ha prodotto le cache.

  --stage probe           uno shard (fase, encoder, compito, dataset, fold) con audit per fit -> <out>/shard.json, shard.oof.npz, audit.jsonl
  --stage aggregate       gli shard di (fase, encoder, compito) -> rapporto nel formato di `channel_readout.py --stage probe`, OOF e audit uniti;
                          con --original anche la parita' con il rapporto storico (fase A)
  --stage parity          rapporto nuovo contro originale: righe, fold, C, predizioni, S; conteggi esatti, nessuna tolleranza
  --stage gate            prerequisiti della fase B: fase A completa (280 record), parita' riprodotta, stessa ricetta fra encoder
  --stage compare-phases  B−R3c per fase, 12000−3000 per encoder, (B−R3c)_12000 − (B−R3c)_3000 sugli STESSI ricampionamenti dei soggetti
  --stage plan            piano JSON, file dei comandi (submit_job.py) e preventivo per semplicione, dallo SHA del codice pubblicato
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.harness import downstream_ab as DA  # noqa: E402
from wearusfm.harness import p1_cross_subject as P1  # noqa: E402
from wearusfm.harness.probe import C_GRID, MAX_ITER, solver_recipe  # noqa: E402


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_probe_convergence", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SC = _script("channel_readout")
CP = SC.CP
AC = _script("arms_compare")
CS = _script("calibration_select")

PROTOCOL = "sonde_convergenza_v1"
PHASES = {"A": MAX_ITER, "B": 12000}  # fissate prima dell'esecuzione: fra le fasi cambia SOLO max_iter
ENCODERS = ("B", "R3c")
TASKS = {"p1": list(P1.P1_DATASETS), "p2": list(CP.P2_DATASETS)}
K_FOLDS = CP.K_FOLDS
N_FITS_PER_FOLD = len(C_GRID) + 1  # sei candidati + il riaddestramento
LAYER, K = "pool", 0
N_BOOT, BOOT_SEED = AC.N_BOOT, AC.BOOT_SEED
READOUT_NOTE = ("se k < larghezza: standardizzazione e PCA (randomized, seme della sonda) a k componenti stimate sul solo train del fold, poi "
                "`linear_probe(scale=False)` sulle componenti (nessuna nuova scala); altrimenti `linear_probe` sulle colonne, come la calibrazione")
# Shard: nome, dataset, fold. DB2 (24.609 righe di train per fold, 49 classi) e' diviso in due meta' per restare sotto le 4 ore della seriale
# anche a 12.000 iterazioni; P2 (EPN612 + UCI, 20 minuti in tutto) sta in un job solo. Ogni (dataset, fold) compare una volta sola per compito.
SHARDS = {"p1": (("db2a", ("ninapro_db2",), (0, 1)), ("db2b", ("ninapro_db2",), (2, 3)),
                 ("db3", ("ninapro_db3",), (0, 1, 2, 3)), ("db6", ("ninapro_db6",), (0, 1, 2, 3))),
          "p2": (("p2", ("epn612", "uci_emg"), (0, 1, 2, 3)),)}
# Tetti di walltime per shard (hh:mm:ss) dai tempi osservati nelle sonde del 28-29/10 (DB2 ~56 min, DB3 ~7, DB6 ~12, P2 ~20, a 8 core): fase A circa
# 1,6x l'osservato; fase B 4x (un fit che tocca 3.000 iterazioni ne puo' fare fino a 12.000: e' il caso peggiore, non una previsione).
SHARD_TIME = {"A": {"db2a": "00:45:00", "db2b": "00:45:00", "db3": "00:20:00", "db6": "00:30:00", "p2": "00:40:00"},
              "B": {"db2a": "02:00:00", "db2b": "02:00:00", "db3": "00:40:00", "db6": "01:00:00", "p2": "01:30:00"}}
SHARD_EXPECTED_MIN = {"db2a": 28, "db2b": 28, "db3": 7, "db6": 12, "p2": 20}  # fase A; fase B fra 1x e 4x


class Problem(ValueError):
    """Un prerequisito non rispettato: il controllo si ferma senza scrivere risultati parziali spacciati per completi."""


def require(ok, message: str) -> None:
    if not ok:
        raise Problem(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def is_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:  # mai sovrascrivere: ogni uscita e' nuova
        json.dump(value, f, indent=1, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def code_roots() -> list[Path]:
    roots = [ROOT]
    work = os.environ.get("WORK")
    if work:
        roots += [Path(work) / "wearusfm", Path(work) / "wearusfm_campaigns"]
    return roots


def fresh(path: Path) -> None:
    """Uscita nuova, assoluta e fuori dal codice (checkout compreso)."""
    require(path.is_absolute() and not path.exists() and not path.is_symlink(), f"uscita non nuova o non assoluta: {path}")
    resolved = path.resolve()
    for r in code_roots():
        rr = r.resolve()
        require(resolved != rr and rr not in resolved.parents, f"uscita dentro il codice ({r}): {path}")


def analysis_code(allow_unfrozen: bool) -> dict:
    """Lo SHA del codice di analisi (dal checkout congelato: `lib_code_dir.sh` esporta WEARUSFM_CODE_SHA). Senza, solo con --allow-unfrozen."""
    sha = os.environ.get("WEARUSFM_CODE_SHA", "")
    require(bool(sha) or allow_unfrozen, "codice di analisi non congelato (WEARUSFM_CODE_SHA assente): eseguire da un checkout di campagna")
    return {"analysis_code_sha": sha, "analysis_code_dir": os.environ.get("WEARUSFM_CODE_DIR", str(ROOT)), "frozen": bool(sha)}


def environment() -> dict:
    import scipy
    import sklearn
    import threadpoolctl

    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "threadpoolctl": threadpoolctl.__version__,
            "threadpools": [{k: v for k, v in info.items() if isinstance(v, (str, int, float, bool)) or v is None}
                            for info in threadpoolctl.threadpool_info()],
            "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "SLURM_CPUS_PER_TASK",
                                                   "SLURM_JOB_ID", "SLURM_JOB_NAME")}}


def versions_of(env: dict) -> dict:
    return {k: env.get(k) for k in ("python", "numpy", "scipy", "sklearn")}


# ----------------------------------------------------------------------------------------------------------------------------------- ingressi

def load_pins(path: Path) -> dict:
    pins = json.loads(Path(path).read_text())
    require({"checkpoints", "reports", "caches"} <= set(pins), f"pin incompleti: {path}")
    return pins


def verify_cache_inputs(features: Path, encoder: str, task: str, datasets: list[str], pins: dict) -> dict:
    """Le cache del teacher a 19.618 che la fase legge: SHA-256 di ogni file uguale al pin, provenienza e ricevuta dell'estrazione presenti e
    coerenti (completa, checkpoint invariato, encoder, modo, passo), checkpoint del pin. Ritorna i pin verificati per dataset."""
    receipt_path = features / "extraction_receipt.json"
    require(receipt_path.is_file(), f"ricevuta dell'estrazione assente: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("complete") is True and receipt.get("checkpoint_unchanged") is True, "ricevuta dell'estrazione incompleta")
    require(receipt.get("model") == encoder and receipt.get("mode") == "channel" and receipt.get("checkpoint_step") == 19618,
            "ricevuta dell'estrazione di un altro encoder, modo o passo")
    ckpt = pins["checkpoints"][encoder]
    require(receipt.get("checkpoint_sha256") == ckpt["sha256"], "ricevuta: checkpoint diverso dal pin")
    out = {}
    for ds in datasets:
        key = f"{encoder}/{task}/{ds}"
        require(key in pins["caches"], f"pin assente per {key}")
        pin = pins["caches"][key]
        path = features / f"{ds}.npz"
        require(path.is_file() and Path(pin["cache"]).name == path.name, f"cache assente o con un altro nome: {path}")
        digest = sha256(path)
        require(digest == pin["cache_sha256"], f"{key}: SHA-256 della cache diverso dal pin: le cache non sono quelle delle sonde originali")
        require(receipt["outputs"].get(ds, {}).get("sha256") == digest, f"{key}: la ricevuta dell'estrazione non attesta questa cache")
        prov_path = Path(str(path) + ".provenance.json")
        require(prov_path.is_file() and sha256(prov_path) == pin["provenance_sha256"], f"{key}: provenienza della cache assente o diversa")
        prov = json.loads(prov_path.read_text())
        require(prov.get("cache_sha256") == digest and prov.get("checkpoint_sha256") == ckpt["sha256"] and prov.get("code_sha") == pin["code_sha"]
                and prov.get("which") == "teacher", f"{key}: provenienza incoerente")
        out[ds] = {"path": str(path), "sha256": digest, "provenance_sha256": pin["provenance_sha256"], "input_sha256": pin["input_sha256"],
                   "splits_sha256": pin["splits_sha256"], "code_sha": pin["code_sha"], "checkpoint_sha256": ckpt["sha256"]}
    return out


def check_gate(path: Path, code: dict) -> dict:
    gate = json.loads(Path(path).read_text())
    require(gate.get("protocol") == PROTOCOL and gate.get("phase") == "A" and gate.get("passed") is True,
            f"la fase A non ha superato il cancello ({path}): la fase B non parte")
    if code["frozen"] and gate.get("analysis_code_sha"):
        require(gate["analysis_code_sha"] == code["analysis_code_sha"], "il cancello viene da un altro codice di analisi")
    return {"path": str(path), "sha256": sha256(path), "solver_recipe_sha256": gate.get("solver_recipe_sha256")}


# -------------------------------------------------------------------------------------------------------------------------------------- shard

def run_probe(a) -> int:
    phase = a.phase
    require(phase in PHASES, "fase sconosciuta")
    max_iter = PHASES[phase]
    code = analysis_code(a.allow_unfrozen)
    out = a.out
    fresh(out)
    gate = None
    if phase == "B":
        require(a.gate is not None, "la fase B richiede --gate (il cancello della fase A)")
        gate = check_gate(a.gate, code)
    datasets = list(a.datasets or TASKS[a.task])
    require(set(datasets) <= set(TASKS[a.task]) and len(set(datasets)) == len(datasets), f"dataset non del compito {a.task}: {datasets}")
    folds = sorted(set(a.folds if a.folds is not None else range(K_FOLDS)))
    require(all(0 <= f < K_FOLDS for f in folds), f"fold fuori da 0..{K_FOLDS - 1}: {folds}")
    pins = load_pins(a.pins)
    p1 = a.task == "p1"
    splits = json.loads(a.splits.read_text()) if p1 else None
    splits_sha = SC._sha256(a.splits) if p1 else CP.P2_SPLITS_SHA256
    inputs = verify_cache_inputs(a.features, a.encoder, a.task, datasets, pins)
    recipe = solver_recipe(max_iter, seed=a.seed)
    recipe_sha = DA.fingerprint(recipe)
    env = environment()
    t0 = time.time()
    audit: list = []
    oof: dict = {}
    entries: dict = {}
    idents = set()
    for ds in datasets:
        cache = SC.load_cache(a.features, ds, a.task, splits, splits_sha, LAYER)
        require(cache["code_sha"] == inputs[ds]["code_sha"], f"{ds}: la cache dichiara un codice diverso dal pin")
        require(cache["checkpoint_sha256"] == inputs[ds]["checkpoint_sha256"], f"{ds}: la cache dichiara un checkpoint diverso dal pin")
        require(cache["input_sha256"] == inputs[ds]["input_sha256"] and splits_sha == inputs[ds]["splits_sha256"],
                f"{ds}: impronta degli ingressi o SHA degli split diversi dal pin")
        x, y, s, pre = cache["x"], cache["y"], cache["s"], cache["subjects"]
        roles = CP.fold_roles(pre)
        context = {"protocol": PROTOCOL, "phase": phase, "encoder": a.encoder, "task": a.task, "dataset": ds, "seed": a.seed,
                   "checkpoint_sha256": cache["checkpoint_sha256"], "cache_sha256": inputs[ds]["sha256"], "cache_code_sha": cache["code_sha"],
                   "analysis_code_sha": code["analysis_code_sha"], "solver_recipe_sha256": recipe_sha}
        before = len(audit)
        res = SC.oof_predictions(x, y, s, roles, a.seed, None, max_iter=max_iter, audit=audit, folds=folds, context=context)
        n_new = len(audit) - before
        require(n_new == len(folds) * N_FITS_PER_FOLD, f"{ds}: {n_new} record di audit invece di {len(folds) * N_FITS_PER_FOLD}")
        fold_of = np.concatenate([np.full(f["n_held"] , f["fold"], dtype=np.int64) for f in res["folds"]])
        require(len(fold_of) == len(res["row"]), f"{ds}: righe OOF e fold non allineati")
        oof[ds] = {"y": res["y"], "pred": res["pred"], "s": res["s"], "row": res["row"], "fold": fold_of}
        idents.add((cache["checkpoint"], cache["checkpoint_sha256"], cache["dim"]))
        entries[ds] = {"input_sha256": cache["input_sha256"], "subjects": pre, "folds": res["folds"], "n_classes": int(len(np.unique(y))),
                       "n_windows": int(len(y)), "n_oof_windows_shard": int(len(res["y"])), "oof_rows_sha256_shard": DA.fingerprint(res["row"]),
                       "readout": {"layer": LAYER, "width": int(x.shape[1]), "channels": cache["channels"], "k": int(x.shape[1]), "reduced": False,
                                   "k_capped_folds": 0, "k_by_fold": [f["k"] for f in res["folds"]]},
                       "features": SC.feature_diagnostics(x, a.seed), "cache": inputs[ds]}
        print(f"[{time.strftime('%H:%M:%S')}] {phase} {a.encoder} {a.task} {ds} fold {folds}: {n_new} fit, "
              f"{sum(bool(r.get('reached_limit')) for r in audit[before:])} al limite di {max_iter}, "
              f"{sum(int(r.get('convergence_warnings', 0)) > 0 for r in audit[before:])} con ConvergenceWarning", flush=True)
    after = verify_cache_inputs(a.features, a.encoder, a.task, datasets, pins)
    require(after == inputs, "le cache sono cambiate durante i fit")
    require(len(idents) == 1, f"cache di checkpoint diversi: {sorted(idents)}")
    (ckpt, ckpt_sha, dim), = idents
    out.mkdir(parents=True, exist_ok=False)
    audit_path = out / "audit.jsonl"
    with audit_path.open("x") as f:
        for record in audit:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    oof_path = out / "shard.oof.npz"
    arrays = {"datasets": np.asarray(datasets)}
    for ds in datasets:
        arrays.update({f"{ds}_{k}": v for k, v in oof[ds].items()})
    np.savez(oof_path, **arrays)
    shard = {"kind": "shard", "protocol": PROTOCOL, "phase": phase, "max_iter": max_iter, "encoder": a.encoder, "task": a.task,
             "datasets": datasets, "folds": folds, "seed": a.seed, "layer": LAYER, "k_requested": K, "checkpoint": ckpt, "checkpoint_sha256": ckpt_sha,
             "model_dim": dim, "features": str(a.features), "splits": str(a.splits) if p1 else None, "splits_sha256": splits_sha,
             "solver_recipe": recipe, "solver_recipe_sha256": recipe_sha, "environment": env, **code,
             "cache_code_sha": inputs[datasets[0]]["code_sha"], "cache_pins": inputs, "pins_file": str(a.pins), "pins_sha256": sha256(a.pins),
             "gate": gate, "entries": entries,
             "audit": {"path": str(audit_path), "sha256": sha256(audit_path), "n_records": len(audit),
                       "expected": len(datasets) * len(folds) * N_FITS_PER_FOLD},
             "oof": {"path": str(oof_path), "sha256": sha256(oof_path)}, "elapsed_s": time.time() - t0}
    write_json(out / "shard.json", shard)
    print(json.dumps({"complete": True, "stage": "probe", "phase": phase, "encoder": a.encoder, "task": a.task, "datasets": datasets,
                      "folds": folds, "n_fits": len(audit), "out": str(out)}), flush=True)
    return 0


# --------------------------------------------------------------------------------------------------------------------------------- aggregato

def _load_shard(path: Path) -> tuple[dict, dict, list]:
    shard = json.loads(path.read_text())
    require(shard.get("kind") == "shard" and shard.get("protocol") == PROTOCOL, f"non e' uno shard del controllo: {path}")
    oof_path, audit_path = Path(shard["oof"]["path"]), Path(shard["audit"]["path"])
    require(sha256(oof_path) == shard["oof"]["sha256"] and sha256(audit_path) == shard["audit"]["sha256"], f"{path}: OOF o audit cambiati")
    with np.load(oof_path, allow_pickle=False) as z:
        require(z["datasets"].tolist() == shard["datasets"], f"{path}: dataset dell'OOF diversi dallo shard")
        arrays = {ds: {k: z[f"{ds}_{k}"] for k in ("y", "pred", "s", "row", "fold")} for ds in shard["datasets"]}
    with audit_path.open() as f:
        audit = [json.loads(line) for line in f if line.strip()]
    require(len(audit) == shard["audit"]["n_records"] == shard["audit"]["expected"], f"{path}: audit incompleto")
    return shard, arrays, audit


def aggregate(shard_paths: list[Path], phase: str, encoder: str, task: str, splits_path: Path | None, pins: dict) -> tuple[dict, dict, list]:
    """Gli shard di (fase, encoder, compito) in un rapporto completo: ogni (dataset, fold) esattamente una volta, stessa ricetta, stesse cache."""
    datasets = TASKS[task]
    loaded = [_load_shard(p) for p in shard_paths]
    first = loaded[0][0]
    for shard, _, _ in loaded:
        require(shard["phase"] == phase and shard["encoder"] == encoder and shard["task"] == task, "shard di un'altra fase, encoder o compito")
        for key in ("max_iter", "seed", "solver_recipe_sha256", "splits_sha256", "checkpoint_sha256", "cache_code_sha", "pins_sha256",
                    "analysis_code_sha"):
            require(shard[key] == first[key], f"shard con {key} diverso: {shard[key]!r} contro {first[key]!r}")
        require(versions_of(shard["environment"]) == versions_of(first["environment"]), "shard eseguiti con versioni diverse delle librerie")
    covered: dict = {ds: {} for ds in datasets}
    for shard, arrays, audit in loaded:
        for ds in shard["datasets"]:
            require(ds in covered, f"dataset {ds} non del compito")
            for f in shard["entries"][ds]["folds"]:
                require(f["fold"] not in covered[ds], f"{ds}: fold {f['fold']} presente in due shard")
                covered[ds][f["fold"]] = (shard, arrays[ds], f)
    for ds in datasets:
        require(sorted(covered[ds]) == list(range(K_FOLDS)), f"{ds}: fold coperti {sorted(covered[ds])}, attesi 0..{K_FOLDS - 1}: output parziale")
    for ds in datasets:
        pin = pins["caches"][f"{encoder}/{task}/{ds}"]
        for shard, _, _ in loaded:
            if ds in shard["datasets"]:
                require(shard["cache_pins"][ds]["sha256"] == pin["cache_sha256"], f"{ds}: lo shard ha letto una cache diversa dal pin")
    p1 = task == "p1"
    splits = json.loads(splits_path.read_text()) if p1 else None
    if p1:
        require(SC._sha256(splits_path) == first["splits_sha256"], "SHA-256 degli split diverso da quello degli shard")
    from sklearn.metrics import balanced_accuracy_score

    report = {"protocol": SC.PROTOCOL, "calibration_protocol": CP.PROTOCOL, "task": task, "k_folds": CP.K_FOLDS, "fold_seed": CP.FOLD_SEED,
              "val_frac": CP.VAL_FRAC, "probe_seed": first["seed"], "feature_set": f"{SC.CR.FEATURE_SET}/{LAYER}",
              "window_s": CP.WINDOW_S if p1 else None,
              "splits": first["splits"] if p1 else f"P2 7:1:2, test esclusi dei semi {list(CP.P2_SPLIT_SEEDS)}",
              "splits_sha256": first["splits_sha256"], "extraction_recipe": SC.recipe(task), "features": first["features"],
              "code_sha": first["analysis_code_sha"], "readout": {"layer": LAYER, "k_requested": K, "reduction": READOUT_NOTE}, "datasets": {}}
    parts, oof, merged_audit, provenance = [], {}, [], {}

    def static(readout: dict) -> dict:  # la parte del readout che non dipende dai fold dello shard
        return {k: v for k, v in readout.items() if k not in ("k_by_fold", "k_capped_folds")}

    for ds in datasets:
        pieces = [covered[ds][i] for i in range(K_FOLDS)]
        entry = pieces[0][0]["entries"][ds]
        for shard, _, _ in pieces:
            e = shard["entries"][ds]
            require(e["input_sha256"] == entry["input_sha256"] and e["subjects"] == entry["subjects"] and static(e["readout"]) == static(entry["readout"])
                    and e["features"] == entry["features"] and e["n_windows"] == entry["n_windows"], f"{ds}: shard incoerenti fra loro")
        ys, ps, ss, rows, folds = [], [], [], [], []
        for i in range(K_FOLDS):
            shard, arrays, f = covered[ds][i]
            sel = arrays["fold"] == i
            require(int(sel.sum()) == f["n_held"], f"{ds}: fold {i}: {int(sel.sum())} righe OOF contro n_held {f['n_held']}")
            ys.append(arrays["y"][sel]), ps.append(arrays["pred"][sel]), ss.append(arrays["s"][sel]), rows.append(arrays["row"][sel])
            folds.append(f)
        y, pred, s, row = (np.concatenate(v) for v in (ys, ps, ss, rows))
        require(len(np.unique(row)) == len(row), f"{ds}: finestre ripetute fra gli shard")
        bacc = float(balanced_accuracy_score(y, pred))
        parts.append((y, pred, s))
        oof.update({f"{ds}_y": y, f"{ds}_pred": pred, f"{ds}_s": s, f"{ds}_row": row,
                    f"{ds}_fold": np.concatenate([np.full(f["n_held"], f["fold"], dtype=np.int64) for f in folds])})
        readout = {**static(entry["readout"]), "k_capped_folds": int(sum(f["k_capped"] for f in folds)), "k_by_fold": [f["k"] for f in folds]}
        report["datasets"][ds] = {"input_sha256": entry["input_sha256"], "subjects": entry["subjects"], "s": bacc, "folds": folds,
                                  "n_classes": entry["n_classes"], "n_windows": entry["n_windows"], "n_oof_windows": int(len(y)),
                                  "oof_rows_sha256": DA.fingerprint(row), "readout": readout, "features": entry["features"]}
        provenance[ds] = entry["cache"]
        for i in range(K_FOLDS):
            shard, _, _ = covered[ds][i]
            audit = next(au for sh, _, au in loaded if sh is shard)
            merged_audit += [r for r in audit if r["dataset"] == ds and r["fold"] == i]
    if p1:
        w = P1.dataset_weights([len(splits["datasets"][d]["test"]) for d in datasets], "subjects")
    else:
        w = P1.dataset_weights([1] * len(datasets), "datasets")
    report.update({"checkpoint": first["checkpoint"], "checkpoint_sha256": first["checkpoint_sha256"], "model_dim": first["model_dim"],
                   "weights": dict(zip(datasets, w.tolist())), "S": float(np.dot(w, [report["datasets"][d]["s"] for d in datasets])),
                   "S_se": P1.mean_bacc_bootstrap_se(parts, seed=first["seed"], weights=w)})
    expected = len(datasets) * K_FOLDS * N_FITS_PER_FOLD
    require(len(merged_audit) == expected, f"audit unito: {len(merged_audit)} record invece di {expected}")
    report.update({"kind": "aggregate", "control_protocol": PROTOCOL, "phase": phase, "max_iter": first["max_iter"], "encoder": encoder,
                   "solver_recipe": first["solver_recipe"], "solver_recipe_sha256": first["solver_recipe_sha256"],
                   "environment": {"common": versions_of(first["environment"]), "shards": [sh["environment"] for sh, _, _ in loaded]},
                   "analysis_code_sha": first["analysis_code_sha"], "cache_code_sha": first["cache_code_sha"], "cache_pins": provenance,
                   "pins_sha256": first["pins_sha256"], "gate": first.get("gate"),
                   "shards": [{"path": str(p), "sha256": sha256(p), "datasets": sh["datasets"], "folds": sh["folds"], "elapsed_s": sh["elapsed_s"],
                               "oof_sha256": sh["oof"]["sha256"], "audit_sha256": sh["audit"]["sha256"]} for p, (sh, _, _) in zip(shard_paths, loaded)],
                   "elapsed_s": float(sum(sh["elapsed_s"] for sh, _, _ in loaded)), "audit_summary": audit_summary(merged_audit)})
    return report, oof, merged_audit


def audit_summary(records: list[dict]) -> dict:
    """Conteggi per il rapporto: fit al limite, con ConvergenceWarning, arresti anomali, per dataset/fold/fase (candidato o riaddestramento), e la
    posizione di ogni fit con warning."""
    out = {"n_records": len(records), "reached_limit": 0, "convergence_warnings": 0, "anomalous_stop": 0, "exceptions": 0,
           "by_dataset": {}, "positions": []}
    for r in records:
        d = out["by_dataset"].setdefault(r["dataset"], {"n_fits": 0, "reached_limit": 0, "convergence_warnings": 0, "selected_reached_limit": 0,
                                                       "refit_reached_limit": 0})
        d["n_fits"] += 1
        lim, cw = bool(r.get("reached_limit")), int(r.get("convergence_warnings", 0)) > 0
        out["reached_limit"] += lim
        out["convergence_warnings"] += cw
        out["anomalous_stop"] += bool(r.get("anomalous_stop"))
        out["exceptions"] += r.get("exception") is not None
        d["reached_limit"] += lim
        d["convergence_warnings"] += cw
        d["selected_reached_limit"] += bool(lim and r.get("phase") == "candidate" and r.get("selected"))
        d["refit_reached_limit"] += bool(lim and r.get("phase") == "refit")
        if cw or lim:
            out["positions"].append({"dataset": r["dataset"], "fold": r["fold"], "phase": r["phase"], "C": r["C"], "n_iter": r.get("n_iter"),
                                     "reached_limit": lim, "convergence_warnings": int(r.get("convergence_warnings", 0)),
                                     "selected": r.get("selected")})
    return out


def write_report(out: Path, report: dict, oof: dict, audit: list, datasets: list[str]) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    oof_path = out.with_suffix(".oof.npz")
    audit_path = Path(str(out) + ".audit.jsonl")
    for p in (oof_path, audit_path):
        require(not p.exists(), f"uscita gia' presente: {p}")
    np.savez(oof_path, datasets=np.asarray(datasets), **oof)
    with audit_path.open("x") as f:
        for record in audit:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    report = {**report, "oof": str(oof_path), "oof_sha256": sha256(oof_path),
              "audit": {"path": str(audit_path), "sha256": sha256(audit_path), "n_records": len(audit)}}
    write_json(out, report)
    return report


def run_aggregate(a) -> int:
    code = analysis_code(a.allow_unfrozen)
    fresh(a.out)
    pins = load_pins(a.pins)
    report, oof, audit = aggregate([Path(p) for p in a.shards], a.phase, a.encoder, a.task, a.splits, pins)
    if code["frozen"]:
        require(report["analysis_code_sha"] == code["analysis_code_sha"], "gli shard vengono da un altro codice di analisi")
    report = write_report(a.out, report, oof, audit, TASKS[a.task])
    summary = report["audit_summary"]
    print(f"[{time.strftime('%H:%M:%S')}] {a.phase} {a.encoder} {a.task}: S = {100 * report['S']:.3f} ± {100 * report['S_se']:.3f}; "
          f"{summary['n_records']} fit, {summary['reached_limit']} al limite, {summary['convergence_warnings']} con ConvergenceWarning", flush=True)
    rc = 0
    if a.original is not None:
        parity = parity_check(Path(a.original), a.original_sha256, a.original_oof_sha256, a.out)
        write_json(a.parity_out, parity)
        rc = 0 if parity["identical"] else 1
        print(f"[{time.strftime('%H:%M:%S')}] parita' con l'originale: {'IDENTICA' if parity['identical'] else 'DIVERSA'} "
              f"({parity['predictions_differing_total']} predizioni diverse, C diversi in {parity['folds_with_different_c']} fold)", flush=True)
    print(json.dumps({"complete": True, "stage": "aggregate", "phase": a.phase, "encoder": a.encoder, "task": a.task, "S": report["S"],
                      "out": str(a.out), "parity_identical": None if a.original is None else rc == 0}), flush=True)
    return rc


# ----------------------------------------------------------------------------------------------------------------------------------- parita'

def parity_check(original: Path, original_sha: str | None, original_oof_sha: str | None, new: Path) -> dict:
    """Rapporto nuovo (fase A) contro quello storico: stesse righe, soggetti, ruoli dei fold, C scelti, predizioni e S. Conteggi esatti delle
    differenze; `identical` solo con zero differenze: una discrepanza e' un problema di riproducibilita' da capire, non da tollerare."""
    orig = json.loads(original.read_text())
    if original_sha:
        require(sha256(original) == original_sha, "rapporto originale: SHA-256 diverso dal pin")
    if original_oof_sha:
        require(orig.get("oof_sha256") == original_oof_sha, "rapporto originale: SHA-256 dell'OOF diverso dal pin")
    rep = json.loads(new.read_text())
    require(rep.get("kind") == "aggregate" and rep.get("phase") == "A", "la parita' si misura sul rapporto aggregato della fase A")
    out = {"protocol": PROTOCOL, "original": str(original), "original_sha256": sha256(original), "new": str(new), "new_sha256": sha256(new),
           "original_legacy_solver_recipe": "solver_recipe_sha256" not in orig, "task": orig.get("task"), "datasets": {}, "problems": []}
    same_params = all(orig.get(k) == rep.get(k) for k in SC.PROBE_KEYS) and orig.get("splits_sha256") == rep.get("splits_sha256")
    same_weights = orig.get("weights") == rep.get("weights")
    same_recipe = orig.get("extraction_recipe") == rep.get("extraction_recipe") and orig.get("feature_set") == rep.get("feature_set")
    same_ckpt = orig.get("checkpoint_sha256") == rep.get("checkpoint_sha256")
    for name, ok in (("probe_params", same_params), ("weights", same_weights), ("extraction_recipe", same_recipe), ("checkpoint", same_ckpt)):
        if not ok:
            out["problems"].append(name)
    oof_o, oof_n = CS.load_oof(orig), CS.load_oof(rep)
    total_diff, c_diff = 0, 0
    for ds in sorted(orig["datasets"]):
        do, dn = orig["datasets"][ds], rep["datasets"].get(ds)
        if dn is None:
            out["problems"].append(f"{ds}: assente nel rapporto nuovo")
            continue
        (y_o, p_o, s_o), (y_n, p_n, s_n) = oof_o[ds], oof_n[ds]
        rows_equal = do.get("oof_rows_sha256") == dn.get("oof_rows_sha256") and do.get("input_sha256") == dn.get("input_sha256")
        arrays_equal = y_o.shape == y_n.shape and np.array_equal(y_o, y_n) and np.array_equal(s_o, s_n)
        roles_equal = [(f["train"], f["val"], f["held"]) for f in do["folds"]] == [(f["train"], f["val"], f["held"]) for f in dn["folds"]]
        c_o, c_n = [f["c"] for f in do["folds"]], [f["c"] for f in dn["folds"]]
        diff = int((p_o != p_n).sum()) if arrays_equal else int(len(y_o))
        n_c = int(sum(a != b for a, b in zip(c_o, c_n))) + abs(len(c_o) - len(c_n))
        total_diff += diff
        c_diff += n_c
        out["datasets"][ds] = {"rows_identical": bool(rows_equal and arrays_equal), "subjects_identical": do.get("subjects") == dn.get("subjects"),
                               "fold_roles_identical": roles_equal, "c_original": c_o, "c_new": c_n, "folds_with_different_c": n_c,
                               "predictions_differing": diff, "n_oof_windows": int(len(y_o)),
                               "s_original": do["s"], "s_new": dn["s"], "s_abs_diff": abs(do["s"] - dn["s"]),
                               "held_bacc_abs_diff": [abs(f["held_bacc"] - g["held_bacc"]) for f, g in zip(do["folds"], dn["folds"])],
                               "val_bacc_abs_diff": [abs(f["val_bacc"] - g["val_bacc"]) for f, g in zip(do["folds"], dn["folds"])]}
        if not (rows_equal and arrays_equal and roles_equal):
            out["problems"].append(f"{ds}: righe, soggetti o ruoli dei fold diversi")
    out.update({"S_original": orig["S"], "S_new": rep["S"], "S_abs_diff": abs(orig["S"] - rep["S"]), "S_se_original": orig.get("S_se"),
                "S_se_new": rep.get("S_se"), "predictions_differing_total": total_diff, "folds_with_different_c": c_diff})
    out["identical"] = bool(not out["problems"] and total_diff == 0 and c_diff == 0 and orig["S"] == rep["S"]
                            and set(orig["datasets"]) == set(rep["datasets"]))
    return out


def run_parity(a) -> int:
    fresh(a.out)
    result = parity_check(Path(a.original), a.original_sha256, a.original_oof_sha256, Path(a.new))
    write_json(a.out, result)
    print(json.dumps({"complete": True, "stage": "parity", "identical": result["identical"], "predictions_differing": result["predictions_differing_total"],
                      "folds_with_different_c": result["folds_with_different_c"], "S_abs_diff": result["S_abs_diff"], "out": str(a.out)}), flush=True)
    return 0 if result["identical"] else 1


# ------------------------------------------------------------------------------------------------------------------------------------ cancello

def load_aggregate(path: Path) -> dict:
    rep = json.loads(Path(path).read_text())
    require(rep.get("kind") == "aggregate" and rep.get("control_protocol") == PROTOCOL, f"non e' un rapporto aggregato del controllo: {path}")
    require(sha256(Path(rep["oof"])) == rep["oof_sha256"] and sha256(Path(rep["audit"]["path"])) == rep["audit"]["sha256"],
            f"{path}: OOF o audit cambiati dopo il rapporto")
    expected = len(TASKS[rep["task"]]) * K_FOLDS * N_FITS_PER_FOLD
    require(rep["audit"]["n_records"] == rep["audit_summary"]["n_records"] == expected, f"{path}: {rep['audit']['n_records']} record invece di {expected}")
    return rep


def run_gate(a) -> int:
    code = analysis_code(a.allow_unfrozen)
    fresh(a.out)
    reports = {}
    for p in a.reports:
        rep = load_aggregate(Path(p))
        require(rep["phase"] == "A", "il cancello legge solo rapporti della fase A")
        reports[(rep["encoder"], rep["task"])] = (Path(p), rep)
    require(set(reports) == {(e, t) for e in ENCODERS for t in TASKS}, f"rapporti della fase A incompleti: {sorted(reports)}")
    recipes = {rep["solver_recipe_sha256"] for _, rep in reports.values()}
    require(len(recipes) == 1, "ricette del solver diverse fra i rapporti della fase A")
    require(all(rep["max_iter"] == PHASES["A"] for _, rep in reports.values()), "fase A con max_iter diverso da quello fissato")
    total = sum(rep["audit"]["n_records"] for _, rep in reports.values())
    parities = {}
    for p in a.parities:
        par = json.loads(Path(p).read_text())
        require(par.get("protocol") == PROTOCOL, f"non e' un file di parita': {p}")
        parities[Path(par["new"]).resolve()] = (Path(p), par)
    missing = [str(p) for p, _ in reports.values() if p.resolve() not in parities]
    require(not missing, f"parita' assente per: {missing}")
    problems = []
    for key, (p, rep) in sorted(reports.items()):
        par_path, par = parities[p.resolve()]
        require(sha256(p) == par["new_sha256"], f"{p}: rapporto cambiato dopo la parita'")
        if not par["identical"]:
            problems.append({"report": str(p), "parity": str(par_path), "predictions_differing": par["predictions_differing_total"],
                             "folds_with_different_c": par["folds_with_different_c"], "problems": par["problems"]})
    if code["frozen"]:
        for p, rep in reports.values():
            require(rep["analysis_code_sha"] == code["analysis_code_sha"], f"{p}: rapporto di un altro codice di analisi")
    expected = len(ENCODERS) * (len(TASKS["p1"]) + len(TASKS["p2"])) * K_FOLDS * N_FITS_PER_FOLD  # 2 x 5 x 4 x 7 = 280
    passed = bool(total == expected and not problems)
    gate = {"protocol": PROTOCOL, "phase": "A", "passed": passed, "n_records": total, "expected_records": expected,
            "solver_recipe_sha256": next(iter(recipes)), **code, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reports": {f"{e}_{t}": {"path": str(p), "sha256": sha256(p), "oof_sha256": rep["oof_sha256"], "S": rep["S"],
                                     "audit_summary": {k: rep["audit_summary"][k] for k in ("n_records", "reached_limit", "convergence_warnings",
                                                                                              "anomalous_stop", "exceptions")}}
                        for (e, t), (p, rep) in sorted(reports.items())},
            "parities": {str(p): {"path": str(pp), "identical": par["identical"]} for p, (pp, par) in parities.items()},
            "problems": problems}
    write_json(a.out, gate)
    print(json.dumps({"complete": True, "stage": "gate", "passed": passed, "n_records": total, "problems": len(problems), "out": str(a.out)}), flush=True)
    return 0 if passed else 1


# ---------------------------------------------------------------------------------------------------------------------------- confronto fasi

def joint_bootstrap(oofs: list[dict], weights: dict, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """S puntuale di ogni rapporto e matrice delle repliche (n_boot x rapporti) con GLI STESSI ricampionamenti dei soggetti di
    `arms_compare.s_comparison` (stesso seme, dataset in ordine alfabetico, una estrazione per replica condivisa da tutti i rapporti), piu' il
    dettaglio per dataset: cosi' ogni contrasto fra rapporti (B−R3c in una fase, 12000−3000 di un encoder, differenza dei due B−R3c) si legge
    sulle stesse repliche, senza sottrarre gli estremi di intervalli separati."""
    datasets = sorted(weights)
    w = np.array([weights[d] for d in datasets], dtype=np.float64)
    rng = np.random.default_rng(seed)
    points = np.zeros((len(datasets), len(oofs)))
    draws = np.zeros((len(datasets), n_boot, len(oofs)))
    for i, d in enumerate(datasets):
        y0, _, s0 = oofs[0][d]
        for o in oofs:
            require(np.array_equal(o[d][0], y0) and np.array_equal(o[d][2], s0), f"{d}: finestre o soggetti diversi fra i rapporti")
        stats = [DA.subject_counts(y0, o[d][1], s0) for o in oofs]
        arrays = [DA._arrays(st) for st in stats]
        subs, n, _ = arrays[0]
        for st, (names, den, _) in zip(stats, arrays):
            require(names == subs and st["classes"] == stats[0]["classes"] and np.array_equal(den, n), f"{d}: soggetti, classi o denominatori diversi")
        points[i] = [DA.bacc(arr[1], arr[2]) for arr in arrays]
        for b in range(n_boot):
            pick = rng.integers(len(subs), size=len(subs))
            draws[i, b] = [DA.bacc(n[pick], arr[2][pick]) for arr in arrays]
    return {"datasets": datasets, "weights": w, "point": points, "draws": draws, "S_point": w @ points,
            "S_draws": np.tensordot(w, draws, axes=(0, 0))}


def contrast(point: np.ndarray, draws: np.ndarray, coef: list[float]) -> dict:
    """Combinazione lineare dei rapporti (es. [-1, 1, 0, 0] = B−R3c nella fase A): stima puntuale, percentili 2,5-97,5 delle repliche, SE."""
    c = np.asarray(coef, dtype=np.float64)
    sample = draws @ c
    return {"delta": float(point @ c), "ci95": np.quantile(sample, [.025, .975]).tolist(), "paired_subject_se": float(np.std(sample, ddof=1))}


CONTRASTS = {"B_minus_R3c_A": [-1, 1, 0, 0], "B_minus_R3c_B": [0, 0, -1, 1], "R3c_B_minus_A": [-1, 0, 1, 0], "B_B_minus_A": [0, -1, 0, 1],
             "delta_of_deltas": [1, -1, -1, 1]}  # ordine dei rapporti: R3c fase A, B fase A, R3c fase B, B fase B
ORDER = (("A", "R3c"), ("A", "B"), ("B", "R3c"), ("B", "B"))


def _recipe_without_max_iter(recipe: dict) -> dict:
    r = json.loads(json.dumps(recipe))
    r.pop("max_iter", None)
    r.get("params", {}).pop("max_iter", None)
    return r


def phase_guard(reps: dict) -> None:
    """Identita' e ricetta dei quattro rapporti di un compito: dentro una fase B e R3c hanno la stessa ricetta del solver (`check_readout_pairing`,
    che ora la esige); fra le fasi e' ammessa la sola differenza di max_iter; righe, soggetti, split, fold, pesi uguali per tutti; mai report
    legacy senza la firma."""
    for key, rep in reps.items():
        require(rep.get("solver_recipe_sha256") and rep.get("solver_recipe"), f"{key}: rapporto senza la ricetta del solver (legacy): escluso")
        require(rep["max_iter"] == PHASES[key[0]] and rep["solver_recipe"]["max_iter"] == PHASES[key[0]], f"{key}: max_iter diverso dalla fase")
    for phase in PHASES:
        SC.check_readout_pairing(reps[(phase, "R3c")], reps[(phase, "B")])
        require(reps[(phase, "R3c")]["checkpoint_sha256"] != reps[(phase, "B")]["checkpoint_sha256"], f"fase {phase}: stesso checkpoint nei due encoder")
    for enc in ENCODERS:
        a, b = reps[("A", enc)], reps[("B", enc)]
        require(a["checkpoint_sha256"] == b["checkpoint_sha256"], f"{enc}: checkpoint diversi fra le fasi")
        require(_recipe_without_max_iter(a["solver_recipe"]) == _recipe_without_max_iter(b["solver_recipe"]),
                f"{enc}: fra le fasi cambia qualcosa oltre a max_iter")
        require(all(a.get(k) == b.get(k) for k in SC.PROBE_KEYS) and a["weights"] == b["weights"] and a["splits_sha256"] == b["splits_sha256"],
                f"{enc}: sonda non accoppiata fra le fasi")
        for ds in a["datasets"]:
            require(all(a["datasets"][ds][k] == b["datasets"][ds][k] for k in ("input_sha256", "oof_rows_sha256")), f"{enc} {ds}: righe diverse fra le fasi")


def _load_audit(rep: dict) -> list[dict]:
    with Path(rep["audit"]["path"]).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def compare_task(task: str, paths: dict, out_dir: Path, run_official: bool) -> dict:
    reps = {key: load_aggregate(paths[key]) for key in ORDER}
    for key, rep in reps.items():
        require(rep["task"] == task and rep["encoder"] == key[1] and rep["phase"] == key[0], f"{key}: rapporto di un altro compito, encoder o fase")
    phase_guard(reps)
    oofs = [CS.load_oof(reps[key]) for key in ORDER]
    weights = reps[ORDER[0]]["weights"]
    jb = joint_bootstrap(oofs, weights)
    result = {"protocol": PROTOCOL, "task": task, "order": [f"{p}_{e}" for p, e in ORDER], "weights": weights,
              "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED,
                            "method": "soggetti ricampionati insieme per i quattro rapporti, per dataset in ordine alfabetico, come arms_compare.s_comparison"},
              "S": {f"{p}_{e}": float(v) for (p, e), v in zip(ORDER, jb["S_point"])},
              "S_se_reports": {f"{p}_{e}": reps[(p, e)]["S_se"] for p, e in ORDER},
              "contrasts": {name: contrast(jb["S_point"], jb["S_draws"], coef) for name, coef in CONTRASTS.items()},
              "datasets": {}}
    for i, d in enumerate(jb["datasets"]):
        result["datasets"][d] = {"weight": float(jb["weights"][i]), "S": {f"{p}_{e}": float(v) for (p, e), v in zip(ORDER, jb["point"][i])},
                                 "contrasts": {name: contrast(jb["point"][i], jb["draws"][i], coef) for name, coef in CONTRASTS.items()}}
    changes = {}
    for enc in ENCODERS:
        a, b = reps[("A", enc)], reps[("B", enc)]
        oa, ob = oofs[ORDER.index(("A", enc))], oofs[ORDER.index(("B", enc))]
        changes[enc] = {}
        for d in sorted(weights):
            pa, pb = oa[d][1], ob[d][1]
            ca, cb = [f["c"] for f in a["datasets"][d]["folds"]], [f["c"] for f in b["datasets"][d]["folds"]]
            changes[enc][d] = {"predictions_changed": int((pa != pb).sum()), "n": int(len(pa)), "fraction_changed": float((pa != pb).mean()),
                               "c_3000": ca, "c_12000": cb, "folds_with_c_shift": int(sum(x != y for x, y in zip(ca, cb))),
                               "s_3000": a["datasets"][d]["s"], "s_12000": b["datasets"][d]["s"]}
    result["changes_12000_vs_3000"] = changes
    result["warnings"] = {f"{p}_{e}": reps[(p, e)]["audit_summary"] for p, e in ORDER}
    result["residual_warnings_at_12000"] = {e: reps[("B", e)]["audit_summary"]["convergence_warnings"] for e in ENCODERS}
    result["reading"] = {"note": "lettura descrittiva di robustezza, un solo seme di pretraining, nessuna nuova soglia di adozione; IC che contiene 0 = "
                                 "inconclusivo, non equivalenza; delta stabile con warning residui = insensibilita' osservata a questo aumento di "
                                 "iterazioni, non convergenza certificata",
                         "delta_of_deltas_ci_contains_zero": bool(result["contrasts"]["delta_of_deltas"]["ci95"][0] <= 0 <= result["contrasts"]["delta_of_deltas"]["ci95"][1]),
                         "sign_B_minus_R3c_A": float(np.sign(result["contrasts"]["B_minus_R3c_A"]["delta"])),
                         "sign_B_minus_R3c_B": float(np.sign(result["contrasts"]["B_minus_R3c_B"]["delta"]))}
    if run_official:  # controllo incrociato: il confronto ufficiale (stesso bootstrap) deve dare gli stessi numeri di B−R3c in ciascuna fase
        result["official"] = {}
        for phase in PHASES:
            out = out_dir / f"ufficiale_{task}_{phase}.json"
            rc = SC.main(["--stage", "compare", "--single-seed", "--models", "--reference", str(paths[(phase, "R3c")]),
                          "--candidate", str(paths[(phase, "B")]), "--out", str(out)])
            require(rc == 0, f"confronto ufficiale della fase {phase} fallito")
            row = json.loads(out.read_text())["comparison"]
            mine = result["contrasts"][f"B_minus_R3c_{phase}"]
            agree = abs(row["delta"] - mine["delta"]) <= 1e-12 and all(abs(x - y) <= 1e-12 for x, y in zip(row["ci95"], mine["ci95"]))
            result["official"][phase] = {"path": str(out), "delta": row["delta"], "ci95": row["ci95"], "agrees_with_joint_bootstrap": bool(agree),
                                         "verdict_label": json.loads(out.read_text())["reading"]["verdict"]}
            require(agree, f"fase {phase}: il bootstrap congiunto non riproduce il confronto ufficiale")
    return result


def run_compare(a) -> int:
    code = analysis_code(a.allow_unfrozen)
    fresh(a.out)
    gate = check_gate(a.gate, code)
    a.out.mkdir(parents=True, exist_ok=False)
    paths = {}
    for p in a.reports:
        rep = load_aggregate(Path(p))
        if code["frozen"]:
            require(rep["analysis_code_sha"] == code["analysis_code_sha"], f"{p}: rapporto di un altro codice di analisi")
        paths[(rep["phase"], rep["encoder"], rep["task"])] = Path(p)
    gate_json = json.loads(Path(a.gate).read_text())
    for key, entry in gate_json["reports"].items():
        enc, task = key.split("_")
        require(("A", enc, task) in paths and sha256(paths[("A", enc, task)]) == entry["sha256"],
                f"rapporto della fase A {key} assente o diverso da quello attestato dal cancello")
    summary = {"protocol": PROTOCOL, "gate": gate, **code, "tasks": {}}
    for task in TASKS:
        task_paths = {(p, e): paths[(p, e, task)] for p, e in ORDER if (p, e, task) in paths}
        require(len(task_paths) == 4, f"{task}: servono i quattro rapporti (due fasi x due encoder), trovati {sorted(task_paths)}")
        result = compare_task(task, task_paths, a.out, run_official=not a.skip_official)
        write_json(a.out / f"confronto_{task}.json", result)
        c = result["contrasts"]
        summary["tasks"][task] = {k: c[k] for k in CONTRASTS}
        summary["tasks"][task]["residual_warnings_at_12000"] = result["residual_warnings_at_12000"]
        print(f"[{time.strftime('%H:%M:%S')}] {task}: B−R3c a 3000 {100 * c['B_minus_R3c_A']['delta']:+.3f} "
              f"[{100 * c['B_minus_R3c_A']['ci95'][0]:+.3f}; {100 * c['B_minus_R3c_A']['ci95'][1]:+.3f}]; a 12000 {100 * c['B_minus_R3c_B']['delta']:+.3f} "
              f"[{100 * c['B_minus_R3c_B']['ci95'][0]:+.3f}; {100 * c['B_minus_R3c_B']['ci95'][1]:+.3f}]; differenza dei delta "
              f"{100 * c['delta_of_deltas']['delta']:+.3f} [{100 * c['delta_of_deltas']['ci95'][0]:+.3f}; {100 * c['delta_of_deltas']['ci95'][1]:+.3f}] punti",
              flush=True)
    write_json(a.out / "riepilogo.json", summary)
    print(json.dumps({"complete": True, "stage": "compare-phases", "out": str(a.out)}), flush=True)
    return 0


# -------------------------------------------------------------------------------------------------------------------------------------- piano

def _hms_to_hours(hms: str) -> float:
    h, m, s = (int(v) for v in hms.split(":"))
    return h + m / 60 + s / 3600


def build_plan(code_sha: str, campaign: str, results_root: str, features_root: str, originals_root: str, pins_rel: str, agent: str,
               ledger_campaign: str, pins: dict, date: str = "20261029") -> dict:
    """Il piano eseguibile: blocchi di job per semplicione (submitter comune), con risorse, tetti e dipendenze; ogni job gira dal checkout di
    campagna `campaign` allo SHA `code_sha` del codice di analisi. I blocchi si sottomettono uno dopo l'altro, a blocco precedente concluso: ogni
    job controlla da solo i prerequisiti (ricevute, cancello), quindi una sottomissione prematura fallisce senza spendere."""
    require(len(code_sha) == 40 and all(c in "0123456789abcdef" for c in code_sha), "code_sha deve essere uno SHA a 40 caratteri")
    checkout = f"/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/{campaign}"
    launcher = f"{checkout}/scripts/slurm/probe_convergence.sbatch"
    pins_path = f"{checkout}/{pins_rel}"
    splits = f"{checkout}/splits/v1/splits_v1.json"
    export = f"--export=ALL,CODE_DIR={checkout},EXPECTED_SHA={code_sha}"
    blocks, jobs = [], []

    def job(block: int, ident: str, cpus: int, mem: str, time_: str, out: str, stage: str, args: list[str], deps: list[str] | None = None,
            expected_min: float | None = None) -> dict:
        argv = ["python3", "scripts/submit_job.py", "--id", ident, "--agente", agent, "--campagna", ledger_campaign, "--gpu-ore", "0"]
        for d in deps or []:
            argv += ["--dopo", d]
        argv += ["--", export, f"--cpus-per-task={cpus}", f"--mem={mem}", f"--time={time_}", launcher, out, stage, *args]
        j = {"block": block, "id": ident, "stage": stage, "cpus": cpus, "mem": mem, "time": time_, "core_hours_cap": cpus * _hms_to_hours(time_),
             "expected_minutes": expected_min, "out": out, "depends_on": deps or [], "command": shlex.join(argv)}
        jobs.append(j)
        return j

    for phase in ("A", "B"):
        base = f"{results_root}/{phase}"
        b_sh = 1 if phase == "A" else 4
        for enc in ENCODERS:
            for task, shards in SHARDS.items():
                for name, datasets, folds in shards:
                    args = ["--phase", phase, "--encoder", enc, "--task", task, "--datasets", *datasets, "--folds", *map(str, folds),
                            "--features", f"{features_root}/{enc}/{task}", "--pins", pins_path, "--splits", splits]
                    if phase == "B":
                        args += ["--gate", f"{results_root}/A/gate.json"]
                    exp = SHARD_EXPECTED_MIN[name] * (1 if phase == "A" else 2.5)
                    job(b_sh, f"PILOT-POSA-SONDE-{phase}-{enc}-{task}-{name}-{date}", 8, "30G", SHARD_TIME[phase][name],
                        f"{base}/{enc}_{task}/shard_{name}", "probe", args, expected_min=exp)
        for enc in ENCODERS:
            for task, shards in SHARDS.items():
                shard_jsons = [f"{base}/{enc}_{task}/shard_{name}/shard.json" for name, _, _ in shards]
                args = ["--phase", phase, "--encoder", enc, "--task", task, "--pins", pins_path, "--splits", splits, "--shards", *shard_jsons]
                if phase == "A":
                    pin = pins["reports"][f"{enc}_{task}"]
                    args += ["--original", f"{originals_root}/{enc}_{task}_pool.json", "--original-sha256", pin["report_sha256"],
                             "--original-oof-sha256", pin["oof_sha256"], "--parity-out", f"{base}/parita_{enc}_{task}.json"]
                job(b_sh + 1, f"PILOT-POSA-SONDE-{phase}-{enc}-{task}-aggregate-{date}", 2, "16G", "00:20:00", f"{base}/{enc}_{task}_pool.json",
                    "aggregate", args, expected_min=3)
        if phase == "A":
            reports = [f"{base}/{enc}_{task}_pool.json" for enc in ENCODERS for task in TASKS]
            parities = [f"{base}/parita_{enc}_{task}.json" for enc in ENCODERS for task in TASKS]
            job(3, f"PILOT-POSA-SONDE-A-gate-{date}", 1, "4G", "00:10:00", f"{base}/gate.json", "gate",
                ["--reports", *reports, "--parities", *parities], expected_min=1)
        else:
            reports = [f"{results_root}/{p}/{enc}_{task}_pool.json" for p in ("A", "B") for enc in ENCODERS for task in TASKS]
            deps = [f"PILOT-POSA-SONDE-B-{enc}-{task}-aggregate-{date}" for enc in ENCODERS for task in TASKS]
            job(5, f"PILOT-POSA-SONDE-confronto-{date}", 2, "16G", "00:30:00", f"{results_root}/confronto", "compare-phases",
                ["--gate", f"{results_root}/A/gate.json", "--reports", *reports], deps=deps, expected_min=5)
    names = {1: "fase A: shard delle sonde a 3.000 iterazioni (10 job, 8 core)", 2: "fase A: aggregazione e parita' con gli originali (4 job)",
             3: "cancello della fase B (1 job)", 4: "fase B: shard a 12.000 iterazioni (10 job, 8 core), ciascuno verifica il cancello",
             5: "fase B: aggregazione (4 job) e confronto fra le fasi (1 job, afterok sui quattro aggregati)"}
    for b in sorted(names):
        in_block = [j for j in jobs if j["block"] == b]
        blocks.append({"block": b, "title": names[b], "n_jobs": len(in_block), "core_hours_cap": sum(j["core_hours_cap"] for j in in_block),
                       "expected_core_hours": sum((j["expected_minutes"] or 0) / 60 * j["cpus"] for j in in_block),
                       "submit_after": "il blocco precedente e' interamente COMPLETED 0:0 (leonardo_watch.py --mode all)" if b > 1 else "subito"})
    caps = {"A": sum(j["core_hours_cap"] for j in jobs if j["block"] in (1, 2, 3)), "B": sum(j["core_hours_cap"] for j in jobs if j["block"] in (4, 5))}
    shard_a = sum(SHARD_EXPECTED_MIN[n] for n, _, _ in SHARDS["p1"] + SHARDS["p2"]) * len(ENCODERS) / 60 * 8  # core-ore osservate (circa)
    small = sum((j["expected_minutes"] or 0) / 60 * j["cpus"] for j in jobs if j["stage"] != "probe")
    return {"schema_version": 1, "protocol": PROTOCOL, "status": "PRONTO_PER_LA_REVIEW_DI_GPT", "analysis_code_sha": code_sha, "campaign_checkout": checkout,
            "cache_code_sha": next(iter(pins["caches"].values()))["code_sha"], "results_root": results_root, "features_root": features_root,
            "originals_root": originals_root, "pins": pins_path, "ledger_campaign": ledger_campaign, "ledger_agent": agent, "gpu_hours_cap": 0,
            "core_hours_cap": {**caps, "total": caps["A"] + caps["B"]},
            "expected_core_hours": {"A": shard_a + small / 2, "B_range": [shard_a + small / 2, 4 * shard_a + small / 2],
                                    "note": "A = tempi osservati nelle sonde del 28-29/10 (8 core); B fra 1x e 4x (fit al limite x4), non una previsione"},
            "serial_partition": {"walltime_limit": "04:00:00", "max_queued_per_user": 10, "note": "ogni blocco ha al piu' 10 job"},
            "phases": {p: {"max_iter": m} for p, m in PHASES.items()}, "shards": {t: [{"name": n, "datasets": list(d), "folds": list(f)} for n, d, f in s]
                                                                          for t, s in SHARDS.items()},
            "automatic_retry": False, "blocks": blocks, "jobs": jobs}


def run_plan(a) -> int:
    pins = load_pins(a.pins)
    plan = build_plan(a.code_sha, a.campaign, a.results_root, a.features_root, a.originals_root, a.pins_rel, a.agente, a.ledger_campaign, pins)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    plan_path, cmd_path = a.out_dir / "piano.json", a.out_dir / "comandi.txt"
    for p in (plan_path, cmd_path):
        if p.exists():
            p.unlink()
    write_json(plan_path, plan)
    lines = [f"# Controllo della convergenza delle sonde B19618/R3c ({PROTOCOL}): comandi per semplicione, dal Mac, UNA riga per comando, nell'ordine.",
             f"# Codice di analisi {a.code_sha} nel checkout di campagna {plan['campaign_checkout']} (da creare prima con campaign_checkout.sbatch).",
             "# Ogni blocco si sottomette quando il precedente e' interamente COMPLETED 0:0; un job che trova un prerequisito mancante esce con errore",
             "# senza spendere (ricevute, cancello). Nessun retry automatico: un job fallito si porta a GPT.", ""]
    for b in plan["blocks"]:
        lines += [f"# ---- blocco {b['block']}: {b['title']} ({b['n_jobs']} job, tetto {b['core_hours_cap']:.2f} core-ore; sottomettere: {b['submit_after']})"]
        lines += [j["command"] for j in plan["jobs"] if j["block"] == b["block"]]
        lines.append("")
    cmd_path.write_text("\n".join(lines))
    print(json.dumps({"complete": True, "stage": "plan", "jobs": len(plan["jobs"]), "core_hours_cap": plan["core_hours_cap"], "plan": str(plan_path),
                      "commands": str(cmd_path)}), flush=True)
    return 0


# --------------------------------------------------------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["probe", "aggregate", "parity", "gate", "compare-phases", "plan"], required=True)
    ap.add_argument("--out", type=Path, help="uscita nuova e assoluta, fuori dal codice (cartella per probe e compare-phases, file altrimenti)")
    ap.add_argument("--phase", choices=sorted(PHASES))
    ap.add_argument("--encoder", choices=ENCODERS)
    ap.add_argument("--task", choices=sorted(TASKS))
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--folds", nargs="+", type=int)
    ap.add_argument("--features", type=Path, help="cartella delle cache del teacher (con extraction_receipt.json e le provenienze)")
    ap.add_argument("--pins", type=Path, help="pin_ingressi.json: SHA delle cache, dei rapporti originali e dei checkpoint")
    ap.add_argument("--splits", type=Path, default=ROOT / "splits" / "v1" / "splits_v1.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", type=Path, help="fase B e compare-phases: il cancello della fase A")
    ap.add_argument("--shards", nargs="+", help="aggregate: gli shard.json della stessa (fase, encoder, compito)")
    ap.add_argument("--original", type=Path, help="aggregate/parity: il rapporto originale (storico) della sonda")
    ap.add_argument("--original-sha256", help="pin dello SHA-256 del rapporto originale")
    ap.add_argument("--original-oof-sha256", help="pin dello SHA-256 delle predizioni fuori fold originali")
    ap.add_argument("--parity-out", type=Path, help="aggregate: dove scrivere la parita' (fase A)")
    ap.add_argument("--new", type=Path, help="parity: il rapporto aggregato nuovo")
    ap.add_argument("--reports", nargs="+", help="gate: i quattro rapporti della fase A; compare-phases: gli otto rapporti")
    ap.add_argument("--parities", nargs="+", help="gate: i quattro file di parita'")
    ap.add_argument("--skip-official", action="store_true", help="compare-phases: senza il confronto ufficiale incrociato (solo prove)")
    ap.add_argument("--allow-unfrozen", action="store_true", help="solo prove locali: codice di analisi non congelato")
    ap.add_argument("--code-sha", help="plan: SHA a 40 caratteri del codice di analisi pubblicato")
    ap.add_argument("--campaign", default="pilot-posa-sonde-20261029", help="plan: nome del checkout di campagna")
    ap.add_argument("--results-root", default="/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pilot_posa/sonde_convergenza_20261029")
    ap.add_argument("--features-root", default="/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/pilot_posa/B19618_20261028")
    ap.add_argument("--originals-root", default="/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pilot_posa/B19618_20261028")
    ap.add_argument("--pins-rel", default="results/infra/pilot_posa/sonde_convergenza_20261029/pin_ingressi.json")
    ap.add_argument("--agente", default="gpt", choices=["claude", "gpt", "simone"], help="plan: categoria del ledger (l'operatore e' semplicione)")
    ap.add_argument("--ledger-campaign", default="pilot-posa-sonde-20261029")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "infra" / "pilot_posa" / "sonde_convergenza_20261029", help="plan")
    a = ap.parse_args(argv)
    if a.stage == "plan":
        if not a.code_sha or not a.pins:
            ap.error("plan: servono --code-sha e --pins")
        return run_plan(a)
    if a.out is None:
        ap.error("serve --out")
    if a.stage == "probe":
        if not (a.phase and a.encoder and a.task and a.features and a.pins):
            ap.error("probe: servono --phase, --encoder, --task, --features e --pins")
        return run_probe(a)
    if a.stage == "aggregate":
        if not (a.phase and a.encoder and a.task and a.shards and a.pins):
            ap.error("aggregate: servono --phase, --encoder, --task, --shards e --pins")
        if (a.original is None) != (a.parity_out is None):
            ap.error("aggregate: --original e --parity-out vanno insieme")
        return run_aggregate(a)
    if a.stage == "parity":
        if not (a.original and a.new):
            ap.error("parity: servono --original e --new")
        return run_parity(a)
    if a.stage == "gate":
        if not (a.reports and a.parities):
            ap.error("gate: servono --reports e --parities")
        return run_gate(a)
    if not (a.reports and a.gate):
        ap.error("compare-phases: servono --reports (otto rapporti) e --gate")
    return run_compare(a)


if __name__ == "__main__":
    sys.exit(main())
