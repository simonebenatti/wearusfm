#!/usr/bin/env python3
"""Diagnostica dei rami PSD e wavelet dopo il 2C (protocollo `rami_diagnostica_v1`, `docs/rami_diagnostica_v1.md`; punto 5 della tabella
comune, consegna del 20/10/2026). Domanda: allenare i rami (raw congelato, 2C) peggiora i rami anche da soli, oppure il peggioramento emerge
soprattutto nella concatenazione col raw? Nessuno dei due esiti dimostra da solo la causa.

Solo cache gia' estratte, nessuna nuova estrazione: le uscite dei rami del 2C (`step2c_frozen_branches_v1`, commit 1c94852: passo 0 =
`branches_init`, allenati = `branches_trained`, 2 x 128 colonne PSD e wavelet) e gli embedding raw della tranche downstream
(`step1_downstream_4000_v1`, 768 colonne), per i quattro teacher dell'A/B. Insiemi di colonne: PW0, PW (soli rami), R (raw), RPW0, RPW (raw
concatenato ai rami). Ogni file si confronta col suo SHA-256 registrato nella provenienza, e le righe con l'impronta di etichette e soggetti.

Metrica: la validation INTERNA della calibrazione (`scripts/calibration_probe.py`, stessa funzione di fold, stessa sonda lineare, stessa S):
P1 sui soli soggetti di pretraining di NinaPro DB2/DB3/DB6 (ruoli train_ e val_ della cache; le righe test_ non si leggono mai), P2 sui
soggetti di EPN-612 e UCI-EMG fuori dai test di tutti e tre gli split ufficiali (le altre righe si scartano subito dopo la lettura). Per fold:
normalizzazione sul train, C scelto sui soggetti di validazione del fold, valutazione sui soggetti tenuti fuori. Nessun numero di test.

Confronti accoppiati per famiglia (A: A_seed0/A_seed1, B: B_seed0/B_seed1; `arms_compare.s_comparison`, soggetti ricampionati insieme):
  rami_da_soli     PW  contro PW0     concatenazione   RPW contro RPW0
  apporto_passo0   RPW0 contro R      apporto_allenati RPW contro R
Lettura (congelata in docs/decisioni.md prima delle sonde): «migliora» con la regola dei bracci (delta >= +0,01, limite inferiore dell'IC al
95% > 0, delta > 0 con tutti e due i teacher), «peggiora» con la regola speculare, altrimenti «entro le soglie» (che non esclude una
differenza, neanche oltre 1 punto: la regola puo' fallire per l'intervallo o per l'accordo dei teacher). La risposta dice solo quale dei
due confronti ha superato la regola. Descrittivi senza soglia: la differenza dei
due delta, (RPW - RPW0) - (PW - PW0), con IC dallo stesso ricampionamento accoppiato degli otto modelli; S per dataset e teacher; rango
effettivo di ogni insieme; ridondanza col raw (CKA lineare; R^2 fuori fold dei rami predetti dal raw con ridge, alpha scelto sui soggetti di
validazione del fold).

  python3 scripts/spectral_diagnostic.py probe --task p1 --arm B_seed0 --sets R PW0 PW --redundancy \
      --eval-root E --downstream-root D --out-dir O          (CPU)
  python3 scripts/spectral_diagnostic.py compare --out-dir O --out esito.json
  python3 scripts/spectral_diagnostic.py plan --code-dir C --expected-sha SHA --out-dir O --out piano.json   (per evaluation_chain.py)
"""

from __future__ import annotations

import argparse
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
from wearusfm.harness import p1_cross_subject as P1  # noqa: E402
from wearusfm.harness.downstream_ab import fingerprint  # noqa: E402
from wearusfm.training.spectral_calibration import sha256_file  # noqa: E402

PROTOCOL = "rami_diagnostica_v1"
STEP2C = {"protocol": "step2c_frozen_branches_v1", "source_commit": "1c9485260aaf32d5936346b50225f5cff1d1b8ea"}
DOWNSTREAM = {"protocol": "step1_downstream_4000_v1", "source_commit": "8dad7e82713ce4ed2741acc9c9d4fd9ac656d2f7"}
SPLIT_SHA = "a18e7565157d5dbba8870ffa67c2342ff90de479da480dccb456922b6f6f4ed7"  # splits/v1/splits_v1.json, come il 2C
CHECKPOINTS = {  # teacher dell'A/B dello Step 1 (scripts/step1_downstream.py)
    "A_seed0": "26ee7e440b01c77b35b9cf198110a4341e267340d377ca32feb80ae6b475b611",
    "A_seed1": "b72a122ea24d01af2830de7f1c76dbcfd7810991e75f9ba015ca5e93bdeeb4d2",
    "B_seed0": "b264df4b75f7047980212ab068fe524148e71dbdef442d4e75e95648f841d3cd",
    "B_seed1": "27b3a819c35f798992b2bcddf7f48998be03d9dc4bc113b79e5170c199577a29",
}
FAMILIES = {"A": ("A_seed0", "A_seed1"), "B": ("B_seed0", "B_seed1")}
DATASETS = {"p1": ("ninapro_db2", "ninapro_db3", "ninapro_db6"), "p2": ("epn612", "uci_emg")}
RAW_DIM, BRANCH_COLS = 768, 256
# insieme -> (raw incluso, modello dei rami o None)
SETS = {"PW0": (False, "init"), "PW": (False, "trained"), "R": (True, None), "RPW0": (True, "init"), "RPW": (True, "trained")}
CONTRASTS = (("rami_da_soli", "PW0", "PW"), ("concatenazione", "RPW0", "RPW"), ("apporto_passo0", "R", "RPW0"),
             ("apporto_allenati", "R", "RPW"))
FEATURE_SET = "cache_2C"  # per l'accoppiamento dei rapporti: stessa sorgente per tutti gli insiemi (le colonne sono in "columns")
RIDGE_ALPHAS = (1.0, 100.0, 10000.0)
N_BOOT, BOOT_SEED = 1000, 20261009  # come arms_compare.s_comparison
# job della catena: (suffisso, compito, insiemi, ridondanza, tempo massimo), nell'ordine dei teacher (famiglia B prima)
JOBS = (("p1-a", "p1", ("R", "PW0", "PW"), True, "04:00:00"), ("p1-b", "p1", ("RPW0",), False, "03:30:00"),
        ("p1-c", "p1", ("RPW",), False, "03:30:00"), ("p2", "p2", ("R", "PW0", "PW", "RPW0", "RPW"), True, "04:00:00"))
ARM_ORDER = ("B_seed0", "B_seed1", "A_seed0", "A_seed1")
NICE = 10000  # abbassa la priorita' fra i job in attesa; se prevalga sull'eta' dipende dai pesi di priorita' di SLURM (non letti)


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_diagnostic", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CP = _script("calibration_probe")


# --- cache e provenienza ---------------------------------------------------------------------------------------------------------

def _prefixes(task: str) -> tuple[str, ...]:
    return ("train_", "val_") if task == "p1" else ("",)  # P1: mai i ruoli test_


def _check_identity(ident: dict, expected: dict, where: Path) -> None:
    bad = {k: (ident.get(k), v) for k, v in expected.items() if ident.get(k) != v}
    if bad:
        raise ValueError(f"{where}: provenienza diversa da quella attesa {bad}")


def provenance(eval_root: Path, downstream_root: Path, task: str, arm: str) -> dict:
    """Provenienze ricontrollate: identita' delle cache dei rami (passo 0 e allenati) e della cache raw, e le loro righe accoppiate."""
    out = {}
    for which in ("init", "trained"):
        path = eval_root / task / arm / f"branches_{which}" / "provenance.json"
        meta = json.loads(path.read_text())
        _check_identity(meta["identity"], {**STEP2C, "task": task, "arm": arm, "variant": f"branches_{which}", "which": "teacher",
                                           "split_sha256": SPLIT_SHA, "source_checkpoint_sha256": CHECKPOINTS[arm]}, path)
        if any(v.get("feature_set") != "branches" or v.get("model_dim") != BRANCH_COLS for v in meta["cache"].values()):
            raise ValueError(f"{path}: non e' una cache dei rami da {BRANCH_COLS} colonne")
        run = meta.get("run") or {}
        expected_run = {**STEP2C, "arm": arm, "collaudo": False, "complete": True, "encoder_identical": True,
                        "source_checkpoint_sha256": CHECKPOINTS[arm]}
        _check_identity(run, expected_run, path)
        out[which] = meta["cache"]
        out[f"run_{which}"] = {k: run.get(k) for k in ("checkpoint_sha256", "init_checkpoint_sha256")}
    if out["run_init"] != out["run_trained"] or not all(out["run_init"].values()):
        raise ValueError(f"{task}/{arm}: rami al passo 0 e allenati non vengono dallo stesso run del 2C")
    path = downstream_root / task / arm / "provenance.json"
    meta = json.loads(path.read_text())
    _check_identity(meta["identity"], {**DOWNSTREAM, "task": task, "arm": arm, "which": "teacher", "split_sha256": SPLIT_SHA,
                                       "checkpoint_sha256": CHECKPOINTS[arm]}, path)
    out["raw"] = meta["cache"]
    keep = ("input_sha256", "rows")
    for which in ("init", "trained"):  # come step2a._paired: stessi ingressi e stesse righe (etichette, soggetti) della cache raw
        if sorted(out[which]) != sorted(out["raw"]) or any({k: out[which][d][k] for k in keep} != {k: out["raw"][d][k] for k in keep}
                                                          for d in out["raw"]):
            raise ValueError(f"{task}/{arm}: rami {which} non accoppiati alla cache raw")
    if sorted(out["raw"]) != sorted(DATASETS[task]):
        raise ValueError(f"{task}/{arm}: dataset {sorted(out['raw'])}")
    return out


def _file(eval_root: Path, downstream_root: Path, task: str, arm: str, part: str, ds: str) -> Path:
    if part == "raw":
        return downstream_root / task / arm / f"{ds}.npz"
    return eval_root / task / arm / f"branches_{part}" / f"{ds}.npz"


def internal_subjects(task: str, ds: str, prov_rows: dict, splits: dict | None) -> list[str]:
    """Soggetti della validation interna. P1: pretraining degli split firmati (= ruoli train_ e val_ della cache), nessuno di test.
    P2: fuori dai test dei tre split ufficiali (`calibration_probe.p2_internal_subjects`), da tutti i soggetti della cache."""
    if task == "p1":
        pre = CP.pretraining_subjects(splits, ds)
        cached = sorted(set(prov_rows["train_"]["subjects"]) | set(prov_rows["val_"]["subjects"]))
        if cached != pre or set(pre) & set(prov_rows["test_"]["subjects"]):
            raise ValueError(f"{ds}: soggetti di train/val della cache diversi dal pretraining degli split")
        return pre
    return CP.p2_internal_subjects(prov_rows[""]["subjects"])


def load(eval_root: Path, downstream_root: Path, task: str, arm: str, parts: set[str], splits: dict | None) -> dict:
    """{ds: {"y", "s", "subjects", "input_sha256", parte: x}} sulle sole righe interne, con file e righe verificati."""
    prov = provenance(eval_root, downstream_root, task, arm)
    out = {}
    for ds in DATASETS[task]:
        rows = prov["raw"][ds]["rows"]
        pre = internal_subjects(task, ds, rows, splits)
        entry = {"subjects": pre, "input_sha256": prov["raw"][ds]["input_sha256"], "files": {}}
        ref = None
        for part in sorted(parts):
            path = _file(eval_root, downstream_root, task, arm, part, ds)
            sha = sha256_file(path)
            if sha != prov[part][ds]["sha256"]:
                raise ValueError(f"{path}: SHA-256 diverso da quello della provenienza")
            entry["files"][part] = {"path": str(path), "sha256": sha}
            xs, ys, ss = [], [], []
            with np.load(path, allow_pickle=False) as z:
                for p in _prefixes(task):
                    y, s = z[p + "y"], z[p + "s"]
                    if fingerprint(y, s) != rows[p]["labels_subjects_sha256"]:
                        raise ValueError(f"{path}: righe {p or 'uniche'} diverse da quelle registrate")
                    x = z[p + "x"]
                    xs.append(x)
                    ys.append(y)
                    ss.append(s)
            x, y, s = np.concatenate(xs), np.concatenate(ys), np.concatenate(ss).astype(str)
            keep = np.isin(s, pre)  # P2: via le finestre dei soggetti di test dei tre split, prima di qualunque calcolo
            x, y, s = x[keep], y[keep], s[keep]
            dim = RAW_DIM if part == "raw" else BRANCH_COLS
            if x.ndim != 2 or x.shape != (len(y), dim) or not np.isfinite(x).all():
                raise ValueError(f"{path}: feature non valide (attese {dim} colonne finite)")
            if ref is None:
                ref = (y, s)
                entry.update(y=y, s=s)
            elif not (np.array_equal(ref[0], y) and np.array_equal(ref[1], s)):
                raise ValueError(f"{task}/{arm}/{ds}: righe non allineate fra raw e rami")
            entry[part] = x
        if sorted(set(entry["s"].tolist())) != pre:
            raise ValueError(f"{ds}: soggetti interni assenti dalla cache")
        out[ds] = entry
    return out


def columns(entry: dict, name: str) -> np.ndarray:
    raw, which = SETS[name]
    blocks = ([entry["raw"]] if raw else []) + ([entry[which]] if which else [])
    return np.concatenate(blocks, 1) if len(blocks) > 1 else blocks[0]


def parts_of(names) -> set[str]:
    out = set()
    for n in names:
        raw, which = SETS[n]
        out |= {"raw"} if raw else set()
        out |= {which} if which else set()
    return out


# --- sonda e ridondanza -----------------------------------------------------------------------------------------------------------

def probe_set(data: dict, task: str, arm: str, name: str, splits: dict | None, splits_sha: str, out: Path, seed: int = 0) -> dict:
    """Rapporto nello stesso formato di `calibration_probe.py` (S, SE, OOF con le righe): lo leggono `calibration_select.load_oof` e
    `arms_compare.s_comparison`, che ne controllano l'accoppiamento."""
    from sklearn.metrics import balanced_accuracy_score

    t0 = time.time()
    report = {"protocol": PROTOCOL, "task": task, "arm": arm, "set": name, "columns": {"raw": SETS[name][0], "branches": SETS[name][1]},
              "k_folds": CP.K_FOLDS, "fold_seed": CP.FOLD_SEED, "val_frac": CP.VAL_FRAC, "probe_seed": seed, "feature_set": FEATURE_SET,
              "window_s": CP.WINDOW_S if task == "p1" else None, "splits_sha256": splits_sha,
              "extraction_recipe": {"branches": STEP2C, "raw": DOWNSTREAM, "split_sha256": SPLIT_SHA}, "datasets": {}}
    parts, oof = [], {}
    for ds, e in data.items():
        x = columns(e, name)
        res = CP.oof_predictions(x, e["y"], e["s"], CP.fold_roles(e["subjects"]), seed)
        bacc = float(balanced_accuracy_score(res["y"], res["pred"]))
        parts.append((res["y"], res["pred"], res["s"]))
        oof.update({f"{ds}_y": res["y"], f"{ds}_pred": res["pred"], f"{ds}_s": res["s"], f"{ds}_row": res["row"]})
        report["datasets"][ds] = {"input_sha256": e["input_sha256"], "files": {p: e["files"][p] for p in parts_of([name])},
                                  "subjects": e["subjects"], "s": bacc, "folds": res["folds"], "n_classes": int(len(np.unique(e["y"]))),
                                  "n_windows": int(len(e["y"])), "n_oof_windows": int(len(res["y"])), "oof_rows_sha256": fingerprint(res["row"]),
                                  "features": CP.feature_diagnostics(x)}
    if task == "p1":
        w = P1.dataset_weights([len(splits["datasets"][d]["test"]) for d in data], "subjects")  # solo i conteggi, pesi della P1
    else:
        w = P1.dataset_weights([1] * len(data), "datasets")
    report.update({"weights": dict(zip(data, w.tolist())), "S": float(np.dot(w, [report["datasets"][d]["s"] for d in data])),
                   "S_se": P1.mean_bacc_bootstrap_se(parts, seed=seed, weights=w)})
    oof_path = out.with_suffix(".oof.npz")
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(oof_path, datasets=np.asarray(list(data)), **oof)
    report.update({"oof": str(oof_path), "oof_sha256": sha256_file(oof_path), "elapsed_s": round(time.time() - t0, 1)})
    out.write_text(json.dumps(report, indent=1, allow_nan=False))
    return report


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    """CKA lineare (Kornblith et al. 2019) fra due insiemi di feature sulle stesse righe: 1 = stessa geometria a meno di rotazioni e scale."""
    a = a.astype(np.float64) - a.mean(0)
    b = b.astype(np.float64) - b.mean(0)
    num = np.linalg.norm(a.T @ b) ** 2
    den = np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)
    return float(num / den) if den > 0 else 0.0


def ridge_r2(x: np.ndarray, t: np.ndarray, s: np.ndarray, roles: list[dict]) -> dict:
    """R^2 fuori fold dei rami `t` predetti dal raw `x` (ridge, alpha scelto sui soggetti di validazione del fold, tutto standardizzato
    sul train del fold): la frazione della varianza dei rami leggibile linearmente dal raw su soggetti nuovi."""
    sse = sst = 0.0
    chosen = []
    for r in roles:
        mt, mv, mh = (np.isin(s, r[k]) for k in ("train", "val", "held"))
        mx, sx = x[mt].mean(0), x[mt].std(0) + 1e-8
        mtg, stg = t[mt].mean(0), t[mt].std(0) + 1e-8
        xt, xv, xh = ((x[m] - mx) / sx for m in (mt, mv, mh))
        tt, tv, th = ((t[m] - mtg) / stg for m in (mt, mv, mh))
        g, c = xt.T @ xt, xt.T @ tt

        def fit(alpha):
            return np.linalg.solve(g + alpha * np.eye(len(g)), c)

        best = min(RIDGE_ALPHAS, key=lambda a: (float(((xv @ fit(a) - tv) ** 2).sum()), -a))
        chosen.append(best)
        sse += float(((xh @ fit(best) - th) ** 2).sum())
        sst += float(((th - th.mean(0)) ** 2).sum())
    return {"r2": 1.0 - sse / sst if sst > 0 else float("nan"), "alphas": chosen}


def redundancy(data: dict) -> dict:
    out = {}
    for ds, e in data.items():
        roles = CP.fold_roles(e["subjects"])
        x = e["raw"].astype(np.float64)
        row = {"cka_R_PW0": linear_cka(x, e["init"]), "cka_R_PW": linear_cka(x, e["trained"]), "cka_PW0_PW": linear_cka(e["init"], e["trained"])}
        for which, key in (("init", "PW0"), ("trained", "PW")):
            r = ridge_r2(x, e[which].astype(np.float64), e["s"], roles)
            row[f"r2_{key}_from_R"], row[f"alphas_{key}"] = r["r2"], r["alphas"]
        out[ds] = row
    return out


def probe(args) -> int:
    splits = json.loads(args.splits.read_text()) if args.task == "p1" else None
    if args.task == "p1" and sha256_file(args.splits) != SPLIT_SHA:
        raise ValueError(f"{args.splits}: non sono gli split firmati del 2C")
    splits_sha = SPLIT_SHA if args.task == "p1" else CP.P2_SPLITS_SHA256
    names = list(args.sets)
    need = parts_of(names) | ({"raw", "init", "trained"} if args.redundancy else set())
    data = load(args.eval_root, args.downstream_root, args.task, args.arm, need, splits)
    folder = args.out_dir / args.task / args.arm
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        rep = probe_set(data, args.task, args.arm, name, splits, splits_sha, folder / f"{name}.json", args.seed)
        print(f"{args.task}/{args.arm}/{name}: S {100 * rep['S']:.2f} +- {100 * rep['S_se']:.2f} "
              "(" + ", ".join(f"{d} {100 * v['s']:.2f}" for d, v in rep["datasets"].items()) + f"), {rep['elapsed_s']:.0f} s", flush=True)
    if args.redundancy:
        t0 = time.time()
        red = {"protocol": PROTOCOL, "task": args.task, "arm": args.arm, "datasets": redundancy(data),
               "effective_rank": {p: {ds: CP.feature_diagnostics(data[ds][p]) for ds in data} for p in ("raw", "init", "trained")}}
        red["elapsed_s"] = round(time.time() - t0, 1)
        (folder / "redundancy.json").write_text(json.dumps(red, indent=1, allow_nan=False))
        print(f"{args.task}/{args.arm}: ridondanza in {red['elapsed_s']:.0f} s", flush=True)
    return 0


# --- confronti --------------------------------------------------------------------------------------------------------------------

def reading(row: dict) -> str:
    """La regola dei bracci e la sua speculare, sul delta = insieme di arrivo - insieme di partenza."""
    if row["beats"]:
        return "migliora"
    if row["delta"] <= -0.01 and row["ci95"][1] < 0 and all(d < 0 for d in row["delta_by_model_seed"]):
        return "peggiora"
    return "entro le soglie"


READINGS = ("peggiora", "entro le soglie", "migliora")
_PHRASE = {"peggiora": "peggioramento oltre le soglie", "migliora": "miglioramento oltre le soglie",
           # seconda review del critico (20/10): esce anche con |delta| > 1 punto e IC che tocca lo zero, o teacher discordi
           "entro le soglie": "regola non superata (stima, intervallo o accordo dei teacher insufficienti): non esclude una differenza, neanche oltre 1 punto"}


def answer(alone: str, concat: str) -> str:
    """Risposta congelata: solo quale confronto ha superato la regola, senza meccanismi ne' confronti fra i due delta (quelli sono il
    descrittivo `differenza_dei_delta`)."""
    return f"soli rami (PW0 -> PW): {_PHRASE[alone]}; concatenazione (RPW0 -> RPW): {_PHRASE[concat]}"


def difference_of_deltas(reps: dict, arms, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Descrittivo senza soglia: (RPW - RPW0) - (PW - PW0), media dei due teacher, con un solo ricampionamento dei soggetti per gli otto
    modelli (stesse righe e soggetti per tutti gli insiemi). Negativo = la concatenazione perde piu' dei soli rami."""
    from wearusfm.harness import downstream_ab as DA

    CS = _script("calibration_select")
    order = [(a, n) for a in arms for n in ("PW0", "PW", "RPW0", "RPW")]
    reports = [reps[a][n] for a, n in order]
    CS.check_pairing(reports)
    if len({json.dumps(r["weights"], sort_keys=True) for r in reports}) != 1:
        raise ValueError("pesi dei dataset diversi")
    oofs = [CS.load_oof(r) for r in reports]
    weights = reports[0]["weights"]
    datasets = sorted(weights)
    w = np.array([weights[d] for d in datasets])
    rng = np.random.default_rng(seed)
    points, teachers, draws, per_ds = [], [], [], {}
    for d in datasets:
        y0, _, s0 = oofs[0][d]
        if any(not (np.array_equal(o[d][0], y0) and np.array_equal(o[d][2], s0)) for o in oofs):
            raise ValueError(f"{d}: finestre o soggetti diversi fra gli insiemi: il confronto non e' accoppiato")
        arrays = [DA._arrays(DA.subject_counts(y0, o[d][1], s0)) for o in oofs]
        subs, n, _ = arrays[0]

        def dd(pick):
            v = np.array([DA.bacc(n[pick], a[2][pick]) for a in arrays]).reshape(len(arms), 4)
            return (v[:, 3] - v[:, 2]) - (v[:, 1] - v[:, 0])

        point = dd(np.arange(len(subs)))
        teachers.append(point)
        points.append(point.mean())
        per_ds[d] = {"delta": float(point.mean()), "delta_by_model_seed": point.tolist()}
        draws.append([dd(rng.integers(len(subs), size=len(subs))).mean() for _ in range(n_boot)])
    samples = w @ np.asarray(draws)
    return {"delta": float(w @ np.array(points)), "delta_by_model_seed": (w @ np.array(teachers)).tolist(),
            "ci95": np.quantile(samples, [.025, .975]).tolist(), "paired_subject_se": float(np.std(samples, ddof=1)), "datasets": per_ds}


def compare(args) -> int:
    AC = _script("arms_compare")
    result = {"protocol": PROTOCOL, "families": {}, "teachers": {}, "redundancy": {}}
    for task in args.tasks:
        for fam, arms in ((f, FAMILIES[f]) for f in args.families):
            reps = {a: {n: json.loads((args.out_dir / task / a / f"{n}.json").read_text()) for n in SETS} for a in arms}
            for a in arms:
                for n, r in reps[a].items():
                    if (r["protocol"], r["task"], r["arm"], r["set"]) != (PROTOCOL, task, a, n):
                        raise ValueError(f"{task}/{a}/{n}: rapporto di un altro protocollo, compito, teacher o insieme")
                    result["teachers"].setdefault(task, {}).setdefault(a, {})[n] = {
                        "S": r["S"], "S_se": r["S_se"], "per_dataset": {d: v["s"] for d, v in r["datasets"].items()},
                        "effective_rank": {d: v["features"]["effective_rank"] for d, v in r["datasets"].items()}}
                red = args.out_dir / task / a / "redundancy.json"
                if red.exists():
                    result["redundancy"].setdefault(task, {})[a] = json.loads(red.read_text())["datasets"]
            rows = {}
            for key, ref, var in CONTRASTS:
                row = AC.s_comparison([reps[arms[0]][ref], reps[arms[1]][ref]], [reps[arms[0]][var], reps[arms[1]][var]])
                rows[key] = {"from": ref, "to": var, **row, "reading": reading(row)}
            result["families"].setdefault(task, {})[fam] = {
                "contrasts": rows, "answer": answer(rows["rami_da_soli"]["reading"], rows["concatenazione"]["reading"]),
                "differenza_dei_delta": difference_of_deltas(reps, arms)}
    result["note"] = ("Validation interna (soggetti di pretraining P1, soggetti fuori dai test P2), nessun numero dei test ufficiali. "
                      "Letture con la regola dei bracci: dicono quale confronto supera la soglia, non perche'; «entro le soglie» non esclude "
                      "una differenza minore (nel 2C, A/P1 ufficiale RPW - RPW0 = -0,84 punti sarebbe «entro le soglie»). 16 letture "
                      "(4 confronti x 2 famiglie x 2 compiti) senza correzione per confronti multipli.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, allow_nan=False))
    for task, fams in result["families"].items():
        for fam, v in fams.items():
            c = v["contrasts"]
            dd = v["differenza_dei_delta"]
            print(f"{task} {fam}: " + "; ".join(f"{k} {100 * c[k]['delta']:+.2f} [{100 * c[k]['ci95'][0]:+.2f}, {100 * c[k]['ci95'][1]:+.2f}] "
                                               f"{c[k]['reading']}" for k, _, _ in CONTRASTS)
                  + f"; differenza dei delta {100 * dd['delta']:+.2f} [{100 * dd['ci95'][0]:+.2f}, {100 * dd['ci95'][1]:+.2f}] -> {v['answer']}")
    return 0


def plan(args) -> int:
    """Piano per `scripts/evaluation_chain.py` (schema eval-plan-v0): 16 sonde e il confronto dopo tutte. La catena sottomette solo entro
    `limits` (job dell'utente in coda e in esecuzione sulla seriale, finestra 1 compresa: la diagnostica si ferma se la coda e' occupata) e
    non mette afterok su predecessori gia' finiti; si richiama a tranche finche' tutto e' sottomesso."""
    out = args.out_dir
    if not str(out).startswith("/"):
        raise ValueError("--out-dir assoluta (e' la cartella dei rapporti sul cluster)")
    nice = [f"--nice={NICE}"]
    nodes, reports = [], []
    for arm in ARM_ORDER:
        for suffix, task, sets, red, time_limit in JOBS:
            outputs = [f"{out}/{task}/{arm}/{n}.json" for n in sets] + ([f"{out}/{task}/{arm}/redundancy.json"] if red else [])
            reports += outputs
            nodes.append({"id": f"RAMI-{arm}-{suffix}", "kind": "probe", "after": [], "queue": "lrd_all_serial", "gpu_hours": 0,
                          "code_dir": args.code_dir, "expected_sha": args.expected_sha, "launcher": "scripts/slurm/spectral_diagnostic.sbatch",
                          "args": [str(out), "probe", "--task", task, "--arm", arm, "--sets", *sets] + (["--redundancy"] if red else []),
                          "exports": {}, "sbatch": [f"--time={time_limit}", *nice], "inputs": [], "outputs": outputs})
    nodes.append({"id": "RAMI-compare", "kind": "compare", "after": [n["id"] for n in nodes], "queue": "lrd_all_serial", "gpu_hours": 0,
                  "code_dir": args.code_dir, "expected_sha": args.expected_sha, "launcher": "scripts/slurm/spectral_diagnostic.sbatch",
                  "args": [str(out), "compare"], "exports": {}, "sbatch": ["--time=00:30:00", "--cpus-per-task=2", *nice],
                  "inputs": [r for r in reports if r.endswith(".json") and not r.endswith("redundancy.json")], "outputs": [f"{out}/esito.json"]})
    doc = {"schema": "eval-plan-v0", "campaign": args.campaign, "agent": "claude", "limits": {"lrd_all_serial": args.max_serial},
           "nodes": nodes}
    _script("evaluation_chain").validate(doc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1))
    print(f"{len(nodes)} nodi, al piu' {args.max_serial} job seriali dell'utente in coda -> {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("probe", allow_abbrev=False)  # niente abbreviazioni: il launcher rifiuta --out-dir e le radici per nome
    p.add_argument("--task", choices=["p1", "p2"], required=True)
    p.add_argument("--arm", choices=sorted(CHECKPOINTS), required=True)
    p.add_argument("--sets", nargs="+", choices=list(SETS), required=True)
    p.add_argument("--redundancy", action="store_true", help="anche CKA, R^2 dal raw e rango effettivo dei blocchi")
    p.add_argument("--eval-root", type=Path, required=True, help="cartella della valutazione del 2C (step2c_1410)")
    p.add_argument("--downstream-root", type=Path, required=True, help="cartella della tranche downstream (step1_downstream_0910)")
    p.add_argument("--splits", type=Path, default=ROOT / "splits" / "v1" / "splits_v1.json")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("compare", allow_abbrev=False)
    c.add_argument("--out-dir", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--tasks", nargs="+", choices=["p1", "p2"], default=["p1", "p2"])
    c.add_argument("--families", nargs="+", choices=sorted(FAMILIES), default=sorted(FAMILIES), help="B si legge prima che A sia finita")
    q = sub.add_parser("plan", allow_abbrev=False)
    q.add_argument("--code-dir", required=True)
    q.add_argument("--expected-sha", required=True)
    q.add_argument("--out-dir", type=Path, required=True, help="cartella dei rapporti sul cluster (assoluta)")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--campaign", default="rami-diag")
    # 2: uno in esecuzione (8 core per utente: uno alla volta) e uno pronto; di piu' non accelera e allungherebbe l'attesa della finestra 1
    q.add_argument("--max-serial", type=int, default=2, help="job seriali dell'utente in coda o in esecuzione oltre i quali non si sottomette")
    args = ap.parse_args(argv)
    return {"probe": probe, "compare": compare, "plan": plan}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
