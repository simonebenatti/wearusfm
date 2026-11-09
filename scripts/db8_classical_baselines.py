"""Baseline classiche di Zanghieri et al. (COINS 2021, [Z21] Tab. I) su NinaPro DB8: passo 1 del piano di diagnosi della replica di TEMPONet.

Piano e cancelli: `docs/temponet_db8_diagnosis_plan.md`, voce del 09/11/2026 in `docs/decisioni.md`. Stessa catena della TCN (cache verificata
per sha256, DoA di Krasoulis eq. S2, A1/A2 per training e validazione, A3 per il test, EMA con alpha dalla validazione, metriche di
`wearusfm.harness.db8_regression`). Modelli di [Z21] sez. II-D2:
- feature: Waveform Length (WL) per canale su finestre di 60 ms (120 campioni), bersaglio all'ultimo campione, 16 feature scalate a varianza
  unitaria sui soli dati di training;
- **SVR** RBF (`gamma="scale"`), una per DoA, finestre a passo 100 ms, C per DoA scelto sulla griglia 0,1-1-10-100 con la MAE di validazione
  dopo EMA (pareggi al C piu' piccolo);
- **MLP** a 3 strati nascosti (100, 100, 100: taglie non dichiarate da [Z21], scelta mia, descrittivo), finestre a passo 25 ms, uscita sulle 5 DoA.
Protocolli: `fold1` (training A1, validazione A2), `fold2` (il contrario) e `full` (training A1 + A2; C e alpha dalle predizioni fuori fold).

Uso (CPU, `scripts/slurm/db8_classical_baselines.sbatch`):
  python3 scripts/db8_classical_baselines.py --cache <cache> --out <cartella NUOVA> --workers 8
  python3 scripts/db8_classical_baselines.py --summarize <cartella del run>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from wearusfm.harness import db8_regression as db8

WL_WINDOW = 120  # 60 ms a 2 kHz
SVR_STEP = 200  # 100 ms
MLP_STEP = 50  # 25 ms
C_GRID = (0.1, 1.0, 10.0, 100.0)
MLP_HIDDEN = (100, 100, 100)
MLP_MAX_ITER = 200
FOLDS = {"fold1": (1, 2), "fold2": (2, 1)}  # (training, validazione)
PROTOCOLS = ("fold1", "fold2", "full")
REFERENCE = {"svr": {"mae_deg": 7.28, "acc_10deg": 0.795, "acc_15deg": 0.883},
             "mlp": {"mae_deg": 7.14, "acc_10deg": 0.799, "acc_15deg": 0.889}}
SVR_REFERENCE, TOLERANCE = 7.28, 0.30  # cancello del passo 1 (decisioni.md, 09/11/2026)
GROUPS = {"normodotati": tuple(range(1, 11)), "amputati": (11, 12)}


def wl_features(emg: np.ndarray, ends: np.ndarray, window: int = WL_WINDOW) -> np.ndarray:
    """WL per canale sulle finestre di `window` campioni che finiscono in `ends`: somma di |x[i] - x[i-1]| dentro la finestra."""
    ends = np.asarray(ends, dtype=np.int64)
    if len(ends) and (ends.min() < window - 1 or ends.max() >= emg.shape[0]):
        raise ValueError("finestre fuori dalla registrazione")
    d = np.abs(np.diff(np.asarray(emg, dtype=np.float64), axis=0))
    cs = np.concatenate([np.zeros((1, d.shape[1])), np.cumsum(d, axis=0)])
    return cs[ends] - cs[ends - (window - 1)]


def windows(rec: db8.Recording, step: int) -> tuple[np.ndarray, np.ndarray]:
    ends = np.arange(WL_WINDOW - 1, rec.n_samples, step, dtype=np.int64)
    return wl_features(rec.emg, ends), rec.doa[ends].astype(np.float64)


class Deadline:
    def __init__(self, t_end: float):
        self.t_end = t_end

    def check(self, what: str) -> None:
        if time.time() > self.t_end:
            raise TimeoutError(f"tempo esaurito prima di: {what}")


def _scaled(train: np.ndarray, *others: np.ndarray):
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(train)
    return (sc.transform(train), *(sc.transform(o) for o in others))


def _entry(val, test, alpha, extra) -> dict:
    out = dict(extra)
    out["alpha"] = [float(a) for a in alpha]
    if val is not None:
        out["val_raw"] = db8.evaluate(*val)
        out["val_ema"] = db8.evaluate(db8.ema(val[0], alpha), val[1])
    out["test_raw"] = db8.evaluate(*test)
    out["test_ema"] = db8.evaluate(db8.ema(test[0], alpha), test[1])
    return out


def svr_subject(data: dict, deadline: Deadline) -> dict:
    from sklearn.svm import SVR

    nd, res, cv = db8.N_DOA, {}, {}
    Xte, Yte = data[db8.TEST_ACQUISITION]["svr"]
    for fold, (tr, va) in FOLDS.items():
        (Xtr, Ytr), (Xva, Yva) = data[tr]["svr"], data[va]["svr"]
        Xtr_s, Xva_s, Xte_s = _scaled(Xtr, Xva, Xte)
        p_va, p_te = np.zeros((len(C_GRID), len(Yva), nd)), np.zeros((len(C_GRID), len(Yte), nd))
        fit_s = np.zeros((len(C_GRID), nd))
        n_sv = np.zeros((len(C_GRID), nd), dtype=np.int64)
        for i, c in enumerate(C_GRID):
            for d in range(nd):
                deadline.check(f"SVR {fold} C={c} DoA {d + 1}")
                t0 = time.time()
                m = SVR(kernel="rbf", C=c, gamma="scale", cache_size=1000).fit(Xtr_s, Ytr[:, d])
                p_va[i, :, d], p_te[i, :, d] = m.predict(Xva_s), m.predict(Xte_s)
                fit_s[i, d], n_sv[i, d] = time.time() - t0, len(m.support_)
        alpha, vmae = np.zeros((len(C_GRID), nd)), np.zeros((len(C_GRID), nd))
        for i in range(len(C_GRID)):
            a, grid_mae = db8.tune_alpha([(p_va[i], Yva)])
            alpha[i], vmae[i] = a, grid_mae.min(axis=0)
        best = vmae.argmin(axis=0)  # primo minimo: C piu' piccolo a parita'
        cols = np.arange(nd)
        sel_va, sel_te = p_va[best, :, cols].T, p_te[best, :, cols].T
        res[fold] = _entry((sel_va, Yva), (sel_te, Yte), alpha[best, cols],
                           {"C": [C_GRID[b] for b in best], "n_train": int(len(Ytr)), "fit_seconds": float(fit_s.sum()),
                            "val_mae_by_C": vmae.tolist(), "support_vectors_fraction": (n_sv[best, cols] / len(Ytr)).tolist()})
        cv[fold] = (vmae, p_va, Yva)
    # full: C per DoA dalla media dei due fold, alpha dalle predizioni fuori fold con quel C, poi training su A1 + A2
    best = ((cv["fold1"][0] + cv["fold2"][0]) / 2).argmin(axis=0)
    cols = np.arange(nd)
    oof = [(cv[f][1][best, :, cols].T, cv[f][2]) for f in FOLDS]
    alpha_full, _ = db8.tune_alpha(oof)
    X12 = np.concatenate([data[1]["svr"][0], data[2]["svr"][0]])
    Y12 = np.concatenate([data[1]["svr"][1], data[2]["svr"][1]])
    X12_s, Xte_s = _scaled(X12, Xte)
    p_te, t0 = np.zeros((len(Yte), nd)), time.time()
    for d in range(nd):
        deadline.check(f"SVR full DoA {d + 1}")
        p_te[:, d] = SVR(kernel="rbf", C=C_GRID[best[d]], gamma="scale", cache_size=1000).fit(X12_s, Y12[:, d]).predict(Xte_s)
    res["full"] = _entry(None, (p_te, Yte), alpha_full,
                         {"C": [C_GRID[b] for b in best], "n_train": int(len(Y12)), "fit_seconds": float(time.time() - t0)})
    return res


def mlp_subject(data: dict, deadline: Deadline, seed: int = 0) -> dict:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor

    def fit(X, Y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return MLPRegressor(hidden_layer_sizes=MLP_HIDDEN, random_state=seed, max_iter=MLP_MAX_ITER).fit(X, Y)

    res, oof = {}, []
    Xte, Yte = data[db8.TEST_ACQUISITION]["mlp"]
    for fold, (tr, va) in FOLDS.items():
        (Xtr, Ytr), (Xva, Yva) = data[tr]["mlp"], data[va]["mlp"]
        Xtr_s, Xva_s, Xte_s = _scaled(Xtr, Xva, Xte)
        deadline.check(f"MLP {fold}")
        t0 = time.time()
        m = fit(Xtr_s, Ytr)
        p_va, p_te = m.predict(Xva_s), m.predict(Xte_s)
        alpha, _ = db8.tune_alpha([(p_va, Yva)])
        res[fold] = _entry((p_va, Yva), (p_te, Yte), alpha, {"n_train": int(len(Ytr)), "fit_seconds": float(time.time() - t0),
                                                            "n_iter": int(m.n_iter_), "converged": bool(m.n_iter_ < MLP_MAX_ITER)})
        oof.append((p_va, Yva))
    alpha_full, _ = db8.tune_alpha(oof)
    X12 = np.concatenate([data[1]["mlp"][0], data[2]["mlp"][0]])
    Y12 = np.concatenate([data[1]["mlp"][1], data[2]["mlp"][1]])
    X12_s, Xte_s = _scaled(X12, Xte)
    deadline.check("MLP full")
    t0 = time.time()
    m = fit(X12_s, Y12)
    res["full"] = _entry(None, (m.predict(Xte_s), Yte), alpha_full, {"n_train": int(len(Y12)), "fit_seconds": float(time.time() - t0),
                                                                      "n_iter": int(m.n_iter_), "converged": bool(m.n_iter_ < MLP_MAX_ITER)})
    return res


def run_subject(job: tuple[int, dict]) -> tuple[int, str]:
    subject, cfg = job
    out = Path(cfg["out"]) / "units"
    try:
        import temponet_db8  # stessa verifica sha256 della cache della TCN

        deadline = Deadline(cfg["deadline_epoch"])
        t0 = time.time()
        provenance = temponet_db8.check_cache(Path(cfg["cache"]), subject)
        data = {}
        for a in db8.ACQUISITIONS:
            rec = db8.load_cached(cfg["cache"], subject, a)
            data[a] = {"svr": windows(rec, SVR_STEP), "mlp": windows(rec, MLP_STEP)}
        result = {"subject": subject, "provenance": provenance, "svr": svr_subject(data, deadline)}
        result["mlp"] = mlp_subject(data, deadline, cfg["mlp_seed"])
        result["seconds"] = round(time.time() - t0, 1)
        (out / f"S{subject}.json").write_text(json.dumps(result, indent=1))
        print(f"S{subject}: SVR fold {np.mean([result['svr'][f]['test_ema']['mae_deg'] for f in FOLDS]):.3f}, "
              f"full {result['svr']['full']['test_ema']['mae_deg']:.3f}; MLP fold "
              f"{np.mean([result['mlp'][f]['test_ema']['mae_deg'] for f in FOLDS]):.3f}, {result['seconds']:.0f} s", flush=True)
        return subject, "ok"
    except Exception:  # registrato; il job esce con errore, nessun nuovo tentativo
        (out / f"S{subject}.error").write_text(traceback.format_exc())
        return subject, "errore"


def gate(svr_fold: float, svr_full: float, ref: float = SVR_REFERENCE, tol: float = TOLERANCE) -> str:
    """Cancello del passo 1 (decisioni.md, 09/11/2026)."""
    if abs(svr_fold - ref) <= tol:
        return "fold coerente"
    if svr_fold > ref + tol and abs(svr_full - ref) <= tol:
        return "indizio H1"
    if svr_fold > ref + tol and svr_full > ref + tol:
        return "indizio H2: stop alla TCN"
    return "non conclusivo"


def summarize(units: list[dict]) -> dict:
    subjects = sorted(u["subject"] for u in units)
    by_s = {u["subject"]: u for u in units}
    complete = subjects == list(range(1, db8.N_SUBJECTS + 1))
    out = {"complete": complete, "subjects": subjects, "reference": REFERENCE, "models": {}}
    for model in ("svr", "mlp"):
        def mae(protocols, subj):
            return float(np.mean([by_s[s][model][p]["test_ema"]["mae_per_doa_deg"] for s in subj for p in protocols]))

        def stat(key, protocols, subj):
            return float(np.mean([by_s[s][model][p]["test_ema"][key] for s in subj for p in protocols]))

        m = {"fold_mae_deg": mae(FOLDS, subjects), "full_mae_deg": mae(["full"], subjects),
             "raw_fold_mae_deg": float(np.mean([by_s[s][model][p]["test_raw"]["mae_deg"] for s in subjects for p in FOLDS])),
             "per_fold_mae_deg": {p: mae([p], subjects) for p in FOLDS},
             "acc_10deg_fold": stat("acc_10deg", FOLDS, subjects), "acc_15deg_fold": stat("acc_15deg", FOLDS, subjects),
             "acc_10deg_full": stat("acc_10deg", ["full"], subjects), "acc_15deg_full": stat("acc_15deg", ["full"], subjects),
             "alpha_mean_fold": float(np.mean([by_s[s][model][p]["alpha"] for s in subjects for p in FOLDS])),
             "per_subject_fold_mae_deg": {str(s): mae(FOLDS, [s]) for s in subjects},
             "per_subject_full_mae_deg": {str(s): mae(["full"], [s]) for s in subjects},
             "groups": {g: {"subjects": [s for s in gs if s in by_s], "fold_mae_deg": mae(FOLDS, [s for s in gs if s in by_s]),
                            "full_mae_deg": mae(["full"], [s for s in gs if s in by_s])}
                        for g, gs in GROUPS.items() if any(s in by_s for s in gs)}}
        if model == "svr":
            m["C_counts_fold"] = {str(c): int(sum(cc == c for s in subjects for p in FOLDS for cc in by_s[s]["svr"][p]["C"])) for c in C_GRID}
        if model == "mlp":
            m["converged_fraction"] = float(np.mean([by_s[s]["mlp"][p]["converged"] for s in subjects for p in PROTOCOLS]))
        out["models"][model] = m
    svr = out["models"]["svr"]
    out["gate"] = gate(svr["fold_mae_deg"], svr["full_mae_deg"]) if complete else "incompleto"
    return out


def to_markdown(s: dict) -> str:
    svr, mlp = s["models"]["svr"], s["models"]["mlp"]
    lines = ["# Baseline classiche su NinaPro DB8 (passo 1 della diagnosi di TEMPONet)", "",
             f"- Cancello (SVR, riferimento 7,28 ± 0,30): **{s['gate']}** (fold {svr['fold_mae_deg']:.3f}, full {svr['full_mae_deg']:.3f}).",
             "", "| Modello | fold con EMA | senza EMA | full con EMA | acc. 10 (fold) | acc. 15 (fold) | alpha (fold) | pubblicato |",
             "|---|---|---|---|---|---|---|---|"]
    for name, m in (("SVR", svr), ("MLP", mlp)):
        ref = REFERENCE[name.lower()]
        lines.append(f"| {name} | {m['fold_mae_deg']:.3f} | {m['raw_fold_mae_deg']:.3f} | {m['full_mae_deg']:.3f} | {m['acc_10deg_fold']:.3f} "
                     f"| {m['acc_15deg_fold']:.3f} | {m['alpha_mean_fold']:.3f} | {ref['mae_deg']:.2f} ({ref['acc_10deg']:.3f}, {ref['acc_15deg']:.3f}) |")
    lines += ["", "| Gruppo | SVR fold | SVR full | MLP fold | MLP full |", "|---|---|---|---|---|"]
    for g in svr["groups"]:
        lines.append(f"| {g} | {svr['groups'][g]['fold_mae_deg']:.3f} | {svr['groups'][g]['full_mae_deg']:.3f} "
                     f"| {mlp['groups'][g]['fold_mae_deg']:.3f} | {mlp['groups'][g]['full_mae_deg']:.3f} |")
    lines += ["", "| Soggetto | SVR fold | SVR full | MLP fold | MLP full |", "|---|---|---|---|---|"]
    for k in svr["per_subject_fold_mae_deg"]:
        tag = " (amputato)" if int(k) in GROUPS["amputati"] else ""
        lines.append(f"| {k}{tag} | {svr['per_subject_fold_mae_deg'][k]:.3f} | {svr['per_subject_full_mae_deg'][k]:.3f} "
                     f"| {mlp['per_subject_fold_mae_deg'][k]:.3f} | {mlp['per_subject_full_mae_deg'][k]:.3f} |")
    lines += ["", f"C scelti (fold): {svr['C_counts_fold']}; MLP arrivati a convergenza: {mlp['converged_fraction']:.2f}."]
    return "\n".join(lines) + "\n"


def write_summary(run_dir: Path) -> dict:
    units = [json.loads(p.read_text()) for p in sorted((run_dir / "units").glob("S*.json"))]
    s = summarize(units)
    (run_dir / "summary.json").write_text(json.dumps(s, indent=1))
    (run_dir / "summary.md").write_text(to_markdown(s))
    print(to_markdown(s))
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache")
    ap.add_argument("--out", help="cartella nuova o vuota")
    ap.add_argument("--summarize", help="rifa' solo la sintesi di un run esistente")
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, db8.N_SUBJECTS + 1)))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mlp-seed", type=int, default=0)
    ap.add_argument("--time-budget-s", type=float, default=0.0, help="nessun fit nuovo dopo questo tempo (0 = nessun limite)")
    args = ap.parse_args(argv)
    if args.summarize:
        write_summary(Path(args.summarize))
        return 0
    if not args.cache or not args.out:
        ap.error("servono --cache e --out (oppure --summarize)")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"{out} non e' vuota: serve una cartella nuova", file=sys.stderr)
        return 2
    (out / "units").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg = {"cache": args.cache, "out": str(out), "mlp_seed": args.mlp_seed,
           "deadline_epoch": (t0 + args.time_budget_s) if args.time_budget_s > 0 else float("inf")}
    import sklearn

    run = {"kind": "db8_classical_baselines_v1", "args": vars(args), "code_sha": os.environ.get("WEARUSFM_CODE_SHA"),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "python": platform.python_version(), "numpy": np.__version__,
           "sklearn": sklearn.__version__, "C_grid": list(C_GRID), "mlp_hidden": list(MLP_HIDDEN), "wl_window": WL_WINDOW,
           "svr_step": SVR_STEP, "mlp_step": MLP_STEP, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / "run.json").write_text(json.dumps(run, indent=1))
    jobs = [(s, cfg) for s in args.subjects]
    if args.workers <= 1:
        status = [run_subject(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
            status = list(pool.map(run_subject, jobs))
    run["status"] = {f"S{s}": st for s, st in status}
    run["seconds"] = round(time.time() - t0, 1)
    (out / "run.json").write_text(json.dumps(run, indent=1))
    bad = [k for k, v in run["status"].items() if v != "ok"]
    if not bad:
        write_summary(out)
    print(f"fine: {len(status) - len(bad)} soggetti su {len(status)} in {run['seconds']:.0f} s" + (f"; non riusciti: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
