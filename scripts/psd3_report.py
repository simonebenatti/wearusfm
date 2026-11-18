#!/usr/bin/env python3
"""Rapporto descrittivo di PSD-3 dagli shard e dal confronto di GPT (nessun fit, nessuna soglia, nessun cancello).

Legge i rapporti per cella (`<modello>_<dataset>_<vista>.json`, prodotti da `scripts/psd3_readout.py run`) e, se c'e', il
`compare.json` della fase `compare` (bootstrap per soggetto di `harness/psd3_compare.py`). Produce un JSON con, per cella e fold:
BAcc OOF per lettore e seme, C scelto e iterazioni per candidato (al limite o no), lambda scelta della MLP con punteggi train/val per
seme e andamento della loss, varianza spiegata dalla PCA, durate; piu' le tabelle in Markdown. I contrasti con IC vengono SOLO dal
`compare.json` (codice congelato di GPT): qui si ricalcola il solo punto OOF, come controllo incrociato.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

MODELS = ("B", "R3c")
VIEWS = ("pool", "dec")
READERS = ("logistic", "mlp")
DATASETS = ("ninapro_db2", "ninapro_db3", "ninapro_db6", "epn612", "uci_emg")
TASKS = {"p1": ("ninapro_db2", "ninapro_db3", "ninapro_db6"), "p2": ("epn612", "uci_emg")}
CURVE_EPOCHS = (1, 10, 30, 60)


def _scalar(value):
    if isinstance(value, list):
        return _scalar(value[0]) if value else None
    return value


def fold_summary(fold: dict) -> dict:
    lg, ml, pp = fold["logistic"], fold["mlp"], fold["preprocessing"]
    candidates = {}
    refit = None
    for rec in lg["audit"]:
        item = {"n_iter": int(_scalar(rec["n_iter"])), "reached_limit": bool(rec["reached_iteration_limit"]),
                "duration_s": float(rec["duration_s"]), "warnings": len(rec.get("warnings") or [])}
        if rec.get("phase") == "refit":
            item["C"] = float(rec["C"])
            refit = item
        else:
            item["val_bacc"] = rec.get("val_bacc")
            candidates[f"{float(rec['C']):g}"] = item
    selected_c = float(lg["selected_hyperparameter"])
    mlp_runs = []
    for rec in ml["audit"]:
        epochs = rec.get("epochs") or []
        by_epoch = {int(e["epoch"]): e for e in epochs}
        mlp_runs.append({"lambda": rec.get("lambda"), "seed": rec.get("seed"), "duration_s": float(rec["duration_s"]),
                         "convergence_claim": rec.get("convergence_claim"), "n_epochs": len(epochs),
                         "train_loss": {str(k): by_epoch[k]["train_loss"] for k in CURVE_EPOCHS if k in by_epoch},
                         "val_bacc": rec.get("val_bacc"), "scores": rec.get("scores")})
    scores = {str(seed): dict(v) for seed, v in ml["scores"].items()}
    gaps = {seed: v["train"] - v["val"] for seed, v in scores.items() if "train" in v and "val" in v}
    ratio = pp.get("explained_variance_ratio") or []
    cumulative = np.cumsum(ratio).tolist() if ratio else []
    return {
        "fold": fold["fold"], "roles": fold["roles"], "dropped_windows": fold.get("dropped_windows"),
        "logistic": {"selected_C": selected_c, "selected_val_bacc": lg["selected_val_score"], "candidates": candidates, "refit": refit,
                     "candidates_reached_limit": sum(c["reached_limit"] for c in candidates.values()),
                     "selected_reached_limit": bool(candidates.get(f"{selected_c:g}", {}).get("reached_limit")),
                     "refit_reached_limit": bool(refit and refit["reached_limit"]),
                     "metric_warnings": len(lg.get("metric_warnings") or []),
                     "duration_s": float(sum(float(r["duration_s"]) for r in lg["audit"]))},
        "mlp": {"selected_lambda": ml["selected_hyperparameter"], "selected_val_bacc": ml["selected_val_score"], "scores": scores,
                "train_minus_val": gaps, "stopping_reason": ml.get("stopping_reason"), "candidates": ml["candidates"], "runs": mlp_runs,
                "metric_warnings": len(ml.get("metric_warnings") or []),
                "duration_s": float(sum(float(r["duration_s"]) for r in ml["audit"]))},
        "pca": {"k": pp["config"]["k"], "variance_explained": pp["variance_explained"],
                "cumulative_at": {str(k): cumulative[k - 1] for k in (16, 32, 64, 128) if len(cumulative) >= k},
                "input_shapes": pp.get("input_shapes"), "seconds": fold.get("preprocessing_seconds")},
    }


def shard_summary(shard: dict) -> dict:
    y = np.asarray(shard["y"])
    bacc = {}
    for reader, by_seed in shard["predictions"].items():
        per_seed = {str(seed): float(balanced_accuracy_score(y, np.asarray(pred))) for seed, pred in by_seed.items()}
        bacc[reader] = {"by_seed": per_seed, "mean": float(np.mean(list(per_seed.values())))}
    folds = [fold_summary(f) for f in shard["folds"]]
    runtime = shard.get("runtime") or {}
    return {
        "model": shard["model"], "dataset": shard["dataset"], "view": shard["view"], "task": shard.get("task"), "status": shard["status"],
        "n_oof": int(len(y)), "n_classes": int(len(set(y.tolist()))), "n_subjects": int(len(set(shard["subjects"]))),
        "recipe_sha256": shard["recipe_sha256"], "plan_sha256": shard.get("plan_sha256"), "checkpoint_sha256": shard["checkpoint_sha256"],
        "threads": runtime.get("threads"), "torch_threads": runtime.get("torch_threads"), "code_sha": runtime.get("code_sha"),
        "bacc_oof": bacc, "folds": folds,
        "totals": {"logistic_s": sum(f["logistic"]["duration_s"] for f in folds), "mlp_s": sum(f["mlp"]["duration_s"] for f in folds),
                   "pca_s": sum(f["pca"]["seconds"] or 0.0 for f in folds),
                   "candidates_reached_limit": sum(f["logistic"]["candidates_reached_limit"] for f in folds),
                   "selected_reached_limit": sum(f["logistic"]["selected_reached_limit"] for f in folds),
                   "refit_reached_limit": sum(f["logistic"]["refit_reached_limit"] for f in folds),
                   "selected_C": [f["logistic"]["selected_C"] for f in folds],
                   "selected_lambda": [f["mlp"]["selected_lambda"] for f in folds],
                   "variance_explained": [f["pca"]["variance_explained"] for f in folds]},
    }


def build_report(shards: list[dict], compare: dict | None = None) -> dict:
    cells = {}
    for shard in shards:
        key = f"{shard['model']}/{shard['dataset']}/{shard['view']}"
        if key in cells:
            raise ValueError(f"cella duplicata: {key}")
        cells[key] = shard_summary(shard)
    expected = {f"{m}/{d}/{v}" for m in MODELS for d in DATASETS for v in VIEWS}
    missing = sorted(expected - set(cells))
    recipes = sorted({c["recipe_sha256"] for c in cells.values()})
    threads = sorted({json.dumps([c["threads"], c["torch_threads"]], sort_keys=True) for c in cells.values()})
    convergence = {
        "candidate_fits": sum(len(f["logistic"]["candidates"]) for c in cells.values() for f in c["folds"]),
        "candidates_reached_limit": sum(c["totals"]["candidates_reached_limit"] for c in cells.values()),
        "selected_reached_limit": sum(c["totals"]["selected_reached_limit"] for c in cells.values()),
        "refit_reached_limit": sum(c["totals"]["refit_reached_limit"] for c in cells.values()),
        "limit_cells": {k: c["totals"]["candidates_reached_limit"] for k, c in cells.items() if c["totals"]["candidates_reached_limit"]},
        "max_n_iter_by_C": {},
        "selected_C_histogram": dict(Counter(f"{x:g}" for c in cells.values() for x in c["totals"]["selected_C"])),
        "selected_lambda_histogram": dict(Counter(str(x) for c in cells.values() for x in c["totals"]["selected_lambda"])),
        "mlp_convergence_claims": sum(bool(r["convergence_claim"]) for c in cells.values() for f in c["folds"] for r in f["mlp"]["runs"]),
        "mlp_train_minus_val_mean": float(np.mean([g for c in cells.values() for f in c["folds"] for g in f["mlp"]["train_minus_val"].values()]))
        if cells else None,
    }
    by_c = {}
    for c in cells.values():
        for f in c["folds"]:
            for cc, item in f["logistic"]["candidates"].items():
                by_c[cc] = max(by_c.get(cc, 0), item["n_iter"])
    convergence["max_n_iter_by_C"] = dict(sorted(by_c.items(), key=lambda kv: float(kv[0])))
    pca = {k: {"variance_explained_by_fold": c["totals"]["variance_explained"]} for k, c in cells.items()}
    costs = {k: {"logistic_s": round(c["totals"]["logistic_s"], 1), "mlp_s": round(c["totals"]["mlp_s"], 1), "pca_s": round(c["totals"]["pca_s"], 1)}
             for k, c in cells.items()}
    report = {"protocol": sorted({s["protocol"] for s in shards}), "n_shards": len(cells), "complete": not missing, "missing_cells": missing,
              "recipe_sha256": recipes, "single_recipe": len(recipes) == 1, "threads": threads, "cells": cells,
              "convergence": convergence, "pca": pca, "costs_s": costs, "compare": None}
    if compare is not None:
        if compare.get("recipe_sha256") not in recipes:
            raise ValueError("compare.json ha una ricetta diversa dagli shard")
        report["compare"] = {"n_boot": compare["n_boot"], "bootstrap_seed": compare["bootstrap_seed"], "units": compare["units"],
                             "datasets": compare["datasets"], "aggregates": compare["aggregates"], "uncertainty": compare.get("uncertainty")}
        check = cross_check(report, compare)
        report["compare"]["cross_check_max_abs_diff"] = check
    return report


def cross_check(report: dict, compare: dict) -> float:
    """Il punto OOF ricalcolato qui deve coincidere con `bacc` delle celle del confronto (stesse predizioni, stessa metrica)."""
    worst = 0.0
    for ds, block in compare["datasets"].items():
        for cell, item in block["cells"].items():
            model, view, reader = cell.split("/")
            mine = report["cells"].get(f"{model}/{ds}/{view}")
            if mine is None:
                continue
            worst = max(worst, abs(mine["bacc_oof"][reader]["mean"] - item["bacc"]))
    return worst


def _pct(x):
    return "n.d." if x is None else f"{100 * x:.2f}"


def markdown(report: dict) -> str:
    lines = ["# PSD-3: tabelle descrittive", "",
             f"Shard: {report['n_shards']}/20 ({'completo' if report['complete'] else 'INCOMPLETO: ' + ', '.join(report['missing_cells'])}); "
             f"ricetta unica: {report['single_recipe']}; thread (BLAS, torch): {report['threads']}.", ""]
    lines += ["## BAcc OOF per dataset e cella (punto ricalcolato dalle predizioni; MLP = media dei due semi, fra parentesi i semi)", "",
              "| dataset | cella | logistica | MLP |", "|---|---|---:|---:|"]
    for ds in DATASETS:
        for m in MODELS:
            for v in VIEWS:
                c = report["cells"].get(f"{m}/{ds}/{v}")
                if not c:
                    continue
                lg = c["bacc_oof"].get("logistic", {}); ml = c["bacc_oof"].get("mlp", {})
                seeds = ", ".join(_pct(x) for x in ml.get("by_seed", {}).values())
                lines.append(f"| {ds} | {m}/{v} | {_pct(lg.get('mean'))} | {_pct(ml.get('mean'))} ({seeds}) |")
    conv = report["convergence"]
    lines += ["", "## Convergenza e selezione", "",
              f"- Fit candidati della logistica: {conv['candidate_fits']}; al limite di 12.000: {conv['candidates_reached_limit']} "
              f"(selezionati al limite: {conv['selected_reached_limit']}, riaddestramenti al limite: {conv['refit_reached_limit']}).",
              f"- Iterazioni massime per C: {conv['max_n_iter_by_C']}.",
              f"- C scelti (istogramma sui fold): {conv['selected_C_histogram']}; lambda MLP scelte: {conv['selected_lambda_histogram']}.",
              f"- MLP: nessuna pretesa di convergenza (budget fisso di 60 epoche); divario medio train − val (BAcc): "
              f"{_pct(conv['mlp_train_minus_val_mean'])} punti."]
    if conv["limit_cells"]:
        lines.append(f"- Celle con fit al limite: {conv['limit_cells']}.")
    lines += ["", "## Dettaglio per cella e fold: C scelto (iterazioni), lambda scelta, train/val MLP per seme, varianza PCA", "",
              "| cella | fold | C (iter.) | al limite | lambda | train/val seme 0 | train/val seme 1 | PCA128 var. |", "|---|---:|---|---:|---|---|---|---:|"]
    for key, c in report["cells"].items():
        for f in c["folds"]:
            lg, ml = f["logistic"], f["mlp"]
            it = lg["candidates"].get(f"{lg['selected_C']:g}", {}).get("n_iter")
            s0 = ml["scores"].get("0", {}); s1 = ml["scores"].get("1", {})
            lines.append(f"| {key} | {f['fold']} | {lg['selected_C']:g} ({it}) | {lg['candidates_reached_limit']} | {ml['selected_lambda']} | "
                         f"{_pct(s0.get('train'))}/{_pct(s0.get('val'))} | {_pct(s1.get('train'))}/{_pct(s1.get('val'))} | {_pct(f['pca']['variance_explained'])} |")
    lines += ["", "## Durate per cella (secondi di fit sommati sui quattro fold)", "", "| cella | logistica | MLP | PCA |", "|---|---:|---:|---:|"]
    for key, c in report["costs_s"].items():
        lines.append(f"| {key} | {c['logistic_s']:.0f} | {c['mlp_s']:.0f} | {c['pca_s']:.0f} |")
    if report["compare"]:
        cmp_ = report["compare"]
        lines += ["", f"## Confronto appaiato (bootstrap per soggetto, {cmp_['n_boot']} repliche, seme {cmp_['bootstrap_seed']}; "
                      f"scarto massimo fra punto ricalcolato e `bacc` del confronto: {cmp_['cross_check_max_abs_diff']:.2e})", ""]
        for task in ("p1", "p2"):
            agg = cmp_["aggregates"][task]
            lines += [f"### Aggregato {task.upper()} (pesi {agg['weights']})", "", "| cella | BAcc | IC95 |", "|---|---:|---|"]
            for cell, item in agg["cells"].items():
                lines.append(f"| {cell} | {_pct(item['bacc'])} | [{_pct(item['ci95'][0])}; {_pct(item['ci95'][1])}] |")
            lines += ["", "| contrasto | delta (punti) | IC95 |", "|---|---:|---|"]
            for name, item in agg["contrasts"].items():
                lines.append(f"| {name} | {item['delta_pp']:+.2f} | [{item['ci95_pp'][0]:+.2f}; {item['ci95_pp'][1]:+.2f}] |")
            lines.append("")
        lines += ["### Contrasti per dataset (delta in punti, IC95)", "", "| dataset | contrasto | delta | IC95 |", "|---|---|---:|---|"]
        for ds, block in cmp_["datasets"].items():
            for name, item in block["contrasts"].items():
                lines.append(f"| {ds} | {name} | {item['delta_pp']:+.2f} | [{item['ci95_pp'][0]:+.2f}; {item['ci95_pp'][1]:+.2f}] |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True, type=Path, help="rapporti per cella (JSON)")
    ap.add_argument("--compare", type=Path, help="compare.json della fase compare (facoltativo)")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    a = ap.parse_args(argv)
    shards = [json.loads(p.read_text()) for p in a.shards]
    compare = json.loads(a.compare.read_text()) if a.compare else None
    report = build_report(shards, compare)
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    a.out_md.write_text(markdown(report))
    print(json.dumps({"n_shards": report["n_shards"], "complete": report["complete"], "single_recipe": report["single_recipe"],
                      "candidates_reached_limit": report["convergence"]["candidates_reached_limit"], "out": str(a.out_md)}))


if __name__ == "__main__":
    main()
