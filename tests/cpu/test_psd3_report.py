"""`scripts/psd3_report.py`: rapporto descrittivo di PSD-3 da shard sintetici con la struttura dei rapporti reali (29/10)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("psd3_report", ROOT / "scripts" / "psd3_report.py")
PR = importlib.util.module_from_spec(spec)
sys.modules["psd3_report"] = PR
spec.loader.exec_module(PR)


def _fold(i, selected_c=0.1, limit_c=None):
    audit = []
    for c in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
        audit.append({"C": c, "phase": "candidate", "n_iter": [12000 if c == limit_c else 100 + int(10 * c)], "duration_s": 1.0,
                      "reached_iteration_limit": c == limit_c, "val_bacc": 0.5 if c == selected_c else 0.4, "warnings": []})
    audit.append({"C": selected_c, "phase": "refit", "n_iter": [120], "duration_s": 1.5, "reached_iteration_limit": False, "warnings": []})
    mlp_audit = [{"lambda": lam, "seed": s, "duration_s": 2.0, "convergence_claim": False, "scores": {"train": 0.7, "val": 0.45},
                  "epochs": [{"epoch": e, "train_loss": 2.0 / e} for e in range(1, 61)]} for lam in (0.0, 1e-3, 1e-2) for s in (0, 1)]
    return {"fold": i, "roles": {"train": ["s1"], "val": ["s2"], "held": ["s3"]}, "dropped_windows": {"val": 0, "held": 0},
            "preprocessing_seconds": 0.5,
            "preprocessing": {"config": {"k": 128}, "variance_explained": 0.99, "explained_variance_ratio": [0.5] + [0.5 / 127] * 127,
                              "input_shapes": [[10, 768], [4, 768], [4, 768]]},
            "logistic": {"audit": audit, "candidates": [{"C": a["C"], "val_bacc": a["val_bacc"]} for a in audit[:-1]],
                         "selected_hyperparameter": selected_c, "selected_val_score": 0.5, "metric_warnings": []},
            "mlp": {"audit": mlp_audit, "candidates": [], "selected_hyperparameter": 1e-2, "selected_val_score": 0.45,
                    "scores": {"0": {"train": 0.7, "val": 0.45}, "1": {"train": 0.72, "val": 0.43}}, "stopping_reason": "fixed_epoch_budget_not_convergence",
                    "metric_warnings": []}}


def _shard(model, dataset, view, *, flip=0, limit_c=None):
    y = [0, 0, 1, 1, 2, 2, 0, 1]
    logistic = list(y)
    logistic[0] = 1 if flip else 0
    mlp0 = list(y); mlp0[2] = 0
    return {"protocol": "psd3-frozen-readers-20261029-v1", "status": "completed", "model": model, "dataset": dataset, "view": view, "task": "p1",
            "recipe_sha256": "r" * 64, "plan_sha256": "p" * 64, "checkpoint_sha256": "c" * 64, "rows": list(range(8)), "y": y,
            "subjects": ["s3", "s3", "s4", "s4", "s5", "s5", "s3", "s4"],
            "predictions": {"logistic": {"0": logistic}, "mlp": {"0": mlp0, "1": list(y)}},
            "runtime": {"threads": 4, "torch_threads": 4, "code_sha": "a" * 40},
            "folds": [_fold(0, limit_c=limit_c), _fold(1, selected_c=1.0)]}


def test_report_scores_selection_limits_and_completeness(tmp_path):
    shards = [_shard("B", "ninapro_db2", "pool"), _shard("B", "ninapro_db2", "dec", flip=1, limit_c=100.0)]
    report = PR.build_report(shards)
    assert report["n_shards"] == 2 and report["complete"] is False and len(report["missing_cells"]) == 18 and report["single_recipe"]
    pool = report["cells"]["B/ninapro_db2/pool"]
    assert pool["bacc_oof"]["logistic"]["mean"] == pytest.approx(1.0)
    assert pool["bacc_oof"]["mlp"]["by_seed"]["1"] == pytest.approx(1.0) and pool["bacc_oof"]["mlp"]["mean"] < 1.0
    assert report["cells"]["B/ninapro_db2/dec"]["bacc_oof"]["logistic"]["mean"] < 1.0
    conv = report["convergence"]
    assert conv["candidate_fits"] == 24 and conv["candidates_reached_limit"] == 1 and conv["selected_reached_limit"] == 0
    assert conv["limit_cells"] == {"B/ninapro_db2/dec": 1} and conv["max_n_iter_by_C"]["100"] == 12000 and conv["max_n_iter_by_C"]["0.001"] == 100
    assert conv["selected_C_histogram"] == {"0.1": 2, "1": 2} and conv["selected_lambda_histogram"] == {"0.01": 4}
    assert conv["mlp_convergence_claims"] == 0 and conv["mlp_train_minus_val_mean"] == pytest.approx(0.27)
    fold = pool["folds"][0]
    assert fold["logistic"]["refit"]["C"] == 0.1 and fold["pca"]["cumulative_at"]["16"] == pytest.approx(0.5 + 15 * 0.5 / 127)
    assert fold["mlp"]["runs"][0]["train_loss"] == {"1": 2.0, "10": 0.2, "30": pytest.approx(2.0 / 30), "60": pytest.approx(2.0 / 60)}
    # cella duplicata rifiutata; markdown con le tabelle principali
    with pytest.raises(ValueError, match="duplicata"):
        PR.build_report(shards + [shards[0]])
    md = PR.markdown(report)
    assert "INCOMPLETO" in md and "| ninapro_db2 | B/pool | 100.00 |" in md and "Celle con fit al limite" in md
    # CLI: scrive JSON e Markdown
    paths = []
    for s in shards:
        p = tmp_path / f"{s['model']}_{s['dataset']}_{s['view']}.json"; p.write_text(json.dumps(s)); paths.append(str(p))
    PR.main(["--shards", *paths, "--out-json", str(tmp_path / "r.json"), "--out-md", str(tmp_path / "r.md")])
    assert json.loads((tmp_path / "r.json").read_text())["n_shards"] == 2 and (tmp_path / "r.md").read_text().startswith("# PSD-3")


def test_report_with_compare_requires_same_recipe_and_cross_checks_points():
    shards = [_shard(m, d, v) for m in PR.MODELS for d in PR.DATASETS for v in PR.VIEWS]
    # MLP: seme 0 sbaglia una finestra della classe 1 su tre -> (1 + 2/3 + 1)/3 = 8/9; seme 1 perfetto; media 17/18
    cells = {f"{m}/{v}/{r}": {"bacc": 1.0 if r == "logistic" else 17 / 18, "ci95": [0.9, 1.0], "subject_bootstrap_se": 0.01,
                              "by_readout_seed": {}} for m in PR.MODELS for v in PR.VIEWS for r in PR.READERS}
    contrasts = {"dec_minus_pool/B/logistic": {"delta_pp": 0.0, "ci95_pp": [-1.0, 1.0], "paired_subject_se_pp": 0.5, "coefficients": {}}}
    compare = {"recipe_sha256": "r" * 64, "n_boot": 2000, "bootstrap_seed": 20261029, "units": {"contrasts": "percentage points"},
               "datasets": {d: {"cells": cells, "contrasts": contrasts} for d in PR.DATASETS},
               "aggregates": {t: {"weights": {}, "cells": cells, "contrasts": contrasts} for t in ("p1", "p2")}}
    report = PR.build_report(shards, compare)
    assert report["complete"] and report["compare"]["cross_check_max_abs_diff"] == pytest.approx(0.0, abs=1e-12)
    assert "Confronto appaiato" in PR.markdown(report) and "| dec_minus_pool/B/logistic | +0.00 | [-1.00; +1.00] |" in PR.markdown(report)
    with pytest.raises(ValueError, match="ricetta"):
        PR.build_report(shards, {**compare, "recipe_sha256": "x" * 64})
