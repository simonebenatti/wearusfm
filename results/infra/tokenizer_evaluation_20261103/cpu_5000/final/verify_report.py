"""Independent arithmetic/provenance audit of the completed teacher5k JSON.
Reads aggregate sufficient statistics only; no remote operation or wearable arrays.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def same(a, b):
    assert np.allclose(a, b, rtol=1e-12, atol=1e-14), (a, b)

report = json.loads((HERE / "report.json").read_text())
plan_path = ROOT / "results/infra/tokenizer_evaluation_20261103/pose_readout_plan_5000.json"
plan = json.loads(plan_path.read_text())
remote = {}
for line in (HERE / "remote_sha256.txt").read_text().splitlines():
    digest, path = line.split(maxsplit=1)
    remote[Path(path).name] = digest
assert sha(HERE / "report.json") == remote["report.json"]
assert report["complete"] is True and report["status"] == "completed"
assert report["step"] == 5000 and report["slurm_job_id"] == "63405791"
assert report["plan_sha256"] == sha(plan_path)
assert report["recipe"] == plan["recipe"]
assert report["recipe"]["offset"] == 6
assert report["input_pins"] == plan["caches"]
assert report["reference_pins"] == plan["references"]
assert report["runtime"]["code_sha"] == "70fd080149c11d10917059875d31a5a2820f176a"
assert report["runtime"]["sources"] == plan["sources"]
assert {p: sha(ROOT / p) for p in plan["sources"]} == plan["sources"]
assert report["runtime"]["torch_threads"] == 8
assert set(report["runtime"]["threads"].values()) == {"8"}
assert all(v["num_threads"] == 8 for v in report["runtime"]["threadpools"])

arms = ["T25-L", "T100-L", "T100-G"]
assert set(report["cells"]) == set(arms)
train = report["inspections"]["T25-L/train"]
val = report["inspections"]["T25-L/val"]
assert len(train["users"]) == 40 and len(val["users"]) == 15
assert set(train["users"]).isdisjoint(val["users"])
summary = {"status": "verified", "report_sha256": sha(HERE / "report.json"),
    "plan_sha256": sha(plan_path), "job_id": "63405791", "which": "teacher", "step": 5000,
    "screening_only": True, "crop": 6, "n_train_users": 40, "n_val_users": 15,
    "gpu_hours": 0.0, "cpu_hours": 134 * 8 / 3600,
    "elapsed_s": 134, "alloc_cpus": 8, "max_rss_kib": 4600864,
    "checks": ["complete report/job/three-arm matrix", "original report remote/local SHA256",
        "plan/runtime/source pins", "six cache/provenance/receipt input pins",
        "remote coefficient/prediction SHA256 equal report", "crop6 and paired target/support identity",
        "train/val users disjoint and fixed CV folds", "resolved alpha boundaries",
        "all per-user sums and scalar metrics", "independent 2000-user bootstrap and contrasts"],
    "cells": {}, "contrasts": {}}
for arm in arms:
    cell = report["cells"][arm]
    assert cell["complete"] and cell["arm"] == arm
    assert cell["checkpoint_step"] == 5000 and cell["which"] == "teacher"
    assert cell["checkpoint_sha256"] == plan["caches"][arm]["train"]["checkpoint_sha256"]
    selection = cell["fit"]["selection"]
    assert selection["boundary_unresolved"] is False
    assert selection["fold_users"] == plan["fold_users"]
    assert selection["alpha"] == min(selection["grid"], key=lambda v: (v["mse"], v["alpha"]))["alpha"]
    assert all(v <= 3 for v in selection["extensions"].values())
    for kind in ("coefficients", "predictions"):
        assert remote[Path(cell[kind]["path"]).name] == cell[kind]["sha256"]
    for split, ref in (("train", train), ("val", val)):
        inspection = report["inspections"][arm + "/" + split]
        assert inspection["users"] == ref["users"]
        assert inspection["targets"] == ref["targets"]
        assert inspection["cache_sha256"] == plan["caches"][arm][split]["sha256"]
    assert cell["fit"]["fit_support_sha256"] == train["targets"]["central_mask"]
    assert cell["analysis"]["support_sha256"] == val["targets"]["central_mask"]
    assert cell["analysis"]["targets_sha256"] == val["targets"]
    groups = {}
    for group, metrics in cell["analysis"]["groups"].items():
        centered = metrics["centered"]
        rows = centered["per_user"]
        for key, total in centered["sums"].items():
            same(sum(v["sums"][key] for v in rows.values()), total)
        sums = centered["sums"]
        ratio = sums["centered"] / sums["vary"]
        same(centered["centered_mse_ratio"], ratio)
        same(centered["variance_ratio"], sums["varp"] / sums["vary"])
        same(centered["pooled_centered_correlation"], sums["cov"] / math.sqrt(sums["varp"] * sums["vary"]))
        same(ratio, 1 + centered["variance_ratio"] - 2 * centered["pooled_centered_correlation"] * math.sqrt(centered["variance_ratio"]))
        groups[group] = {"E_centered": ratio,
            "E_ci95": report["paired"][group]["cell_ratio_ci95"][arm],
            "mae_rad": metrics["mae_rad"], "variance_ratio": centered["variance_ratio"],
            "correlation": centered["pooled_centered_correlation"],
            "n_selected_windows": metrics["n_selected_windows"],
            "n_supported_windows": centered["n_windows"],
            "n_valid_elements": centered["n_valid_elements"],
            "median_user_E_centered": float(np.median([v["sums"]["centered"] / v["sums"]["vary"] for v in rows.values()]))}
    summary["cells"][arm] = {"alpha": selection["alpha"], "alpha_extensions": selection["extensions"], "groups": groups}

for group in ("dynamic", "all", "static"):
    recorded = report["paired"][group]
    users = sorted(recorded["users"])
    assert users == val["users"]
    rows = {a: report["cells"][a]["analysis"]["groups"][group]["centered"]["per_user"] for a in arms}
    base = rows[arms[0]]
    for arm in arms:
        for user in users:
            for key in ("n_windows", "n_window_joints", "n_valid_elements"):
                assert rows[arm][user][key] == base[user][key]
            same(rows[arm][user]["sums"]["vary"], base[user]["sums"]["vary"])
    denominator = np.array([base[u]["sums"]["vary"] for u in users])
    draws = np.random.default_rng(20261102).integers(0, len(users), (2000, len(users)))
    den = denominator[draws].sum(axis=1)
    assert np.isfinite(den).all() and (den > 0).all()
    errors = {a: np.array([rows[a][u]["sums"]["centered"] for u in users]) for a in arms}
    for arm in arms:
        same(np.percentile(errors[arm][draws].sum(axis=1) / den, [2.5, 97.5]), recorded["cell_ratio_ci95"][arm])
    summary["contrasts"][group] = {}
    for label, contrast in recorded["contrasts"].items():
        delta = errors[contrast["candidate"]] - errors[contrast["control"]]
        same(delta.sum() / denominator.sum(), contrast["delta_ratio"])
        same(np.percentile(delta[draws].sum(axis=1) / den, [2.5, 97.5]), contrast["ci95"])
        same(-delta.sum() / errors[contrast["control"]].sum(), contrast["relative_error_reduction"])
        summary["contrasts"][group][label] = {
            k: v for k, v in contrast.items() if k != "per_user"}
        summary["contrasts"][group][label].update(
            users_better=int((delta < 0).sum()), users_equal=int((delta == 0).sum()),
            users_worse=int((delta > 0).sum()),
            median_user_delta_E=float(np.median(delta / denominator)))
(HERE / "verification_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
print(json.dumps({"status": summary["status"], "cells": summary["cells"], "dynamic_contrasts": summary["contrasts"]["dynamic"]}, indent=2))
