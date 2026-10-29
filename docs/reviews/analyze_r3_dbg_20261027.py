"""Read-only recalculation of R3 dbg logs and an explicitly hypothetical Adam model.

Usage: python analyze_r3_dbg_20261027.py /path/to/claude-checkout [--out report.json]
No training, raw signal access, network, or job submission. Output contains aggregates only.
The Adam calculation assumes fixed isotropic features and a constant initial second moment;
it is an illustration, not an estimate of the real model's convergence time.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def analyze(repo: Path) -> dict:
    root = repo / "results/infra/pilot_posa/collaudo_r3_dbg_62429050"
    names = ("metrics.jsonl", "fixed_eval.jsonl", "fixed_eval_set.json", "config.json", "summary.json")
    blobs = {name: (root / name).read_bytes() for name in names}
    rows = [json.loads(x) for x in blobs["metrics.jsonl"].splitlines()]
    fixed = [json.loads(x) for x in blobs["fixed_eval.jsonl"].splitlines()]
    panel = json.loads(blobs["fixed_eval_set.json"])
    config, summary = json.loads(blobs["config.json"]), json.loads(blobs["summary.json"])
    step = np.array([r["step"] for r in rows])
    loss = np.array([r["pose"] for r in rows])
    count = np.array([r["pose_windows"] for r in rows])
    lr = np.array([r["lr"] for r in rows])
    norm = np.array([r["grad_norm"] for r in rows])
    dose = np.cumsum(lr)
    valid = count > 0
    intervals = []
    for start, end in ((1, 100), (500, 599), (1201, 1300)):
        sel = (step >= start) & (step <= end) & valid
        intervals.append({"steps": [start, end], "n_records": int(sel.sum()),
                          "pose_mean": float(loss[sel].mean()), "pose_sd": float(loss[sel].std(ddof=1)),
                          "se_if_iid": float(loss[sel].std(ddof=1) / math.sqrt(sel.sum())),
                          "pose_windows_global": int(count[sel].sum()),
                          "window_weighted_mean_descriptive": float(np.average(loss[sel], weights=count[sel])),
                          # The loss is measured BEFORE the optimizer update recorded in this row.
                          "mean_prior_lr_dose": float((dose - lr)[sel].mean())})
    pose_rows = [w for w in panel["windows"] if w[0].startswith("emg2pose/")]
    recs = Counter(w[0] for w in pose_rows)
    users = Counter(w[0].split("/")[1] for w in pose_rows)
    total_length, union_length, overlapping_pairs = 0, 0, 0
    for rec in recs:
        spans = sorted((w[2], w[3]) for w in pose_rows if w[0] == rec)
        end = -1
        for start, stop in spans:
            total_length += stop - start
            union_length += max(0, stop - max(start, end))
            end = max(end, stop)
        for i, (start, stop) in enumerate(spans):
            overlapping_pairs += sum(max(start, a) < min(stop, b) for a, b in spans[i + 1:])
    fixed_stats = []
    for row in fixed:
        delta = [row[f"pose_mae_j{j:02d}"] - fixed[0][f"pose_mae_j{j:02d}"] for j in range(20)]
        fixed_stats.append({"step": row["step"], "pose": row["pose"], "pose_mae": row["pose_mae"],
                            "pose_mae_std": row["pose_mae_std"], "pose_windows": row["pose_windows"],
                            "mse_change_pct": 100 * (row["pose"] / fixed[0]["pose"] - 1),
                            "mae_change_pct": 100 * (row["pose_mae"] / fixed[0]["pose_mae"] - 1),
                            "joints_with_lower_mae": sum(d < 0 for d in delta), "joint_mae_delta_rad": delta})
    models = []
    a, z = intervals[1]["mean_prior_lr_dose"], intervals[2]["mean_prior_lr_dose"]
    for q in (0.1, 0.2, 0.5, 1.0):
        record = {"explainable_fraction_q": q}
        for batch in (float(count.mean()), 28.0):
            kappa = math.sqrt(batch) / math.sqrt(1 + batch * q / 384)
            l0 = 1 - q + q * math.exp(-2 * kappa * a)
            l1 = 1 - q + q * math.exp(-2 * kappa * z)
            record[f"B={batch:.6f}"] = {
                "full_lr_steps_for_10pct_from_zero": None if q <= 0.1 else -math.log1p(-0.1 / q) / (2e-4 * kappa),
                "predicted_gate_drop_pct": 100 * (1 - l1 / l0),
            }
        models.append(record)
    peak = step[np.isclose(lr, config["lr"], rtol=0, atol=1e-12)]
    return {"run_sha": summary["code"]["sha"],
            "source_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()},
            "geometry": {"patch_ms": config["loader"]["patch_ms"], "align_ms": config["loader"]["align_ms"]},
            "pose_global": {"windows": int(count.sum()), "mean_per_step": float(count.mean()),
                            "min_max": [int(count.min()), int(count.max())], "zero_steps": step[~valid].tolist()},
            "schedule": {"lr_sum": float(lr.sum()), "peak_lr_equivalent_steps": float(lr.sum() / config["lr"]),
                         "peak_lr_steps": [int(peak[0]), int(peak[-1])], "peak_lr_n_steps": int(len(peak)),
                         "gate_delta_prior_dose_equivalent_steps": (z - a) / config["lr"],
                         "clip_active_steps": int((norm > config["grad_clip"]).sum()),
                         "mean_clip_factor_first_last_100": [float((1 / norm[:100]).mean()), float((1 / norm[-100:]).mean())],
                         "pure_weight_decay_factor": float(np.prod(1 - config["weight_decay"] * lr))},
            "gate_intervals": intervals,
            "correlations_pose_present_only": {name: float(np.corrcoef(loss[valid], x[valid])[0, 1])
                                               for name, x in (("step", step), ("lr", lr), ("count", count))},
            "fixed_panel": {"all_windows": len(panel["windows"]), "emg2pose_windows": len(pose_rows),
                            "unique_emg2pose_windows": len(set(tuple(w) for w in pose_rows)), "users": len(users),
                            "recordings": len(recs), "windows_per_recording": sorted(recs.values()),
                            "side_counts": dict(Counter(w[0].rsplit("_", 1)[-1] for w in pose_rows)),
                            "summed_samples": total_length, "union_samples": union_length,
                            "overlapping_pairs": overlapping_pairs, "metrics": fixed_stats},
            "adam_toy_not_prediction": models}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = json.dumps(analyze(args.repo), indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(report)
    else:
        print(report, end="")
