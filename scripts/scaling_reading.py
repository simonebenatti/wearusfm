"""Prespecified descriptive reading of the paired 300M/30M 5k comparison.

Consumes audited summary contrasts; it does not compute probes or bootstrap,
certify their provenance, authorize a continuation, or submit any job.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROTOCOL = "scaling-reading-5000-20261104-v1"
POLICY = {
    "protocol": PROTOCOL,
    "checkpoint_step": 5000,
    "candidate": "T100-G-300M",
    "control": "T100-G-30M",
    "bootstrap": {"unit": "subject", "replicates": 2000, "seed": 20261102},
    "epn_margin_pp": 0.0,
    "pose_practical_point_reduction": 0.02,
    "p1_p2_noninferiority_margin_pp": 1.0,
    "historical_pair_is_noise_floor": False,
    "continuation_is_authorized": False,
    "continuation_endpoint": 19618,
    "continuation_additional_steps": 14618,
    "budget_formula": "4/3600 * ceil900(14618 * max(trainer300M5000_seconds/5000, preflight300M_112_update_mean_seconds) * 1.20 + 1200)",
    "budget_freeze": "numeric training, extraction GPU and reader CPU caps before opening 5000 downstream scores",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "nonfinite or nonnumeric statistic")
    return float(value)


def contrast(row, key):
    point = number(row[key])
    ci = row["ci95"]
    require(isinstance(ci, list) and len(ci) == 2, "two CI bounds required")
    lo, hi = map(number, ci)
    require(lo <= hi, "reversed confidence interval")
    # Percentile bootstrap intervals need not enclose the original point estimate.
    return point, lo, hi


def read_comparison(summary):
    require(summary.get("protocol") == "scaling-comparison-5000-20261104-v1", "wrong summary protocol")
    require(type(summary.get("checkpoint_step")) is int and summary["checkpoint_step"] == 5000, "mixed or wrong dose")
    require(summary.get("candidate") == POLICY["candidate"] and summary.get("control") == POLICY["control"], "wrong contrast")
    require(summary.get("bootstrap") == POLICY["bootstrap"], "bootstrap recipe differs")
    quality = summary.get("quality", {})
    expected = {"artifact_integrity", "paired_support", "solver_converged"}
    require(set(quality) == expected and all(type(v) is bool for v in quality.values()), "explicit audit statuses required")
    result = {"protocol": PROTOCOL, "policy": POLICY, "reading_valid": all(quality.values()),
              "continuation_authorized": False, "source_quality": quality}
    if not result["reading_valid"]:
        return {**result, "recommendation": "repair_evaluation_no_scientific_reading",
                "failed_checks": sorted(k for k, v in quality.items() if not v)}
    _, el, eu = contrast(summary["epn"], "delta_pp")
    epn = "better" if el > 0 else "worse" if eu < 0 else "unresolved"
    pose = summary["pose"]
    control, candidate = number(pose["control_E_dyn"]), number(pose["candidate_E_dyn"])
    require(control > 0 and candidate >= 0, "invalid dynamic error ratios")
    delta, pl, pu = contrast(pose, "delta_E_dyn")
    require(math.isclose(delta, candidate - control, rel_tol=1e-10, abs_tol=1e-12), "pose point and delta disagree")
    pose_reading = "better" if pu < 0 else "worse" if pl > 0 else "unresolved"
    relative = (control - candidate) / control
    ni = {}
    for task in ("p1", "p2"):
        _, lo, hi = contrast(summary[task], "delta_pp")
        ni[task] = "supported" if lo > -1 else "loss_beyond_margin" if hi < -1 else "unresolved"
    if "worse" in (epn, pose_reading) or "loss_beyond_margin" in ni.values():
        next_step = "pause_this_recipe_review_before_19618"
    elif "better" in (epn, pose_reading):
        next_step = "propose_19618_confirmation"
    else:
        next_step = "propose_19618_dose_question"
    return {**result, "primary_reading": {"epn": epn, "pose": pose_reading},
            "pose_relative_point_reduction": relative,
            "pose_practical_signal": relative >= 0.02 and pu < 0,
            "classification_noninferiority": ni, "recommendation": next_step,
            "scope": "Conditional on one training seed and development subjects. Separate marginal CIs; no joint 95% guarantee. A pause concerns this recipe, not scaling generally."}


def continuation_training_cap(trainer5000_seconds, preflight_update_mean_seconds):
    trainer, update = number(trainer5000_seconds), number(preflight_update_mean_seconds)
    require(trainer > 0 and update > 0, "positive measured timings required")
    seconds = 900 * math.ceil((14618 * max(trainer / 5000, update) * 1.20 + 1200) / 900)
    return {"additional_steps": 14618, "walltime_seconds": seconds, "gpu_count": 4,
            "gpu_hours_cap": 4 * seconds / 3600,
            "extractions_and_cpu_included": False, "submission_authorized": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = read_comparison(json.loads(args.summary.read_text()))
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
