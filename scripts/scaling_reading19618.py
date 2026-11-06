"""Descriptive reading of the paired 300M/30M comparison at 19,618 updates.

Derived from scaling_reading.py (frozen 5,000 policy) by the replacements declared in
scaling_derive19618.py. The S8 definitions are unchanged: EPN-612 and dynamic pose error as
separate primaries, practical pose signal at a 2% point reduction with the interval below zero,
P1/P2 non-inferiority margin of 1 pp, subject bootstrap 2,000/20261102. The S8 operational table
decided the continuation from the 5,000 result and is not reapplied: this reading describes the
19,618 endpoint, recommends no dose, selects no checkpoint and authorizes no further training.
Consumes audited summary contrasts; it does not compute probes or bootstrap,
certify their provenance or submit any job.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROTOCOL = "scaling-reading-19618-20261106-v1"
POLICY = {
    "protocol": PROTOCOL,
    "checkpoint_step": 19618,
    "candidate": "T100-G-300M",
    "control": "T100-G-30M",
    "bootstrap": {"unit": "subject", "replicates": 2000, "seed": 20261102},
    "epn_margin_pp": 0.0,
    "pose_practical_point_reduction": 0.02,
    "p1_p2_noninferiority_margin_pp": 1.0,
    "historical_pair_is_noise_floor": False,
    "definitions_inherited_from": "scaling-reading-5000-20261104-v1",
    "s8_operational_table_reapplied": False,
    "recommendation_emitted": False,
    "checkpoint_selection_between_doses": False,
    "further_training_authorized": False,
    "pretraining_seeds": 1,
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
    require(summary.get("protocol") == "scaling-comparison-19618-20261106-v1", "wrong summary protocol")
    require(type(summary.get("checkpoint_step")) is int and summary["checkpoint_step"] == 19618, "mixed or wrong dose")
    require(summary.get("candidate") == POLICY["candidate"] and summary.get("control") == POLICY["control"], "wrong contrast")
    require(summary.get("bootstrap") == POLICY["bootstrap"], "bootstrap recipe differs")
    quality = summary.get("quality", {})
    expected = {"artifact_integrity", "paired_support", "solver_converged"}
    require(set(quality) == expected and all(type(v) is bool for v in quality.values()), "explicit audit statuses required")
    result = {"protocol": PROTOCOL, "policy": POLICY, "reading_valid": all(quality.values()),
              "further_training_authorized": False, "source_quality": quality}
    if not result["reading_valid"]:
        return {**result, "reading_status": "repair_evaluation_no_scientific_reading",
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
    return {**result, "reading_status": "descriptive_endpoint_reading",
            "primary_reading": {"epn": epn, "pose": pose_reading},
            "pose_relative_point_reduction": relative,
            "pose_practical_signal": relative >= 0.02 and pu < 0,
            "classification_noninferiority": ni,
            "scope": "Conditional on one training seed and development subjects. Separate marginal CIs; no joint 95% guarantee. "
                     "An unresolved outcome is not equivalence; a worse outcome concerns this recipe at this dose, not scaling generally. "
                     "The 5000 and 19618 readings stay side by side: no rule selects between doses."}


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
