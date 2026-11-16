"""Prepare matched review configurations and read train-only calibration evidence.

This command never submits jobs or approves training. Training weights remain null
until a separately reviewed calibration receipt and release fill them explicitly.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wearusfm.training import run as R
from wearusfm.training.contrastive_objectives import ContrastiveConfig
from wearusfm.data.pretraining_loader import LoaderConfig
from wearusfm.data.virtual_montage import VirtualSpec
from wearusfm.training.masking import MaskSpec

BASELINE = ROOT / "results/infra/tokenizer_execution_20261103/19618/T100-G/config.json"
BASELINE_SHA256 = "f9d0318462fd30a5592087ef72c63d3036a88ff1d07fc0dad8624e36253c2cfa"
PLAN = ROOT / "results/infra/contrastive_comparison_20261116/plan.json"
WORLD_SIZE, WORKERS_PER_RANK, CALIBRATION_STEPS, FILL_STEPS = 4, 6, 512, 256


def _tuples(data, names):
    for name in names:
        if name in data and data[name] is not None:
            data[name] = tuple(data[name])


def config_from_dict(data: dict) -> R.RunConfig:
    """Restore every nested configuration dataclass without changing the recipe."""
    if not isinstance(data, dict):
        raise ValueError("run config must be a dictionary")
    d = copy.deepcopy(data)
    model = d.pop("model")
    _tuples(model, ("spectral_branches",))
    jepa = d.pop("jepa")
    loader = d.pop("loader")
    mask = loader.pop("mask")
    _tuples(mask, ("short", "medium", "long"))
    loader["mask"] = MaskSpec(**mask)
    _tuples(loader, ("filter_band_hz", "notch_hz"))
    if loader.get("virtual") is not None:
        virtual = loader["virtual"]
        if "shapes" in virtual:
            virtual["shapes"] = tuple(tuple(s) for s in virtual["shapes"])
        _tuples(virtual, ("strides", "classes", "exclude_datasets", "no_bipolar_datasets"))
        loader["virtual"] = VirtualSpec(**virtual)
    if "schedule" in d:
        schedule = d["schedule"]
        _tuples(schedule, ("keep_checkpoints",))
        d["schedule"] = R.Schedule(**schedule)
    if "alarm" in d:
        d["alarm"] = R.AlarmRule(**d["alarm"])
    _tuples(d, ("datasets",))
    return R.RunConfig(model=R.FMConfig(**model), jepa=R.JEPAConfig(**jepa), loader=LoaderConfig(**loader), **d)


def _jsonable(value):
    return json.loads(json.dumps(value, allow_nan=False))


def build_review_configs(baseline: dict) -> dict[str, dict]:
    """Three full review recipes and an inert-weight 512-batch calibration recipe."""
    base = R.config_to_dict(config_from_dict(baseline))
    expected = {"batch_size": 32, "accum_steps": 1, "lr": 1e-4, "warmup_steps": 1962,
                "weight_decay": .05, "grad_clip": 1., "max_steps": 19618, "seed": 0}
    if any(base.get(k) != v for k, v in expected.items()):
        raise ValueError("baseline optimization/data-dose is not the frozen T100-G recipe")
    m, j = base["model"], base["jepa"]
    if (m["dim"], m["window_ms"], m["patch_ms"], m["tokenizer_activation"], m["pose_head"]) != (384, 100., 25., "gelu", True):
        raise ValueError("baseline architecture is not T100-G")
    if m.get("tokenizer_logsquare", False) or m.get("contrastive_projection", False):
        raise ValueError("baseline must not have extra tokenizer or contrastive branches")
    if (j["jepa_weight"], j["anchor_weight"], j["pose_weight"], j["ema_momentum"]) != (0., .2, 1., .996):
        raise ValueError("baseline objective differs")
    if base["schedule"] != {"kind": "wsd", "total_steps": 49046, "decay_fraction": .2, "keep_checkpoints": [5000, 19618]}:
        raise ValueError("baseline schedule differs")
    base["model"]["contrastive_projection"] = True
    output = {}
    for arm in ("C0", "CN", "CR"):
        cfg = copy.deepcopy(base)
        cc = asdict(ContrastiveConfig(arm=arm))
        cc["nce_weight"] = cc["rnc_weight"] = None
        cfg["contrastive"] = cc
        output[arm] = cfg
    calibration = copy.deepcopy(base)
    calibration["max_steps"] = CALIBRATION_STEPS
    calibration["contrastive"] = asdict(ContrastiveConfig(arm="C0", nce_weight=1., rnc_weight=1., calibration_only=True))
    # These values do not weight the recorded component gradients; C0 applies
    # both terms at zero and calibration_only performs no optimizer/EMA update.
    output["calibration"] = calibration
    return _jsonable(output)


def _read(path):
    def bad_constant(value):
        raise ValueError(f"nonfinite JSON constant {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=bad_constant, object_pairs_hook=unique)


def _pin(path):
    p = Path(path)
    data = p.read_bytes()
    return {"path": str(p), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _reader_provenance():
    """Pin local reader sources, without claiming they attest the remote run code."""
    paths = sorted((ROOT / "src/wearusfm").rglob("*.py")) + sorted((ROOT / "scripts").glob("contrastive_*.py"))
    pins = {str(p.relative_to(ROOT)): {k: v for k, v in _pin(p).items() if k != "path"} for p in paths}
    digest = hashlib.sha256(json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"scope": "reader checkout only; frozen run SHA/source equality remains a release check",
            "plan": _pin(PLAN), "source_manifest_sha256": digest, "sources": pins}


def _coverage(summary_contrastive):
    expected = summary_contrastive.get("expected_datasets")
    if not isinstance(expected, list) or not expected or any(not isinstance(v, str) or not v for v in expected) or len(set(expected)) != len(expected):
        raise ValueError("calibration summary must declare all positively weighted datasets")
    if summary_contrastive.get("coverage_completed_blocks") != []:
        raise ValueError("512-step calibration cannot complete a 1024-update coverage block")
    block = summary_contrastive["coverage_partial_block"]
    if block["updates"] != 256 or block["through_step"] != 511 or set(block["datasets"]) != set(expected):
        raise ValueError("calibration coverage support/steps differ from the frozen panel")
    total, good = 0, 0
    details = {}
    for dataset, row in sorted(block["datasets"].items()):
        n, k = row["queries"], row["eligible"]
        if type(n) is not int or type(k) is not int or not 0 <= k <= n:
            raise ValueError("invalid coverage counts")
        fraction = k / max(n, 1)
        if not math.isclose(row["fraction"], fraction, abs_tol=1e-12):
            raise ValueError("dataset coverage fraction inconsistent with counts")
        details[dataset] = {"queries": n, "eligible": k, "fraction": fraction, "passed": n >= 32 and fraction >= .8}
        total, good = total + n, good + k
    if total != 256 * 32 or not math.isclose(block["nce_fraction"], good / total, abs_tol=1e-12):
        raise ValueError("coverage query count is not the complete global panel")
    rnc = block["rnc_active_fraction"]
    if type(rnc) not in (int, float) or not math.isfinite(rnc) or not 0 <= rnc <= 1 or not math.isclose(rnc * 256, round(rnc * 256), abs_tol=1e-8):
        raise ValueError("invalid RnC coverage fraction")
    passed = good / total >= .9 and rnc >= .9 and all(d["passed"] for d in details.values())
    if block.get("passed") != passed:
        raise ValueError("coverage verdict differs from recomputed frozen gates")
    return {"passed": passed, "nce_fraction": good / total, "rnc_active_fraction": rnc, "datasets": details}


def calibration_report(run_dir, baseline_path=BASELINE):
    """Read measured evidence; PASSED_FOR_REVIEW does not authorize a training."""
    folder, baseline_path = Path(run_dir), Path(baseline_path)
    if _pin(baseline_path)["sha256"] != BASELINE_SHA256:
        raise ValueError("historical baseline hash changed")
    cfg, summary = _read(folder / "config.json"), _read(folder / "summary.json")
    expected = build_review_configs(_read(baseline_path))["calibration"]
    if R.config_to_dict(config_from_dict(cfg)) != expected:
        raise ValueError("calibration run config differs from the frozen 512-batch recipe")
    if summary.get("steps") != CALIBRATION_STEPS or summary.get("stopped") != "max_steps":
        raise ValueError("calibration did not finish the complete frozen panel")
    code = summary.get("code")
    run_sha = code.get("sha") if isinstance(code, dict) else None
    if not isinstance(run_sha, str) or len(run_sha) != 40 or any(c not in "0123456789abcdef" for c in run_sha):
        raise ValueError("calibration summary requires a full hexadecimal frozen run code SHA")
    if summary.get("world_size") != WORLD_SIZE or summary.get("batch_per_process") != 8 or summary.get("workers_per_rank") != WORKERS_PER_RANK:
        raise ValueError("calibration requires four ranks, eight windows and six loader workers per rank")
    c = summary["contrastive"]
    if c.get("config") != cfg["contrastive"] or c.get("optimizer_updates") != 0 or c.get("queue_last_step") != 511:
        raise ValueError("calibration performed optimizer updates or has inconsistent queue/config")
    sha = c.get("initial_sha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(v not in "0123456789abcdef" for v in sha) or c.get("final_student_sha256") != sha or c.get("final_teacher_sha256") != sha:
        raise ValueError("calibration changed the initial student/teacher weights")
    records = [_read_json_line(line) for line in (folder / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if len(records) != CALIBRATION_STEPS or [r.get("step") for r in records] != list(range(1, CALIBRATION_STEPS + 1)):
        raise ValueError("calibration metrics are missing, duplicated or out of order")
    if any(r.get("contrastive_calibration_only") is not True for r in records):
        raise ValueError("non-calibration metric in panel")
    decisions, selected = [], []
    for r in records[FILL_STEPS:]:
        norms = {k: r.get("contrast_global_grad_" + k) for k in ("base", "nce", "rnc")}
        reasons = []
        if not _positive(r.get("contrast_nce_eligible")):
            reasons.append("no_nce_support")
        if not _positive(r.get("contrast_rnc_windows")):
            reasons.append("no_rnc_support")
        reasons += ["invalid_gradient_" + k for k, value in norms.items() if not _positive(value)]
        informative = not reasons
        take = informative and len(selected) < 32
        decision = {"step": r["step"], "informative": informative, "selected": take,
                    "reasons": reasons, "gradient_norms": norms,
                    "nce_eligible": r.get("contrast_nce_eligible"), "rnc_windows": r.get("contrast_rnc_windows"),
                    "cos_nce_base": r.get("contrast_global_cos_nce_base"), "cos_rnc_base": r.get("contrast_global_cos_rnc_base")}
        decisions.append(decision)
        if take:
            selected.append(decision)
    coverage = _coverage(c)
    failure_reasons = []
    if len(selected) != 32:
        failure_reasons.append("fewer_than_32_jointly_informative_batches")
    weights, ratios = {}, {}
    if len(selected) == 32:
        for name in ("nce", "rnc"):
            ratios[name] = [r["gradient_norms"]["base"] / r["gradient_norms"][name] for r in selected]
            weights[name] = .1 * statistics.median(ratios[name])
            if not math.isfinite(weights[name]) or not 1e-4 <= weights[name] <= 1:
                failure_reasons.append("weight_outside_frozen_range_" + name)
    if not coverage["passed"]:
        failure_reasons.append("coverage_gate_failed")
    report = {"schema": 1, "status": "FAILED_REQUIRES_REVIEW" if failure_reasons else "PASSED_FOR_REVIEW",
              "training_approved": False, "failure_reasons": failure_reasons, "weights": weights,
              "selected_steps": [r["step"] for r in selected], "gradient_ratios": ratios, "batch_decisions": decisions,
              "coverage": coverage, "initial_state_sha256": sha, "optimizer_updates": 0,
              "world_size": WORLD_SIZE, "workers_per_rank_required": WORKERS_PER_RANK,
              "run_code": code, "reader_provenance": _reader_provenance(),
              "inputs": {name: _pin(folder / name) for name in ("config.json", "summary.json", "metrics.jsonl")},
              "baseline": _pin(baseline_path)}
    # Invalid numeric evidence must not be hidden by non-standard JSON serialization.
    _jsonable(report)
    return report


def _read_json_line(line):
    def bad(value):
        raise ValueError(f"nonfinite calibration metric {value}")
    return json.loads(line, parse_constant=bad)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--baseline", type=Path, default=BASELINE)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("calibration-report")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.command == "prepare":
        if _pin(a.baseline)["sha256"] != BASELINE_SHA256:
            ap.error("historical baseline hash changed")
        configs = build_review_configs(_read(a.baseline))
        a.out.mkdir(parents=True, exist_ok=False)
        for name, cfg in configs.items():
            (a.out / (name + ".config.json")).write_text(json.dumps(cfg, indent=2, allow_nan=False) + "\n")
        report = {"schema": 1, "status": "REVIEW_REQUIRED", "training_approved": False,
                  "baseline": _pin(a.baseline), "world_size": WORLD_SIZE, "workers_per_rank": WORKERS_PER_RANK,
                  "training_weights": "null until calibration review and dedicated release",
                  "calibration_weights": "inert placeholders, C0 and calibration_only, no optimizer/EMA updates",
                  "configs": {name: _pin(a.out / (name + ".config.json")) for name in configs}}
        (a.out / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    else:
        report = calibration_report(a.run_dir)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("x") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
    print(json.dumps({"status": report["status"], "training_approved": False}))


if __name__ == "__main__":
    main()
