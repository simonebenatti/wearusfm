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


def policy_reference(section, name):
    pin = _read(PLAN).get(section, {}).get(name)
    if not isinstance(pin, dict) or not isinstance(pin.get("path"), str):
        raise ValueError("missing frozen policy reference: " + name)
    path = ROOT / pin["path"]
    if path.is_absolute() and not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("policy reference must be repository-relative")
    if _pin(path)["sha256"] != pin.get("sha256"):
        raise ValueError("policy reference hash changed: " + name)
    return _read(path)


def historical_gradient_multiplier():
    evidence = policy_reference("calibration", "historical_reference")
    if evidence.get("reference_updates") != [513, 19618] or evidence.get("initial_updates") != [1, 8]:
        raise ValueError("historical gradient reference range differs from the frozen policy")
    if not _positive(evidence.get("reference_mean")) or not _positive(evidence.get("initial_mean")):
        raise ValueError("invalid historical gradient reference")
    multiplier = evidence["reference_mean"] / evidence["initial_mean"]
    if not math.isclose(multiplier, evidence.get("multiplier", 0), rel_tol=1e-14):
        raise ValueError("historical gradient multiplier disagrees with its pinned means")
    return multiplier


def validate_input_bindings(pins):
    """Bind D1 quotas to their exact manifest/durations; paths may relocate."""
    if not isinstance(pins, dict) or set(pins) != {"manifest", "scales", "window_seconds", "pose_stats"}:
        raise ValueError("complete comparison input bindings required")
    reference = policy_reference("coverage", "manifest_reference")
    for name, source in (("manifest", "source_manifest"), ("window_seconds", "source_window_seconds")):
        if pins[name] != reference[source]["sha256"]:
            raise ValueError("execution input differs from coverage-policy source: " + name)
    frozen = _read(PLAN).get("input_sha256")
    if frozen is not None and pins != frozen:
        raise ValueError("execution input differs from frozen comparison plan")


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
    coverage = policy_reference("coverage", "manifest_reference")
    shared = {"coverage_quotas": coverage["quota_window"], "grad_every": 100, "grad_dense_start": 513, "grad_dense_end": 576,
              "coverage_exempt_datasets": tuple(coverage["exempt_datasets"])}
    output = {}
    for arm in ("C0", "CN", "CR"):
        cfg = copy.deepcopy(base)
        cc = asdict(ContrastiveConfig(arm=arm, nce_weight=1., rnc_weight=1., **shared))
        cc["nce_weight"] = cc["rnc_weight"] = None
        cfg["contrastive"] = cc
        output[arm] = cfg
    calibration = copy.deepcopy(base)
    calibration["max_steps"] = CALIBRATION_STEPS
    calibration["contrastive"] = asdict(ContrastiveConfig(arm="C0", nce_weight=1., rnc_weight=1., calibration_only=True, **shared))
    # These values do not weight the recorded component gradients; C0 applies
    # both terms at zero and calibration_only performs no optimizer/EMA update.
    output["calibration"] = calibration
    return _jsonable(output)


def _loads(raw):
    def bad_constant(value):
        raise ValueError(f"nonfinite JSON constant {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result
    result = json.loads(raw, parse_constant=bad_constant, object_pairs_hook=unique)
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        if isinstance(value, dict):
            for item in value.values(): finite(item)
        elif isinstance(value, list):
            for item in value: finite(item)
    finite(result)  # JSON 1e999 otherwise silently becomes infinity.
    return result


def _read(path):
    return _loads(Path(path).read_text())


def _pin(path, relative_to=None):
    p = Path(path)
    data = p.read_bytes()
    base = ROOT if relative_to is None else Path(relative_to)
    try:
        name = str(p.resolve().relative_to(base.resolve()))
    except ValueError:
        name = str(p.resolve())
    return {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def source_digest():
    """Canonical execution source identity, shared by audit, reader and runner."""
    paths = sorted((ROOT / "src/wearusfm").rglob("*.py"))
    paths += sorted((ROOT / "scripts").glob("contrastive_*.py"))
    paths += sorted((ROOT / "scripts/slurm").glob("contrastive*.sbatch"))
    paths += [ROOT / "scripts/train_jepa.py", ROOT / "scripts/slurm/lib_code_dir.sh"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + bytes.fromhex(_pin(path)["sha256"]))
    return digest.hexdigest()


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _count(value, maximum=None):
    return (type(value) in (int, float) and math.isfinite(value) and value >= 0
            and int(value) == value and (maximum is None or value <= maximum))


def _reader_provenance():
    """Pin local reader sources, without claiming they attest the remote run code."""
    paths = sorted((ROOT / "src/wearusfm").rglob("*.py")) + sorted((ROOT / "scripts").glob("contrastive_*.py"))
    pins = {str(p.relative_to(ROOT)): {k: v for k, v in _pin(p).items() if k != "path"} for p in paths}
    digest = hashlib.sha256(json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"scope": "reader checkout only; frozen run SHA/source equality remains a release check",
            "plan": _pin(PLAN), "source_manifest_sha256": digest, "sources": pins}


def _coverage(summary_contrastive):
    config = ContrastiveConfig(**summary_contrastive["config"])
    expected = summary_contrastive.get("expected_datasets")
    if not isinstance(expected, list) or not expected or any(not isinstance(v, str) or not v for v in expected) or len(set(expected)) != len(expected):
        raise ValueError("calibration summary must declare all positively weighted datasets")
    if config.coverage_quotas is not None and set(expected) != set(config.coverage_quotas):
        raise ValueError("calibration dataset set differs from the manifest-bound coverage policy")
    if summary_contrastive.get("coverage_completed_blocks") != []:
        raise ValueError("512-step calibration cannot complete a 1024-update coverage block")
    block = summary_contrastive["coverage_partial_block"]
    if type(block.get("updates")) is not int or type(block.get("through_step")) is not int or block["updates"] != 256 or block["through_step"] != 511 or set(block["datasets"]) != set(expected):
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
        exempt = dataset in config.coverage_exempt_datasets
        if row.get("exempt") is not exempt:
            raise ValueError("dataset exemption differs from the frozen manifest policy")
        details[dataset] = {"queries": n, "eligible": k, "fraction": fraction, "exempt": exempt,
                            "passed": exempt or (n >= config.coverage_min_queries and fraction >= config.coverage_dataset_fraction)}
        total, good = total + n, good + k
    if total != 256 * 32 or not math.isclose(block["nce_fraction"], good / total, abs_tol=1e-12):
        raise ValueError("coverage query count is not the complete global panel")
    rnc = block["rnc_active_fraction"]
    if type(rnc) not in (int, float) or not math.isfinite(rnc) or not 0 <= rnc <= 1 or not math.isclose(rnc * 256, round(rnc * 256), abs_tol=1e-8):
        raise ValueError("invalid RnC coverage fraction")
    if block.get("exempt_datasets") != list(config.coverage_exempt_datasets):
        raise ValueError("coverage exemption list differs from the frozen policy")
    policy = {k: getattr(config, k) for k in ("coverage_min_queries", "coverage_dataset_fraction", "coverage_nce_fraction",
                                            "coverage_rnc_fraction", "coverage_global_batch_size", "coverage_quotas")}
    if block.get("policy") != policy:
        raise ValueError("coverage thresholds differ from the frozen policy")
    passed = good / total >= config.coverage_nce_fraction and rnc >= config.coverage_rnc_fraction and all(d["passed"] for d in details.values())
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
    if type(summary.get("steps")) is not int or summary["steps"] != CALIBRATION_STEPS or summary.get("stopped") != "max_steps":
        raise ValueError("calibration did not finish the complete frozen panel")
    code = summary.get("code")
    run_sha = code.get("sha") if isinstance(code, dict) else None
    if not isinstance(run_sha, str) or len(run_sha) != 40 or any(c not in "0123456789abcdef" for c in run_sha):
        raise ValueError("calibration summary requires a full hexadecimal frozen run code SHA")
    if any(type(summary.get(k)) is not int or summary[k] != n for k, n in
           (("world_size", WORLD_SIZE), ("batch_per_process", 8), ("workers_per_rank", WORKERS_PER_RANK))):
        raise ValueError("calibration requires four ranks, eight windows and six loader workers per rank")
    c = summary["contrastive"]
    if c.get("config") != cfg["contrastive"] or type(c.get("optimizer_updates")) is not int or c["optimizer_updates"] != 0 or type(c.get("queue_last_step")) is not int or c["queue_last_step"] != 511:
        raise ValueError("calibration performed optimizer updates or has inconsistent queue/config")
    provenance = c.get("provenance", {})
    if provenance.get("source_sha256") != source_digest():
        raise ValueError("calibration execution sources differ from this frozen reader/runner")
    input_bindings = provenance.get("input_sha256")
    if (not isinstance(input_bindings, dict) or set(input_bindings) != {"manifest", "scales", "window_seconds", "pose_stats"} or
            any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in input_bindings.values())):
        raise ValueError("calibration requires complete execution input bindings")
    validate_input_bindings(input_bindings)
    sha = c.get("initial_sha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(v not in "0123456789abcdef" for v in sha) or c.get("final_student_sha256") != sha or c.get("final_teacher_sha256") != sha:
        raise ValueError("calibration changed the initial student/teacher weights")
    records = [_read_json_line(line) for line in (folder / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if len(records) != CALIBRATION_STEPS or any(type(r.get("step")) is not int for r in records) or [r.get("step") for r in records] != list(range(1, CALIBRATION_STEPS + 1)):
        raise ValueError("calibration metrics are missing, duplicated or out of order")
    if any(r.get("contrastive_calibration_only") is not True for r in records):
        raise ValueError("non-calibration metric in panel")
    for record in records:
        for name in ("contrast_nce_eligible", "contrast_rnc_windows"):
            if not _count(record.get(name), 32):
                raise ValueError("invalid integral support count " + name)
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
    panel = records[FILL_STEPS:]
    if sum(r["contrast_nce_eligible"] for r in panel) != sum(d["eligible"] for d in coverage["datasets"].values()):
        raise ValueError("metric NCE counts disagree with summary coverage")
    if sum(r["contrast_rnc_windows"] > 0 for r in panel) != round(coverage["rnc_active_fraction"] * len(panel)):
        raise ValueError("metric RnC support disagrees with summary coverage")
    for dataset, row in coverage["datasets"].items():
        for prefix, field in (("contrast_nce_queries__", "queries"), ("contrast_nce_eligible__", "eligible")):
            values = [r.get(prefix + dataset, 0) for r in panel]
            if any(not _count(v, 32) for v in values) or sum(values) != row[field]:
                raise ValueError("metric dataset counts disagree with summary coverage: " + dataset)
    failure_reasons = []
    if len(selected) != 32:
        failure_reasons.append("fewer_than_32_jointly_informative_batches")
    weights, ratios = {}, {}
    multiplier = historical_gradient_multiplier()
    if len(selected) == 32:
        for name in ("nce", "rnc"):
            ratios[name] = [r["gradient_norms"]["base"] / r["gradient_norms"][name] for r in selected]
            finite = all(math.isfinite(v) for v in ratios[name])
            weights[name] = .1 * multiplier * statistics.median(ratios[name]) if finite else None
            if weights[name] is None or not math.isfinite(weights[name]) or not 1e-4 <= weights[name] <= 1:
                failure_reasons.append("weight_outside_frozen_range_" + name)
                if weights[name] is not None and not math.isfinite(weights[name]):
                    weights[name] = None
            ratios[name] = [v if math.isfinite(v) else None for v in ratios[name]]
    if not coverage["passed"]:
        failure_reasons.append("coverage_gate_failed")
    report = {"schema": 1, "status": "FAILED_REQUIRES_REVIEW" if failure_reasons else "PASSED_FOR_REVIEW",
              "training_approved": False, "failure_reasons": failure_reasons, "weights": weights,
              "selected_steps": [r["step"] for r in selected], "gradient_ratios": ratios, "batch_decisions": decisions,
              "coverage": coverage, "initial_state_sha256": sha, "optimizer_updates": 0,
              "historical_gradient_multiplier": multiplier,
              "world_size": WORLD_SIZE, "workers_per_rank_required": WORKERS_PER_RANK,
              "run_code": code, "source_sha256": source_digest(), "reader_provenance": _reader_provenance(),
              "input_sha256": input_bindings,
              "inputs": {name: _pin(folder / name, folder) for name in ("config.json", "summary.json", "metrics.jsonl")},
              "baseline": _pin(baseline_path)}
    # Invalid numeric evidence must not be hidden by non-standard JSON serialization.
    _jsonable(report)
    return report


def _read_json_line(line):
    return _loads(line)


def authenticate_calibration(receipt, run_dir):
    """Recompute the decision from all pinned evidence; status/weights are not trust anchors."""
    if not isinstance(receipt, dict) or run_dir is None:
        raise ValueError("calibrated recipe requires a receipt and its pinned evidence directory")
    if receipt.get("status") != "PASSED_FOR_REVIEW" or receipt.get("failure_reasons") != []:
        raise ValueError("calibration receipt did not pass every frozen gate")
    measured = calibration_report(run_dir)
    if _jsonable(receipt) != measured:
        raise ValueError("calibration receipt differs from authenticated recalculation")
    return measured


def calibrated_configs(receipt, run_dir, endpoint=576):
    """All three candidates originate from one authenticated receipt."""
    measured = authenticate_calibration(receipt, run_dir)
    if type(endpoint) is not int or endpoint not in (544, 576, 5000, 19618):
        raise ValueError("invalid frozen endpoint")
    configs = build_review_configs(_read(BASELINE))
    result = {}
    for arm in ("C0", "CN", "CR"):
        cfg = configs[arm]
        cfg["max_steps"] = endpoint
        cfg["contrastive"].update({name + "_weight": measured["weights"][name] for name in ("nce", "rnc")})
        result[arm] = cfg
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--baseline", type=Path, default=BASELINE)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("calibration-report")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("calibrated-configs")
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--endpoint", type=int, choices=(576, 5000, 19618), default=576)
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
    elif a.command == "calibration-report":
        report = calibration_report(a.run_dir)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("x") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
    else:
        receipt = _read(a.receipt)
        configs = calibrated_configs(receipt, a.run_dir, a.endpoint)
        a.out.mkdir(parents=True, exist_ok=False)
        for arm, cfg in configs.items():
            (a.out / (arm + ".config.json")).write_text(json.dumps(cfg, indent=2, allow_nan=False) + "\n")
            if a.endpoint == 576:
                for endpoint, suffix in ((544, "split"), (5000, "5000"), (19618, "19618")):
                    candidate = copy.deepcopy(cfg)
                    candidate["max_steps"] = endpoint
                    (a.out / (arm + "." + suffix + ".config.json")).write_text(json.dumps(candidate, indent=2, allow_nan=False) + "\n")
        report = {"schema": 1, "status": "CALIBRATED_COHORT_FOR_REVIEW", "training_approved": False,
                  "endpoint": a.endpoint, "calibration_sha256": _pin(a.receipt)["sha256"],
                  "source_sha256": source_digest(), "initial_state_sha256": receipt["initial_state_sha256"],
                  "configs": {arm: _pin(a.out / (arm + ".config.json"), a.out) for arm in configs}}
        (a.out / "cohort.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "training_approved": False}))


if __name__ == "__main__":
    main()
