#!/usr/bin/env python3
"""Fixed Step 1 A/B pilot. No submission, auto-resume, or hyperparameter search.

Protocol: docs/step1_ab_pilot.md. Train requires a fresh external run directory;
compare verifies all four delivered streams, checkpoints and frozen caches.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from spectral_step1 import output_path, source_commit
from wearusfm.harness.spectral_ab import paired_spectral_comparison, validate_training_pair
from wearusfm.harness.spectral_probe import validate_splits
from wearusfm.training import run as R
from wearusfm.training.spectral_calibration import calibration_signature, sha256_file

STEPS = 4000
WEIGHTS = {"A": 0., "B": .05}
TIME_LIMITS = {"A": 14100, "B": 21300}  # 5 min margin before 4h/6h SLURM wall time
INPUT_SHA256 = {
    "manifest": "853f35ac5252ef901be390385164c6dc5658f45270441eb74557c6be317ed6d3",
    "scales": "76486d882b5deeca66f152c3541534e9f35efca35f944f6f8677b6478b877486",
    "window_seconds": "e660e324a28a339c038ed07dbaa7d4c358f78f613cd32226eea3f522119a48b0",
    "calibration": "4640dfcf2c6819fb8ba2870e1506f08e4b024cdf38dda267c4cd37652cbc1f89",
    "splits": "e064a1102bb4a7dd1b01a70caefa8fa2c58d8c9c751bafeafe6bd2892fff7a30",
}
SUBSET_SHA256 = "9ba2ddbab3b18558b64f6c3bea186e1e90d49d5e89dca308a26af4df940890d2"


def checked_commit(expected):
    actual = source_commit()
    if actual != expected:
        raise ValueError("checkout differs from frozen expected commit")
    return actual


def build_config(arm, seed, calibration):
    cfg = R.with_window1_rules(R.sanity_config(max_steps=STEPS))
    cfg = replace(cfg, seed=seed, keep_grad_every=500, pairing_audit=True)
    return R.with_spectral_keep(cfg, calibration, WEIGHTS[arm])


def _finite_state(value):
    if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite final checkpoint tensor")
    elif isinstance(value, dict):
        for item in value.values():
            _finite_state(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite_state(item)


def read_run(path, expected_commit):
    run = {name: json.loads((path/f"{name}.json").read_text()) for name in ("config", "summary", "pairing")}
    run["metrics"] = [json.loads(line) for line in (path/"metrics.jsonl").read_text().splitlines()]
    provenance = json.loads((path/"run_provenance.json").read_text())
    if provenance["source_commit"] != expected_commit or provenance["input_sha256"] != INPUT_SHA256:
        raise ValueError("training code/input provenance mismatch")
    if (path/"STOP").exists() or sha256_file(path/"checkpoint.pt") != provenance["checkpoint_sha256"]:
        raise ValueError("STOP or changed checkpoint")
    run["provenance"] = provenance
    return run


def validate_report_targets(report, values):
    test = values["split"] == "test"
    subjects = values["subject"][test]
    targets = values["y"][test].astype(np.float64)
    names = sorted(set(subjects.tolist()))
    for view in report["views"].values():
        if [row["coordinate"] for row in view["coordinates"]] != list(range(32)):
            raise ValueError("32 ordered report coordinates required")
        for j, row in enumerate(view["coordinates"]):
            if sorted(row.get("per_subject", {})) != names:
                raise ValueError("ridge report subjects differ from feature cache")
            for name in names:
                y = targets[subjects == name, j]
                sub = row["per_subject"][name]
                if sub["n"] != len(y) or not np.allclose(
                    [sub["target_mean"], sub["target_m2"]],
                    [y.mean(), np.square(y-y.mean()).sum()], rtol=1e-12, atol=1e-12,
                ):
                    raise ValueError("ridge target sufficient statistics differ from feature cache")


def audit_runs(runs_root, commit):
    runs, audits = {}, {}
    for model_seed in (0, 1):
        pair = []
        for arm in ("A", "B"):
            name = f"{arm}_seed{model_seed}"
            run = read_run(runs_root/name, commit)
            if run["provenance"]["arm"] != arm or run["config"]["seed"] != model_seed:
                raise ValueError("run arm/model seed mislabeled")
            runs[name] = run
            pair.append(run)
        audits[str(model_seed)] = validate_training_pair(*pair)
    return runs, audits


def train_arm(args):
    commit = checked_commit(args.expected_commit)
    out = output_path(args.out_dir)  # validates NEW, absolute-resolved, outside checkout
    hashes = {name: sha256_file(getattr(args, name)) for name in INPUT_SHA256}
    if hashes != INPUT_SHA256:
        raise ValueError("input hash differs from frozen protocol")
    cfg = build_config(args.arm, args.seed, args.calibration)
    scales = json.loads(args.scales.read_text())["scales"]
    window_s = json.loads(args.window_seconds.read_text())["window_s"]
    signature = calibration_signature(cfg, args.manifest, scales, window_s)
    print(json.dumps({"source_commit": commit, "arm": args.arm, "seed": args.seed,
                      "input_sha256": hashes, "config": R.config_to_dict(cfg)}), flush=True)
    summary = R.train(cfg, args.manifest, args.root, out, scales=scales, window_s=window_s,
                      device="cuda", num_workers=6, time_limit_s=TIME_LIMITS[args.arm],
                      log=lambda message: print(message, flush=True))
    provenance = {"source_commit": commit, "arm": args.arm, "model_seed": args.seed,
                  "input_sha256": hashes, "extraction_signature": signature,
                  "checkpoint_sha256": sha256_file(out/"checkpoint.pt")
                  if (out/"checkpoint.pt").exists() else None}
    (out/"run_provenance.json").write_text(json.dumps(provenance, indent=2))
    if summary["steps"] != STEPS or summary["stopped"] != "max_steps" or (out/"STOP").exists():
        raise ValueError("incomplete/stopped arm: block afterok evaluation; do not resume automatically")
    state = torch.load(out/"checkpoint.pt", map_location="cpu", weights_only=False)
    if state["step"] != STEPS or state["config"] != R.config_to_dict(cfg):
        raise ValueError("final checkpoint endpoint/config mismatch")
    _finite_state(state)
    checked_commit(commit)
    if {name: sha256_file(getattr(args, name)) for name in INPUT_SHA256} != hashes:
        raise ValueError("input changed during training")
    return 0


def compare(args):
    commit = checked_commit(args.expected_commit)
    out = output_path(args.out)
    runs, audits = audit_runs(args.runs_root, commit)
    reports, artifacts, reference = [], {}, None
    for model_seed in (0, 1):
        arms = []
        for arm in ("A", "B"):
            name = f"{arm}_seed{model_seed}"
            run = runs[name]
            feature_path, report_path = args.eval_root/f"{name}.npz", args.eval_root/f"{name}.json"
            report = json.loads(report_path.read_text())
            with np.load(feature_path, allow_pickle=False) as data:
                meta = json.loads(str(data["metadata_json"].item()))
                values = {k: data[k].copy() for k in ("y", "valid", "subject", "split")}
                x = data["X"]
                if x.shape != (10000, 768) or not np.isfinite(x).all():
                    raise ValueError("expected 10000x768 finite feature cache")
            if (values["y"].shape != (10000, 32) or values["valid"].shape != (10000, 32)
                    or values["valid"].dtype != bool or not values["valid"].all() or not np.isfinite(values["y"]).all()):
                raise ValueError("complete 10000x32 observed targets required")
            validate_splits(values["subject"], values["split"])
            for role, rows, subjects in (("train", 8000, 80), ("val", 1000, 10), ("test", 1000, 10)):
                group = values["subject"][values["split"] == role]
                _, counts = np.unique(group, return_counts=True)
                if len(group) != rows or len(counts) != subjects or not (counts == 100).all():
                    raise ValueError("expected 80/10/10 subjects with 100 windows each")
            if report["metadata"] != meta or meta["checkpoint_sha256"] != run["provenance"]["checkpoint_sha256"]:
                raise ValueError("report/cache/checkpoint identity mismatch")
            if report.get("features_sha256") != sha256_file(feature_path):
                raise ValueError("ridge was not fitted on this feature cache")
            if any(meta.get(k) != value for k, value in run["provenance"]["extraction_signature"].items()):
                raise ValueError("training/extraction preprocessing signature mismatch")
            if (meta["source_commit"] != commit or meta["subset_sha256"] != SUBSET_SHA256
                    or meta["split_sha256"] != INPUT_SHA256["splits"] or meta["manifest_sha256"] != INPUT_SHA256["manifest"]
                    or meta["which"] != "teacher" or meta["seed"] != 0 or meta["windows_per_subject"] != 100
                    or meta["target_version"] != run["config"]["target_version"]
                    or meta["readout_version"] != run["config"]["readout_version"]):
                raise ValueError("frozen extraction protocol mismatch")
            if reference is None:
                reference = values
            elif any(not np.array_equal(reference[key], values[key]) for key in values):
                raise ValueError("different evaluation targets/masks/subjects/splits between arms/seeds")
            validate_report_targets(report, values)
            arms.append(report)
            artifacts[name] = {"checkpoint_sha256": run["provenance"]["checkpoint_sha256"],
                               "features_sha256": sha256_file(feature_path), "report_sha256": sha256_file(report_path)}
        reports.append(arms)
    views = {view: paired_spectral_comparison([(arms[0]["views"][view], arms[1]["views"][view]) for arms in reports])
             for view in ("backbone", "local", "p1p2")}
    result = {"protocol": "step1_ab_4000_v1", "source_commit": commit, "pairing": audits,
              "artifacts": artifacts, "primary_view": "p1p2", "views": views,
              "decision": views["p1p2"]["decision"], "downstream_status": "P1/P2 require separate authorization"}
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result["views"]["p1p2"]["primary"]), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--arm", choices=WEIGHTS, required=True)
    train.add_argument("--seed", type=int, choices=(0, 1), required=True)
    train.add_argument("--root", type=Path, action="append", required=True)
    train.add_argument("--out-dir", type=Path, required=True)
    for name in INPUT_SHA256:
        train.add_argument("--"+name.replace("_", "-"), type=Path, required=True)
    paired = sub.add_parser("compare")
    paired.add_argument("--runs-root", type=Path, required=True)
    paired.add_argument("--eval-root", type=Path, required=True)
    paired.add_argument("--out", type=Path, required=True)
    audit = sub.add_parser("audit", help="read-only delivered-stream gate BEFORE spending on extraction")
    audit.add_argument("--runs-root", type=Path, required=True)
    for command in (train, paired, audit):
        command.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    if args.command == "audit":
        _, audits = audit_runs(args.runs_root, checked_commit(args.expected_commit))
        print(json.dumps(audits, indent=2))
        return 0
    return train_arm(args) if args.command == "train" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
