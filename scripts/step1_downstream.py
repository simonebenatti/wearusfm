#!/usr/bin/env python3
"""Fixed downstream tranche for step1_ab_4000_v1. All artifacts outside Git."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch  # before sklearn on macOS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import probe_p1
import probe_p2
from spectral_step1 import output_path, source_commit
from wearusfm.harness.downstream_ab import compare_reports, fingerprint
from wearusfm.training.spectral_calibration import sha256_file

TRAIN_COMMIT = "69fa2da8757b322ea74875b4e9d104a89bbf6d5d"
SPLIT_SHA = "a18e7565157d5dbba8870ffa67c2342ff90de479da480dccb456922b6f6f4ed7"
SPECTRAL_SHA = "e2ca2d71118755f9389cacf5747fd845b377659d78e1e54004d9946e9c280e44"
CHECKPOINTS = {
    "A_seed0": "26ee7e440b01c77b35b9cf198110a4341e267340d377ca32feb80ae6b475b611",
    "A_seed1": "b72a122ea24d01af2830de7f1c76dbcfd7810991e75f9ba015ca5e93bdeeb4d2",
    "B_seed0": "b264df4b75f7047980212ab068fe524148e71dbdef442d4e75e95648f841d3cd",
    "B_seed1": "27b3a819c35f798992b2bcddf7f48998be03d9dc4bc113b79e5170c199577a29",
}
DATASETS = {"p1": ("ninapro_db2", "ninapro_db3", "ninapro_db6"), "p2": ("epn612", "uci_emg")}
PROTOCOL = "step1_downstream_4000_v1"


def check_source(expected):
    if source_commit() != expected or sha256_file(ROOT / "splits/v1/splits_v1.json") != SPLIT_SHA:
        raise ValueError("source commit or signed split changed")


def cache_inventory(folder, task):
    inventory = {}
    for ds in DATASETS[task]:
        path = folder / f"{ds}.npz"
        with np.load(path, allow_pickle=False) as z:
            if "input_sha256" not in z or len(str(z["input_sha256"].item())) != 64:
                raise ValueError("missing audited input hash")
            prefixes = ("train_", "val_", "test_") if task == "p1" else ("",)
            rows, seen = {}, set()
            for prefix in prefixes:
                x, y, s = (z[prefix+k] for k in "xys")
                if x.ndim != 2 or x.shape != (len(y), 768) or s.shape != y.shape or y.ndim != 1 or not len(y) or not np.isfinite(x).all():
                    raise ValueError("invalid feature cache dimensions/values")
                names = set(s.tolist())
                if seen & names:
                    raise ValueError("subject leakage across roles")
                seen.update(names)
                rows[prefix] = {"n": len(y), "labels_subjects_sha256": fingerprint(y, s), "subjects": sorted(names)}
            inventory[ds] = {"sha256": sha256_file(path), "input_sha256": str(z["input_sha256"].item()), "rows": rows}
    return inventory


def run(args):
    check_source(args.expected_commit)
    ckpt = args.runs_root / args.arm / "checkpoint.pt"
    if sha256_file(ckpt) != CHECKPOINTS[args.arm]:
        raise ValueError("checkpoint differs from frozen final teacher")
    provenance = json.loads((ckpt.parent / "run_provenance.json").read_text())
    if provenance["source_commit"] != TRAIN_COMMIT or provenance["checkpoint_sha256"] != CHECKPOINTS[args.arm]:
        raise ValueError("training provenance changed")
    folder = args.eval_root / args.task / args.arm
    report_path = args.eval_root / args.task / f"{args.arm}.json"
    metadata_path = folder / "provenance.json"
    identity = {"protocol": PROTOCOL, "source_commit": args.expected_commit, "training_commit": TRAIN_COMMIT,
                "checkpoint_sha256": CHECKPOINTS[args.arm], "arm": args.arm, "task": args.task,
                "which": "teacher", "probe_seed": 0, "split_sha256": SPLIT_SHA}
    argv = ["--checkpoint", str(ckpt), "--features", str(folder), "--stage", args.stage, "--which", "teacher", "--seed", "0"]
    if args.task == "p1":
        for root in args.root or []:
            argv += ["--root", str(root)]
        if not args.root:
            raise ValueError("P1 roots required")
        argv += ["--window-s", "1", "--val-frac", "0.1", "--weighting", "subjects"]
    elif args.stage == "extract":
        if not args.epn_root or not args.uci_root:
            raise ValueError("P2 raw roots required")
        for root in args.epn_root:
            argv += ["--epn-root", str(root)]
        argv += ["--uci-root", str(args.uci_root)]
    if args.stage == "extract":
        output_path(folder)
        argv += ["--audit-inputs", "--batch", "64", "--device", "cuda"]
    else:
        output_path(report_path)
        metadata = json.loads(metadata_path.read_text())
        if metadata["identity"] != identity or cache_inventory(folder, args.task) != metadata["cache"]:
            raise ValueError("cache provenance mismatch")
        argv += ["--out", str(report_path)]
    runner = probe_p1 if args.task == "p1" else probe_p2
    if runner.main(argv) != 0:
        raise ValueError("probe stage failed")
    check_source(args.expected_commit)
    if sha256_file(ckpt) != CHECKPOINTS[args.arm]:
        raise ValueError("checkpoint changed during stage")
    inventory = cache_inventory(folder, args.task)
    if args.stage == "extract":
        metadata_path.write_text(json.dumps({"identity": identity, "cache": inventory}, indent=2, allow_nan=False))
    else:
        if inventory != metadata["cache"]:
            raise ValueError("cache changed during probe")
        report = json.loads(report_path.read_text())
        report["provenance"] = metadata
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    if args.compare_after:
        compare(args)


def compare(args):
    check_source(args.expected_commit)
    out = output_path(args.eval_root / "comparison.json")
    if sha256_file(args.spectral_report) != SPECTRAL_SHA:
        raise ValueError("spectral predecessor changed")
    reports, hashes = {}, {}
    for task in DATASETS:
        reports[task] = {}
        reference = None
        for arm in CHECKPOINTS:
            path = args.eval_root / task / f"{arm}.json"
            r = json.loads(path.read_text())
            identity = r["provenance"]["identity"]
            if identity != {"protocol": PROTOCOL, "source_commit": args.expected_commit, "training_commit": TRAIN_COMMIT,
                            "checkpoint_sha256": CHECKPOINTS[arm], "arm": arm, "task": task,
                            "which": "teacher", "probe_seed": 0, "split_sha256": SPLIT_SHA}:
                raise ValueError("report provenance mismatch")
            inventory = cache_inventory(args.eval_root / task / arm, task)
            if inventory != r["provenance"]["cache"]:
                raise ValueError("report cache changed")
            paired = {ds: {k: v for k, v in row.items() if k != "sha256"} for ds, row in inventory.items()}
            if reference is not None and reference != paired:
                raise ValueError("actual windows/labels/subjects differ between models")
            reference = paired
            hashes[f"{task}/{arm}"] = sha256_file(path)
            reports[task][arm] = r
    result = compare_reports(reports)
    result.update(protocol=PROTOCOL, source_commit=args.expected_commit, training_commit=TRAIN_COMMIT,
                  spectral_report_sha256=SPECTRAL_SHA, report_sha256=hashes, checkpoint_sha256=CHECKPOINTS)
    check_source(args.expected_commit)
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["extract", "probe", "compare"], required=True)
    ap.add_argument("--task", choices=DATASETS)
    ap.add_argument("--arm", choices=CHECKPOINTS)
    ap.add_argument("--expected-commit", required=True)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--spectral-report", type=Path)
    ap.add_argument("--root", type=Path, action="append")
    ap.add_argument("--epn-root", type=Path, action="append")
    ap.add_argument("--uci-root", type=Path)
    ap.add_argument("--compare-after", action="store_true")
    args = ap.parse_args(argv)
    if args.stage == "compare" or args.compare_after:
        if args.spectral_report is None:
            ap.error("comparison requires --spectral-report")
    if args.compare_after and args.stage != "probe":
        ap.error("--compare-after requires probe")
    if args.stage == "compare":
        compare(args)
    elif args.task is None or args.arm is None:
        ap.error("extract/probe require task and arm")
    else:
        run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
