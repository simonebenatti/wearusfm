#!/usr/bin/env python3
"""Step 1 offline split/calibrate/extract/probe; outputs outside Git.

Spectral diagnostic split uses PRETRAINING subjects, not downstream P1/P2 test.
Native loader crops are 1-4s: do not reuse long-window P1/P2 feature caches.
"""
import argparse
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.data import pretraining_loader as L
from wearusfm.harness.fm_features import load_model
from wearusfm.harness.spectral_probe import spectral_probe, validate_splits
from wearusfm.model.spectral_step1 import READOUT_VERSION, global_spectral_targets, pool_p1p2
from wearusfm.model.spectral_targets import TARGET_VERSION
from wearusfm.training import run as R
from wearusfm.training.spectral_calibration import calibration_signature, json_sha256, sha256_file


def source_commit():
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("commit code before producing real artifacts")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def output_path(path):
    path = path.resolve()
    if path == ROOT or ROOT in path.parents or path.exists():
        raise ValueError("output must be NEW and outside repository")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def subset(index, subjects):
    ii = [i for i, r in enumerate(index.rows) if r["dataset"]+"/"+r["subject"] in subjects]
    if not ii:
        raise ValueError("no matching pretraining subjects")
    weights = index.weights[ii]
    return L.ManifestIndex([index.rows[i] for i in ii], weights/weights.sum(), index.roots)


def read_roles(path, index):
    roles = json.loads(path.read_text())
    if set(roles) != {"train", "val", "test"} or any(not isinstance(s, list) or not s for s in roles.values()):
        raise ValueError("split must contain nonempty train/val/test lists")
    subjects = [s for r in roles for s in roles[r]]
    labels = [r for r in roles for _ in roles[r]]
    validate_splits(np.array(subjects), np.array(labels))
    if len(subjects) != len(set(subjects)):
        raise ValueError("duplicate subjects")
    allowed = {r["dataset"]+"/"+r["subject"] for r in index.rows}
    if not set(subjects) <= allowed:
        raise ValueError("diagnostic subjects must be PRETRAINING, not P1/P2 test")
    return roles


def iter_batches(index, cfg, roles, n, seed, scales, window_s, names):
    if n <= 0:
        raise ValueError("windows per subject must be positive")
    for role in names:
        for subject in sorted(roles[role]):
            loader = L.PretrainLoader(subset(index, {subject}), cfg.loader, dict(scales), window_s)
            rng = np.random.default_rng([seed, int(json_sha256(subject)[:8], 16)])
            for start in range(0, n, 8):
                yield role, subject, loader.batch(min(8, n-start), rng)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for command in ("split", "calibrate", "extract"):
        p = sub.add_parser(command)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--root", type=Path, action="append", required=True)
        p.add_argument("--datasets", default="emg2qwerty")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", type=Path, required=True)
        if command != "split":
            p.add_argument("--splits", type=Path, required=True)
            p.add_argument("--scales", type=Path, required=True)
            p.add_argument("--window-seconds", type=Path, required=True)
            p.add_argument("--windows-per-subject", type=int, required=True)
        if command == "extract":
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--which", choices=["student", "teacher"], default="teacher")
            p.add_argument("--device", default="cpu")
    p = sub.add_parser("probe")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-checkpoint-sha256", required=True)
    args = ap.parse_args(argv)
    out = output_path(args.out)
    if args.command == "probe":
        with np.load(args.data, allow_pickle=False) as data:
            x, y, mask, subjects, split = (data[k] for k in ("X", "y", "valid", "subject", "split"))
            meta = json.loads(str(data["metadata_json"].item()))
        required = ("source_commit", "manifest_sha256", "split_sha256", "subset_sha256", "loader", "which")
        if any(not meta.get(k) for k in required) or meta.get("checkpoint_sha256") != args.expected_checkpoint_sha256:
            raise ValueError("missing/incompatible probe provenance")
        if meta.get("target_version") != TARGET_VERSION or meta.get("readout_version") != READOUT_VERSION:
            raise ValueError("stale targets/features: regenerate cache")
        d = x.shape[1]//2
        if x.ndim != 2 or x.shape[1] != 2*d or d == 0:
            raise ValueError("probe features must have even positive dimension (2d)")
        report = {"metadata": meta, "views": {name: spectral_probe(f, y, mask, subjects, split)
                   for name, f in (("backbone", x[:, :d]), ("local", x[:, d:]), ("p1p2", x))}}
        out.write_text(json.dumps(report, indent=2, allow_nan=False))
        return 0
    cfg = replace(R.with_window1_rules(R.sanity_config()), datasets=None if args.datasets == "all" else tuple(args.datasets.split(",")))
    index = R.load_index(args.manifest, args.root, cfg.datasets)
    if args.command == "split":
        subjects = sorted({r["dataset"]+"/"+r["subject"] for r in index.rows})
        if len(subjects) < 3:
            raise ValueError("at least three subjects required")
        shuffled = np.random.default_rng(args.seed).permutation(subjects).tolist()
        n = max(1, len(subjects)//10)
        out.write_text(json.dumps({"val": sorted(shuffled[:n]), "test": sorted(shuffled[n:2*n]), "train": sorted(shuffled[2*n:])}, indent=2))
        print(f"Freeze split before calibration/training: {sha256_file(out)}")
        return 0
    roles = read_roles(args.splits, index)
    scales = json.loads(args.scales.read_text())["scales"]
    window_s = json.loads(args.window_seconds.read_text())["window_s"]
    meta = calibration_signature(cfg, args.manifest, scales, window_s)
    meta.update(source_commit=source_commit(), split_sha256=sha256_file(args.splits), seed=args.seed, windows_per_subject=args.windows_per_subject)
    ys, masks, xs, subjects, splits, identities = [], [], [], [], [], []
    model = None
    if args.command == "extract":
        checkpoint_hash = sha256_file(args.checkpoint)
        model = load_model(str(args.checkpoint), device=args.device, which=args.which)
        if sha256_file(args.checkpoint) != checkpoint_hash:
            raise ValueError("checkpoint changed while loading: freeze it before extraction")
        meta.update(checkpoint_sha256=checkpoint_hash, which=args.which,
                    interpretation="spectral probe split; encoder may have seen all subjects without labels")
    names = ["train"] if args.command == "calibrate" else ["train", "val", "test"]
    for role, subject, batch in iter_batches(index, cfg, roles, args.windows_per_subject, args.seed, scales, window_s, names):
        y, mask = global_spectral_targets(batch.anchor_targets, batch.counts, batch.qc_valid)
        ys.append(y); masks.append(mask)
        subjects.extend([subject]*len(y)); splits.extend([role]*len(y))
        for row, win, presented in zip(batch.rows, batch.windows, batch.presented):
            identities.append([row["dataset"], row["subject"], row["session"], asdict(win), presented])
        if model is not None:
            inp, _, _ = R._to_device(*L.to_model_inputs(batch), args.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                xs.append(pool_p1p2(model.encode(inp, None), inp.counts, inp.qc_valid).cpu().numpy())
        print(f"{role} {subject}: {len(y)} windows", flush=True)
    meta.update(subset_sha256=json_sha256(identities), subjects=sorted(set(subjects)), role="train" if model is None else "spectral_diagnostic_pretraining_subjects")
    arrays = {"y": np.concatenate(ys), "valid": np.concatenate(masks), "subject": np.array(subjects), "split": np.array(splits),
              "metadata_json": np.array(json.dumps(meta, allow_nan=False))}
    if xs:
        arrays["X"] = np.concatenate(xs)
    with out.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
