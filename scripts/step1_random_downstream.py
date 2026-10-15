#!/usr/bin/env python3
"""Random-initial-teacher baseline for the closed step1_downstream_4000_v1."""
import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import torch  # before sklearn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import step1_downstream as old
from spectral_step1 import output_path
from wearusfm.harness.downstream_ab import fingerprint
from wearusfm.harness.random_encoder import compare_to_random, initial_teacher
from wearusfm.training.spectral_calibration import sha256_file

PROTOCOL = "step1_random_downstream_4000_v1"
EVAL_COMMIT = "8dad7e82713ce4ed2741acc9c9d4fd9ac656d2f7"
COMPARISON_SHA = "7c93c1b700b7462ce46eeb27a3de65abd38d95caef38b6dc03c76520957527c1"
INITIAL_SHA = {
    0: "fe4cfe34930e4f0836ab8107bf203f0e0ce95db8cfa0cdf1aaa239bcf9dacdb0",
    1: "f02883fd0cc0ac8d744ee7ae9036b7b1346c16c686f08a6de3a714c1d3570446",
}
# Encoder and probe implementations must be identical to the historical campaign.
# Targets/losses/resume changed since then but are not used for frozen inference.
SAME_FILES = ["scripts/probe_p1.py", "scripts/probe_p2.py"] + [
    f"src/wearusfm/harness/{name}.py" for name in
    ("fm_features", "p1_cross_subject", "probe", "benchmarks", "splits", "downstream_ab")
] + [f"src/wearusfm/model/{name}.py" for name in
     ("fm", "frontend", "patch_tokens", "channel_identity", "channel_codes", "geometry", "local_encoder", "perceiver", "backbone")]
SAME_FUNCTIONS = {"src/wearusfm/model/spectral_step1.py": ("pool_p1p2",),
                  "src/wearusfm/data/pretraining_loader.py": ("_filter",)}


def check_pipeline(expected):
    old.check_source(expected)
    for path in SAME_FILES + list(SAME_FUNCTIONS):
        frozen = subprocess.check_output(["git", "show", f"{EVAL_COMMIT}:{path}"], cwd=ROOT)
        current = (ROOT / path).read_bytes()
        if path in SAME_FUNCTIONS:
            names = SAME_FUNCTIONS[path]
            def selected(source):
                return [ast.dump(n, include_attributes=False) for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name in names]
            if len(selected(current)) != len(names) or selected(current) != selected(frozen):
                raise ValueError(f"historical inference function changed: {path}")
        elif current != frozen:
            raise ValueError(f"historical encoder/probe changed: {path}")


def identity(args):
    return {"protocol": PROTOCOL, "source_commit": args.expected_commit, "task": args.task,
            "model_seed": args.model_seed, "optimization_updates": 0, "which": "teacher", "probe_seed": 0,
            "split_sha256": old.SPLIT_SHA, "initial_state_sha256": INITIAL_SHA[args.model_seed],
            "reference_checkpoint_sha256": old.CHECKPOINTS[f"A_seed{args.model_seed}"]}


def paired_inventory(cache):
    return {ds: {k: v for k, v in row.items() if k != "sha256"} for ds, row in cache.items()}


def run(args):
    check_pipeline(args.expected_commit)
    folder = args.eval_root / args.task / f"Random_seed{args.model_seed}"
    ckpt, metadata_path = folder / "initial_teacher.pt", folder / "provenance.json"
    report_path = args.eval_root / args.task / f"Random_seed{args.model_seed}.json"
    if args.stage == "extract":
        output_path(folder)
        model, cfg = initial_teacher(args.runs_root / f"A_seed{args.model_seed}/checkpoint.pt",
                                     reference_sha256=identity(args)["reference_checkpoint_sha256"],
                                     seed=args.model_seed, initial_sha256=INITIAL_SHA[args.model_seed])
        folder.mkdir()
        torch.save({"artifact_role": "untrained_random_encoder", "optimization_updates": 0, "step": 0,
                    "config": cfg, "teacher": model.state_dict()}, ckpt)
        metadata = {"identity": identity(args), "model_config_sha256": fingerprint(cfg["model"]),
                    "checkpoint_sha256": sha256_file(ckpt), "calibration_sha256": cfg["keep_calibration_sha256"]}
        del model
    else:
        output_path(report_path)
        metadata = json.loads(metadata_path.read_text())
        if metadata["identity"] != identity(args) or old.cache_inventory(folder, args.task) != metadata["cache"]:
            raise ValueError("random cache provenance mismatch")
    if sha256_file(ckpt) != metadata["checkpoint_sha256"]:
        raise ValueError("random checkpoint changed")
    argv = ["--checkpoint", str(ckpt), "--features", str(folder), "--stage", args.stage,
            "--which", "teacher", "--seed", "0"]
    if args.task == "p1":
        if not args.root:
            raise ValueError("P1 roots required")
        for root in args.root:
            argv += ["--root", str(root)]
        argv += ["--window-s", "1", "--val-frac", "0.1", "--weighting", "subjects"]
    elif args.stage == "extract":
        if not args.epn_root or not args.uci_root:
            raise ValueError("P2 roots required")
        for root in args.epn_root:
            argv += ["--epn-root", str(root)]
        argv += ["--uci-root", str(args.uci_root)]
    argv += (["--audit-inputs", "--batch", "64", "--device", "cuda"] if args.stage == "extract"
             else ["--out", str(report_path)])
    if (old.probe_p1 if args.task == "p1" else old.probe_p2).main(argv) != 0:
        raise ValueError("random probe failed")
    check_pipeline(args.expected_commit)
    if sha256_file(ckpt) != metadata["checkpoint_sha256"]:
        raise ValueError("random checkpoint changed during stage")
    inventory = old.cache_inventory(folder, args.task)
    if args.stage == "extract":
        metadata["cache"] = inventory
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False))
    else:
        if inventory != metadata["cache"]:
            raise ValueError("random cache changed during probe")
        report = json.loads(report_path.read_text())
        report["provenance"] = metadata
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    if args.compare_after:
        compare(args)


def compare(args):
    check_pipeline(args.expected_commit)
    frozen = args.learned_root / "comparison.json"
    if sha256_file(frozen) != COMPARISON_SHA:
        raise ValueError("closed downstream comparison changed")
    original = json.loads(frozen.read_text())
    out = output_path(args.eval_root / "comparison.json")
    random, learned, hashes = {}, {arm: {} for arm in ("A", "B")}, {}
    for task in old.DATASETS:
        random[task] = {}
        for arm in learned:
            learned[arm][task] = {}
        for seed in (0, 1):
            args.task, args.model_seed = task, seed
            path = args.eval_root / task / f"Random_seed{seed}.json"
            rr = json.loads(path.read_text())
            meta = rr["provenance"]
            folder = path.parent / f"Random_seed{seed}"
            if meta["identity"] != identity(args) or old.cache_inventory(folder, task) != meta["cache"]:
                raise ValueError("random report provenance mismatch")
            if sha256_file(folder / "initial_teacher.pt") != meta["checkpoint_sha256"]:
                raise ValueError("random checkpoint changed before comparison")
            random[task][seed] = rr
            hashes[f"{task}/Random_seed{seed}"] = sha256_file(path)
            for arm in learned:
                path = args.learned_root / task / f"{arm}_seed{seed}.json"
                checksum = sha256_file(path)
                if checksum != original["report_sha256"][f"{task}/{arm}_seed{seed}"]:
                    raise ValueError("closed learned report changed")
                lr = json.loads(path.read_text())
                if paired_inventory(lr["provenance"]["cache"]) != paired_inventory(meta["cache"]):
                    raise ValueError("actual windows/labels/subjects differ from the learned campaign")
                learned[arm][task][seed] = lr
                hashes[f"{task}/{arm}_seed{seed}"] = checksum
    result = {"protocol": PROTOCOL, "source_commit": args.expected_commit, "historical_eval_commit": EVAL_COMMIT,
              "predecessor_sha256": COMPARISON_SHA, "initial_state_sha256": INITIAL_SHA, "report_sha256": hashes,
              "comparisons": {arm: compare_to_random(random, learned[arm]) for arm in learned}}
    check_pipeline(args.expected_commit)
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("extract", "probe", "compare"), required=True)
    ap.add_argument("--expected-commit", required=True)
    ap.add_argument("--task", choices=old.DATASETS)
    ap.add_argument("--model-seed", type=int, choices=(0, 1))
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--learned-root", type=Path)
    ap.add_argument("--root", type=Path, action="append")
    ap.add_argument("--epn-root", type=Path, action="append")
    ap.add_argument("--uci-root", type=Path)
    ap.add_argument("--compare-after", action="store_true")
    args = ap.parse_args(argv)
    if args.compare_after and args.stage != "probe":
        ap.error("compare-after requires probe")
    if (args.stage == "compare" or args.compare_after) and not args.learned_root:
        ap.error("comparison requires learned-root")
    if args.stage == "compare":
        compare(args)
    elif args.task is None or args.model_seed is None:
        ap.error("extract/probe require task and model-seed")
    else:
        run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
