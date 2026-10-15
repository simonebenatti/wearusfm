#!/usr/bin/env python3
"""Audit existing Step2C metrics on CPU; no training or threshold fitting."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.harness.training_stability import audit_curve


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=METRICS_JSONL")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error("output already exists")
    runs = {}
    for spec in args.run:
        name, sep, filename = spec.partition("=")
        if not sep or not name or name in runs:
            ap.error("run names must be nonempty, unique, and followed by =PATH")
        path = Path(filename).resolve()
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        runs[name] = {"metrics_sha256": hashlib.sha256(raw).hexdigest(), "metrics_path": str(path),
                      **audit_curve(rows)}
    result = {"audit": "step2c_training_stability_descriptive_v1", "runs": runs,
              "note": "The historical collaudo verdict and downstream criteria are unchanged."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, r in runs.items():
        if not r["finite_nonnegative"]:
            print(f"{name}: numerical failure at updates {r['invalid_updates']}")
            continue
        c = r["critical_window"]
        value = r["v1"]["max_rise"]
        ratio_text = f"{value:.3f}" if value is not None else "undefined/nonfinite"
        print(f"{name}: v1 ratio={ratio_text}; "
              f"critical {c['first_update']}..{c['last_update']}, mean={c['mean']:.5f}, "
              f"median={c['median']:.5f}, ratio without max={c['ratio_without_largest_batch']}; "
              f"clipped={r['gradient']['clipped_fraction']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
