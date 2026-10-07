#!/usr/bin/env python3
"""Compare historical and corrected LONG masks locally; no signal files needed.

Historical code is loaded from a pinned local Git commit, without changing the
checkout. Outputs are descriptive: token coverage is not expected to be uniform
when interval starts are uniform. Requires NumPy and Matplotlib (for figures).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wearusfm.ingest import capgmyo, emg2pose  # noqa: E402
from wearusfm.ingest.common import montage_to_dict  # noqa: E402
from wearusfm.model import channel_codes as CC  # noqa: E402
from wearusfm.training import masking as current  # noqa: E402

BEFORE = "28674c9f3c931af606b479ce4911ae14d77f9a49"
MODULE_PATH = "src/wearusfm/training/masking.py"


def historical_module(ref: str):
    sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--verify", f"{ref}^{{commit}}"], text=True).strip()
    source = subprocess.check_output(["git", "-C", str(REPO), "show", f"{sha}:{MODULE_PATH}"], text=True)
    module = types.ModuleType("_wearusfm_historical_masking")
    sys.modules[module.__name__] = module  # dataclasses resolves its defining module
    exec(compile(source, f"git:{sha}:{MODULE_PATH}", "exec"), module.__dict__)
    return module, sha, hashlib.sha256(source.encode()).hexdigest()


def layout_for(topology: str):
    montage = (emg2pose.build_montage_metadata("diagnostic", "synthetic", "left") if topology == "ring"
               else capgmyo.build_montage_metadata(1))
    meta = montage_to_dict(montage, [True] * montage.n_channels)
    return CC.layout_from_montage(meta), CC.anatomy_codes(meta).compartment_weights.argmax(axis=1)


def measure(module, topology: str, patches: int, seeds: int, mode: str):
    layout, compartment = layout_for(topology)
    spec = (module.MaskSpec(0.5, 0, 0, 1, 0) if mode == "long_only"
            else module.MaskSpec.d10_proposal(0.5))
    rvq_on = mode == "d10_rvq_on"
    coverage = np.zeros(patches, dtype=np.int64)
    fractions, examples, intervals = [], [], []
    original = module.draw_tube

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[4] == spec.long[0]:  # LONG tubes; slabs do not call this helper
            intervals.append([int(result[1]), int(result[2])])
        return result

    module.draw_tube = record
    try:
        for seed in range(seeds):
            visible, kind = module.generate_mask(layout, compartment, patches, patches + 16, rvq_on, spec,
                                                 np.random.default_rng(seed))
            if not np.array_equal(visible, kind == module.VISIBLE) or (kind[:, patches:] != module.VISIBLE).any():
                raise AssertionError("visibility/padding invariant violated")
            coverage += (kind[:, :patches] == module.LONG).sum(axis=0)
            fractions.append(float((kind[:, :patches] != module.VISIBLE).mean()))
            if seed == 0:
                examples = (kind[:, :patches] == module.LONG).astype(np.uint8)
    finally:
        module.draw_tube = original
    half = patches // 2
    first, second = int(coverage[:half].sum()), int(coverage[half:].sum())
    cap = min(spec.long[1], half)
    intervals = np.asarray(intervals, dtype=int).reshape(-1, 2)
    durations = intervals[:, 1] - intervals[:, 0]
    if not len(durations) or not ((durations >= spec.long[0]) & (durations <= cap)).all():
        raise AssertionError("LONG tube duration invariant violated")
    if not ((intervals[:, 0] >= 0) & (intervals[:, 1] <= patches)).all():
        raise AssertionError("LONG tube outside valid window")
    if max(fractions) > spec.ratio * 1.1 + 1e-12:
        raise AssertionError("configured budget tolerance violated")
    result = {
        "topology": topology, "channels": layout.n_channels, "patches": patches,
        "window_ms": patches * 25, "seeds": list(range(seeds)), "mode": mode,
        "long_tokens_first_half": first, "long_tokens_second_half": second,
        "share_long_tokens_second_half": second / (first + second) if first + second else None,
        "long_coverage_per_patch": (coverage / (seeds * layout.n_channels)).tolist(),
        "hidden_fraction_mean": float(np.mean(fractions)),
        "hidden_fraction_min": min(fractions), "hidden_fraction_max": max(fractions),
        "proposed_long_tubes": len(intervals), "long_duration_min": int(durations.min()),
        "long_duration_max": int(durations.max()), "long_start_min": int(intervals[:, 0].min()),
        "long_start_max": int(intervals[:, 0].max()), "long_end_max": int(intervals[:, 1].max()),
        "checks": {"individual_duration_cap": True, "padding": True, "visibility": True, "budget_upper_bound": True},
    }
    return result, examples


def figures(out: Path, cases: list[dict], examples: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    lengths = [40, 80, 160]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), layout="constrained")
    for row, patches in enumerate(lengths):
        for col, version in enumerate(("before", "after")):
            ax = axes[row, col]
            ax.imshow(examples[("ring", patches, "long_only", version)], origin="upper", aspect="auto",
                      extent=[0, patches * 25, 16.5, 0.5], cmap=ListedColormap(["#f1f4f8", "#326b9b"]), vmin=0, vmax=1)
            ax.axvline(patches * 12.5, color="#d45e2c", ls="--", lw=1)
            ax.set_title(f"{'Previous' if version == 'before' else 'Corrected'} | {patches * 25 / 1000:g} s | seed 0")
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("channel")
    fig.suptitle("LONG masks: blue = hidden, dashed line = half window | 16-channel ring, LONG only")
    fig.savefig(out / "long_examples.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), layout="constrained")
    for row, patches in enumerate(lengths):
        for col, topology in enumerate(("ring", "grid")):
            ax = axes[row, col]
            case = next(c for c in cases if c["topology"] == topology and c["patches"] == patches and c["mode"] == "long_only")
            time_ms = (np.arange(patches) + 0.5) * 25
            for version, color in (("before", "#a34b38"), ("after", "#326b9b")):
                ax.plot(time_ms, case[version]["long_coverage_per_patch"], label=version, color=color)
            ax.axvline(patches * 12.5, color="grey", ls="--", lw=1)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{topology} | {patches * 25 / 1000:g} s | {len(case['after']['seeds'])} seeds")
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("fraction of channels hidden as LONG")
            ax.grid(alpha=0.2)
            ax.legend()
    fig.suptitle("Realized LONG coverage | uniform interval starts do not imply uniform token coverage")
    fig.savefig(out / "long_coverage.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before-ref", default=BEFORE)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    if args.seeds < 1:
        ap.error("seeds must be positive")
    out = args.out_dir.resolve()
    if out == REPO or REPO in out.parents:
        ap.error("use an output directory outside the repository")
    if out.exists() and any(out.iterdir()):
        ap.error("output must be new/empty")
    before, before_sha, before_hash = historical_module(args.before_ref)
    cases, examples = [], {}
    for topology in ("ring", "grid"):
        for patches in (40, 41, 80, 160):
            for mode in ("long_only", "d10_rvq_off", "d10_rvq_on"):
                case = {"topology": topology, "patches": patches, "mode": mode}
                for version, module in (("before", before), ("after", current)):
                    result, example = measure(module, topology, patches, args.seeds, mode)
                    case[version] = result
                    examples[(topology, patches, mode, version)] = example
                cases.append(case)
        print(f"{topology}: compared", flush=True)
    report = {
        "kind": "descriptive local replication of LONG placement; no training or signal data",
        "before_commit": before_sha, "before_masking_sha256": before_hash,
        "after_code_base_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "after_masking_sha256": hashlib.sha256((REPO / MODULE_PATH).read_bytes()).hexdigest(),
        "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameters": {"patch_ms": 25, "ratio": 0.5, "seeds": list(range(args.seeds)), "padding_patches": 16},
        "limitations": ["No measure of learning quality or downstream accuracy.",
                        "Same seeds/configuration do not produce identical random draw sequences after the fix.",
                        "Overlap and rejection alter realized durations/type shares; limits apply to individual proposed tubes.",
                        "Uniform interval starts do not imply uniform per-token coverage.",
                        "Only LONG tubes are counted in LONG metrics; RVQ slabs have a separate SLAB label.",
                        "Grid metadata uses the current ingest layout; this test does not establish its anatomical orientation."],
        "cases": cases,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not args.no_plots:
        figures(out, cases, examples)
    print(json.dumps({"output": str(out), "cases": len(cases), "mask_generations": len(cases) * args.seeds * 2}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
