#!/usr/bin/env python3
"""Describe independent-run scalar divergence; no gates or recipes are changed.

Run from any directory; only the adjacent results.json is written. Uses stdlib.
Metrics are read from the tracked gzip archives, not local untracked copies.
"""
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import tempfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "benchmarks_completed_bf16_20261118"
RUNS = ("benchmark/C0", "benchmark/CN", "benchmark/CR", "resume/CN", "resume/CR")
FIELDS = ("total", "grad_norm", "pose", "log_rms", "band_shape", "log_env",
          "ms_fast", "ms_slow", "contrast_global_grad_base",
          "contrast_global_grad_nce", "contrast_global_grad_rnc",
          "contrast_projector_grad_norm", "contrast_nce", "contrast_rnc",
          "contrast_global_cos_nce_base", "contrast_global_cos_rnc_base",
          "contrast_nce_weight", "contrast_rnc_weight",
          "grad_clip_factor", "jepa")
JOBS = dict(zip(RUNS, (64771457, 64771478, 64771595, 64771859, 64772690)))
RTOL, ATOL = 1e-4, 1e-6


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {k: v for key, item in value.items()
                for k, v in flatten(item, f"{prefix}.{key}" if prefix else key).items()}
    return {prefix: value}


def differences(a, b):
    aa, bb = flatten(a), flatten(b)
    return {k: {"left": aa.get(k), "right": bb.get(k)}
            for k in sorted(aa.keys() | bb.keys()) if aa.get(k) != bb.get(k)}


def quantiles(vals):
    if not vals:
        return None
    vals = sorted(vals)
    result = {}
    for q in (0, .5, .9, .95, .99, 1):
        h = (len(vals) - 1) * q
        lo, hi = math.floor(h), math.ceil(h)
        result[str(q)] = vals[lo] + (h - lo) * (vals[hi] - vals[lo])
    return result


def field_stats(left, right, field, lo, hi):
    entries = []
    missing = []
    for step in range(lo, hi + 1):
        a, b = left[step], right[step]
        if field not in a or field not in b:
            missing.append(step)
            continue
        aa, bb = float(a[field]), float(b[field])
        if not math.isfinite(aa) or not math.isfinite(bb):
            raise ValueError((field, step, aa, bb))
        diff = abs(aa - bb)
        scale = max(abs(aa), abs(bb))
        entries.append({"step": step, "left": aa, "right": bb,
                        "absolute_difference": diff,
                        "symmetric_relative_difference": diff / scale if scale else 0.,
                        "relative_to_left": diff / abs(aa) if aa else (0. if bb == 0 else None),
                        "exact_equal": aa == bb,
                        "within_frozen_tolerance": math.isclose(aa, bb, rel_tol=RTOL, abs_tol=ATOL)})
    if not entries:
        return {"n": 0, "missing_steps": missing}
    exact = [e for e in entries if not e["exact_equal"]]
    fail = [e for e in entries if not e["within_frozen_tolerance"]]
    return {
        "n": len(entries), "missing_steps": missing,
        "first_exact_difference": exact[0] if exact else None,
        "exact_difference_count": len(exact),
        "first_outside_frozen_tolerance": fail[0] if fail else None,
        "outside_frozen_tolerance_count": len(fail),
        "outside_frozen_tolerance_steps": [e["step"] for e in fail],
        "left_quantiles": quantiles([e["left"] for e in entries]),
        "right_quantiles": quantiles([e["right"] for e in entries]),
        "absolute_difference_quantiles": quantiles([e["absolute_difference"] for e in entries]),
        "symmetric_relative_difference_quantiles": quantiles([e["symmetric_relative_difference"] for e in entries]),
        "relative_to_left_quantiles": quantiles([e["relative_to_left"] for e in entries if e["relative_to_left"] is not None]),
        "max_absolute": max(entries, key=lambda e: e["absolute_difference"]),
        "max_symmetric_relative": max(entries, key=lambda e: e["symmetric_relative_difference"]),
    }


def compare(name_a, name_b, lo, hi, rows, configs):
    a, b = rows[name_a], rows[name_b]
    fields = {f: field_stats(a, b, f, lo, hi) for f in FIELDS}
    origin_diff = [s for s in range(lo, hi + 1) if a[s]["origin"] != b[s]["origin"]]
    return {
        "left": name_a, "right": name_b, "first_step": lo, "last_step": hi,
        "origins_identical": not origin_diff, "origin_difference_steps": origin_diff,
        "lr_identical": all(a[s]["lr"] == b[s]["lr"] for s in range(lo, hi + 1)),
        "config_differences": differences(configs[name_a], configs[name_b]),
        "frozen_total_or_grad_norm_failing_steps": sorted(set(
            fields["total"]["outside_frozen_tolerance_steps"] + fields["grad_norm"]["outside_frozen_tolerance_steps"])),
        "fields": fields,
    }


def main():
    rows, configs, run_meta, pins = {}, {}, {}, {}
    for name in RUNS:
        p = SOURCE / name
        raw_gz = (p / "metrics.jsonl.gz").read_bytes()
        raw = gzip.decompress(raw_gz)
        rr = [json.loads(line) for line in raw.splitlines() if line]
        rows[name] = {r["step"]: r for r in rr}
        assert len(rr) == 576 and list(rows[name]) == list(range(1, 577)), name
        configs[name] = json.loads((p / "config.json").read_bytes())
        summary = json.loads((p / "summary.json").read_bytes())
        for file in ("metrics.jsonl.gz", "config.json", "summary.json", "origin_sessions.json"):
            data = (p / file).read_bytes()
            pins[f"{name}/{file}"] = {"bytes": len(data), "sha256": digest(data)}
        flat = flatten(configs[name])
        settings = {k: v for k, v in flat.items()
                    if any(s in k.lower() for s in ("dropout", "amp_bf16", "determin", "seed", "checkpoint"))}
        run_meta[name] = {
            "job_id": JOBS[name], "records": len(rr),
            "decompressed_metrics_sha256": digest(raw),
            "uncompressed_local_matches_archive": ((p / "metrics.jsonl").read_bytes() == raw
                                                    if (p / "metrics.jsonl").exists() else None),
            "origin_sessions_sha256": digest((p / "origin_sessions.json").read_bytes()),
            "origin_stream_sha256": digest(canonical([r["origin"] for r in rr])),
            "zero_effective_nce_and_rnc_weights_steps_1_256": all(
                r["contrast_nce_weight"] == r["contrast_rnc_weight"] == 0 for r in rr[:256]),
            "projector_norm_zero_steps_1_256_when_observed": all(
                r["contrast_projector_grad_norm"] == 0 for r in rr[:256] if "contrast_projector_grad_norm" in r),
            "settings_from_config": settings,
            "deterministic_algorithms_runtime_setting": "not recorded in config, summary or metrics",
            "environment": summary["environment"], "code": summary["code"],
            "provenance": summary["contrastive"]["provenance"],
            "initial_sha256": summary["contrastive"]["initial_sha256"],
            "world_size": summary["world_size"], "batch_per_process": summary["batch_per_process"],
            "summary_steps": summary["steps"], "summary_stopped": summary["stopped"],
            "hostname": "not recorded in collected logs/accounting",
            "per_step_wall_clock_timestamp": "not recorded; t_step_s and t_data_s are durations",
        }
    for filename in ["accounting.txt"] + [f"contrastive_gpu_{j}.out" for j in JOBS.values()]:
        data = (SOURCE / filename).read_bytes()
        pins[filename] = {"bytes": len(data), "sha256": digest(data)}
    pairwise = [compare(a, b, 1, 256, rows, configs) for a, b in itertools.combinations(RUNS, 2)]
    same_arm = {}
    for arm in ("CN", "CR"):
        a, b = "benchmark/" + arm, "resume/" + arm
        same_arm[arm] = {
            "before_restart_1_544": compare(a, b, 1, 544, rows, configs),
            "last32_before_restart_513_544": compare(a, b, 513, 544, rows, configs),
            "after_restart_545_576": compare(a, b, 545, 576, rows, configs),
            "boundary_values": {str(s): {f: {"continuous": rows[a][s].get(f), "split": rows[b][s].get(f)}
                                          for f in ("total", "grad_norm", "contrast_global_grad_base")}
                                for s in (543, 544, 545, 546)},
        }
    cohort = HERE.parent / "calibrated_cohort_20261118_bf16"
    split_config_checks = {}
    for arm in ("CN", "CR"):
        full = json.loads((cohort / (arm + ".config.json")).read_bytes())
        split = json.loads((cohort / (arm + ".split.config.json")).read_bytes())
        split_config_checks[arm] = {"cohort_full_vs_split": differences(full, split),
                                    "saved_vs_cohort_full": differences(configs["benchmark/"+arm], full)}
        for file in (arm + ".config.json", arm + ".split.config.json"):
            data = (cohort / file).read_bytes()
            pins[f"../calibrated_cohort_20261118_bf16/{file}"] = {"bytes": len(data), "sha256": digest(data)}
    manifest_path = HERE.parent / "compatibility_20261118/manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    repo = HERE.parents[3]
    source_rows = [{"path": r["path"], "sha256": digest((repo / r["path"]).read_bytes())}
                   for r in manifest["current_sources"]]
    current_digest = digest(b"".join(r["path"].encode() + b"\0" + bytes.fromhex(r["sha256"])
                                   for r in source_rows))
    result = {
        "purpose": "Post hoc descriptive local diagnosis, not a replacement gate or proof of cause.",
        "frozen_scalar_tolerance": {"rtol": RTOL, "atol": ATOL, "rule": "math.isclose, max absolute pair scale"},
        "quantiles": "linear interpolation at (n-1)*q; relative differences are fractions, not percent",
        "metrics_note": "Effective contrastive weights are zero through 256; unweighted losses and diagnostic component gradients may be nonzero.",
        "resume_note": "The split job independently trains steps 1..544 before restarting; its state at 544 is not the continuous job's state. Divergence already present before 544 prevents attributing post-544 differences uniquely to resume.",
        "limits": ["No checkpoint tensors, masks, activations or gradient vectors are analyzed here.",
                   "Identical origin IDs do not independently certify identical stochastic augmentation/masks.",
                   "These runs do not isolate numerical precision, kernel nondeterminism or implementation errors.",
                   "No causal attribution to bf16, no significance testing, no gate or threshold change."],
        "source_directory_relative_to_script": "../benchmarks_completed_bf16_20261118",
        "source_files": pins, "runs": run_meta,
        "frozen_source_check": {
            "manifest_sha256": digest(manifest_path.read_bytes()),
            "source_count": len(source_rows), "source_sha256_recomputed": current_digest,
            "all_pinned_files_identical": source_rows == manifest["current_sources"],
            "all_summary_source_digests_match": all(
                current_digest == m["provenance"]["source_sha256"] for m in run_meta.values()),
            "digest_matches_manifest": current_digest == manifest["current_source_sha256"],
        },
        "accounting_collection_timestamp": (SOURCE / "accounting.txt").read_text().splitlines()[0],
        "split_config_checks": split_config_checks,
        "all_pairs_zero_weight_1_256": pairwise,
        "same_arm_independent_runs": same_arm,
        "all_five_origin_streams_identical_1_576": len({x["origin_stream_sha256"] for x in run_meta.values()}) == 1,
        "all_five_origin_session_maps_identical": len({x["origin_sessions_sha256"] for x in run_meta.values()}) == 1,
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if "--plot" in sys.argv:
        # Optional figure; numerical results above need no third-party dependencies.
        with tempfile.TemporaryDirectory(prefix=".plot-config-", dir=HERE) as cache:
            os.environ["MPLCONFIGDIR"] = cache
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
            for ax, arm, color in zip(axes, ("CN", "CR"), ("#1765a4", "#a54b08")):
                a, b = rows["benchmark/" + arm], rows["resume/" + arm]
                xx = list(range(1, 577))
                yy = [100 * abs(a[s]["grad_norm"] - b[s]["grad_norm"]) /
                      max(abs(a[s]["grad_norm"]), abs(b[s]["grad_norm"])) for s in xx]
                ax.plot(xx, yy, color=color, linewidth=.9)
                ax.axvline(256, color="#666666", linestyle="--", linewidth=1)
                ax.axvline(544, color="#222222", linestyle="--", linewidth=1)
                ax.axvspan(544, 576, color="#999999", alpha=.12)
                ax.set_title(f"{arm}: esecuzione continua contro esecuzione indipendente con ripresa", loc="left", fontsize=11)
                ax.set_ylim(0, max(yy) * 1.15)
                ax.set_ylabel("Scarto della norma (%)")
                ax.grid(axis="y", alpha=.2)
                ax.spines[["top", "right"]].set_visible(False)
            axes[0].annotate("fine peso contrastivo zero", xy=(256, .96), xycoords=("data", "axes fraction"),
                             xytext=(-4, 0), textcoords="offset points", fontsize=9, ha="right", va="top")
            axes[0].annotate("riavvio dopo 544", xy=(544, .96), xycoords=("data", "axes fraction"),
                             xytext=(-4, 0), textcoords="offset points", fontsize=9, ha="right", va="top")
            axes[1].set_xlabel("Passo di training")
            axes[1].set_xticks([1, 100, 200, 256, 350, 450, 544, 576])
            axes[1].set_xlim(1, 576)
            fig.suptitle("La divergenza precede il riavvio · diagnostica post hoc", fontsize=14, y=.98)
            fig.text(.07, .015, "Scarto = 100 × |norma continua − norma split| / max(|norma continua|, |norma split|).\n"
                     "Stesso stream e ricetta per ogni coppia; il grafico non identifica la causa e non sostituisce i criteri congelati.", fontsize=9)
            fig.tight_layout(rect=(0, .08, 1, .94), h_pad=2.3)
            fig.savefig(HERE / "gradient_divergence.png", dpi=170)
            plt.close(fig)
    for pair in pairwise:
        f = pair["fields"]["grad_norm"]
        print(pair["left"], pair["right"], "grad_fail", f["outside_frozen_tolerance_count"],
              "grad_max_relative_pct", f["symmetric_relative_difference_quantiles"]["1"] * 100,
              "loss_fail", pair["fields"]["total"]["outside_frozen_tolerance_count"])
    for arm, regions in same_arm.items():
        for region in ("before_restart_1_544", "last32_before_restart_513_544", "after_restart_545_576"):
            r = regions[region]
            print(arm, region, {f: {"fail": r["fields"][f]["outside_frozen_tolerance_count"],
                                  "p50_pct": 100*r["fields"][f]["symmetric_relative_difference_quantiles"]["0.5"],
                                  "max_pct": 100*r["fields"][f]["symmetric_relative_difference_quantiles"]["1"]}
                                for f in ("total", "grad_norm")})


if __name__ == "__main__":
    main()
