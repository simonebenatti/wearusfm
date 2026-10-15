#!/usr/bin/env python3
"""Describe the sparse, historical Step 1 weighted-gradient diagnostics on CPU."""
import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TRAIN_COMMIT = "69fa2da8757b322ea74875b4e9d104a89bbf6d5d"
TRAINER_PATH = "src/wearusfm/training/run.py"
TRAINER_SHA256 = "e5aaafe7440464bcfc17a2e95845fae1e8b2f05d534a87088619bb1ac754d20e"
RUNS = ("A_seed0", "A_seed1", "B_seed0", "B_seed1")
CHECKPOINTS = {
    "A_seed0": "26ee7e440b01c77b35b9cf198110a4341e267340d377ca32feb80ae6b475b611",
    "A_seed1": "b72a122ea24d01af2830de7f1c76dbcfd7810991e75f9ba015ca5e93bdeeb4d2",
    "B_seed0": "b264df4b75f7047980212ab068fe524148e71dbdef442d4e75e95648f841d3cd",
    "B_seed1": "27b3a819c35f798992b2bcddf7f48998be03d9dc4bc113b79e5170c199577a29",
}
INPUT_SHA256 = {
    "manifest": "853f35ac5252ef901be390385164c6dc5658f45270441eb74557c6be317ed6d3",
    "scales": "76486d882b5deeca66f152c3541534e9f35efca35f944f6f8677b6478b877486",
    "window_seconds": "e660e324a28a339c038ed07dbaa7d4c358f78f613cd32226eea3f522119a48b0",
    "calibration": "4640dfcf2c6819fb8ba2870e1506f08e4b024cdf38dda267c4cd37652cbc1f89",
    "splits": "e064a1102bb4a7dd1b01a70caefa8fa2c58d8c9c751bafeafe6bd2892fff7a30",
}
STEPS = 4000
CADENCE = 500
EXPECTED_LOG_STEPS = list(range(1, STEPS + 1, CADENCE))
GRAD_KEYS = ("shared_grad_jepa", "shared_grad_anchors", "shared_grad_keep")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value, what):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError(f"{what}: expected a SHA256 digest")
    return value


def historical_source():
    raw = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{TRAIN_COMMIT}:{TRAINER_PATH}"])
    if sha256(raw) != TRAINER_SHA256:
        raise ValueError("historical trainer source hash changed")
    return {"commit": TRAIN_COMMIT, "path": TRAINER_PATH, "sha256": TRAINER_SHA256,
            "gradient_lines": [306, 315], "optimizer_and_log_lines": [316, 328]}


def validate_config(name, cfg, provenance):
    arm, seed = name[0], int(name[-1])
    expected = {"seed": seed, "max_steps": STEPS, "batch_size": 32, "keep_grad_every": CADENCE,
                "grad_clip": 1.0, "pairing_audit": True, "datasets": ["emg2qwerty"],
                "keep_calibration_sha256": INPUT_SHA256["calibration"]}
    if any(cfg.get(k) != value for k, value in expected.items()):
        raise ValueError(f"{name}: configuration differs from frozen Step 1 protocol")
    expected_jepa = {"target": "b", "anchor_weight": 0.2, "keep_weight": 0.0 if arm == "A" else 0.05,
                     "loss_per_window": True, "ema_momentum": 0.996, "rvq_targets": "slab"}
    if any(cfg.get("jepa", {}).get(k) != value for k, value in expected_jepa.items()):
        raise ValueError(f"{name}: loss weights/definition differ from frozen Step 1 protocol")
    model = cfg.get("model", {})
    if model.get("keep_readout") is not True or model.get("multiscale_anchor") is not True or model.get("rvq_codes") is not None:
        raise ValueError(f"{name}: unexpected optional heads")
    if cfg.get("loader", {}).get("multiscale_anchor") is not True:
        raise ValueError(f"{name}: missing multiscale targets")
    if (provenance.get("source_commit") != TRAIN_COMMIT or provenance.get("arm") != arm
            or provenance.get("model_seed") != seed or provenance.get("input_sha256") != INPUT_SHA256
            or provenance.get("checkpoint_sha256") != CHECKPOINTS[name]):
        raise ValueError(f"{name}: training provenance differs from frozen artifacts")


def summary(values):
    return {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def analyze_rows(name, rows, cfg):
    """All norms are PRE-clipping; grad_norm covers all parameters, not just the shared encoder."""
    expected = [] if name.startswith("A") else EXPECTED_LOG_STEPS
    if [row.get("step") for row in rows] != expected:
        raise ValueError(f"{name}: expected diagnostic log steps {expected}")
    if not rows:
        return {"status": "excluded_by_design", "diagnostic_count": 0,
                "reason": "The historical diagnostic requires keep_weight > 0; A has weight zero. No A component norms were recorded."}
    samples = []
    for row in rows:
        step = row["step"]
        for key in (*GRAD_KEYS, "grad_norm"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} step {step}: {key} must be finite and nonnegative")
        if row["shared_grad_jepa"] <= 0:
            raise ValueError(f"{name} step {step}: zero JEPA norm prevents finite norm ratios")
        sample = {"log_step": step, "update_index": step - 1,
                  **{k: row[k] for k in GRAD_KEYS}, "grad_norm_all_parameters": row["grad_norm"],
                  "anchors_over_jepa": row["shared_grad_anchors"] / row["shared_grad_jepa"],
                  "keep_over_jepa": row["shared_grad_keep"] / row["shared_grad_jepa"],
                  "all_parameter_gradient_clipped": row["grad_norm"] > cfg["grad_clip"]}
        if any(not math.isfinite(sample[key]) for key in ("anchors_over_jepa", "keep_over_jepa")):
            raise ValueError(f"{name} step {step}: nonfinite norm ratio")
        samples.append(sample)
    return {"status": "complete_sparse_diagnostics", "diagnostic_count": len(samples), "finite_nonnegative": True,
            "coverage": {"logged_updates": STEPS, "sampled_update_indices": [s["update_index"] for s in samples],
                         "fraction": len(samples) / STEPS},
            **aggregate(samples, cfg["grad_clip"]), "samples": samples}


def aggregate(samples, grad_clip):
    fields = (*GRAD_KEYS, "anchors_over_jepa", "keep_over_jepa", "grad_norm_all_parameters")
    clipped = [s["log_step"] for s in samples if s["all_parameter_gradient_clipped"]]
    return {"norms_and_ratios": {key: summary([s[key] for s in samples]) for key in fields},
            "clipping_at_diagnostic_samples": {"threshold": grad_clip, "count": len(clipped),
                                                "fraction": len(clipped) / len(samples), "log_steps": clipped}}


def analyze_export(export_dir):
    export_dir = Path(export_dir).resolve()
    manifest_path = export_dir / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if set(manifest["runs"]) != set(RUNS):
        raise ValueError("export manifest must identify exactly the four frozen runs")
    results, normalized = {}, []
    for name in RUNS:
        metrics = export_dir / f"{name}.jsonl"
        raw = metrics.read_bytes()
        meta = manifest["runs"][name]
        _digest(meta["metrics_sha256"], f"{name} full metrics")
        if meta["metrics_lines"] != STEPS or meta["export_sha256"] != sha256(raw):
            raise ValueError(f"{name}: full log endpoint or sparse export hash mismatch")
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        if meta["export_lines"] != len(rows):
            raise ValueError(f"{name}: sparse export line count mismatch")
        config_path, provenance_path = (export_dir / name / fn for fn in ("config.json", "run_provenance.json"))
        config_raw, provenance_raw = config_path.read_bytes(), provenance_path.read_bytes()
        cfg, provenance = json.loads(config_raw), json.loads(provenance_raw)
        validate_config(name, cfg, provenance)
        comparable = {**cfg, "seed": 0, "jepa": {**cfg["jepa"], "keep_weight": 0.0}}
        normalized.append(comparable)
        results[name] = {"weights": {"jepa": 1.0, "anchors": cfg["jepa"]["anchor_weight"], "keep": cfg["jepa"]["keep_weight"]},
                         "provenance": {"metrics_full_sha256_reported_by_exporter": meta["metrics_sha256"],
                                        "metrics_full_lines_reported_by_exporter": meta["metrics_lines"],
                                        "sparse_export_sha256": sha256(raw), "sparse_export_lines": len(rows),
                                        "config_sha256": sha256(config_raw), "run_provenance_sha256": sha256(provenance_raw),
                                        "source_commit": provenance["source_commit"], "checkpoint_sha256": provenance["checkpoint_sha256"]},
                         **analyze_rows(name, rows, cfg)}
    if any(cfg != normalized[0] for cfg in normalized[1:]):
        raise ValueError("run configurations differ beyond model seed and keep weight")
    pooled = [sample for name in ("B_seed0", "B_seed1") for sample in results[name]["samples"]]
    return {"analysis": "step1_weighted_shared_gradients_descriptive_v1", "training_source": historical_source(),
            "export_manifest_sha256": sha256(manifest_raw), "runs": results,
            "pooled_B_descriptive": aggregate(pooled, 1.0),
            "conventions": {"shared_parameters": ["tokenizer", "identity", "local", "pool", "backbone"],
                            "jepa": "norm(grad(jepa_loss))",
                            "anchors": "norm(grad((total - jepa) - keep_weight * keep_loss)); anchor_weight already included",
                            "keep": "norm(grad(keep_weight * keep_loss)); do not multiply by keep_weight again",
                            "step": "log step 501 reports update index 500; norms measured before backward/clipping/optimizer.step",
                            "grad_norm": "Pre-clipping norm of ALL trainable parameters, as returned by clip_grad_norm_"},
            "limits": ["Only eight batches per B seed (0.2% of its 4000 updates); no A component-gradient observations.",
                       "Norms do not add; ratios are not percentage contributions or evidence about gradient alignment/conflict.",
                       "The shared-encoder norms exclude the decoder and prediction heads; grad_norm includes them.",
                       "Clipping rates here describe diagnostic samples only, not all 4000 updates.",
                       "Pre-clipping gradient norms do not determine AdamW parameter updates, moments, or weight decay.",
                       "Sparse measurements cannot establish dominance at every update, causal effects, or downstream utility.",
                       "Full-log hashes/counts are reported by the read-only exporter; only sparse exports and metadata are local."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True, help="Directory containing manifest.json, sparse logs and per-run metadata")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output already exists")
    result = analyze_export(args.export_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, run in result["runs"].items():
        if run["status"] == "excluded_by_design":
            print(f"{name}: no component diagnostics by design")
        else:
            ratios = run["norms_and_ratios"]
            print(f"{name}: anchors/JEPA median={ratios['anchors_over_jepa']['median']:.6g}, "
                  f"keep/JEPA median={ratios['keep_over_jepa']['median']:.6g}; "
                  f"clipped samples={run['clipping_at_diagnostic_samples']['count']}/8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
