"""CPU train-stream audit: identities, masking/QC and pose support, no encoder.

Replays the real logical rank/worker streams serially (no GPU or submission).
It checks candidate availability, not projection finiteness or signal quality.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from collections import Counter
import hashlib
import json
import resource
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.training import run as R
from wearusfm.training.contrastive_objectives import ContrastiveConfig, ContrastiveRuntime, physical_pair_mask
from wearusfm.training.contrastive_memory import identities_from_batch, admissible_mask, candidate_diagnostics
from wearusfm.data.pose_targets import N_JOINTS
from wearusfm.model.rank_contrast import anchor_support, rank_contrast_count


def audit_batches(global_batches, config, expected_datasets, patch_ms=25.):
    runtime = ContrastiveRuntime(asdict(config))
    runtime.expected_datasets = tuple(sorted(expected_datasets))
    records, digest = [], hashlib.sha256()
    calibration_panel = None
    for step, batches in enumerate(global_batches):
        ids, query, key, masks = [], [], [], []
        for batch in batches:
            if batch.index != step:
                raise ValueError("audit stream index differs from optimizer step")
            ids.extend(identities_from_batch(batch))
            offset = 0
            for s, (c, p) in enumerate(zip(batch.counts, batch.n_patches)):
                qc = batch.qc_valid[offset:offset+c]
                key.append(bool(qc.any()))
                query.append(bool((batch.visible[offset:offset+c, :p] & qc[:, None]).any()))
                if batch.pose is None or batch.pose[s] is None:
                    masks.append(torch.zeros(p, N_JOINTS, dtype=torch.bool))
                else:
                    masks.append(torch.as_tensor(batch.pose[s][1], dtype=torch.bool))
                offset += c
        k = torch.zeros(len(ids), 1)  # identity-only placeholders, NEVER training features
        kv = torch.tensor(key)
        snapshot = runtime.queue.snapshot(step, k, ids, kv)
        negatives = admissible_mask(ids, snapshot.identities, gap_seconds=config.gap_seconds)
        eligible = torch.tensor(query) & kv & (negatives.sum(1) >= config.min_negatives)
        pmax = max(m.shape[0] for m in masks)
        valid = torch.stack([torch.cat((m, torch.zeros(pmax-m.shape[0], N_JOINTS, dtype=torch.bool))) for m in masks])
        si, pi, *_ = anchor_support(valid, config.rnc_stride, "all")
        allowed, points = physical_pair_mask(ids, si, pi, patch_ms, "cpu")
        _, _, _, pairs, _, active = anchor_support(valid, config.rnc_stride, "all", allowed)
        rnc_windows = int(rank_contrast_count(valid, stride=config.rnc_stride, per_window=True, pair_allowed=allowed))
        stats = {d: [sum(i.dataset == d for i in ids), sum(bool(e) for i, e in zip(ids, eligible) if i.dataset == d)]
                 for d in sorted({i.dataset for i in ids})}
        runtime.step = step
        runtime.pending = (k, ids, kv)
        runtime.pending_stats = {"rnc_active": rnc_windows > 0, "datasets": stats}
        failure = runtime.before_update()
        if failure is None:
            runtime.commit()
        if step == 511:
            calibration_panel = runtime.coverage_report()
        record = {"step": step, "queries": len(ids), "eligible_nce": int(eligible.sum()), "datasets": stats,
                  "rnc_windows": rnc_windows, "rnc_anchors": int(active.sum()), "rnc_candidate_anchors": len(points),
                  "rnc_pairs": int(pairs[active].sum()), "duplicate_instants": len(points)-len(set(points)),
                  "rnc_max_multiplicity": max(Counter(points).values(), default=0),
                  "pose_recordings": len({ids[s].recording for s in si.tolist()}),
                  "pose_subjects": len({(ids[s].dataset, ids[s].subject) for s in si.tolist()})}
        record.update(candidate_diagnostics(ids, snapshot, negatives, step, datasets=expected_datasets))
        record.update({"contrast_rnc_" + key: record["rnc_" + key] for key in
                       ("windows", "anchors", "candidate_anchors", "pairs")})
        record["contrast_rnc_max_instant_multiplicity"] = record["rnc_max_multiplicity"]
        record["contrast_rnc_pairs_excluded_same_instant"] = sum(n * (n - 1) for n in Counter(points).values())
        record["contrast_rnc_duplicate_instants"] = record["duplicate_instants"]
        record["contrast_rnc_recordings"] = record["pose_recordings"]
        record["contrast_rnc_subjects"] = record["pose_subjects"]
        for suffix in ("", *("__" + dataset for dataset in expected_datasets)):
            denominator = record["contrast_nce_negative_count" + suffix]
            for numerator, metric in (("negative_age_sum", "negative_age_mean"), ("negative_current", "negative_current_fraction"),
                                      ("negative_same_subject", "negative_same_subject_fraction"),
                                      ("negative_same_recording", "negative_same_recording_fraction")):
                record["contrast_nce_" + metric + suffix] = record["contrast_nce_" + numerator + suffix] / max(denominator, 1)
        digest.update(json.dumps([i.to_dict() for i in ids], sort_keys=True).encode())
        digest.update(valid.numpy().tobytes())
        digest.update(np.asarray([query, key], dtype=bool).tobytes())
        records.append(record)
        if failure:
            break
    if not records:
        raise ValueError("empty audit stream")
    partial = runtime.coverage_report()
    return {"scope": "train candidate/support audit; no encoder, projection or training",
            "steps": len(records), "origin_support_sha256": digest.hexdigest(), "records": records,
            "expected_datasets": list(runtime.expected_datasets), "calibration_panel": calibration_panel,
            "coverage_completed_blocks": runtime.coverage_reports, "coverage_partial_block": partial,
            "passed": all(r["passed"] for r in runtime.coverage_reports)
                      and (partial["passed"] if partial["updates"] else bool(runtime.coverage_reports))
                      and (calibration_panel is None or calibration_panel["passed"])}


def descriptor_budget(cfg, world_size=4, workers=6, cache_cap=4):
    """Bound simultaneous EMG + pose mappings including live block-pool views.

    Reopening a cache entry is deterministic: scales/accepted starts use identity
    seeds, and the sampler never consumes RNG to maintain these caches.
    """
    if type(cache_cap) is not int or cache_cap < 1:
        raise ValueError("audit cache cap must be a positive integer")
    if cfg.loader.rvq_codes_root:
        raise ValueError("audit descriptor bound does not cover RVQ caches")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    cap = min(cfg.loader.open_sessions, cache_cap)
    per_loader = cap + cfg.loader.block_pool + 2 + (cache_cap + 2 if cfg.loader.pose_root else 0)
    bound = 128 + world_size * workers * per_loader
    if soft != resource.RLIM_INFINITY and bound > soft:
        raise ValueError(f"audit RLIMIT_NOFILE {soft} below conservative bound {bound}; no stream opened")
    return {"soft_limit": soft, "hard_limit": hard, "estimated_open_descriptor_bound": bound,
            "emg_cache_per_loader": cap, "pose_cache_per_loader": cache_cap,
            "logical_loaders": world_size * workers, "reserve": 128}


def replay(cfg, index, scales, window_s, steps, world_size=4, workers=6, cache_cap=4):
    """Same worker identities/seeds as DataLoader, interleaved without processes."""
    if cfg.batch_size % world_size or workers < 1:
        raise ValueError("invalid audit rank/worker geometry")
    budget = descriptor_budget(cfg, world_size, workers, cache_cap)
    streams = []
    for rank in range(world_size):
        rank_streams = []
        for worker in range(workers):
            loader = R.L.PretrainLoader(index, replace(cfg.loader, open_sessions=budget["emg_cache_per_loader"]), dict(scales), window_s)
            if loader.pose_store is not None:
                loader.pose_store._cache_size = budget["pose_cache_per_loader"]
            rank_streams.append(iter(R.L.batch_stream(loader, cfg.batch_size // world_size, seed=cfg.seed,
                worker=worker, n_workers=workers, segment=cfg.stream_segment, rank=rank)))
        streams.append(rank_streams)
    for step in range(steps):
        yield [next(streams[r][step % workers]) for r in range(world_size)]


def main():
    from contrastive_run import validate_recipe
    from contrastive_prepare import _read, source_digest, validate_input_bindings
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, action="append", required=True)
    ap.add_argument("--scales", type=Path, required=True)
    ap.add_argument("--window-seconds", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists() or a.steps != 1024:
        ap.error("new output and frozen 1024-step audit required")
    inputs = {"manifest": a.manifest, "scales": a.scales, "window_seconds": a.window_seconds}
    cfg = validate_recipe(_read(a.config), "calibration")
    pose_stats = Path(cfg.loader.pose_root) / "pose_stats.json"
    inputs["pose_stats"] = pose_stats
    pins = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in inputs.items()}
    validate_input_bindings(pins)
    frozen_source = source_digest()
    config_bytes_pin = hashlib.sha256(a.config.read_bytes()).hexdigest()
    cc = ContrastiveConfig(**cfg.contrastive)
    R.check_input_files(a.manifest, a.window_seconds, cfg.nested, pins["manifest"])
    index = R.load_index(a.manifest, a.root, cfg.datasets, cfg.nested)
    scales, windows = _read(a.scales)["scales"], _read(a.window_seconds)["window_s"]
    R.check_derived_binding(index.params, cfg, windows)
    descriptors = descriptor_budget(cfg)
    report = audit_batches(replay(cfg, index, scales, windows, a.steps), cc,
                           {r["dataset"] for r, w in zip(index.rows, index.weights) if w > 0}, cfg.model.patch_ms)
    after = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in inputs.items()}
    if after != pins or frozen_source != source_digest() or config_bytes_pin != hashlib.sha256(a.config.read_bytes()).hexdigest():
        raise ValueError("audit inputs changed during replay; no report accepted")
    report["inputs"] = pins
    report["config_sha256"] = hashlib.sha256(json.dumps(_read(a.config), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report["source_sha256"] = frozen_source
    report["descriptor_budget"] = descriptors
    report["world_size"], report["workers_per_rank"] = 4, 6
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({"passed": report["passed"], "steps": report["steps"],
                      "calibration_panel_passed": report["calibration_panel"]["passed"],
                      "descriptor_budget": descriptors}))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
