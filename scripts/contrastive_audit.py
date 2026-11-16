"""CPU train-stream audit: identities, masking/QC and pose support, no encoder.

Replays the real logical rank/worker streams serially (no GPU or submission).
It checks candidate availability, not projection finiteness or signal quality.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.training import run as R
from wearusfm.training.contrastive_objectives import ContrastiveConfig, ContrastiveRuntime, physical_pair_mask
from wearusfm.training.contrastive_memory import identities_from_batch, admissible_mask
from wearusfm.model.rank_contrast import anchor_support, rank_contrast_count


def audit_batches(global_batches, config, expected_datasets, patch_ms=25.):
    runtime = ContrastiveRuntime(asdict(config))
    runtime.expected_datasets = tuple(sorted(expected_datasets))
    records, digest = [], hashlib.sha256()
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
                    masks.append(torch.zeros(p, 20, dtype=torch.bool))
                else:
                    masks.append(torch.as_tensor(batch.pose[s][1], dtype=torch.bool))
                offset += c
        k = torch.zeros(len(ids), 1)  # identity-only placeholders, NEVER training features
        kv = torch.tensor(key)
        snapshot = runtime.queue.snapshot(step, k, ids, kv)
        negatives = admissible_mask(ids, snapshot.identities, gap_seconds=config.gap_seconds)
        eligible = torch.tensor(query) & kv & (negatives.sum(1) >= config.min_negatives)
        pmax = max(m.shape[0] for m in masks)
        valid = torch.stack([torch.cat((m, torch.zeros(pmax-m.shape[0], 20, dtype=torch.bool))) for m in masks])
        si, pi, *_ = anchor_support(valid, config.rnc_stride, "all")
        allowed, points = physical_pair_mask(ids, si, pi, patch_ms, "cpu")
        rnc_windows = int(rank_contrast_count(valid, stride=config.rnc_stride, per_window=True, pair_allowed=allowed))
        stats = {d: [sum(i.dataset == d for i in ids), sum(bool(e) for i, e in zip(ids, eligible) if i.dataset == d)]
                 for d in sorted({i.dataset for i in ids})}
        runtime.step = step
        runtime.pending = (k, ids, kv)
        runtime.pending_stats = {"rnc_active": rnc_windows > 0, "datasets": stats}
        failure = runtime.commit()
        record = {"step": step, "queries": len(ids), "eligible_nce": int(eligible.sum()), "datasets": stats,
                  "rnc_windows": rnc_windows, "rnc_anchors": len(points), "duplicate_instants": len(points)-len(set(points)),
                  "pose_recordings": len({ids[s].recording for s in si.tolist()}),
                  "pose_subjects": len({(ids[s].dataset, ids[s].subject) for s in si.tolist()})}
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
            "coverage_completed_blocks": runtime.coverage_reports, "coverage_partial_block": partial,
            "passed": all(r["passed"] for r in runtime.coverage_reports) and (partial["passed"] if partial["updates"] else bool(runtime.coverage_reports))}


def replay(cfg, index, scales, window_s, steps, world_size=4, workers=6):
    """Same worker identities/seeds as DataLoader, interleaved without processes."""
    if cfg.batch_size % world_size or workers < 1:
        raise ValueError("invalid audit rank/worker geometry")
    streams = []
    for rank in range(world_size):
        rank_streams = []
        for worker in range(workers):
            loader = R.L.PretrainLoader(index, cfg.loader, dict(scales), window_s)
            rank_streams.append(iter(R.L.batch_stream(loader, cfg.batch_size // world_size, seed=cfg.seed,
                worker=worker, n_workers=workers, segment=cfg.stream_segment, rank=rank)))
        streams.append(rank_streams)
    for step in range(steps):
        yield [next(streams[r][step % workers]) for r in range(world_size)]


def main():
    from contrastive_run import validate_recipe
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
    inputs = (a.config, a.manifest, a.scales, a.window_seconds)
    pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    cfg = validate_recipe(json.loads(a.config.read_text()), "calibration")
    pose_stats = Path(cfg.loader.pose_root) / "pose_stats.json"
    inputs += (pose_stats,)
    pins[str(pose_stats)] = hashlib.sha256(pose_stats.read_bytes()).hexdigest()
    cc = ContrastiveConfig(**cfg.contrastive)
    R.check_input_files(a.manifest, a.window_seconds, cfg.nested, pins[str(a.manifest)])
    index = R.load_index(a.manifest, a.root, cfg.datasets, cfg.nested)
    scales, windows = json.loads(a.scales.read_text())["scales"], json.loads(a.window_seconds.read_text())["window_s"]
    R.check_derived_binding(index.params, cfg, windows)
    report = audit_batches(replay(cfg, index, scales, windows, a.steps), cc,
                           {r["dataset"] for r, w in zip(index.rows, index.weights) if w > 0}, cfg.model.patch_ms)
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    if after != pins:
        raise ValueError("audit inputs changed during replay; no report accepted")
    report["inputs"] = pins
    report["world_size"], report["workers_per_rank"] = 4, 6
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
