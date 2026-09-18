#!/usr/bin/env python3
"""Misura l'asse L (layout del batch: padding / packing / bucketing), spec §7-§8.

Richiede CUDA + flash_attn: non gira sul Mac. Va lanciato su Leonardo.

**Semplificazione di questa prima iterazione, dichiarata esplicitamente**: il batch e'
costruito solo con montaggi che condividono la STESSA frequenza al front-end (default
1000 Hz), per isolare il confronto padding/packing/bucketing senza affrontare ancora il
raggruppamento per frequenza del collate reale (spec §4, stadio 5). Con la frequenza di
default (1000 Hz) il pool copre comunque le tre classi di quota: sparso (NinaPro
standard/DB6), anello (emg2pose), griglia (CapgMyo) - il caso che piu' conta per il
confronto (v10 §5).

Il bottleneck Perceiver attende sui token canale-patch dell'INTERO campione in un colpo
solo, non per singolo patch temporale (v10 §5 la vorrebbe fattorizzata): vedi il
docstring di wearusfm.model.proxy per la giustificazione della semplificazione.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from wearusfm.data.manifest import (  # noqa: E402
    MONTAGES,
    QuotaClass,
    front_end_fs_hz,
    full_montage_weights,
)
from wearusfm.data.synthetic import generate_sample, sample_montage  # noqa: E402
from wearusfm.model.geometry import build_neighbors  # noqa: E402
from wearusfm.model.proxy import ProxyConfig, ProxyModel  # noqa: E402


def eligible_montages(target_fs_hz: float, all_native_fs: bool) -> tuple:
    return tuple(m for m in MONTAGES if front_end_fs_hz(m, all_native=all_native_fs) == target_fs_hz)


def draw_batch(rng, montages, weights, batch_size, context_s, all_native_fs, p_piena, dgp, dcp):
    samples = []
    for _ in range(batch_size):
        m = sample_montage(rng, montages, weights)
        s = generate_sample(
            m, rng, context_s=context_s, all_native_fs=all_native_fs, p_piena=p_piena,
            montage_dropout_group_p=dgp, montage_dropout_channel_p=dcp,
        )
        samples.append(s)
    return samples


def to_device_tensor(data: np.ndarray, device, dtype) -> torch.Tensor:
    # data: (T, C) int16 -> (C, T) float
    return torch.from_numpy(np.ascontiguousarray(data.T).astype(np.float32)).to(device=device, dtype=dtype)


def run_arm_padding(model, samples, n_patches, k_neighbors, rng, device, dtype):
    b = len(samples)
    c_max = max(s.c_presented for s in samples)
    t = samples[0].n_samples

    padded = torch.zeros(b, c_max, t, device=device, dtype=dtype)
    for i, s in enumerate(samples):
        padded[i, : s.c_presented, :] = to_device_tensor(s.data, device, dtype)

    tokens = model.front_end(padded.reshape(b * c_max, t), n_patches).reshape(b, c_max, n_patches, -1)
    d = tokens.shape[-1]

    per_sample = []
    for i, s in enumerate(samples):
        real = tokens[i, : s.c_presented]
        idx, dist = build_neighbors(s.groups_presented, k_neighbors, rng)
        idx_t = torch.from_numpy(idx).to(device)
        dist_t = torch.from_numpy(dist).to(device=device, dtype=dtype)
        enc = model.local_encoder(real, idx_t, dist_t)
        per_sample.append(enc.reshape(-1, d))

    l_sizes = [pt.shape[0] for pt in per_sample]
    l_max = max(l_sizes)
    perceiver_in = torch.zeros(b, l_max, d, device=device, dtype=dtype)
    valid = torch.zeros(b, l_max, dtype=torch.bool, device=device)
    for i, pt in enumerate(per_sample):
        perceiver_in[i, : pt.shape[0]] = pt
        valid[i, : pt.shape[0]] = True

    perceiver_out = model.perceiver.forward_padded(perceiver_in, valid)
    out = model.finish(perceiver_out)
    padding_fraction = 1.0 - (sum(l_sizes) / (b * l_max))
    return out, padding_fraction


def run_arm_packing(model, samples, n_patches, k_neighbors, rng, device, dtype):
    b = len(samples)
    t = samples[0].n_samples
    lengths_c = [s.c_presented for s in samples]
    cu_c = np.cumsum([0] + lengths_c)

    packed_raw = torch.cat([to_device_tensor(s.data, device, dtype) for s in samples], dim=0)
    tokens = model.front_end(packed_raw, n_patches)
    d = tokens.shape[-1]

    per_sample = []
    for i, s in enumerate(samples):
        seg = tokens[cu_c[i] : cu_c[i + 1]]
        idx, dist = build_neighbors(s.groups_presented, k_neighbors, rng)
        idx_t = torch.from_numpy(idx).to(device)
        dist_t = torch.from_numpy(dist).to(device=device, dtype=dtype)
        enc = model.local_encoder(seg, idx_t, dist_t)
        per_sample.append(enc.reshape(-1, d))

    packed_l = torch.cat(per_sample, dim=0)
    l_sizes = [pt.shape[0] for pt in per_sample]
    cu_l = torch.tensor(np.cumsum([0] + l_sizes), dtype=torch.int32, device=device)

    perceiver_out = model.perceiver.forward_packed(packed_l, cu_l, batch_size=b)
    out = model.finish(perceiver_out)
    return out, 0.0


def run_arm_bucketing(model, samples, n_patches, k_neighbors, rng, device, dtype):
    by_c: dict[int, list] = {}
    for s in samples:
        by_c.setdefault(s.c_presented, []).append(s)

    t = samples[0].n_samples
    bucket_outs = []
    for c, bucket_samples in by_c.items():
        bk = len(bucket_samples)
        raw = torch.stack([to_device_tensor(s.data, device, dtype) for s in bucket_samples])  # (Bk, C, T)
        tokens = model.front_end(raw.reshape(bk * c, t), n_patches).reshape(bk, c, n_patches, -1)
        d = tokens.shape[-1]

        per_sample = []
        for i, s in enumerate(bucket_samples):
            idx, dist = build_neighbors(s.groups_presented, k_neighbors, rng)
            idx_t = torch.from_numpy(idx).to(device)
            dist_t = torch.from_numpy(dist).to(device=device, dtype=dtype)
            enc = model.local_encoder(tokens[i], idx_t, dist_t)
            per_sample.append(enc.reshape(-1, d))
        stacked = torch.stack(per_sample, dim=0)  # (Bk, C*P, d) - uniforme nel bucket
        bucket_outs.append(model.perceiver.forward_bucket(stacked))

    perceiver_out = torch.cat(bucket_outs, dim=0)
    out = model.finish(perceiver_out)
    return out, 0.0


ARM_FUNCS = {"padding": run_arm_padding, "packing": run_arm_packing, "bucketing": run_arm_bucketing}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=list(ARM_FUNCS), required=True)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--n-local-layers", type=int, default=2)
    p.add_argument("--n-backbone-layers", type=int, default=10)
    p.add_argument("--k-latents", type=int, default=64)
    p.add_argument("--k-neighbors", type=int, default=8)
    p.add_argument("--target-fs-hz", type=float, default=1000.0)
    p.add_argument("--all-native-fs", action="store_true")
    p.add_argument("--patch-ms", type=float, default=25.0)
    p.add_argument("--context-s", type=float, default=4.0)
    p.add_argument("--quote-a", type=float, default=40.0)
    p.add_argument("--quote-b", type=float, default=35.0)
    p.add_argument("--quote-c", type=float, default=25.0)
    p.add_argument("--p-piena", type=float, default=0.5)
    p.add_argument("--dropout-group-p", type=float, default=0.2)
    p.add_argument("--dropout-channel-p", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--warmup-batches", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print(json.dumps({"status": "FAIL", "error": "CUDA non disponibile"}))
        return 1

    device = torch.device("cuda")
    dtype = torch.bfloat16

    montages = eligible_montages(args.target_fs_hz, args.all_native_fs)
    if not montages:
        print(json.dumps({"status": "FAIL", "error": f"nessun montaggio a {args.target_fs_hz} Hz"}))
        return 1
    quota_weights = {QuotaClass.A_RADI: args.quote_a, QuotaClass.B_ANELLI: args.quote_b, QuotaClass.C_GRIGLIE: args.quote_c}
    full_weights = full_montage_weights(quota_weights, c_pessimistic=True)
    weights = {m.name: full_weights.get(m.name, 0.0) for m in montages}
    total_w = sum(weights.values())
    if total_w <= 0:
        weights = {m.name: 1.0 for m in montages}
        total_w = len(montages)
    weights = {k: v / total_w for k, v in weights.items()}

    n_patches = max(1, round(args.context_s * 1000.0 / args.patch_ms))

    cfg = ProxyConfig(
        d_model=args.d_model, n_heads=args.n_heads, n_local_layers=args.n_local_layers,
        n_backbone_layers=args.n_backbone_layers, k_latents=args.k_latents, k_neighbors=args.k_neighbors,
    )
    model = ProxyModel(cfg).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())

    run_fn = ARM_FUNCS[args.arm]
    rng = np.random.default_rng(args.seed)

    wall_times = []
    padding_fractions = []
    peak_mem_bytes = []
    total_batches = args.warmup_batches + args.n_batches

    for b in range(total_batches):
        samples = draw_batch(
            rng, montages, weights, args.batch_size, args.context_s, args.all_native_fs,
            args.p_piena, args.dropout_group_p, args.dropout_channel_p,
        )
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        out, padding_fraction = run_fn(model, samples, n_patches, args.k_neighbors, rng, device, dtype)
        loss = out.float().pow(2).mean()
        model.zero_grad(set_to_none=True)
        loss.backward()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if b >= args.warmup_batches:
            wall_times.append(elapsed)
            padding_fractions.append(padding_fraction)
            peak_mem_bytes.append(torch.cuda.max_memory_allocated(device))

    wall = np.array(wall_times)
    tokens_per_batch = args.batch_size * n_patches  # token utili (canale-patch reali) circa: vedi nota
    result = {
        "status": "OK",
        "arm": args.arm,
        "n_params": int(n_params),
        "d_model": args.d_model,
        "n_backbone_layers": args.n_backbone_layers,
        "target_fs_hz": args.target_fs_hz,
        "batch_size": args.batch_size,
        "n_batches": args.n_batches,
        "n_patches": n_patches,
        "p50_step_s": float(np.percentile(wall, 50)),
        "p95_step_s": float(np.percentile(wall, 95)),
        "padding_fraction_mean": float(np.mean(padding_fractions)),
        "peak_memory_mb": float(np.max(peak_mem_bytes) / 1e6),
        "steps_per_s": float(1.0 / np.mean(wall)),
        "gpu_name": torch.cuda.get_device_name(0),
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
