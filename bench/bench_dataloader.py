#!/usr/bin/env python3
"""Benchmark sintetico del dataloader - passo 0 (docs/dataloader_bench_spec.md).

Due sottocomandi:
  write-shards  - scrive un pool di shard sintetici su disco (asse M, spec §5).
                  Va lanciato PRIMA di `measure`, in un job precedente, cosi' la prima
                  misura di `measure` e' davvero a cache fredda.
  measure       - legge dal pool con `n_workers` processi paralleli, misura il tempo di
                  fetch per batch e calcola le metriche dell'asse M (spec §8).

Misura solo l'asse M (montaggi: al volo / precompute), lato CPU e I/O, con un
consumatore simulato - nessuna GPU richiesta. L'asse L (layout: padding / packing /
bucketing) richiede il proxy su GPU (bench/proxy_model.py, da eseguire su Leonardo) e
non e' in questo script.

Esempio (smoke test locale, pool piccolo):
    python bench/bench_dataloader.py write-shards --root /tmp/wearusfm_bench \\
        --arm al_volo --n-shards 100 --seed 0
    python bench/bench_dataloader.py measure --root /tmp/wearusfm_bench \\
        --arm al_volo --n-batches 20 --batch-size 16 --n-workers 4 --n-params 30e6 \\
        --out /tmp/wearusfm_bench/results.json

Sul cluster: root su $WORK o $FAST (spec §5, entrambi da misurare), n-shards e
n-batches molto piu' grandi, n-workers coerente con CLAUDE.md/piano (8 per GPU).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.data.consumer import compute_asse_m_metrics, required_windows_per_s  # noqa: E402
from wearusfm.data.manifest import (  # noqa: E402
    MONTAGES,
    MONTAGES_BY_NAME,
    QuotaClass,
    full_montage_weights,
)
from wearusfm.data.pipeline import (  # noqa: E402
    generate_precomputed_sample_for_shard,
    generate_raw_sample_for_shard,
    run_pipeline_on_the_fly,
    run_pipeline_precompute,
)
from wearusfm.data.shards import write_shard  # noqa: E402


def _quota_weights(quote_a: float, quote_b: float, quote_c: float) -> dict[QuotaClass, float]:
    return {
        QuotaClass.A_RADI: quote_a,
        QuotaClass.B_ANELLI: quote_b,
        QuotaClass.C_GRIGLIE: quote_c,
    }


def cmd_write_shards(args: argparse.Namespace) -> None:
    root = Path(args.root)
    weights = full_montage_weights(
        _quota_weights(args.quote_a, args.quote_b, args.quote_c),
        c_pessimistic=not args.class_c_mixed,
    )
    names = list(weights.keys())
    probs = np.array([weights[n] for n in names], dtype=np.float64)
    probs /= probs.sum()

    rng = np.random.default_rng(args.seed)
    index: list[dict] = []
    t0 = time.perf_counter()
    for i in range(args.n_shards):
        name = names[rng.choice(len(names), p=probs)]
        montage = MONTAGES_BY_NAME[name]
        shard_id = f"shard_{i:06d}"
        if args.arm == "al_volo":
            sample = generate_raw_sample_for_shard(
                montage, rng, context_s=args.context_s, all_native_fs=args.all_native_fs
            )
        else:
            sample = generate_precomputed_sample_for_shard(
                montage, rng, context_s=args.context_s, all_native_fs=args.all_native_fs,
                p_piena=args.p_piena, montage_dropout_group_p=args.dropout_group_p,
                montage_dropout_channel_p=args.dropout_channel_p,
            )
        write_shard(root, shard_id, sample)
        index.append({"shard_id": shard_id, "montage_name": name, "fs_hz": sample.front_end_fs_hz})

    (root / "index.json").write_text(json.dumps(index))
    elapsed = time.perf_counter() - t0
    total_bytes = sum((root / f"{e['shard_id']}.bin").stat().st_size for e in index)
    print(
        f"scritti {len(index)} shard ({args.arm}) in {elapsed:.1f}s, "
        f"{total_bytes / 1e9:.2f} GB totali, su {root}"
    )


def _measure_one(root: Path, shard_id: str, arm: str, seed: int, rvq_slab: bool) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    if arm == "al_volo":
        sample, stages = run_pipeline_on_the_fly(root, shard_id, rng=rng, rvq_slab=rvq_slab)
    else:
        sample, stages = run_pipeline_precompute(root, shard_id, rng=rng, rvq_slab=rvq_slab)
    stages["c_presented"] = float(sample.c_presented)
    return stages


def cmd_measure(args: argparse.Namespace) -> None:
    root = Path(args.root)
    index = json.loads((root / "index.json").read_text())
    shard_ids = [e["shard_id"] for e in index]

    t_passo_s = args.t_passo_s
    if t_passo_s is None:
        t_passo_s = 1.0 / required_windows_per_s(args.n_params)

    rng = np.random.default_rng(args.seed)
    batch_fetch_times: list[float] = []
    stage_samples: dict[str, list[float]] = {}
    c_values: list[int] = []

    warmup = args.warmup_batches
    total_batches = warmup + args.n_batches

    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        for b in range(total_batches):
            ids = rng.choice(shard_ids, size=args.batch_size, replace=True)
            seeds = rng.integers(0, 2**31 - 1, size=args.batch_size)
            t0 = time.perf_counter()
            futures = [
                pool.submit(_measure_one, root, sid, args.arm, int(sd), args.rvq_slab)
                for sid, sd in zip(ids, seeds)
            ]
            results = [f.result() for f in futures]
            elapsed = time.perf_counter() - t0

            if b >= warmup:
                batch_fetch_times.append(elapsed)
                for r in results:
                    c_values.append(int(r.pop("c_presented")))
                    for k, v in r.items():
                        stage_samples.setdefault(k, []).append(v)

    metrics = compute_asse_m_metrics(np.array(batch_fetch_times), t_passo_s, batch_size=args.batch_size)
    stage_p50 = {k: float(np.percentile(v, 50)) for k, v in stage_samples.items()}

    out = {
        "arm": args.arm,
        "root": str(root),
        "n_batches": args.n_batches,
        "batch_size": args.batch_size,
        "n_workers": args.n_workers,
        "t_passo_s": t_passo_s,  # tempo richiesto per UNA finestra
        "t_passo_batch_s": metrics.t_passo_batch_s,  # soglia per batch usata nel confronto
        "n_params": args.n_params,
        "waiting_fraction": metrics.waiting_fraction,
        "p50_wall_s": metrics.p50_wall_s,
        "p95_wall_s": metrics.p95_wall_s,
        "p99_wall_s": metrics.p99_wall_s,
        "windows_per_s": metrics.windows_per_s,
        "stage_p50_s": stage_p50,
        "c_mean": float(np.mean(c_values)),
        "c_max": int(np.max(c_values)),
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"scritto {out_path}")
    print(json.dumps(out, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    ws = sub.add_parser("write-shards", help="scrive il pool di shard sintetici")
    ws.add_argument("--root", required=True)
    ws.add_argument("--arm", choices=["al_volo", "precompute"], required=True)
    ws.add_argument("--n-shards", type=int, default=200)
    ws.add_argument("--quote-a", type=float, default=40.0)
    ws.add_argument("--quote-b", type=float, default=35.0)
    ws.add_argument("--quote-c", type=float, default=25.0)
    ws.add_argument("--class-c-mixed", action="store_true", help="default: pessimistico (solo Hyser)")
    ws.add_argument("--p-piena", type=float, default=0.5)
    ws.add_argument("--dropout-group-p", type=float, default=0.2)
    ws.add_argument("--dropout-channel-p", type=float, default=0.1)
    ws.add_argument("--context-s", type=float, default=4.0)
    ws.add_argument("--all-native-fs", action="store_true")
    ws.add_argument("--seed", type=int, default=0)
    ws.set_defaults(func=cmd_write_shards)

    ms = sub.add_parser("measure", help="misura il fetch a partire da un pool gia' scritto")
    ms.add_argument("--root", required=True)
    ms.add_argument("--arm", choices=["al_volo", "precompute"], required=True)
    ms.add_argument("--n-batches", type=int, default=500)
    ms.add_argument("--warmup-batches", type=int, default=50)
    ms.add_argument("--batch-size", type=int, default=64)
    ms.add_argument("--n-workers", type=int, default=8)
    ms.add_argument("--n-params", type=float, default=30e6, help="per il ritmo richiesto, spec §9")
    ms.add_argument(
        "--t-passo-s", type=float, default=None,
        help="sovrascrive il ritmo richiesto calcolato; e' il tempo per UNA finestra, non per batch",
    )
    ms.add_argument("--rvq-slab", action="store_true", help="default spento, spec §4")
    ms.add_argument("--seed", type=int, default=0)
    ms.add_argument("--out", default=None, help="results/step0/<braccio>/<configurazione>.json")
    ms.set_defaults(func=cmd_measure)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
