"""Catena completa per campione (spec §4, stadi 1-5). Lo stadio 6 (pinning e
trasferimento GPU) non e' qui: richiede CUDA e vive nel proxy del passo su Leonardo.

Le due varianti (spec §6, asse M):
- `run_pipeline_on_the_fly`: legge lo shard RAW (montaggio nativo intero, senza
  sottocampionamento) e applica sottocampionamento HD + montage dropout ad ogni lettura
  (stocastico per costruzione, spec §3-§4).
- `run_pipeline_precompute`: legge uno shard gia' RIDOTTO al montaggio presentato
  (sottocampionamento e dropout gia' applicati offline, in un job precedente): meno
  byte da leggere, ma il montaggio e' fisso, non piu' augmentation stocastica
  (v10 §4.6 - conseguenza dichiarata su `D_c`, non decisa da questo modulo).

Ogni funzione ritorna (SyntheticSample, dict tempo-per-stadio-in-secondi): il tempo per
stadio serve a rendere azionabile un fallimento (spec §4).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from wearusfm.data.manifest import MONTAGES_BY_NAME, Montage, front_end_fs_hz
from wearusfm.data.masking import rvq_slab_mask, spatial_mask, temporal_mask
from wearusfm.data.shards import read_shard
from wearusfm.data.synthetic import SyntheticSample, generate_sample
from wearusfm.data.transforms import (
    add_realistic_noise,
    apply_montage_dropout_with_indices,
    hd_virtual_bipolar_montage,
    presented_class_for_virtual_montage,
    time_warp,
)


def _augment_and_mask(sample: SyntheticSample, rng: np.random.Generator, *, rvq_slab: bool) -> dict[str, float]:
    stages: dict[str, float] = {}

    t = time.perf_counter()
    data_f = sample.data.astype(np.float64)
    data_f = add_realistic_noise(data_f, sample.front_end_fs_hz, rng)
    data_f = time_warp(data_f, rng)
    stages["augment"] = time.perf_counter() - t

    t = time.perf_counter()
    _ = temporal_mask(sample.n_samples, sample.front_end_fs_hz, rng)
    _ = spatial_mask(sample.groups_presented, rng)
    if rvq_slab:
        _ = rvq_slab_mask(sample.n_samples, sample.front_end_fs_hz, rng)
    stages["masking"] = time.perf_counter() - t

    return stages


def run_pipeline_on_the_fly(
    root: Path,
    shard_id: str,
    *,
    rng: np.random.Generator,
    p_piena: float = 0.5,
    montage_dropout_group_p: float = 0.2,
    montage_dropout_channel_p: float = 0.1,
    rvq_slab: bool = False,
) -> tuple[SyntheticSample, dict[str, float]]:
    """Arm M1: lo shard raw e' il montaggio NATIVO intero (v10 §2.7). Sottocampionamento
    HD (per la classe C) e montage dropout si applicano qui, a ogni lettura: e'
    l'augmentation stocastica che l'arm 'precompute' invece bake-a nello shard."""
    stages: dict[str, float] = {}

    t = time.perf_counter()
    raw = read_shard(root, shard_id)
    stages["read"] = time.perf_counter() - t

    t = time.perf_counter()
    montage = MONTAGES_BY_NAME[raw.montage_name]
    data = np.asarray(raw.data)

    if montage.quota_class.value == "C" and rng.random() >= p_piena:
        virtual = hd_virtual_bipolar_montage(raw.groups_presented, rng)
        idx = rng.choice(data.shape[1], size=virtual[0].n_channels, replace=False)
        idx.sort()
        data = data[:, idx]
        groups = virtual
        quota_class_presented = presented_class_for_virtual_montage(virtual[0].n_channels)
    else:
        groups = raw.groups_presented
        quota_class_presented = raw.quota_class_presented

    groups_after_dropout, keep_idx = apply_montage_dropout_with_indices(
        groups, rng, group_drop_p=montage_dropout_group_p, channel_drop_p=montage_dropout_channel_p
    )
    data = data[:, keep_idx]

    sample = SyntheticSample(
        montage_name=raw.montage_name,
        quota_class_origin=raw.quota_class_origin,
        quota_class_presented=quota_class_presented,
        groups_presented=groups_after_dropout,
        fs_native_hz=raw.fs_native_hz,
        front_end_fs_hz=raw.front_end_fs_hz,
        context_s=raw.context_s,
        data=data,
    )
    stages["subsample_dropout"] = time.perf_counter() - t

    stages.update(_augment_and_mask(sample, rng, rvq_slab=rvq_slab))
    return sample, stages


def run_pipeline_precompute(
    root: Path,
    shard_id: str,
    *,
    rng: np.random.Generator,
    rvq_slab: bool = False,
) -> tuple[SyntheticSample, dict[str, float]]:
    stages: dict[str, float] = {}

    t = time.perf_counter()
    sample = read_shard(root, shard_id)  # gia' ridotto al montaggio presentato
    stages["read"] = time.perf_counter() - t
    stages["subsample_dropout"] = 0.0  # bakato nello shard, spec §6

    stages.update(_augment_and_mask(sample, rng, rvq_slab=rvq_slab))
    return sample, stages


def generate_raw_sample_for_shard(
    montage: Montage,
    rng: np.random.Generator,
    *,
    context_s: float,
    all_native_fs: bool,
) -> SyntheticSample:
    """Campione RAW da scrivere su disco per l'arm 'al volo': montaggio nativo intero,
    nessun sottocampionamento HD ne' montage dropout (si applicano alla lettura)."""
    fe_fs = front_end_fs_hz(montage, all_native=all_native_fs)
    n_samples = max(1, round(fe_fs * context_s))
    c = montage.n_channels
    data = rng.integers(-2000, 2000, size=(n_samples, c), dtype=np.int16)
    return SyntheticSample(
        montage_name=montage.name,
        quota_class_origin=montage.quota_class,
        quota_class_presented=montage.quota_class,
        groups_presented=montage.groups,
        fs_native_hz=montage.fs_native_hz,
        front_end_fs_hz=fe_fs,
        context_s=context_s,
        data=data,
    )


def generate_precomputed_sample_for_shard(
    montage: Montage,
    rng: np.random.Generator,
    *,
    context_s: float,
    all_native_fs: bool,
    p_piena: float,
    montage_dropout_group_p: float,
    montage_dropout_channel_p: float,
) -> SyntheticSample:
    """Campione gia' ridotto al montaggio presentato, per l'arm 'precompute'."""
    return generate_sample(
        montage,
        rng,
        context_s=context_s,
        all_native_fs=all_native_fs,
        p_piena=p_piena,
        montage_dropout_group_p=montage_dropout_group_p,
        montage_dropout_channel_p=montage_dropout_channel_p,
    )
