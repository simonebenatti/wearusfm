"""Scrittura e lettura degli shard sintetici su disco (v10 §4.3, spec §5).

Formato: int16 memory-mapped, layout tempo x canali (una finestra = una lettura
contigua, spec §5). Un sidecar JSON accanto al binario porta i metadati necessari a
ricostruire il campione (nome montaggio, classi, gruppi, frequenze).

Il contenuto e' casuale (nessun byte di EMG reale): questo modulo serve a rendere
vero l'I/O del benchmark, non a produrre dati utilizzabili altrove (spec §5: "senza
I/O il precompute non si vede").
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wearusfm.data.manifest import ChannelGroup, QuotaClass
from wearusfm.data.synthetic import SyntheticSample

_DTYPE = np.int16


def shard_paths(root: Path, shard_id: str) -> tuple[Path, Path]:
    root = Path(root)
    return root / f"{shard_id}.bin", root / f"{shard_id}.json"


def write_shard(root: Path, shard_id: str, sample: SyntheticSample) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    bin_path, json_path = shard_paths(root, shard_id)

    data = np.ascontiguousarray(sample.data, dtype=_DTYPE)
    data.tofile(bin_path)

    meta = {
        "montage_name": sample.montage_name,
        "quota_class_origin": sample.quota_class_origin.value,
        "quota_class_presented": sample.quota_class_presented.value,
        "groups_presented": [
            {"topology": g.topology, "n_channels": g.n_channels} for g in sample.groups_presented
        ],
        "fs_native_hz": sample.fs_native_hz,
        "front_end_fs_hz": sample.front_end_fs_hz,
        "context_s": sample.context_s,
        "n_samples": sample.n_samples,
        "c_presented": sample.c_presented,
        "dtype": str(np.dtype(_DTYPE)),
    }
    json_path.write_text(json.dumps(meta))


def read_shard(root: Path, shard_id: str, *, mmap: bool = True) -> SyntheticSample:
    root = Path(root)
    bin_path, json_path = shard_paths(root, shard_id)
    meta = json.loads(json_path.read_text())

    shape = (meta["n_samples"], meta["c_presented"])
    if mmap:
        data = np.memmap(bin_path, dtype=_DTYPE, mode="r", shape=shape)
    else:
        data = np.fromfile(bin_path, dtype=_DTYPE).reshape(shape)

    groups = tuple(
        ChannelGroup(g["topology"], g["n_channels"]) for g in meta["groups_presented"]
    )
    return SyntheticSample(
        montage_name=meta["montage_name"],
        quota_class_origin=QuotaClass(meta["quota_class_origin"]),
        quota_class_presented=QuotaClass(meta["quota_class_presented"]),
        groups_presented=groups,
        fs_native_hz=meta["fs_native_hz"],
        front_end_fs_hz=meta["front_end_fs_hz"],
        context_s=meta["context_s"],
        data=data,
    )
