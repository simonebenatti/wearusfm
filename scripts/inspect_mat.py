#!/usr/bin/env python3
"""Ispezione in SOLA LETTURA di un file .mat (prima di scrivere un parser di ingest): variabili con forma e tipo, valori degli scalari, e per la
variabile EMG la deviazione standard per colonna e la quota di potenza sopra una frequenza (es. 555 Hz per DB8: se l'EMG e' stato acquisito a 1111 Hz
e sovracampionato a 2 kHz, sopra 555 Hz non deve esserci quasi nulla). Legge sia i .mat classici (scipy) sia i v7.3 (HDF5, h5py). Non scrive niente.

  python scripts/inspect_mat.py S1_E1_A1.mat --fs 2000 --above-hz 555 --seconds 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MAX_SCALAR_SIZE = 4  # variabili con al piu' tanti elementi si stampano per intero


def is_v73(path: Path) -> bool:
    with open(path, "rb") as f:
        return b"MATLAB 7.3" in f.read(128)


def list_variables(path: Path) -> list[dict]:
    if is_v73(path):
        import h5py

        out = []
        with h5py.File(path, "r") as f:
            for k, v in f.items():
                if isinstance(v, h5py.Dataset):
                    out.append({"name": k, "shape": list(v.shape[::-1]), "dtype": str(v.dtype)})  # MATLAB salva in ordine di colonna
        return out
    import scipy.io as sio

    return [{"name": n, "shape": list(s), "dtype": t} for n, s, t in sio.whosmat(str(path))]


def load_variables(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    if is_v73(path):
        import h5py

        with h5py.File(path, "r") as f:
            return {n: np.asarray(f[n]).T for n in names if n in f}
    import scipy.io as sio

    mat = sio.loadmat(str(path), variable_names=names)
    return {n: np.asarray(mat[n]) for n in names if n in mat}


def power_fraction_above(x: np.ndarray, fs: float, above_hz: float) -> np.ndarray:
    """Per colonna di x (T, C): potenza sopra `above_hz` / potenza totale (media tolta)."""
    x = x - x.mean(axis=0, keepdims=True)
    p = np.abs(np.fft.rfft(x, axis=0)) ** 2
    f = np.fft.rfftfreq(x.shape[0], d=1.0 / fs)
    tot = p.sum(axis=0)
    return np.where(tot > 0, p[f > above_hz].sum(axis=0) / np.where(tot > 0, tot, 1.0), np.nan)


def inspect(path: Path, emg_var: str, fs: float, above_hz: float | None, seconds: float) -> dict:
    variables = list_variables(path)
    small = [v["name"] for v in variables if int(np.prod(v["shape"])) <= MAX_SCALAR_SIZE]
    label_like = [v["name"] for v in variables if v["name"] in ("stimulus", "restimulus", "repetition", "rerepetition")]
    data = load_variables(path, small + label_like + [emg_var])
    out: dict = {"file": str(path), "format": "v7.3 (HDF5)" if is_v73(path) else "classico", "variables": variables,
                 "scalars": {n: np.asarray(data[n]).ravel().tolist() if data[n].dtype.kind in "biuf" else str(data[n].ravel().tolist())
                             for n in small if n in data}}
    out["labels"] = {n: {"length": int(data[n].size), "n_distinct": int(np.unique(data[n]).size), "min": int(data[n].min()), "max": int(data[n].max())}
                     for n in label_like if n in data}
    if emg_var in data:
        emg = np.asarray(data[emg_var], dtype=np.float64)
        out["emg"] = {"shape": list(emg.shape), "std_per_column": emg.std(axis=0).tolist(), "max_abs": float(np.abs(emg).max()),
                      "n_constant_columns": int((emg.std(axis=0) == 0).sum()), "duration_s_at_fs": emg.shape[0] / fs}
        if above_hz is not None:
            chunk = emg[: int(seconds * fs)]
            out["emg"][f"power_fraction_above_{above_hz:g}_hz_first_{seconds:g}_s"] = power_fraction_above(chunk, fs, above_hz).tolist()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--emg-var", default="emg")
    ap.add_argument("--fs", type=float, default=2000.0, help="frequenza della griglia del file (per la quota di potenza)")
    ap.add_argument("--above-hz", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=60.0, help="secondi iniziali su cui calcolare lo spettro")
    args = ap.parse_args(argv)
    print(json.dumps(inspect(args.path, args.emg_var, args.fs, args.above_hz, args.seconds), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
