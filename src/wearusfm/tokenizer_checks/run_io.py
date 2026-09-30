"""Pezzi comuni agli script di analisi del run V1-V4 (`scripts/step1bis_continuous_features.py`, `scripts/rvq_branch_probe.py`): il checkpoint
verificato, la lettura degli array salvati dal run, la ricostruzione della riga di comando per i sotto-processi. Solo numpy: si importa anche nei
processi che non devono caricare torch."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

# NeuroRVQ_EMG_tokenizer_v1.pt da Hugging Face, 574.583.162 byte: lo stesso file del run 59048369 (results/step1bis/RIEPILOGO.md)
CHECKPOINT_SHA256 = "0d255bcc9f1c75ccc374cba06eab476f15bf5fb2a87115d6dc8d2dc0adadce49"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load_arrays(arrays_dir: Path) -> dict:
    """Gli array salvati dal run (`arrays_<jobid>/<dataset>.npz`): {dataset: {codes, tokens, group_subject}}."""
    out = {}
    for p in sorted(Path(arrays_dir).glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        out[p.stem] = {k: z[k] for k in ("codes", "tokens", "group_subject")}
    if not out:
        raise SystemExit(f"nessun .npz in {arrays_dir}")
    return out


def argv_without_option(argv: list[str], option: str) -> list[str]:
    """La riga di comando senza `option` e il suo valore (forme `--opt val` e `--opt=val`): serve a rilanciare lo script con un altro valore."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a == option:
            skip = True
        elif not a.startswith(option + "="):
            out.append(a)
    return out
