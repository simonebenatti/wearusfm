"""Codici dell'ancora RVQ per il pretraining (passo 6; D5b firmata e congelata il 30/09/2026: livello 0 del ramo 0 del tokenizer NeuroRVQ-EMG
congelato, sui dataset con l'ancora accesa). Solo numpy: la logica si prova sul Mac; il tokenizer gira su Leonardo (`scripts/precompute_rvq_codes.py`).

**Stesso ingresso di V2** (D5a, `tokenizer_checks`): vista canonica per prova (passa-banda 20-400 Hz, 1 kHz), una scala di sessione (deviazione
standard sui canali QC-validi), fattore di arrivo 21,718 (calibrato su emg2pose nel run del 29/09 e verificato identico in ogni run di V2), un canale
alla volta con l'indice spaziale fisso, **campioni da 16 patch da 200 ms** (3,2 s) come i campioni di V2. Qui ogni prova si divide in blocchi
consecutivi di 16 patch dall'inizio della prova (l'ultimo puo' essere piu' corto): il codice di una patch dipende dal suo blocco, come in V2
dipendeva dal suo campione. Un canale scartato dal QC non ha codici (-1).

**Allineamento** con le finestre del dataloader: la griglia canonica e' ancorata all'inizio di ogni prova (`tokenizer_checks.sessions`), e il
dataloader fa cominciare ogni finestra su un multiplo di 200 ms dall'inizio della prova (D10, decisione 11). La finestra che comincia k passi da 200
ms dopo l'inizio della prova t ha come blocco RVQ w la patch canonica k + w della prova t. Patch oltre la fine della vista canonica: -1.

Archivio: un `.npz` per sessione, `<radice>/<dataset>/<soggetto>/<sessione>.npz`, con `codes` (C, A) int16 (prove concatenate) e, per prova,
`trial_offset` (prima patch della prova nell'array, -1 se la prova non da' patch) e `trial_patches`.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

from wearusfm.preprocessing.canonical_view import (
    PATCH_SAMPLES,
    apply_scale,
    bandpass_resample,
    min_filter_samples,
    patchify,
    resample_ratio,
    session_scale,
)

V2_SCALE_FACTOR = 21.718382449214758  # fattore di arrivo di V2 (run del 29/09; «comparability» identico nei run 59078235, 59104658, 59108493, 59183099)
V2_CHECKPOINT_SHA256 = "0d255bcc9f1c75ccc374cba06eab476f15bf5fb2a87115d6dc8d2dc0adadce49"
BLOCK_PATCHES = 16  # campioni di V2: 16 patch x 200 ms
RVQ_BLOCK_S = 0.2
MISSING = -1


def canonical_trials(segments: list[np.ndarray], fs: float, qc_valid: np.ndarray) -> tuple[list[np.ndarray | None], float]:
    """Le prove di una sessione nella vista canonica, scalate, a patch: una voce (C, A_t, 200) float32 per prova, None se la prova non da' patch;
    piu' la scala di sessione. Stessi passi di `tokenizer_checks.sessions.to_canonical` (un test lo confronta), ma per prova."""
    resample_ratio(fs)
    min_n = min_filter_samples(fs)
    views = [bandpass_resample(seg, fs).astype(np.float32) if seg.shape[0] >= min_n else None for seg in segments]
    usable = [v for v in views if v is not None and v.shape[0] >= PATCH_SAMPLES]
    if not usable:
        raise ValueError("nessuna prova con almeno una patch")
    scale = session_scale(usable, qc_valid)
    out = [patchify(apply_scale(v, scale)).astype(np.float32) if v is not None and v.shape[0] >= PATCH_SAMPLES else None for v in views]
    return out, scale


def blocks_of(n_patches: int, block: int = BLOCK_PATCHES) -> list[tuple[int, int]]:
    """Blocchi consecutivi [inizio, fine) di al piu' `block` patch dall'inizio della prova."""
    return [(a, min(a + block, n_patches)) for a in range(0, n_patches, block)]


def session_codes(trials: list[np.ndarray | None], qc_valid: np.ndarray, encode, factor: float = V2_SCALE_FACTOR,
                  batch: int = 1024) -> dict[str, np.ndarray]:
    """Codici (livello 0 del ramo 0) di una sessione. `encode(x)`: x (B, n_time * 200) float32 di UN canale -> (B, n_time) codici interi (il
    runner del tokenizer su Leonardo, un finto runner nei test). I blocchi della stessa lunghezza si raggruppano in batch."""
    c = len(qc_valid)
    offs, counts, off = [], [], 0
    for t in trials:
        if t is None:
            offs.append(MISSING)
            counts.append(0)
        else:
            offs.append(off)
            counts.append(t.shape[1])
            off += t.shape[1]
    codes = np.full((c, off), MISSING, dtype=np.int16)
    jobs: dict[int, list[tuple[int, int, int, int]]] = {}  # lunghezza del blocco -> [(canale, prova, inizio, fine)]
    for ti, t in enumerate(trials):
        if t is None:
            continue
        for a, b in blocks_of(t.shape[1]):
            for ch in np.flatnonzero(qc_valid):
                jobs.setdefault(b - a, []).append((int(ch), ti, a, b))
    for n_time, items in jobs.items():
        for i in range(0, len(items), batch):
            part = items[i:i + batch]
            x = np.stack([trials[ti][ch, a:b].reshape(-1) for ch, ti, a, b in part]).astype(np.float32) * np.float32(factor)
            got = np.asarray(encode(x))
            if got.shape != (len(part), n_time):
                raise ValueError(f"codici con forma {got.shape}, attesa {(len(part), n_time)}")
            for (ch, ti, a, b), row in zip(part, got):
                codes[ch, offs[ti] + a: offs[ti] + b] = row
    return {"codes": codes, "trial_offset": np.asarray(offs, dtype=np.int64), "trial_patches": np.asarray(counts, dtype=np.int64)}


def code_path(root: Path, dataset: str, subject: str, session: str) -> Path:
    return Path(root) / dataset / subject / f"{session}.npz"


class RVQCodeStore:
    """Codici per sessione letti dall'archivio (cache delle ultime `cache` sessioni)."""

    def __init__(self, root: str | Path, cache: int = 64):
        self.root, self.cache = Path(root), cache
        self._open: OrderedDict = OrderedDict()

    def get(self, row: dict) -> dict | None:
        key = (row["dataset"], row["subject"], row["session"])
        if key in self._open:
            self._open.move_to_end(key)
            return self._open[key]
        p = code_path(self.root, *key)
        entry = dict(np.load(p)) if p.exists() else None
        self._open[key] = entry
        if len(self._open) > self.cache:
            self._open.popitem(last=False)
        return entry


def trial_of(view, win) -> tuple[int, int]:
    """(indice della prova, primo campione della prova nell'array) per una finestra del dataloader: array 3D -> la prova della finestra;
    2D con `trials` -> la prova che contiene l'inizio; 2D senza prove -> (0, 0)."""
    if win.trial is not None:
        return int(win.trial), 0
    trials = view.meta.get("trials") or []
    if trials and all("offset" in t for t in trials):
        starts = [int(t["offset"]) for t in trials]
        i = int(np.searchsorted(starts, win.start, side="right") - 1)
        return i, starts[i]
    return 0, 0


def window_codes(entry: dict | None, view, win, n_channels: int, patch_ms: float = 25.0) -> np.ndarray:
    """(C, W) codici per i blocchi da 200 ms della finestra (W = patch della finestra // 8), -1 dove mancano (sessione senza codici, canale
    scartato, oltre la fine della vista canonica)."""
    per = int(round(RVQ_BLOCK_S * 1000.0 / patch_ms))
    w = win.n_patches // per
    out = np.full((n_channels, w), MISSING, dtype=np.int64)
    if entry is None or w == 0:
        return out
    ti, anchor = trial_of(view, win)
    off, n = int(entry["trial_offset"][ti]), int(entry["trial_patches"][ti])
    if off < 0:
        return out
    k = (win.start - anchor) / (RVQ_BLOCK_S * view.fs)
    k0 = int(round(k))
    if abs(k - k0) > 1e-3 * max(1.0, abs(k)) + 0.01:
        raise ValueError(f"finestra che non comincia sulla griglia da 200 ms della prova ({k:.4f} passi): allineamento impossibile")
    j = np.arange(k0, k0 + w)
    ok = (j >= 0) & (j < n)
    out[:, ok] = entry["codes"][:, off + j[ok]]
    return out


def pack_codes(per_sample: list[np.ndarray]) -> np.ndarray:
    """Codici per campione (C_s, W_s) -> (C_tot, W_max), -1 come riempimento: stesso ordine dei canali impacchettati."""
    w = max((x.shape[1] for x in per_sample), default=0)
    return np.concatenate([np.pad(x, ((0, 0), (0, w - x.shape[1])), constant_values=MISSING) for x in per_sample], axis=0)


def expected_tokens(hours: float, n_channels: int) -> int:
    """Token da calcolare (per la stima del costo): canali x blocchi da 200 ms."""
    return int(math.ceil(hours * 3600.0 / RVQ_BLOCK_S)) * n_channels
