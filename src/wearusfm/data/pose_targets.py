"""Target di posa per finestra e per patch (R3, seconda ondata del pilot degli obiettivi: `docs/pilot_posa_v1.md` §2 «Bersaglio di posa»).

L'ingest (`ingest.emg2pose_pose`) scrive per ogni registrazione di emg2pose i blocchi a 200 Hz della posa (`pose_200hz_f16.npy`, (B, 20)
float16, blocco j = campioni grezzi [10 j, 10 j + 10) dell'EMG processato) e la loro validita' (`pose_valid.npy`, (B,) bool), in un albero
parallelo a quello processato dell'EMG: `<pose_root>/u<utente>/<stem>/`, cioe' le ultime due componenti del percorso della sessione. Qui:

- `PoseStore`: apre quei file in memoria mappata per sessione (cache), legge `pose_stats.json` (media e deviazione per giunto sullo split di
  pretraining, per lato e insieme, col criterio congelato per la standardizzazione per lato) e ne tiene lo SHA-256 per il config del run.
  Una sessione di emg2pose senza file di posa e' un ERRORE (`FileNotFoundError`), non un salto silenzioso.
- `window_pose`: per una finestra (`Window` del loader: `start`, `n_patches`, a `fs` nativa) il target per patch: la media dei blocchi validi
  il cui centro (10 j + 5) cade nella patch, sulla griglia del front-end ancorata al primo campione della finestra (`ceil(p * patch_s * fs
  - 1e-9)`, la stessa di `model.fm.zero_hidden_samples`): in una patch da 25 ms a 2 kHz cadono esattamente 5 centri, salvo ai bordi;
  sfasamento massimo di mezzo blocco (2,5 ms), dichiarato nel protocollo. Standardizzazione per giunto; maschera (P, 20) True dove almeno un
  blocco valido cade nella patch (la validita' e' per blocco, quindi uguale sui 20 giunti). Solo numpy.

Il default del training non passa di qui: senza `LoaderConfig.pose_root` il loader non apre nulla e `PretrainBatch.pose` resta None.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

POSE_DATASET = "emg2pose"
N_JOINTS = 20
BLOCK = 10  # campioni grezzi per blocco: 2000 Hz -> 200 Hz
POSE_FILE, VALID_FILE, META_FILE, STATS_FILE = "pose_200hz_f16.npy", "pose_valid.npy", "pose_meta.json", "pose_stats.json"
STD_FLOOR = 1e-6


def pose_dir(pose_root: Path, session_path: Path) -> Path:
    """`<pose_root>/u<utente>/<stem>`: le ultime due componenti del percorso della sessione processata (`.../emg2pose/u<utente>/<stem>`)."""
    return Path(pose_root) / session_path.parent.name / session_path.name


class PoseStore:
    """I file di posa per sessione (memoria mappata, cache) e le statistiche di standardizzazione."""

    def __init__(self, root: str | Path, cache: int = 64, include_unprompted: bool = True):
        self.root = Path(root)
        # registrazioni col bracciale sulla mano NON sollecitata a muoversi in quella fase (`moving_hand` diverso da `side` e da `both`;
        # 23% delle ore di emg2pose nel pretraining): tenute nel termine per la firma del 26/10/2026; `--pose-unprompted off` le maschera
        self.include_unprompted = include_unprompted
        raw = (self.root / STATS_FILE).read_bytes()
        self.stats = json.loads(raw)
        self.stats_sha256 = hashlib.sha256(raw).hexdigest()
        if self.stats.get("all", {}).get("mean") is None:
            raise ValueError(f"{self.root / STATS_FILE}: statistiche vuote (nessuna registrazione dello split di pretraining)")
        self.per_side = bool((self.stats.get("side_criterion") or {}).get("per_side"))
        self.mean_all = np.asarray(self.stats["all"]["mean"], dtype=np.float32)
        self.std_all = np.maximum(np.asarray(self.stats["all"]["std"], dtype=np.float32), STD_FLOOR)
        self.by_side = {}
        for side, st in (self.stats.get("by_side") or {}).items():
            if st.get("mean") is not None:
                self.by_side[side] = (np.asarray(st["mean"], dtype=np.float32), np.maximum(np.asarray(st["std"], dtype=np.float32), STD_FLOOR))
        if self.per_side and not {"left", "right"} <= set(self.by_side):
            raise ValueError("standardizzazione per lato richiesta ma le statistiche per lato mancano")
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache

    @staticmethod
    def applies(row: dict) -> bool:
        return row.get("dataset") == POSE_DATASET

    @staticmethod
    def unprompted(view) -> bool:
        """Vero se in quella fase la mano del bracciale non era sollecitata a muoversi (dal `csv_row` del sidecar dell'EMG)."""
        row = view.meta.get("csv_row") or {}
        side, moving = (row.get("side") or "").lower(), (row.get("moving_hand") or "").lower()
        return bool(side) and bool(moving) and moving not in (side, "both")

    def norm(self, view) -> tuple[np.ndarray, np.ndarray]:
        """(media, deviazione) per giunto: per lato (dal `csv_row.side` del sidecar dell'EMG) se il criterio congelato lo ha deciso."""
        if not self.per_side:
            return self.mean_all, self.std_all
        side = ((view.meta.get("csv_row") or {}).get("side") or "").lower()
        if side not in self.by_side:
            raise ValueError(f"{view.path}: lato {side!r} senza statistiche di posa (standardizzazione per lato)")
        return self.by_side[side]

    def entry(self, view) -> tuple[np.ndarray, np.ndarray]:
        d = pose_dir(self.root, Path(view.path))
        key = str(d)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if not (d / POSE_FILE).exists() or not (d / VALID_FILE).exists():
            raise FileNotFoundError(f"{view.path}: posa mancante in {d} (sessione di {POSE_DATASET} senza ingest della posa)")
        blocks = np.load(d / POSE_FILE, mmap_mode="r")
        valid = np.load(d / VALID_FILE)
        if blocks.ndim != 2 or blocks.shape[1] != N_JOINTS or valid.shape != (blocks.shape[0],):
            raise ValueError(f"{d}: forme di posa inattese {blocks.shape} / {valid.shape}")
        self._cache[key] = (blocks, valid)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return blocks, valid

    @staticmethod
    def check_length(blocks: np.ndarray, view) -> None:
        """I blocchi di posa coprono l'EMG processato della sessione (array continuo): T // BLOCK blocchi, ne' uno di piu' ne' uno di meno."""
        arr = getattr(view, "arr", None)
        if arr is None or arr.ndim != 2:
            return
        n_blocks = int(arr.shape[0] // BLOCK)
        if blocks.shape[0] != n_blocks:
            raise ValueError(f"{view.path}: {blocks.shape[0]} blocchi di posa per {n_blocks} attesi dall'EMG ({arr.shape[0]} campioni)")

    def window(self, row: dict, view, win, patch_ms: float) -> tuple[np.ndarray, np.ndarray] | None:
        """(target (P, 20) float32 standardizzato, maschera (P, 20) bool) per una finestra di emg2pose; None per gli altri dataset."""
        if not self.applies(row):
            return None
        blocks, valid = self.entry(view)
        self.check_length(blocks, view)
        if not self.include_unprompted and self.unprompted(view):  # finestra di emg2pose senza target: contribuisce zero, conta zero
            p = int(win.n_patches)
            return np.zeros((p, N_JOINTS), dtype=np.float32), np.zeros((p, N_JOINTS), dtype=bool)
        mean, std = self.norm(view)
        return window_pose(blocks, valid, win, view.fs, patch_ms, mean, std)


def patch_bounds(n_patches: int, patch_ms: float, fs: float) -> np.ndarray:
    """Inizi delle patch 0..P (l'ultimo = fine della P-1) in campioni dal primo campione della finestra: la griglia del front-end."""
    return np.ceil(np.arange(n_patches + 1) * (patch_ms / 1000.0) * fs - 1e-9).astype(np.int64)


def window_pose(blocks: np.ndarray, valid: np.ndarray, win, fs: float, patch_ms: float, mean: np.ndarray, std: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray]:
    """Target per patch (P, 20) float32 standardizzato e maschera (P, 20): media dei blocchi validi col centro dentro la patch."""
    if getattr(win, "trial", None) is not None:
        raise ValueError("posa: sessioni a prove non previste (emg2pose e' un array continuo)")
    if abs(fs * (BLOCK / 2000.0) - BLOCK * fs / 2000.0) > 1e-9 or fs <= 0:
        raise ValueError("fs non valida")
    p = int(win.n_patches)
    bounds = patch_bounds(p, patch_ms, fs) + int(win.start)  # assoluti, in campioni grezzi
    target = np.zeros((p, N_JOINTS), dtype=np.float32)
    mask = np.zeros((p, N_JOINTS), dtype=bool)
    if p == 0 or blocks.shape[0] == 0:
        return target, mask
    half = BLOCK / 2.0
    j_lo = max(0, int(math.floor((bounds[0] - half) / BLOCK)))
    j_hi = min(blocks.shape[0], int(math.ceil((bounds[-1] - half) / BLOCK)) + 1)
    if j_hi <= j_lo:
        return target, mask
    j = np.arange(j_lo, j_hi)
    centers = j * BLOCK + half
    p_of = np.searchsorted(bounds, centers, side="right") - 1  # patch del centro di ogni blocco
    ok = (p_of >= 0) & (p_of < p) & (centers >= bounds[0]) & (centers < bounds[-1]) & np.asarray(valid[j_lo:j_hi], dtype=bool)
    if not ok.any():
        return target, mask
    pj, bj = p_of[ok], j[ok]
    acc = np.zeros((p, N_JOINTS), dtype=np.float64)
    cnt = np.zeros(p, dtype=np.int64)
    np.add.at(acc, pj, np.asarray(blocks[bj], dtype=np.float64))
    np.add.at(cnt, pj, 1)
    has = cnt > 0
    target[has] = ((acc[has] / cnt[has, None] - mean[None, :]) / std[None, :]).astype(np.float32)
    mask[has] = True
    return target, mask
