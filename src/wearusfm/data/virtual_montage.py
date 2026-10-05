"""Montaggi virtuali dalle griglie HD, al volo nel dataloader (D6a; Simone, 04/10/2026, decisione 15: «montaggi virtuali dalle griglie HD al volo
nel dataloader»; v10 §2.7: sottocampionamento a montaggi virtuali <= 32 canali come **augmentation stocastica**, non riduzione obbligatoria; v10
§4.6). Proposta di AG, da firmare (`docs/foglio_d6a_montaggi_virtuali.md`); nel codice e' un'opzione spenta (`LoaderConfig.virtual = None`).

Un campione di una griglia, con probabilita' `p`, si presenta come una **sottogriglia** di una sola griglia del montaggio: una finestra
rettangolare di `shape` elettrodi presi ogni `stride` righe e colonne, in posizione a caso (solo le forme che ci stanno).
- **monopolare:** gli elettrodi della finestra;
- **bipolare:** per ogni elettrodo della finestra, la differenza «compagno meno elettrodo» col compagno a `stride` passi lungo l'asse scelto
  (righe o colonne): polarita' canonica (v10 §3.3: orientamento noto, si canonicalizza). Il canale sta nel punto medio della coppia; vale se
  valgono tutti e due gli elettrodi.
La topologia presentata resta la griglia (classe C): gli elettrodi delle griglie non hanno identita' di muscolo, e come montaggio sparso (identita'
solo anatomica, v10 §3.5) i canali sarebbero indistinguibili. Le coordinate restano quelle della griglia d'origine (`grid_extent` per la
posizione nel sistema del sensore): una sottogriglia ogni 2 righe ha gli elettrodi a 2 passi, come sono.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

AXES = ("row", "col")


@dataclass(frozen=True)
class VirtualSpec:
    p: float = 0.5  # probabilita' che un campione di griglia sia presentato come montaggio virtuale
    p_bipolar: float = 0.5  # fra i virtuali, quota bipolare
    shapes: tuple[tuple[int, int], ...] = ((4, 8), (8, 4), (4, 4))  # righe x colonne di elettrodi (<= 32 canali, v10 §2.7)
    strides: tuple[int, ...] = (1, 2)  # passo fra gli elettrodi, in passi della griglia
    classes: tuple[str, ...] = ("C",)  # classi di quota del manifest a cui si applica


@dataclass(frozen=True)
class VirtualMontage:
    kind: str  # "mono" | "bip"
    a: np.ndarray  # (C',) colonna dell'array di ogni elettrodo
    b: np.ndarray | None  # (C',) colonna del compagno (bipolare)
    montage: dict  # montaggio presentato: un gruppo grid_2d
    qc_valid: np.ndarray  # (C',)
    axis: str | None
    stride: int
    label: str  # per il registro della topologia presentata

    @property
    def scale_key(self) -> str | None:
        """Chiave della scala di sessione: la monopolare usa quella della sessione; la bipolare quella della stessa derivazione."""
        return None if self.kind == "mono" else bipolar_scale_key(self.axis, self.stride)


def bipolar_scale_key(axis: str, stride: int) -> str:
    return f"bip:{axis}:{stride}"


def grids(montage: dict) -> list[tuple[int, int, dict]]:
    """(indice del gruppo, prima colonna dell'array, gruppo) delle griglie del montaggio, nell'ordine delle colonne."""
    out, col = [], 0
    for gi, g in enumerate(montage["groups"]):
        if g["topology"] == "grid_2d":
            out.append((gi, col, g))
        col += len(g["channels"])
    return out


def _cells(g: dict, col0: int) -> dict[tuple[int, int], int]:
    """(riga, colonna) -> colonna dell'array. Coordinate di griglia intere (gli adattatori le danno cosi')."""
    return {(int(c["sensor_coords"]["grid_row"]), int(c["sensor_coords"]["grid_col"])): col0 + i for i, c in enumerate(g["channels"])}


def _placements(cells: dict, shape: tuple[int, int], stride: int, step: tuple[int, int]) -> list[tuple[int, int]]:
    """Angoli (riga, colonna) in cui la finestra, e i compagni a `step` (bipolare; (0, 0) = monopolare), stanno tutti nella griglia."""
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    out = []
    for r0 in range(min(rows), max(rows) + 1):
        for c0 in range(min(cols), max(cols) + 1):
            ok = True
            for i in range(shape[0]):
                for j in range(shape[1]):
                    r, c = r0 + i * stride, c0 + j * stride
                    if (r, c) not in cells or (r + step[0], c + step[1]) not in cells:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                out.append((r0, c0))
    return out


def all_pairs(montage: dict, axis: str, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Tutte le coppie (elettrodo, compagno a `stride` passi lungo `axis`) delle griglie: per la scala di sessione della derivazione bipolare."""
    step = (stride, 0) if axis == "row" else (0, stride)
    a, b = [], []
    for _, col0, g in grids(montage):
        cells = _cells(g, col0)
        for (r, c), k in cells.items():
            if (r + step[0], c + step[1]) in cells:
                a.append(k)
                b.append(cells[(r + step[0], c + step[1])])
    return np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)


def draw(montage: dict, qc_valid: np.ndarray, spec: VirtualSpec, rng: np.random.Generator) -> VirtualMontage | None:
    """Un montaggio virtuale, oppure None (campione presentato intero: con probabilita' 1 - p, o se nessuna forma ci sta)."""
    gs = grids(montage)
    if not gs or rng.random() >= spec.p:
        return None
    kind = "bip" if rng.random() < spec.p_bipolar else "mono"
    gi, col0, g = gs[int(rng.integers(len(gs)))]
    axis = AXES[int(rng.integers(2))] if kind == "bip" else None
    cells = _cells(g, col0)
    options = []
    for shape in spec.shapes:
        for stride in spec.strides:
            step = (0, 0) if axis is None else ((stride, 0) if axis == "row" else (0, stride))
            places = _placements(cells, shape, stride, step)
            if places:
                options.append((shape, stride, step, places))
    if not options:
        return None
    shape, stride, step, places = options[int(rng.integers(len(options)))]
    r0, c0 = places[int(rng.integers(len(places)))]
    pos = [(r0 + i * stride, c0 + j * stride) for i in range(shape[0]) for j in range(shape[1])]
    a = np.asarray([cells[p] for p in pos], dtype=np.int64)
    b = np.asarray([cells[(r + step[0], c + step[1])] for r, c in pos], dtype=np.int64) if kind == "bip" else None
    qc = np.asarray(qc_valid, dtype=bool)
    valid = qc[a] & qc[b] if b is not None else qc[a].copy()
    src = g["channels"]
    rows = [int(c["sensor_coords"]["grid_row"]) for c in src]
    cols = [int(c["sensor_coords"]["grid_col"]) for c in src]
    chans = []
    for k, (r, c) in enumerate(pos):
        ch = copy.deepcopy(src[a[k] - col0])
        sc = ch["sensor_coords"]
        sc["channel_index"] = k
        sc["grid_row"], sc["grid_col"] = r + step[0] / 2.0, c + step[1] / 2.0
        if b is not None:
            hi_a, hi_b = ch["effective_band_hz"], src[b[k] - col0]["effective_band_hz"]
            ch["effective_band_hz"] = [max(hi_a[0], hi_b[0]), min(hi_a[1], hi_b[1])]
            ch["bipolar_orientation"] = f"{axis}+{stride}"
        ch["qc_valid"] = bool(valid[k])
        chans.append(ch)
    group = {k: v for k, v in g.items() if k != "channels"}
    group["grid_extent"] = [max(rows), max(cols)]  # indici massimi della griglia d'origine (come `_sensor_pos` sulla griglia intera)
    group["virtual"] = kind
    presented = {k: v for k, v in montage.items() if k != "groups"}
    presented["groups"] = [dict(group, channels=chans)]
    label = f"{kind} {shape[0]}x{shape[1]} s{stride}" + (f" {axis}" if axis else "")
    return VirtualMontage(kind, a, b, presented, valid, axis, stride, label)


def apply(x: np.ndarray, vm: VirtualMontage) -> np.ndarray:
    """(C, T) del montaggio d'origine -> (C', T) del montaggio virtuale."""
    return x[vm.a] if vm.b is None else x[vm.b] - x[vm.a]


def band_limit_hz(montage: dict, fs: float) -> np.ndarray:
    """Limite superiore di banda per canale (min fra la banda dichiarata e la Nyquist), come `SessionView.band_limit_hz`."""
    hi = [float(c["effective_band_hz"][1]) for g in montage["groups"] for c in g["channels"]]
    return np.minimum(np.asarray(hi), fs / 2.0)
