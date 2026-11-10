"""TEMPONet su emg2pose, regressione della posa: protocollo ufficiale di valutazione e pezzi del training.

Riferimenti: `docs/temponet_emg2pose_plan.md`; regole ed emendamento delle finestre in `docs/decisioni.md`, 10/11/2026. Solo numpy, e torch dove
indicato: si prova su CPU.

Dal codice di emg2pose (`facebookresearch/emg2pose`, letto il 10/11/2026):
- `utils.get_ik_failures_mask`: un fallimento della cinematica inversa e' una riga coi 20 angoli `np.isclose` a zero;
- `utils.get_contiguous_ones`: tratti (inizio, fine INCLUSA) dei valori veri;
- `data.WindowedEmgDataset` con `skip_ik_failures=True`: si tengono i tratti con `t1 - t0 >= finestra`; ogni tratto da' `(t1 - t0 - finestra)
  // passo + 1` finestre, che partono da `t0 + k * passo`;
- `test_analysis`: finestra = 10.000 campioni predetti + contesto del modello, passo 10.000, nessun padding; il modello predice gli ultimi
  10.000 campioni;
- `metrics.AngleMAE`: L1 sugli angoli, in radianti. Qui l'errore si riporta in gradi.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

FS_HZ = 2000.0
N_JOINTS = 20
N_CHANNELS = 16
WINDOW = 256  # finestra d'ingresso di TEMPONet ([Z21])
CONTEXT = WINDOW - 1  # campioni di storia per istante predetto
EVAL_PREDICTED = 10_000  # campioni predetti per finestra ufficiale (`test_analysis`, window_length)
EVAL_STRIDE = 10_000
EVAL_STEP = 10  # istanti valutati ogni 10 campioni (200 Hz)
BLOCK = 10  # campioni per blocco della posa a 200 Hz (`data.pose_targets`)
TRAIN_STRIDE_BLOCKS = 8  # finestre di training ogni 8 blocchi (40 ms)
VAL_EPOCH_STRIDE_BLOCKS = 4  # validazione grezza per epoca, solo descrittiva: istanti a 50 Hz
RAD2DEG = 180.0 / math.pi
SCALE_SD_RANGE = (3.0, 30.0)
GATE_MAX_RATIO = 0.9
ALPHA_GRID = np.round(np.arange(0.0, 1.0, 0.01), 2)  # 0,00 ... 0,99, come su DB8
CONDITIONS = ("user", "stage", "user_stage")
TAB4_REGRESSION_DEG = {  # [E24] Tab. 4, media fra utenti (letta il 10/11/2026)
    "user": {"vemg2pose": 12.2, "NeuroPose": 13.2, "SensingDynamics": 15.5},
    "stage": {"vemg2pose": 15.2, "NeuroPose": 17.2, "SensingDynamics": 18.8},
    "user_stage": {"vemg2pose": 15.8, "NeuroPose": 17.5, "SensingDynamics": 18.7},
}


# --- protocollo ufficiale ---------------------------------------------------------------------------------------------------------------

def no_ik_failure(angles: np.ndarray) -> np.ndarray:
    """(T,) True dove la cinematica inversa non e' fallita: come `utils.get_ik_failures_mask`."""
    a = np.asarray(angles)
    return ~np.all(np.isclose(a, np.zeros_like(a)), axis=-1)


def contiguous_ones(binary: np.ndarray) -> list[tuple[int, int]]:
    """[(inizio, fine INCLUSA)] dei tratti di valori veri, come `utils.get_contiguous_ones`."""
    b = np.asarray(binary)
    if (b == 0).all():
        return []
    ones = np.where(b)[0]
    boundaries = np.where(np.diff(ones) != 1)[0]
    return [(int(ones[i]), int(ones[j])) for i, j in zip(np.insert(boundaries + 1, 0, 0), np.append(boundaries, len(ones) - 1))]


def official_window_starts(mask: np.ndarray, predicted: int = EVAL_PREDICTED, context: int = CONTEXT, stride: int = EVAL_STRIDE) -> np.ndarray:
    """Inizi o delle finestre effettive [o, o + predicted + context): `WindowedEmgDataset(window_length=predicted + context, stride=predicted,
    skip_ik_failures=True)` di `test_analysis`."""
    eff = predicted + context
    starts: list[int] = []
    for t0, t1 in contiguous_ones(mask):
        if t1 - t0 >= eff:
            n = (t1 - t0 - eff) // stride + 1
            starts.extend(t0 + k * stride for k in range(n))
    return np.asarray(starts, dtype=np.int64)


def eval_instants(starts: np.ndarray, predicted: int = EVAL_PREDICTED, context: int = CONTEXT, step: int = EVAL_STEP) -> np.ndarray:
    """(finestre, predicted // step) campioni valutati: o + context + step * m. Ognuno ha `context` campioni di storia dentro la finestra."""
    m = np.arange(0, predicted, step, dtype=np.int64)
    return np.asarray(starts, dtype=np.int64)[:, None] + context + m[None, :]


# --- ingresso e training ----------------------------------------------------------------------------------------------------------------

def scale_exponent(median_sd: float, lo: float = SCALE_SD_RANGE[0], hi: float = SCALE_SD_RANGE[1]) -> int:
    """k intero con median_sd * 10^k in [lo, hi]; se piu' d'uno, il piu' vicino a 10 in scala logaritmica (regola congelata)."""
    if not (median_sd > 0 and math.isfinite(median_sd)):
        raise ValueError(f"deviazione standard mediana non valida: {median_sd}")
    cands = [k for k in range(-15, 16) if lo <= median_sd * 10.0 ** k <= hi]
    if not cands:
        raise ValueError(f"nessuna potenza di 10 porta {median_sd} in [{lo}, {hi}]")
    return min(cands, key=lambda k: (abs(math.log10(median_sd * 10.0 ** k) - 1.0), k))


def train_blocks(valid: np.ndarray, n_samples: int, stride_blocks: int = TRAIN_STRIDE_BLOCKS, block: int = BLOCK,
                 window: int = WINDOW) -> np.ndarray:
    """Blocchi j del training: validi, j multiplo di `stride_blocks`, ultimo campione 10 j + 9 con almeno window - 1 campioni di storia nella
    registrazione e dentro la registrazione. La finestra finisce in 10 j + 9; il bersaglio e' la media del blocco."""
    v = np.asarray(valid, dtype=bool)
    j = np.arange(0, v.size, stride_blocks, dtype=np.int64)
    j = j[v[j]]
    end = block * j + block - 1
    return j[(end >= window - 1) & (end < n_samples)]


def block_end(j: np.ndarray, block: int = BLOCK) -> np.ndarray:
    return block * np.asarray(j, dtype=np.int64) + block - 1


# --- registrazioni ----------------------------------------------------------------------------------------------------------------------

def read_rows(csv_path: str | Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def safe(name: str) -> str:
    from wearusfm.ingest.emg2pose import _SAFE_RE

    return _SAFE_RE.sub("_", name)


def recording_key(row: dict) -> tuple[str, str]:
    """(cartella dell'utente, cartella della registrazione) come negli alberi dell'ingest: `u<utente>/<stem>`."""
    return f"u{safe(row['user'])}", safe(Path(row["filename"]).stem)


def is_moving_side(row: dict) -> bool:
    return row.get("moving_hand") in (row.get("side"), "both")


def pose_stats_all(pose_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Media e deviazione per giunto (radianti) sullo split di training: chiave `all` di `pose_stats.json` (per_side = false)."""
    st = json.loads((Path(pose_root) / "pose_stats.json").read_text())
    if st.get("split") != "train":
        raise ValueError("pose_stats.json non e' calcolato sullo split train")
    return np.asarray(st["all"]["mean"], dtype=np.float64), np.asarray(st["all"]["std"], dtype=np.float64)


# --- EMA e alpha (torch) ----------------------------------------------------------------------------------------------------------------

def ema_windows(pred, alpha):
    """EMA causale di [Z21] eq. 2 lungo l'asse 1 di pred (finestre, istanti, giunti), y'_t = a y'_{t-1} + (1 - a) y_t con y'_0 = y_0: riparte a
    ogni finestra. alpha scalare o per giunto (D,). Torch."""
    import torch

    a = torch.as_tensor(alpha, dtype=pred.dtype, device=pred.device).reshape(-1)
    a = a.expand(pred.shape[-1]) if a.numel() == 1 else a
    out = torch.empty_like(pred)
    out[:, 0] = pred[:, 0]
    for t in range(1, pred.shape[1]):
        out[:, t] = a * out[:, t - 1] + (1 - a) * pred[:, t]
    return out


def tune_alpha_windows(pred, target, grid=ALPHA_GRID):
    """Per giunto, l'alpha della griglia col MAE minimo su tutte le finestre (l'EMA riparte a ogni finestra); a parita' l'alpha piu' piccolo.
    Somme in float64. Ritorna (alpha (D,) numpy, MAE (len(grid), D) numpy in radianti)."""
    import torch

    g = torch.as_tensor(np.asarray(grid, dtype=np.float64), device=pred.device)[:, None, None]  # (G, 1, 1)
    p = pred.to(torch.float64)
    y = target.to(torch.float64)
    state = p[:, 0].unsqueeze(0).expand(len(grid), -1, -1).clone()  # (G, finestre, giunti)
    err = (state - y[:, 0].unsqueeze(0)).abs().sum(dim=1)
    for t in range(1, p.shape[1]):
        state.mul_(g).add_((1 - g) * p[:, t].unsqueeze(0))
        err += (state - y[:, t].unsqueeze(0)).abs().sum(dim=1)
    mae = (err / (p.shape[0] * p.shape[1])).cpu().numpy()
    best = np.argmin(mae, axis=0)  # il primo minimo: l'alpha piu' piccolo
    return np.asarray(grid)[best], mae


# --- metriche ---------------------------------------------------------------------------------------------------------------------------

def abs_err_sums(pred, target) -> tuple[float, int]:
    """(somma degli errori assoluti in radianti su istanti e giunti, numero di valori). Torch o numpy; somma in float64."""
    import torch

    if torch.is_tensor(pred):
        d = pred.to(torch.float64) - torch.as_tensor(target, device=pred.device).to(torch.float64)
        return float(d.abs().sum()), int(d.numel())
    d = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.abs(d).sum()), int(d.size)


def summarize_users(per_user: dict[str, tuple[float, int]]) -> dict:
    """per_user: {utente: (somma degli errori in radianti, numero di valori)}. MAE per utente in gradi; media e dev. std. (ddof 1) fra utenti;
    media pesata per istanti."""
    users = sorted(per_user)
    mae = np.array([per_user[u][0] / per_user[u][1] * RAD2DEG for u in users]) if users else np.zeros(0)
    tot_s = sum(per_user[u][0] for u in users)
    tot_n = sum(per_user[u][1] for u in users)
    return {"n_users": len(users), "mean_over_users_deg": float(mae.mean()) if len(mae) else None,
            "sd_over_users_deg": float(mae.std(ddof=1)) if len(mae) > 1 else None,
            "pooled_deg": tot_s / tot_n * RAD2DEG if tot_n else None, "n_values": tot_n,
            "per_user_deg": {u: float(v) for u, v in zip(users, mae)}}


def gate(model_users: dict, const_users: dict, max_ratio: float = GATE_MAX_RATIO) -> dict:
    """Cancello prima del test: media fra utenti della MAE del modello / media fra utenti della MAE del predittore costante < max_ratio."""
    m = summarize_users(model_users)["mean_over_users_deg"]
    c = summarize_users(const_users)["mean_over_users_deg"]
    ratio = m / c
    return {"model_deg": m, "constant_deg": c, "ratio": ratio, "max_ratio": max_ratio, "passed": bool(ratio < max_ratio)}


def reading(mae_deg: float, condition: str) -> str:
    """Lettura descrittiva congelata: dove cade la MAE rispetto alle tre baseline della Tab. 4 di [E24]."""
    t = TAB4_REGRESSION_DEG[condition]
    if mae_deg < t["vemg2pose"]:
        return "sotto vemg2pose"
    if mae_deg < t["NeuroPose"]:
        return "fra vemg2pose e NeuroPose"
    if mae_deg < t["SensingDynamics"]:
        return "fra NeuroPose e SensingDynamics"
    return "sopra SensingDynamics"
