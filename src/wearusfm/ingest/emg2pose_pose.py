"""Ingest degli angoli articolari di emg2pose (la POSA), per la seconda ondata del pilot degli obiettivi (`docs/pilot_posa_v1.md` §2 e §6).

Stessi HDF5 dell'ingest EMG (`ingest.emg2pose`: gruppo `emg2pose`, dataset composto `timeseries` con `time`, `joint_angles` (20,), `emg` (16,)),
letti in accesso casuale dal tar non compresso. Qui si legge SOLO `joint_angles`. Che cosa sono (fact-checker del 26/10/2026,
`docs/fatti_da_verificare.md` n. 28, stato `raccolto`): angoli da motion capture ottico a 60 Hz con cinematica inversa offline, interpolati
linearmente a 2 kHz e filtrati passa-basso a 15 Hz (ordine non dichiarato); 20 gradi di liberta' nell'ordine di `constants.py` del repository
pubblico; radianti secondo i commenti del codice (da confermare: il collaudo misura il massimo assoluto per giunto); i fallimenti della
cinematica inversa sono righe coi 20 angoli tutti a zero (12,7% dei frame nel paper), senza maschera nel file.

Per registrazione si scrive, FUORI dall'albero processato dell'EMG (che il manifest v1.1 congela):
  <out_root>/u<utente>/<stem>/pose_200hz_f16.npy   (T // 10, 20) float16: media per blocco di 10 campioni (200 Hz), allineata per indice di
                                                   campione all'EMG processato (blocco j = campioni [10 j, 10 j + 10))
  <out_root>/u<utente>/<stem>/pose_valid.npy       (T // 10,) bool: blocco valido solo se tutti i suoi 10 campioni sono validi
  <out_root>/u<utente>/<stem>/pose_meta.json       misure e parametri (sotto)

Regola di validita' dei campioni (nucleo congelato nella proposta; il MARGINE si congela in decisioni.md dopo il collaudo dell'ingest): un
campione e' non valido se la riga e' tutta a zero (|angolo| <= `ZERO_ATOL` su tutti i giunti), se ha un valore non finito, o se sta entro
`zero_margin_samples` da una riga tutta a zero. Per dimensionare il margine senza assumere l'ipotesi delle rampe, il collaudo misura il profilo di
|differenza prima| in funzione della distanza dal bordo dei blocchi di zeri (fino a `PROFILE_MAX_LAG` campioni, 200 ms) contro la distribuzione
lontano dai bordi, e conta i minimi simultanei dei 20 giunti senza righe nulle (le «V» di un frame fallito isolato interpolato).

Le statistiche per giunto (`pose_stats.json`, `aggregate_stats`) si calcolano sui soli blocchi validi delle registrazioni dello split di
pretraining (`split == "train"` nel CSV dei metadati: 17.136 registrazioni, 158 utenti, le stesse del manifest v1.1), per lato e insieme, col
criterio congelato per decidere la standardizzazione per lato (§2 (iv) del protocollo). Solo numpy e h5py: si prova sul Mac.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np

from wearusfm.data.pose_targets import BLOCK, META_FILE, N_JOINTS, POSE_FILE, STATS_FILE, VALID_FILE  # un'unica definizione, lato dati
from wearusfm.ingest.emg2pose import NATIVE_FS_HZ, RecordingRef, _SAFE_RE, load_metadata_csv  # noqa: F401  (riesportati per il driver)
from wearusfm.ingest.kaifosh import time_axis_report

POSE_FS_HZ = NATIVE_FS_HZ / BLOCK
MOCAP_FS_HZ = 60.0
FRAME_SAMPLES = NATIVE_FS_HZ / MOCAP_FS_HZ  # 33,3 campioni per frame di motion capture
ZERO_ATOL = 1e-8  # `np.isclose(x, 0)` del repository pubblico: |x| <= 1e-8
PROFILE_MAX_LAG = 400  # 200 ms a 2 kHz
PROFILE_BIN = 10  # un blocco a 200 Hz
DIP_REL = 0.05  # minimo simultaneo: norma sotto il 5% della mediana della registrazione
DIP_MARGIN = int(round(2 * FRAME_SAMPLES))  # lontano almeno due frame da una riga a zero
# ordine dei 20 gradi di liberta' (constants.py del repository pubblico, fatto 28); AA = abdu-adduzione, FE = flesso-estensione
JOINTS = ("THUMB_CMC_FE", "THUMB_CMC_AA", "THUMB_MCP_FE", "THUMB_IP_FE",
          *[f"{f}_{j}" for f in ("INDEX", "MIDDLE", "RING", "PINKY") for j in ("MCP_AA", "MCP_FE", "PIP_FE", "DIP_FE")])
PRETRAINING_SPLIT = "train"
# criterio congelato per la standardizzazione per lato (docs/pilot_posa_v1.md §2 (iv)): per lato se per almeno un giunto |media sx - media dx|
# supera SIDE_MEAN_SD deviazioni standard congiunte, o se il rapporto fra le deviazioni dei due lati supera SIDE_SD_RATIO
SIDE_MEAN_SD, SIDE_SD_RATIO = 0.5, 1.5


# --- lettura ----------------------------------------------------------------------------------------------------------------------------

def read_pose(fileobj) -> dict:
    """{angles (T, 20) float32, time (T,), attrs}: legge SOLO `joint_angles` (e `time`), non l'EMG."""
    with h5py.File(fileobj, "r") as f:
        grp = f["emg2pose"]
        dset = grp["timeseries"]
        if not {"joint_angles", "time"} <= set(dset.dtype.names or ()):
            raise ValueError(f"campi di `timeseries` inattesi: {dset.dtype.names}")
        attrs = {}
        for k, v in grp.attrs.items():
            attrs[k] = v.decode() if isinstance(v, bytes) else (v.item() if isinstance(v, np.generic) else v)
        if float(attrs.get("sample_rate", NATIVE_FS_HZ)) != NATIVE_FS_HZ:
            raise ValueError(f"sample_rate {attrs.get('sample_rate')} diversa da {NATIVE_FS_HZ}")
        angles = np.ascontiguousarray(dset.fields("joint_angles")[:]) if hasattr(dset, "fields") else np.asarray(dset[:]["joint_angles"])
        time_s = np.asarray(dset.fields("time")[:] if hasattr(dset, "fields") else dset[:]["time"], dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float32)
    if angles.ndim != 2 or angles.shape[1] != N_JOINTS:
        raise ValueError(f"forma di joint_angles inattesa {angles.shape}")
    return {"angles": angles, "time": time_s, "attrs": attrs}


# --- validita' --------------------------------------------------------------------------------------------------------------------------

def zero_rows(angles: np.ndarray, atol: float = ZERO_ATOL) -> np.ndarray:
    """(T,) True dove i 20 angoli sono tutti a zero (fallimento della cinematica inversa, fatto 28)."""
    return np.all(np.abs(angles) <= atol, axis=1)


def nonfinite_rows(angles: np.ndarray) -> np.ndarray:
    return ~np.all(np.isfinite(angles), axis=1)


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(inizio, fine)) dei tratti consecutivi di True."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    edges = np.flatnonzero(np.diff(np.concatenate([[False], m, [False]]).astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2])]


def distance_to(mask: np.ndarray) -> np.ndarray:
    """(T,) distanza in campioni dalla riga True piu' vicina (0 sulle righe True; T se non ce n'e' nessuna)."""
    m = np.asarray(mask, dtype=bool)
    n = m.size
    inf = n
    fwd = np.empty(n, dtype=np.int64)
    last = -inf
    for i in range(n):
        if m[i]:
            last = i
        fwd[i] = i - last
    bwd = np.empty(n, dtype=np.int64)
    nxt = n + inf
    for i in range(n - 1, -1, -1):
        if m[i]:
            nxt = i
        bwd[i] = nxt - i
    return np.minimum(np.minimum(fwd, bwd), inf)


def invalid_rows(angles: np.ndarray, zero_margin_samples: int = 0) -> np.ndarray:
    """(T,) True = campione non valido: riga tutta a zero, valore non finito, o entro `zero_margin_samples` da una riga tutta a zero."""
    z = zero_rows(angles)
    bad = z | nonfinite_rows(angles)
    if zero_margin_samples > 0 and z.any():
        bad |= distance_to(z) <= zero_margin_samples
    return bad


def block_reduce(angles: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Blocchi di `BLOCK` campioni: media (float16) e validita' (tutti i campioni validi). Il resto finale (T mod BLOCK) si scarta.
    La media di un blocco non valido si calcola sui suoi campioni validi, o e' 0 se non ne ha: non va mai usata (la maschera lo dice)."""
    t = (angles.shape[0] // BLOCK) * BLOCK
    a = np.asarray(angles[:t], dtype=np.float64).reshape(-1, BLOCK, N_JOINTS)
    v = np.asarray(valid[:t], dtype=bool).reshape(-1, BLOCK)
    w = v[..., None].astype(np.float64)
    cnt = w.sum(axis=1)
    mean = np.where(cnt > 0, (np.nan_to_num(a) * w).sum(axis=1) / np.maximum(cnt, 1), 0.0)
    return mean.astype(np.float16), v.all(axis=1)


# --- misure per il collaudo -------------------------------------------------------------------------------------------------------------

def edge_profile(angles: np.ndarray, zero: np.ndarray, *, max_lag: int = PROFILE_MAX_LAG, bin_size: int = PROFILE_BIN) -> dict:
    """Profilo di |differenza prima| (media sui 20 giunti) in funzione della distanza dalla riga a zero piu' vicina, in bin di `bin_size`
    campioni fino a `max_lag`, contro la stessa grandezza lontano (> max_lag) dai bordi. Serve a misurare la contaminazione dei bordi senza
    assumere la forma (rampa lineare, transitorio del filtro). Le righe a zero e quelle non finite non entrano.
    In piu' (27/10/2026, R5), per OGNI blocco di zeri: il suo profilo (conteggi e somme per bin, sui campioni che hanno quel blocco come riga a
    zero piu' vicina) e l'altezza del salto `h` = norma RMS media della posa a `PLATEAU_LAGS` campioni dal bordo (None se non ci sono campioni)."""
    finite = np.all(np.isfinite(angles), axis=1)
    n_rows = angles.shape[0]
    d = np.full(n_rows, np.nan)
    diff = np.abs(np.diff(np.asarray(angles, dtype=np.float64), axis=0)).mean(axis=1)  # fra la riga i e la i+1
    ok = finite[1:] & finite[:-1] & ~zero[1:] & ~zero[:-1]
    d[1:][ok] = diff[ok]
    runs = runs_of(zero)
    out = {"bin_samples": bin_size, "max_lag_samples": max_lag, "n_zero_rows": int(zero.sum()), "n_zero_runs": len(runs)}
    good = np.isfinite(d)
    if not zero.any():
        far = d[good]
        out.update({"bins": [], "far": _dist_summary(far), "runs": []})
        return out
    dist = distance_to(zero)
    bins = []
    for lo in range(1, max_lag + 1, bin_size):
        hi = min(lo + bin_size - 1, max_lag)
        sel = good & (dist >= lo) & (dist <= hi)
        bins.append({"lag_from": lo, "lag_to": hi, "n": int(sel.sum()), "mean_abs_diff": float(d[sel].mean()) if sel.any() else None})
    far = d[good & (dist > max_lag)]
    out.update({"bins": bins, "far": _dist_summary(far)})
    # per blocco: ogni campione entro max_lag va al blocco della sua riga a zero piu' vicina
    n_bins = len(bins)
    zpos = np.flatnonzero(zero)
    starts = np.array([a for a, _ in runs], dtype=np.int64)
    idx = np.flatnonzero((dist >= 1) & (dist <= max_lag))
    j = np.searchsorted(zpos, idx)
    prev_pos = zpos[np.clip(j - 1, 0, zpos.size - 1)]
    next_pos = zpos[np.clip(j, 0, zpos.size - 1)]
    d_prev = np.where(j > 0, idx - prev_pos, np.iinfo(np.int64).max)
    d_next = np.where(j < zpos.size, next_pos - idx, np.iinfo(np.int64).max)
    nearest = np.where(d_prev <= d_next, prev_pos, next_pos)
    rid = np.searchsorted(starts, nearest, side="right") - 1
    bin_idx = np.minimum((dist[idx] - 1) // bin_size, n_bins - 1)
    norm = np.sqrt(np.mean(np.square(np.nan_to_num(np.asarray(angles, dtype=np.float64))), axis=1))
    in_plateau = (dist[idx] >= PLATEAU_LAGS[0]) & (dist[idx] <= PLATEAU_LAGS[1]) & finite[idx]
    good_idx = good[idx]
    per_run = []
    for k, (a, b) in enumerate(runs):
        sel = rid == k
        cnt = np.bincount(bin_idx[sel & good_idx], minlength=n_bins)
        sm = np.bincount(bin_idx[sel & good_idx], weights=d[idx][sel & good_idx], minlength=n_bins)
        hsel = sel & in_plateau
        per_run.append({"start": int(a), "end": int(b), "len": int(b - a), "h": float(norm[idx][hsel].mean()) if hsel.any() else None,
                        "n_h": int(hsel.sum()), "n_per_bin": cnt.astype(int).tolist(), "sum_per_bin": sm.tolist()})
    out["runs"] = per_run
    return out


def _dist_summary(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)), "p90": float(np.quantile(x, 0.9)),
            "p99": float(np.quantile(x, 0.99))}


def simultaneous_dips(angles: np.ndarray, zero: np.ndarray, *, rel: float = DIP_REL, margin: int = DIP_MARGIN) -> dict:
    """Minimi simultanei dei 20 giunti SENZA righe a zero: tratti in cui la norma quadratica media degli angoli scende sotto `rel` volte la
    mediana della registrazione, lontano almeno `margin` campioni da ogni riga a zero. Sono le «V» che lascerebbe un frame fallito isolato
    interpolato linearmente (fatto 28, ipotesi): il collaudo li conta."""
    finite = np.all(np.isfinite(angles), axis=1)
    norm = np.sqrt(np.mean(np.square(np.nan_to_num(np.asarray(angles, dtype=np.float64))), axis=1))
    base = finite & ~zero
    med = float(np.median(norm[base])) if base.any() else 0.0
    dip = base & (norm < rel * med) if med > 0 else np.zeros_like(base)
    if zero.any():
        dip &= distance_to(zero) > margin
    runs = runs_of(dip)
    return {"median_norm": med, "rel": rel, "margin_samples": margin, "n_dip_rows": int(dip.sum()), "n_dip_runs": len(runs),
            "dip_run_lengths": [b - a for a, b in runs[:50]]}


def joint_stats(blocks: np.ndarray, valid: np.ndarray) -> dict:
    """Somme per giunto sui blocchi validi (per aggregare fra registrazioni): n, somma, somma dei quadrati, minimo, massimo, in float64."""
    b = np.asarray(blocks[valid], dtype=np.float64)
    if b.shape[0] == 0:
        return {"n": 0, "sum": [0.0] * N_JOINTS, "sumsq": [0.0] * N_JOINTS, "min": [None] * N_JOINTS, "max": [None] * N_JOINTS}
    return {"n": int(b.shape[0]), "sum": b.sum(axis=0).tolist(), "sumsq": np.square(b).sum(axis=0).tolist(),
            "min": b.min(axis=0).tolist(), "max": b.max(axis=0).tolist()}


# --- una registrazione ------------------------------------------------------------------------------------------------------------------

def out_dir_for(out_root: Path, user: str, stem: str) -> Path:
    return Path(out_root) / f"u{_SAFE_RE.sub('_', user)}" / _SAFE_RE.sub("_", stem)


def already_done(out_dir: Path, zero_margin_samples: int, atol: float = ZERO_ATOL) -> bool:
    """Vero se `pose_meta.json` esiste ed e' stato scritto con la STESSA regola di validita' (margine e tolleranza): le 40 registrazioni del
    collaudo (margine 0) non devono entrare nell'ingest intero con un altro margine (critico del 26/10, rilievo 3)."""
    meta_path = Path(out_dir) / META_FILE
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except ValueError:
        return False
    return int(meta.get("zero_margin_samples", -1)) == int(zero_margin_samples) and float(meta.get("zero_atol", -1.0)) == float(atol) \
        and (Path(out_dir) / POSE_FILE).exists() and (Path(out_dir) / VALID_FILE).exists()


def ingest_pose_recording(fileobj, ref: RecordingRef, out_root: Path, meta_row: dict | None, *, zero_margin_samples: int = 0,
                          measure: bool = True, fields: dict | None = None, profile_max_lag: int = PROFILE_MAX_LAG) -> dict:
    """Una registrazione -> `pose_200hz_f16.npy`, `pose_valid.npy`, `pose_meta.json` in `<out_root>/u<utente>/<stem>/`. Con `measure` calcola
    anche profilo dei bordi (fino a `profile_max_lag` campioni) e minimi simultanei (collaudo); senza, solo validita', blocchi e statistiche."""
    rec = read_pose(fileobj)
    angles = rec["angles"]
    user = (meta_row or {}).get("user", "sconosciuto")
    zero = zero_rows(angles)
    nonfin = nonfinite_rows(angles)
    bad = invalid_rows(angles, zero_margin_samples)
    blocks, valid = block_reduce(angles, ~bad)
    tax = time_axis_report(rec["time"])
    out_dir = out_dir_for(out_root, user, ref.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / POSE_FILE, blocks)
    np.save(out_dir / VALID_FILE, valid)
    finite_abs = np.abs(np.nan_to_num(angles, nan=0.0, posinf=0.0, neginf=0.0))
    per_joint_var = np.var(np.asarray(blocks[valid], dtype=np.float64), axis=0).tolist() if valid.any() else [None] * N_JOINTS
    meta = {
        "source_member": ref.member, "stem": ref.stem, "user": user, "csv_row": meta_row, "attrs": rec["attrs"],
        "side": (meta_row or {}).get("side"), "moving_hand": (meta_row or {}).get("moving_hand"), "split": (meta_row or {}).get("split"),
        "n_samples": int(angles.shape[0]), "native_fs_hz": NATIVE_FS_HZ, "pose_fs_hz": POSE_FS_HZ, "block_samples": BLOCK,
        "n_blocks": int(blocks.shape[0]), "n_valid_blocks": int(valid.sum()),
        "n_zero_rows": int(zero.sum()), "frac_zero_rows": float(zero.mean()) if zero.size else 0.0, "n_zero_runs": len(runs_of(zero)),
        "zero_run_lengths_head": [b - a for a, b in runs_of(zero)[:50]], "n_nonfinite_rows": int(nonfin.sum()),
        "zero_margin_samples": int(zero_margin_samples), "zero_atol": ZERO_ATOL, "n_invalid_rows": int(bad.sum()),
        "time_axis": tax, "max_abs_per_joint": finite_abs.max(axis=0).tolist() if angles.shape[0] else [None] * N_JOINTS,
        "per_joint_var_valid_blocks": per_joint_var, "joint_stats_valid_blocks": joint_stats(blocks, valid), "joints": list(JOINTS),
        "fields_in_file": fields,
    }
    if measure:
        meta["edge_profile"] = edge_profile(angles, zero, max_lag=int(profile_max_lag))
        meta["simultaneous_dips"] = simultaneous_dips(angles, zero)
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=1, default=str))
    return {k: meta[k] for k in ("user", "stem", "side", "moving_hand", "split", "n_samples", "n_blocks", "n_valid_blocks", "n_zero_rows",
                                 "frac_zero_rows", "n_zero_runs", "n_nonfinite_rows", "n_invalid_rows", "max_abs_per_joint")} | \
        ({"dips": meta["simultaneous_dips"]["n_dip_runs"]} if measure else {})


def fields_in_file(fileobj) -> list[str]:
    """I campi del dataset `timeseries` e gli attributi del gruppo: il collaudo li riporta per cercare maschere o campi di validita'."""
    with h5py.File(fileobj, "r") as f:
        grp = f["emg2pose"]
        return {"timeseries_fields": list(grp["timeseries"].dtype.names or ()), "group_attrs": sorted(grp.attrs.keys()),
                "group_members": sorted(grp.keys()), "root_members": sorted(f.keys())}


# --- selezione per il collaudo ----------------------------------------------------------------------------------------------------------

class CollaudoPicker:
    """Accetta le registrazioni nell'ordine del tar finche' le quote per (lato, moving_hand) non sono piene: `limit` in tutto, con `both_sides`
    meta' per lato e dentro ogni lato quote uguali per i valori di `moving_hand` (3 attesi: left, right, both). Con `max_per_user` al massimo
    tante registrazioni per utente (27/10/2026: il primo collaudo ha preso 40 registrazioni da 2 utenti soli)."""

    def __init__(self, limit: int, both_sides: bool, moving_values: tuple[str, ...] = ("left", "right", "both"), max_per_user: int | None = None,
                 exclude_users: set[str] | None = None):
        self.limit, self.both_sides, self.max_per_user = limit, both_sides, max_per_user
        self.exclude_users = {str(u) for u in (exclude_users or ())}  # es. gli utenti del primo collaudo: il campione nuovo non li riusa
        self.per_side = math.ceil(limit / 2) if both_sides else limit
        self.per_bucket = math.ceil(self.per_side / len(moving_values)) if both_sides else limit
        self.taken: dict[tuple[str, str], int] = {}
        self.per_user: dict[str, int] = {}
        self.total = 0

    def accept(self, meta_row: dict | None) -> bool:
        if self.total >= self.limit:
            return False
        side = (meta_row or {}).get("side", "?")
        mh = (meta_row or {}).get("moving_hand", "?")
        user = str((meta_row or {}).get("user", "?"))
        if self.both_sides:
            if sum(v for (s, _), v in self.taken.items() if s == side) >= self.per_side:
                return False
            if self.taken.get((side, mh), 0) >= self.per_bucket:
                return False
        if user in self.exclude_users:
            return False
        if self.max_per_user is not None and self.per_user.get(user, 0) >= self.max_per_user:
            return False
        self.taken[(side, mh)] = self.taken.get((side, mh), 0) + 1
        self.per_user[user] = self.per_user.get(user, 0) + 1
        self.total += 1
        return True

    @property
    def full(self) -> bool:
        return self.total >= self.limit


# --- statistiche per giunto -------------------------------------------------------------------------------------------------------------

def _finish(acc: dict) -> dict:
    n = acc["n"]
    if n < 2:
        return {"n": n, "mean": None, "std": None}
    mean = np.asarray(acc["sum"]) / n
    var = np.maximum(np.asarray(acc["sumsq"]) / n - mean ** 2, 0.0)
    return {"n": n, "mean": mean.tolist(), "std": np.sqrt(var).tolist(), "min": acc["min"], "max": acc["max"]}


def _accumulate(acc: dict, js: dict) -> None:
    if js["n"] == 0:
        return
    if acc["n"] == 0:
        acc.update({"n": js["n"], "sum": list(js["sum"]), "sumsq": list(js["sumsq"]), "min": list(js["min"]), "max": list(js["max"])})
        return
    acc["n"] += js["n"]
    acc["sum"] = [a + b for a, b in zip(acc["sum"], js["sum"])]
    acc["sumsq"] = [a + b for a, b in zip(acc["sumsq"], js["sumsq"])]
    acc["min"] = [min(a, b) for a, b in zip(acc["min"], js["min"])]
    acc["max"] = [max(a, b) for a, b in zip(acc["max"], js["max"])]


def aggregate_stats(out_root: Path, *, split: str = PRETRAINING_SPLIT, mean_sd: float = SIDE_MEAN_SD, sd_ratio: float = SIDE_SD_RATIO) -> dict:
    """`pose_stats.json`: media e deviazione standard per giunto sui blocchi validi delle registrazioni dello split, per lato e insieme, piu'
    il criterio congelato per la standardizzazione per lato. Si scrive in `<out_root>/pose_stats.json` e si ritorna."""
    empty = lambda: {"n": 0, "sum": [0.0] * N_JOINTS, "sumsq": [0.0] * N_JOINTS, "min": None, "max": None}  # noqa: E731
    by_side, overall, n_rec, n_rec_side, skipped = {}, empty(), 0, {}, 0
    for meta_path in sorted(Path(out_root).rglob(META_FILE)):
        meta = json.loads(meta_path.read_text())
        if meta.get("split") != split:
            skipped += 1
            continue
        side = meta.get("side") or "?"
        _accumulate(by_side.setdefault(side, empty()), meta["joint_stats_valid_blocks"])
        _accumulate(overall, meta["joint_stats_valid_blocks"])
        n_rec += 1
        n_rec_side[side] = n_rec_side.get(side, 0) + 1
    sides = {s: _finish(a) for s, a in by_side.items()}
    allst = _finish(overall)
    crit = {"mean_sd_threshold": mean_sd, "sd_ratio_threshold": sd_ratio, "per_side": None, "max_mean_gap_sd": None, "max_sd_ratio": None}
    if {"left", "right"} <= set(sides) and sides["left"]["std"] is not None and sides["right"]["std"] is not None:
        ml, mr = np.asarray(sides["left"]["mean"]), np.asarray(sides["right"]["mean"])
        sl, sr = np.asarray(sides["left"]["std"]), np.asarray(sides["right"]["std"])
        pooled = np.sqrt((sl ** 2 + sr ** 2) / 2)
        gap = np.abs(ml - mr) / np.where(pooled > 0, pooled, np.inf)
        ratio = np.maximum(sl, sr) / np.where(np.minimum(sl, sr) > 0, np.minimum(sl, sr), np.inf)
        crit.update({"max_mean_gap_sd": float(gap.max()), "max_sd_ratio": float(ratio.max()),
                     "per_side": bool(gap.max() > mean_sd or ratio.max() > sd_ratio),
                     "gap_sd_per_joint": gap.tolist(), "sd_ratio_per_joint": ratio.tolist()})
    stats = {"split": split, "n_recordings": n_rec, "n_recordings_per_side": n_rec_side, "n_recordings_other_splits_skipped": skipped,
             "joints": list(JOINTS), "all": allst, "by_side": sides, "side_criterion": crit, "rule": "media e deviazione per giunto sui blocchi "
             "validi (200 Hz) delle registrazioni dello split di pretraining; standardizzazione per lato se il criterio lo dice (docs/pilot_posa_v1.md §2 (iv))",
             "generated_unix": time.time()}
    text = json.dumps(stats, indent=1)
    stats["sha256_without_this_field"] = hashlib.sha256(text.encode()).hexdigest()
    (Path(out_root) / STATS_FILE).write_text(json.dumps(stats, indent=1))
    return stats


def load_stats(root: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, str]:
    """Media e deviazione per giunto da `pose_stats.json`: (mean_all (20,), mean_by_side {side: (20,)} o None, std..., sha256 del file)."""
    p = Path(root) / STATS_FILE
    raw = p.read_bytes()
    stats = json.loads(raw)
    return stats, hashlib.sha256(raw).hexdigest()


# --- regola per il margine M, dal rapporto del collaudo (congelata il 26/10/2026 PRIMA di leggere il rapporto; `docs/decisioni.md`) -----------
MARGIN_TOL = 0.25  # un bin del profilo e' «fuori» se la sua |differenza prima| media si scosta piu' del 25% dal valore lontano dai bordi
MARGIN_MIN_N = 200  # un bin con meno campioni non conta (ne' dentro ne' fuori)
DIP_FRAC_MAX = 1e-3  # «V» tollerate: righe nei minimi simultanei fino allo 0,1% delle righe non nulle; oltre serve una regola in piu'
ZERO_FRAC_RANGE = (0.02, 0.40)  # frazione di righe a zero plausibile sul collaudo (paper: 12,7%)
MAX_ABS_RADIANS = 7.0  # oltre 2 pi (piu' un margine) gli angoli non sono radianti: fermarsi
SIDE_MIN_RECORDINGS = 15  # sulle 20 per lato chieste al collaudo
# Regola di riserva R1-R5 (scritta il 27/10/2026 DOPO aver visto il profilo a 200 ms del collaudo 62364572, su proposta del `critico`, PRIMA del
# collaudo col profilo lungo; e' una regola POST HOC informata da quel profilo, non una pre-registrazione: decisioni.md). Si CALCOLA solo se la v1
# e' non determinata o da' M > V1_MAX_M. R1: L = media del profilo nei bin PLATEAU_LAGS (150-200 ms). R2, persistenza a due lati: la lettura «il
# plateau e' comportamento, non artefatto» regge solo se TUTTI i bin contati oltre 400 campioni stanno fra 0,80 L e 1,25 L (pavimento e tetto su
# tutto il profilo misurato, quinta review) e ci sono bin contati almeno fino a 2.000 campioni; una discesa sotto 0,80 L, presto o tardi, e' il
# segno di un artefatto che decade o finisce, e allora la riserva NON si applica (vale la v1 sul profilo lungo; se da' M > 400 ci si ferma e si
# porta a Simone con la perdita). R3: M3 = `lag_to` dell'ultimo bin entro 300 campioni sopra (1 + V2_TOL) L. R5: profilo per altezza del salto verso lo zero (terzili
# della norma della posa a 150-200 ms dal bordo, ogni terzile sul proprio L): un artefatto d'interpolazione o di filtro scala col salto, il
# comportamento e il rumore della cinematica inversa no; M5 = ultimo bin entro 300 in cui alto / basso supera 1 + JUMP_RATIO_TOL. M = max(M3, M5);
# se R5 non e' calcolabile (terzili assenti, plateau magro) M = M3, dichiarato nella proposta.
# Seconda, terza e quarta review del critico (27/10): la riserva si CALCOLA se la v1 e' «non determinata» o da' M > V1_MAX_M, ma si APPLICA solo
# se NESSUN bin contato del profilo lungo «rientra», dove rientrare e' A UN LATO SOLO: rapporto al lontano <= 1 + MARGIN_TOL = 1,25 (anche sotto
# il lontano: una contaminazione che finisce con un gradino verso il basso e' finita). Predicato proprio, indipendente dall'ultimo bin e dallo
# stato della v1: un profilo che rientra e poi riesce non e' «mai rientrato» e ci si ferma; se la v1 risulta determinata solo perche' gli ultimi
# bin non sono contati, vale il predicato sui bin contati. R2 tiene il pavimento 0,80 L e il tetto 1,25 L su TUTTI i bin contati oltre 400 (quinta
# review: anche dopo 2.000, cosi' un livello fra 1,25 x lontano e 0,80 L ferma sempre, senza zona grigia). Se la v1 e' determinata con M > V1_MAX_M
# (il profilo rientra solo dopo 200 ms) la riserva e' solo informativa e ci si ferma portando a Simone la v1 con la perdita e R3 / R5.
PLATEAU_LAGS = (301, 400)  # campioni dal bordo: 150-200 ms
V1_MAX_M = 400  # oltre 200 ms di margine la v1 non decide da sola (perdita > 10% delle righe sul collaudo): fermarsi, decisione di Simone
LONG_COLLAUDO_MIN_USERS = 10  # collaudo lungo: almeno 10 utenti (il primo ne aveva 2); si passa a margin_from_collaudo(min_users=...)
V2_TOL = 0.10  # R3: soglia sul plateau; PROPOSTA di Claude del 27/10 con la tabella di sensibilita' in decisioni.md (0,25 -> 140; 0,10 -> 200; 0,05 -> 240)
V2_PERSIST_FROM = 401  # R2: banda a due lati su tutti i bin contati da qui in poi (quinta review: pavimento e tetto su tutto il misurato)
V2_PERSIST_BAND = (0.80, 1.25)  # R2: [0,80 L; 1,25 L] su ogni bin contato oltre 400
V2_MIN_COUNTED_LAG = 2000  # R2: bin CONTATI almeno fino a 1 s; su un profilo corto la riserva da' solo un valore provvisorio
JUMP_RATIO_TOL = 0.25  # R5: alto / basso (ciascuno normalizzato al proprio plateau) sopra 1,25 = contaminazione che scala col salto
JUMP_MIN_RUNS = 9  # R5: terzili solo con almeno 9 blocchi di zeri con altezza misurabile


def _ratio_bins(raw: list, far, min_n: int) -> list:
    """Bin del profilo con il rapporto al valore lontano (None se il bin non conta: pochi campioni, media assente o `far` nullo)."""
    out = []
    for b in raw or []:
        m = b.get("mean_abs_diff")
        counted = bool(far) and m is not None and int(b["n"]) >= min_n
        out.append({"lag_from": int(b["lag_from"]), "lag_to": int(b["lag_to"]), "n": int(b["n"]), "mean_abs_diff": m,
                    "ratio": (m / far) if counted else None, "conta": counted})
    return out


def _level(rbins: list, lags: tuple[int, int], key: str = "ratio") -> tuple[float | None, int]:
    """Media (non pesata) di `key` sui bin contati dentro `lags` (lag_from >= lo e lag_to <= hi) e quanti sono."""
    vals = [b[key] for b in rbins if b["conta"] and b["lag_from"] >= lags[0] and b["lag_to"] <= lags[1] and b.get(key) is not None]
    return (sum(vals) / len(vals) if vals else None), len(vals)


def jump_profile_margin(terciles: dict, *, min_n: int = MARGIN_MIN_N, jump_tol: float = JUMP_RATIO_TOL) -> dict:
    """R5: dal profilo per terzile di altezza del salto (`by_jump_tercile` del rapporto: `low`, `mid`, `high`, ognuno con `bins` assoluti),
    ogni terzile e' normalizzato al proprio plateau (PLATEAU_LAGS); il bin e' contaminato se alto / basso > 1 + jump_tol (bin contati in
    entrambi). M5 = `lag_to` dell'ultimo bin contaminato entro PLATEAU_LAGS[0] - 1. Indipendente dal valore lontano e dal comportamento comune."""
    lo, hi = terciles.get("low") or {}, terciles.get("high") or {}
    lb = [{"lag_from": int(b["lag_from"]), "lag_to": int(b["lag_to"]), "n": int(b["n"]), "mean_abs_diff": b.get("mean_abs_diff"),
           "conta": b.get("mean_abs_diff") is not None and int(b["n"]) >= min_n} for b in lo.get("bins") or []]
    hb = [{"lag_from": int(b["lag_from"]), "lag_to": int(b["lag_to"]), "n": int(b["n"]), "mean_abs_diff": b.get("mean_abs_diff"),
           "conta": b.get("mean_abs_diff") is not None and int(b["n"]) >= min_n} for b in hi.get("bins") or []]
    l_lo, n_lo = _level(lb, PLATEAU_LAGS, "mean_abs_diff")
    l_hi, n_hi = _level(hb, PLATEAU_LAGS, "mean_abs_diff")
    if not lb or not hb or n_lo < 5 or n_hi < 5 or not l_lo or not l_hi:
        return {"applicabile": False, "motivo": "terzili assenti o plateau non misurabile in uno dei due", "M5": None}
    rows, last = [], None
    for a, b in zip(lb, hb):
        if a["conta"] and b["conta"] and a["mean_abs_diff"] and a["lag_from"] == b["lag_from"]:
            r = (b["mean_abs_diff"] / l_hi) / (a["mean_abs_diff"] / l_lo)
            contaminated = r > 1 + jump_tol
            rows.append({"lag_from": a["lag_from"], "lag_to": a["lag_to"], "high_over_low": r, "contaminato": contaminated})
            if contaminated and a["lag_to"] < PLATEAU_LAGS[0]:
                last = a["lag_to"]
    return {"applicabile": True, "M5": int(last) if last is not None else 0, "plateau_low": l_lo, "plateau_high": l_hi,
            "h_range_low": [lo.get("h_min"), lo.get("h_max")], "h_range_high": [hi.get("h_min"), hi.get("h_max")],
            "n_runs_low": lo.get("n_runs"), "n_runs_high": hi.get("n_runs"), "jump_tol": jump_tol, "bins": rows}


def margin_from_collaudo(report: dict, *, tol: float = MARGIN_TOL, min_n: int = MARGIN_MIN_N, dip_frac_max: float = DIP_FRAC_MAX,
                         v2_tol: float = V2_TOL, jump_tol: float = JUMP_RATIO_TOL, min_users: int | None = None) -> dict:
    """Legge il rapporto JSON del collaudo (`scripts/ingest_emg2pose_pose.py --limit N --both-sides`) e applica, nell'ordine:
    precondizioni (se una manca: FERMARSI, nessun M; `min_users` = LONG_COLLAUDO_MIN_USERS per il collaudo lungo); la regola v1 congelata il
    26/10/2026 (M = `lag_to` dell'ULTIMO bin del profilo dei bordi fuori tolleranza |bin / lontano - 1| > tol, bin con almeno min_n campioni; 0 se
    nessuno; «non determinato» se fuori anche l'ultimo bin); se la v1 e' non determinata o da' M > V1_MAX_M, si CALCOLA la riserva R1-R5 (post
    hoc, 27/10: vedi le costanti), che si APPLICA solo se nessun bin contato rientra sotto 1 + tol volte il lontano (a un lato solo) e la
    persistenza R2 regge su tutto il profilo lungo; altrimenti ci si ferma e la proposta porta a Simone la v1 (con la perdita) e R3 / R5
    informativi; la regola per le «V» (nessuna in
    piu' se le righe nei minimi simultanei sono <= dip_frac_max delle righe non nulle). Riporta il tetto della perdita di righe dovuta a M."""
    col = report.get("collaudo") or {}
    recs = report.get("per_recording") or []
    problems = []
    per_side = col.get("per_side") or {}
    sides_ok = {s: n for s, n in per_side.items() if n >= SIDE_MIN_RECORDINGS}
    if len(sides_ok) < 2:
        problems.append(f"lati con almeno {SIDE_MIN_RECORDINGS} registrazioni: {sides_ok} (ne servono due)")
    fz = col.get("frac_zero_rows")
    if fz is None or not (ZERO_FRAC_RANGE[0] <= fz <= ZERO_FRAC_RANGE[1]):
        problems.append(f"frazione di righe a zero {fz} fuori da {ZERO_FRAC_RANGE}")
    mx = col.get("max_abs_overall")
    if mx is None or mx > MAX_ABS_RADIANS:
        problems.append(f"massimo assoluto {mx} oltre {MAX_ABS_RADIANS}: non radianti?")
    n_failed = len(report.get("failed") or {})
    if n_failed:
        problems.append(f"{n_failed} registrazioni fallite")
    users_all = sorted({str(r.get("user")) for r in recs})
    if min_users is not None and len(users_all) < min_users:
        problems.append(f"utenti {len(users_all)} < {min_users} richiesti per il collaudo lungo")
    rows = sum(int(r["n_samples"]) for r in recs)
    zero_rows_total = sum(int(r["n_zero_rows"]) for r in recs)
    runs_total = sum(int(r["n_zero_runs"]) for r in recs)
    invalid0 = sum(int(r["n_invalid_rows"]) for r in recs)
    users = sorted({str(r.get("user")) for r in recs})

    def loss(m):
        return (min(1.0, 2 * m * runs_total / rows) if rows else None) if m else (0.0 if m == 0 else None)

    out = {"precondizioni": {"ok": not problems, "problemi": problems, "per_side": per_side, "frac_zero_rows": fz, "max_abs_overall": mx,
                             "n_nonfinite_rows": col.get("n_nonfinite_rows"), "n_failed": n_failed, "n_recordings": len(recs),
                             "n_users": len(users), "per_user": col.get("per_user")},
           "soglie": {"tol": tol, "min_n": min_n, "dip_frac_max": dip_frac_max, "zero_frac_range": list(ZERO_FRAC_RANGE),
                      "max_abs_radians": MAX_ABS_RADIANS, "side_min_recordings": SIDE_MIN_RECORDINGS, "v1_max_m": V1_MAX_M,
                      "plateau_lags": list(PLATEAU_LAGS), "v2_tol": v2_tol, "v2_persist_from": V2_PERSIST_FROM, "soglia_rientro_ratio": 1 + tol,
                      "v2_persist_band": list(V2_PERSIST_BAND), "v2_min_counted_lag": V2_MIN_COUNTED_LAG, "jump_ratio_tol": jump_tol}}
    prof = col.get("edge_profile") or {}
    far = prof.get("far_mean_abs_diff")
    bins = _ratio_bins(prof.get("bins"), far, min_n)
    # --- v1 (26/10/2026), invariata
    last_out = None
    for b in bins:
        b["fuori"] = bool(b["conta"] and abs(b["ratio"] - 1.0) > tol)
        if b["fuori"]:
            last_out = b["lag_to"]
    if not far or not bins:
        stato, M, why = "non_determinato", None, "profilo dei bordi assente o valore lontano nullo"
    elif bins[-1]["fuori"]:
        stato, M = "non_determinato", None
        why = f"anche l'ultimo bin (fino a {bins[-1]['lag_to']} campioni) e' fuori tolleranza: il profilo non torna entro il massimo misurato"
    elif last_out is None:
        stato, M, why = "nessuna_contaminazione", 0, "nessun bin fuori tolleranza"
    else:
        stato, M, why = "determinato", last_out, f"ultimo bin fuori tolleranza: fino a {last_out} campioni dal bordo"
    out["profilo"] = {"far_mean_abs_diff": far, "far_n": prof.get("far_n"), "bins": bins, "n_bins_non_contati": sum(1 for b in bins if not b["conta"]),
                      "max_lag_measured": int(bins[-1]["lag_to"]) if bins else None,
                      "max_lag_counted": max((b["lag_to"] for b in bins if b["conta"]), default=0)}
    out["v1"] = {"stato": stato, "M": M, "motivo": why, "perdita_tetto": loss(M)}
    inside_bins = [b["lag_to"] for b in bins if b["conta"] and b["ratio"] <= 1 + tol]  # «rientra», a un lato solo: anche sotto il lontano
    never_returns = bool(bins) and not inside_bins
    # --- riserva R1-R5 (27/10/2026, post hoc): si calcola se la v1 non decide o da' M > V1_MAX_M; si applica solo se il profilo non rientra mai
    v2 = {"calcolata": False}
    if bins and (stato == "non_determinato" or (stato == "determinato" and M > V1_MAX_M)):
        v1_stato, v1_M = stato, M
        level, n_pl = _level(bins, PLATEAU_LAGS)
        counted_max = out["profilo"]["max_lag_counted"]
        if n_pl < 5 or not level:
            v2 = {"calcolata": False, "motivo": f"meno di 5 bin contati nel plateau {PLATEAU_LAGS}", "v1_stato": v1_stato, "v1_M": v1_M}
            stato, M, why = "non_determinato", None, why + "; riserva non calcolabile: plateau non misurabile; decide Simone"
        else:
            persist = [b for b in bins if b["conta"] and b["lag_from"] >= V2_PERSIST_FROM]
            below = [b["lag_to"] for b in persist if b["ratio"] < V2_PERSIST_BAND[0] * level]
            above = [b["lag_to"] for b in persist if b["ratio"] > V2_PERSIST_BAND[1] * level]
            short = counted_max < V2_MIN_COUNTED_LAG
            inside = [b["lag_to"] for b in bins if b["conta"] and b["lag_to"] < PLATEAU_LAGS[0] and b["ratio"] > (1 + v2_tol) * level]
            m3 = max(inside) if inside else 0
            r5 = jump_profile_margin(prof.get("by_jump_tercile") or {}, min_n=min_n, jump_tol=jump_tol)
            m5 = r5.get("M5")
            m_res = max(m3, m5) if m5 is not None else m3
            applicabile = never_returns and not short and not below and not above
            v2 = {"calcolata": True, "applicabile": applicabile, "v1_stato": v1_stato, "v1_M": v1_M, "v1_perdita_tetto": loss(v1_M),
                  "profilo_rientra": not never_returns, "bins_rientrati": inside_bins[:50], "n_bins_rientrati": len(inside_bins),
                  "soglia_rientro_ratio": 1 + tol, "R5_calcolabile": m5 is not None,
                  "R1_plateau_level_ratio_to_far": level, "R1_n_plateau_bins": n_pl,
                  "R2_profilo_corto": short, "R2_max_lag_counted": counted_max, "R2_bins_sotto_pavimento_oltre_400": below,
                  "R2_bins_sopra_tetto_oltre_400": above, "R3_M3": int(m3), "R5": r5, "M": int(m_res),
                  "sensibilita_R3": {str(tt): (max([b["lag_to"] for b in bins if b["conta"] and b["lag_to"] < PLATEAU_LAGS[0]
                                                    and b["ratio"] > (1 + tt) * level] or [0])) for tt in (0.25, 0.20, 0.15, 0.10, 0.05)}}
            if applicabile:
                stato, M = "determinato_riserva", int(m_res)
                why = (f"v1 {v1_stato}" + (f" (M = {v1_M})" if v1_M is not None else "") + f"; riserva: plateau {PLATEAU_LAGS[0]}-{PLATEAU_LAGS[1]} = "
                       f"{level:.2f} x lontano, persistenza R2 ok fino a {counted_max} campioni; R3 (> {1 + v2_tol:.2f} x plateau) = {m3}; "
                       f"R5 (salto alto / basso > {1 + jump_tol:.2f}) = {m5}; M = max")
            else:
                stato, M = "non_determinato", None
                reasons = []
                if not never_returns:
                    if v1_stato == "determinato":
                        reasons.append(f"la v1 e' determinata con M = {v1_M} > {V1_MAX_M} (il profilo rientra sotto {1 + tol:.2f} x lontano solo dopo "
                                       "200 ms): decide Simone con la perdita")
                    else:
                        re_exits = any(b["conta"] and b["ratio"] > 1 + tol and b["lag_to"] > min(inside_bins) for b in bins)
                        if re_exits:
                            reasons.append(f"il profilo rientra sotto {1 + tol:.2f} x lontano e poi riesce ({len(inside_bins)} bin contati dentro, "
                                           f"fino a {max(inside_bins)} campioni): la riserva vale solo se non rientra mai; decide Simone")
                        else:
                            reasons.append(f"il profilo rientra sotto {1 + tol:.2f} x lontano da {min(inside_bins)} campioni e non riesce piu': la "
                                           f"contaminazione e' finita li' ma la v1 non la chiude (ultimo bin fuori); decide Simone")
                elif v1_stato == "determinato":
                    reasons.append(f"la v1 risulta determinata (M = {v1_M}) solo per bin finali NON contati: nessun bin contato rientra, quel M non e' "
                                   "un esito")
                if short:
                    reasons.append(f"profilo contato solo fino a {counted_max} campioni (< {V2_MIN_COUNTED_LAG}): valore PROVVISORIO M = {m_res}")
                if below:
                    reasons.append(f"R2 caduta: {len(below)} bin contati oltre {V2_PERSIST_FROM - 1} sotto {V2_PERSIST_BAND[0]:.2f} x plateau (fino a "
                                   f"{max(below)}): segno di artefatto che decade o finisce")
                if above:
                    reasons.append(f"R2 caduta: {len(above)} bin oltre 400 sopra {V2_PERSIST_BAND[1]:.2f} x plateau")
                reasons.append(f"valori informativi della riserva: R3 = {m3}, R5 = {m5}")
                why = (f"v1 {v1_stato}" + (f" con M = {v1_M}" if v1_M is not None else f": {why}") + "; riserva non applicata: " + "; ".join(reasons))
    out["v2"] = v2
    out["margine"] = {"M": M, "stato": stato, "motivo": why}
    out["perdita"] = {"rows": rows, "frac_zero_rows": zero_rows_total / rows if rows else None, "n_zero_runs": runs_total,
                      "frac_invalid_m0": invalid0 / rows if rows else None, "frac_rows_margine_tetto": loss(M)}
    dips = col.get("simultaneous_dips") or {}
    nonzero = rows - zero_rows_total
    frac_dip = (int(dips.get("n_dip_rows_total", 0)) / nonzero) if nonzero else None
    out["V"] = {"n_dip_rows_total": dips.get("n_dip_rows_total"), "n_dip_runs_total": dips.get("n_dip_runs_total"),
                "recordings_with_dips": dips.get("recordings_with_dips"), "frac_dip_rows_over_nonzero": frac_dip,
                "regola_aggiuntiva_necessaria": None if frac_dip is None else bool(frac_dip > dip_frac_max)}
    if problems:
        out["proposta"] = "FERMARSI: precondizioni non soddisfatte (" + "; ".join(problems) + ")"
    elif stato == "non_determinato":
        info = f"; valori informativi della riserva: R3 = {v2.get('R3_M3')}, R5 = {(v2.get('R5') or {}).get('M5')}" if v2.get("calcolata") else ""
        if v2.get("calcolata") and v2.get("R2_profilo_corto"):
            out["proposta"] = "FERMARSI: margine non determinato (" + why + "); profilo lungo da misurare e poi rileggere"
        elif v2.get("calcolata") and v2.get("v1_M") is not None and v2.get("profilo_rientra"):
            out["proposta"] = (f"FERMARSI: la riserva non si applica e la v1 sul profilo lungo da' M = {v2['v1_M']} campioni (perdita fino al "
                               f"{100 * (v2['v1_perdita_tetto'] or 0):.1f}% delle righe): decisione di Simone, con la perdita in chiaro" + info)
        elif v2.get("calcolata") and v2.get("v1_M") is not None:
            out["proposta"] = ("FERMARSI: nessun bin contato rientra ma la riserva non si applica (" + why + "); la v1 e' determinata solo per bin "
                               "finali non contati e il suo M non e' un esito: decisione di Simone" + info)
        elif v2.get("calcolata"):
            out["proposta"] = "FERMARSI: margine non determinato (" + why + "): decisione di Simone" + info
        else:
            out["proposta"] = "FERMARSI: margine non determinato (" + why + ")"
    elif out["V"]["regola_aggiuntiva_necessaria"]:
        out["proposta"] = (f"M = {M} campioni, MA le «V» superano la soglia ({frac_dip:.2e} > {dip_frac_max}): serve una regola in piu' "
                           "prima dell'ingest intero")
    elif stato == "determinato_riserva":
        out["proposta"] = (f"M = {M} campioni ({M / NATIVE_FS_HZ * 1000:.0f} ms per lato di ogni blocco di zeri) con la regola di riserva "
                           f"R1-R5 (post hoc): R3 = {v2['R3_M3']}, R5 = {v2['R5'].get('M5')}"
                           + ("" if v2["R5_calcolabile"] else " (R5 NON calcolabile: M = M3, dichiarato)") + "; nessuna regola per le «V»")
    else:
        out["proposta"] = f"M = {M} campioni ({M / NATIVE_FS_HZ * 1000:.0f} ms per lato di ogni blocco di zeri); nessuna regola per le «V»"
    return out


if __name__ == "__main__":  # python3 -m wearusfm.ingest.emg2pose_pose <rapporto del collaudo>.json [--lungo]  (--lungo: almeno 10 utenti)
    import sys

    _min_users = LONG_COLLAUDO_MIN_USERS if "--lungo" in sys.argv[2:] else None
    print(json.dumps(margin_from_collaudo(json.loads(Path(sys.argv[1]).read_text()), min_users=_min_users), indent=1))
