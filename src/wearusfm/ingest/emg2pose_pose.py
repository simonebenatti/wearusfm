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
    assumere la forma (rampa lineare, transitorio del filtro). Le righe a zero e quelle non finite non entrano."""
    finite = np.all(np.isfinite(angles), axis=1)
    d = np.full(angles.shape[0], np.nan)
    diff = np.abs(np.diff(np.asarray(angles, dtype=np.float64), axis=0)).mean(axis=1)  # fra la riga i e la i+1
    ok = finite[1:] & finite[:-1] & ~zero[1:] & ~zero[:-1]
    d[1:][ok] = diff[ok]
    out = {"bin_samples": bin_size, "max_lag_samples": max_lag, "n_zero_rows": int(zero.sum()), "n_zero_runs": len(runs_of(zero))}
    good = np.isfinite(d)
    if not zero.any():
        far = d[good]
        out.update({"bins": [], "far": _dist_summary(far)})
        return out
    dist = distance_to(zero)
    bins = []
    for lo in range(1, max_lag + 1, bin_size):
        hi = min(lo + bin_size - 1, max_lag)
        sel = good & (dist >= lo) & (dist <= hi)
        bins.append({"lag_from": lo, "lag_to": hi, "n": int(sel.sum()), "mean_abs_diff": float(d[sel].mean()) if sel.any() else None})
    far = d[good & (dist > max_lag)]
    out.update({"bins": bins, "far": _dist_summary(far)})
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
                          measure: bool = True, fields: dict | None = None) -> dict:
    """Una registrazione -> `pose_200hz_f16.npy`, `pose_valid.npy`, `pose_meta.json` in `<out_root>/u<utente>/<stem>/`. Con `measure` calcola
    anche profilo dei bordi e minimi simultanei (collaudo); senza, solo validita', blocchi e statistiche."""
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
        meta["edge_profile"] = edge_profile(angles, zero)
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
    meta' per lato e dentro ogni lato quote uguali per i valori di `moving_hand` (3 attesi: left, right, both)."""

    def __init__(self, limit: int, both_sides: bool, moving_values: tuple[str, ...] = ("left", "right", "both")):
        self.limit, self.both_sides = limit, both_sides
        self.per_side = math.ceil(limit / 2) if both_sides else limit
        self.per_bucket = math.ceil(self.per_side / len(moving_values)) if both_sides else limit
        self.taken: dict[tuple[str, str], int] = {}
        self.total = 0

    def accept(self, meta_row: dict | None) -> bool:
        if self.total >= self.limit:
            return False
        side = (meta_row or {}).get("side", "?")
        mh = (meta_row or {}).get("moving_hand", "?")
        if self.both_sides:
            if sum(v for (s, _), v in self.taken.items() if s == side) >= self.per_side:
                return False
            if self.taken.get((side, mh), 0) >= self.per_bucket:
                return False
        self.taken[(side, mh)] = self.taken.get((side, mh), 0) + 1
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
