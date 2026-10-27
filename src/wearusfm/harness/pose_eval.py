"""Valutazione della posa FUORI CAMPIONE (emendamento 1 a `pilot_posa_v1`, firmato da Simone il 26/10/2026; `docs/emendamento_pilot_posa_v1.md`):

- E1: la testa di posa di R3 sugli utenti dello split `val` di emg2pose (15 utenti assenti dal training), contro il **predittore costante** (la
  media del train per giunto, cioe' 0 nello spazio standardizzato): errore assoluto medio per giunto nell'unita' degli angoli, per utente e
  complessivo, su finestre dinamiche e poco dinamiche; **regola congelata**: rapporto MAE / MAE_costante sulle finestre dinamiche con limite
  superiore dell'IC95 (bootstrap sugli utenti, 1.000 ricampionamenti, seme 20261009) < 0,9 e mediana per utente < 0,9.
- E2: **controllo disallineato in valutazione**: lo stesso modello, target di posa di un'altra finestra dello stesso utente, della stessa fase e
  dello stesso lato (stessa registrazione a >= 8 s, altrimenti un'altra SESSIONE dello stesso utente; senza candidati la finestra si esclude e si
  conta), coppie quasi identiche scartate (distanza standardizzata sotto il 10° percentile); **regola congelata**: limite inferiore dell'IC95 di
  MAE_disallineato / MAE_costante > 0,95, altrimenti (a) non e' interpretabile. Il **gap** MAE_disallineato - MAE_allineato si calcola sullo
  STESSO supporto (stesse finestre dinamiche con coppia conservata, stessa intersezione delle maschere, stessi pesi), per utente e complessivo
  (correzioni del 27/10/2026 dalla review di GPT, `docs/reviews/r3_updates_review_20261027.md`: la E1 e il gate di E2 restano invariati).
- E3: **accessibilita'**: regressione ridge lineare per istante dai latenti congelati del backbone (la media sui K latenti, la parte del backbone
  di `pool_p1p2` prima della media sul tempo) alla posa standardizzata, allenata su un sottoinsieme fisso del train e letta su `val` con le
  stesse metriche: vale per R3, R3c e R0 (anche senza testa).

Estrazione: finestre fisse (10 per registrazione, 4 s, sulla griglia a 200 ms del loader, seme 20261026, senza sovrapposizione), lo stesso
preprocessing del training (filtro del loader e scala di sessione di `estimate_session_scale`), encoder senza maschera (come la sonda). I
risultati stanno fuori da Git; il rapporto JSON e' piccolo e si copia sul Mac. Nessuna soglia qui dentro e' nuova: sono quelle dell'emendamento.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wearusfm.data import pose_targets as PT
from wearusfm.data import pretraining_loader as L
from wearusfm.ingest.emg2pose import _SAFE_RE

SEED_WINDOWS = 20261026
SEED_BOOTSTRAP = 20261009
N_BOOT = 1000
RULE = {"learned_ci_upper": 0.9, "learned_median_user": 0.9, "misaligned_ci_lower": 0.95, "misaligned_min_gap_s": 8.0,
        "misaligned_drop_quantile": 0.10, "dynamic_split": "mediana della varianza della posa per finestra su val",
        "ridge_alphas": (0.1, 1.0, 10.0, 100.0), "instants_per_window": 8}
DATASET = "emg2pose"


@dataclass
class EvalConfig:
    n_windows: int = 10
    window_s: float = 4.0
    seed: int = SEED_WINDOWS
    batch: int = 32
    device: str = "cpu"
    instants_per_window: int = RULE["instants_per_window"]
    band_hz: tuple[float, float] | None = (20.0, 450.0)
    loader: L.LoaderConfig = field(default_factory=lambda: L.signed_config())


# --- registrazioni ----------------------------------------------------------------------------------------------------------------------

def recordings_for(csv_path: Path, processed_root: Path, split: str, *, n_users: int | None = None, n_recordings: int | None = None,
                   seed: int = SEED_WINDOWS) -> list[dict]:
    """Le registrazioni di emg2pose dello `split` del CSV presenti nell'albero processato (`<root>/u<utente>/<stem>/metadata.json`), con
    utente, sessione, fase, lato. Con `n_users`/`n_recordings` un sottoinsieme fisso (seme): prima gli utenti, poi le registrazioni spalmate sugli
    utenti scelti (il sottoinsieme del train per E3)."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] != split:
                continue
            stem = Path(r["filename"]).stem
            d = Path(processed_root) / f"u{_SAFE_RE.sub('_', r['user'])}" / _SAFE_RE.sub("_", stem)
            if (d / "metadata.json").exists():
                rows.append({"user": r["user"], "session": r["session"], "stage": r["stage"], "side": r["side"],
                             "moving_hand": r["moving_hand"], "stem": stem, "path": str(d)})
    rows.sort(key=lambda r: (r["user"], r["session"], r["stem"]))
    rng = np.random.default_rng(seed)
    if n_users is not None:
        users = sorted({r["user"] for r in rows})
        keep = set(rng.choice(users, size=min(n_users, len(users)), replace=False).tolist())
        rows = [r for r in rows if r["user"] in keep]
    if n_recordings is not None and len(rows) > n_recordings:
        by_user: dict[str, list] = {}
        for r in rows:
            by_user.setdefault(r["user"], []).append(r)
        for u in by_user:
            rng.shuffle(by_user[u])
        picked, k = [], 0
        while len(picked) < n_recordings and any(len(v) > k for v in by_user.values()):
            for u in sorted(by_user):
                if len(by_user[u]) > k and len(picked) < n_recordings:
                    picked.append(by_user[u][k])
            k += 1
        rows = sorted(picked, key=lambda r: (r["user"], r["session"], r["stem"]))
    return rows


def fixed_windows(view, cfg: L.LoaderConfig, n: int, window_s: float, seed: int) -> list:
    """`n` finestre lunghe `window_s` sulla griglia del loader (`admissible_starts`), senza sovrapposizione, spalmate sulla registrazione con uno
    scostamento casuale (seme): le stesse finestre a ogni estrazione (confronti appaiati fra modelli)."""
    fs, patch = view.fs, cfg.patch_ms / 1000.0
    need = int(round(window_s / patch))
    cands = []
    for cand in L.admissible_starts(view, cfg):
        trial, s, e, anchor, k0, ks = cand
        for k in ks.tolist():
            w = L.window_at(view, cfg, cand, k)
            if w is not None and w.n_patches >= need:
                cands.append(L.Window(w.trial, w.start, w.start + int(math.ceil(need * patch * fs - 1e-9)), w.span, need))
    cands.sort(key=lambda w: w.start)
    if not cands:
        return []
    rng = np.random.default_rng([seed, len(cands)])
    gap = int(math.ceil(window_s * fs))
    chosen, last_end = [], -1
    stride = max(1, len(cands) // max(n, 1))
    offset = int(rng.integers(0, stride))
    for w in cands[offset::stride]:
        if w.start >= last_end:
            chosen.append(w)
            last_end = w.start + gap
        if len(chosen) >= n:
            break
    return chosen


# --- estrazione ---------------------------------------------------------------------------------------------------------------------------

def extract(model, recordings: list[dict], pose_store: PT.PoseStore, ev: EvalConfig, log=print) -> dict:
    """Latenti per istante, predizione della testa (se c'e'), target e maschere di posa, metadati, per tutte le finestre fisse delle
    registrazioni. Tutto in memoria: ~20.000 finestre x 160 patch x 384 in float16 = ~2,4 GB per `val`."""
    import torch

    from wearusfm.model import channel_codes as CC
    from wearusfm.model.channel_identity import codes_to_tensors
    from wearusfm.model.fm import ModelInputs
    from wearusfm.model.local_encoder import sets_to_tensors
    from wearusfm.model.pose_head import latent_mean

    model.eval()
    lat, pred, target, mask, std, meta, pose_var = [], [], [], [], [], [], []
    has_head = getattr(model, "pose_head", None) is not None
    d = model.cfg.dim
    device = ev.device
    for ri, rec in enumerate(recordings):
        view = L.SessionView.open(Path(rec["path"]), ev.loader.split_at_gaps)
        rng = np.random.default_rng([ev.seed, ri])
        try:
            scale = L.estimate_session_scale(view, ev.loader, rng)
        except ValueError as e:  # scala non calcolabile (segnale costante): la registrazione si salta e si conta
            log(f"  {rec['stem']}: saltata ({e})")
            continue
        wins = fixed_windows(view, ev.loader, ev.n_windows, ev.window_s, ev.seed)
        if not wins:
            log(f"  {rec['stem']}: nessuna finestra ammessa")
            continue
        layout, code = CC.layout_from_montage(view.montage), CC.anatomy_codes(view.montage)
        sets = CC.attention_sets(layout, ev.loader.k_neighbors)
        c = layout.n_channels
        row = {"dataset": DATASET}
        for i in range(0, len(wins), ev.batch):
            chunk = wins[i:i + ev.batch]
            xs = [L.read_window(view, w.trial, w.start, w.stop, w.span, ev.loader).astype(np.float32) / np.float32(scale) for w in chunk]
            n = len(xs)
            inp = ModelInputs([torch.from_numpy(x).to(device) for x in xs], [view.fs] * n,
                              {k: v.to(device) for k, v in codes_to_tensors(CC.pack_codes([code] * n)).items()},
                              {k: v.to(device) for k, v in sets_to_tensors(CC.pack_attention_sets([sets] * n)).items()}, [c] * n,
                              torch.as_tensor(layout.qc_valid).repeat(n).to(device))
            with torch.no_grad():
                enc = model.encode(inp, None)
                z = latent_mean(enc.z).float()  # (n, P, d)
                p = model.pose_head(z).float().cpu().numpy() if has_head else None
            z = z.cpu().numpy()
            for j, w in enumerate(chunk):
                t, m, sd = pose_store.window(row, view, w, ev.loader.patch_ms)
                P = w.n_patches
                lat.append(z[j, :P].astype(np.float16))
                target.append(t.astype(np.float32))
                mask.append(m)
                std.append(sd.astype(np.float32))
                pred.append(p[j, :P].astype(np.float32) if has_head else np.zeros((P, PT.N_JOINTS), dtype=np.float32))
                rows_ok = m.any(axis=1)
                pose_var.append(float(t[rows_ok].var(axis=0).mean()) if rows_ok.sum() > 1 else float("nan"))
                meta.append({"user": rec["user"], "session": rec["session"], "stage": rec["stage"], "side": rec["side"], "stem": rec["stem"],
                             "recording": ri, "start": int(w.start), "fs": float(view.fs), "n_patches": int(P)})
        log(f"  [{ri + 1}/{len(recordings)}] {rec['stem']}: {len(wins)} finestre")
    if not lat:
        raise ValueError("nessuna finestra estratta")
    P = max(x.shape[0] for x in lat)

    def pad(arrs, fill, dtype):
        out = np.full((len(arrs), P, arrs[0].shape[1]), fill, dtype=dtype)
        for i, a in enumerate(arrs):
            out[i, :a.shape[0]] = a
        return out

    return {"lat": pad(lat, 0, np.float16), "pred": pad(pred, 0, np.float32), "target": pad(target, 0, np.float32), "mask": pad(mask, False, bool),
            "std": np.stack(std), "pose_var": np.asarray(pose_var, dtype=np.float32), "has_head": has_head, "dim": d,
            "meta": meta, "n_recordings": len(recordings), "config": {"n_windows": ev.n_windows, "window_s": ev.window_s, "seed": ev.seed,
                                                                      "pose_stats_sha256": pose_store.stats_sha256, "per_side": pose_store.per_side}}


def save_npz(path: Path, data: dict) -> None:
    np.savez_compressed(path, lat=data["lat"], pred=data["pred"], target=data["target"], mask=data["mask"], std=data["std"],
                        pose_var=data["pose_var"], has_head=np.asarray(bool(data["has_head"])), dim=np.asarray(int(data["dim"])),
                        meta_json=np.asarray(json.dumps(data["meta"])), n_recordings=np.asarray(int(data["n_recordings"])),
                        config_json=np.asarray(json.dumps(data["config"])))


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {"lat": z["lat"], "pred": z["pred"], "target": z["target"], "mask": z["mask"], "std": z["std"], "pose_var": z["pose_var"],
                "has_head": bool(z["has_head"]), "dim": int(z["dim"]), "meta": json.loads(str(z["meta_json"])),
                "n_recordings": int(z["n_recordings"]), "config": json.loads(str(z["config_json"]))}


# --- metriche -----------------------------------------------------------------------------------------------------------------------------

def _window_errors(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per finestra: errore assoluto medio (nell'unita' degli angoli) del modello e del costante, e il numero di elementi validi."""
    e = np.abs(pred - target) * std[:, None, :] * mask
    c = np.abs(target) * std[:, None, :] * mask  # costante = media del train = 0 standardizzato
    n = mask.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, e.sum(axis=(1, 2)) / np.maximum(n, 1), np.nan), np.where(n > 0, c.sum(axis=(1, 2)) / np.maximum(n, 1), np.nan), n


def _pooled_ratio(err: np.ndarray, const: np.ndarray, n: np.ndarray, sel: np.ndarray) -> float:
    s = sel & (n > 0)
    den = float((const[s] * n[s]).sum())
    return float((err[s] * n[s]).sum()) / den if den > 0 else float("nan")


def _bootstrap_users(err, const, n, sel, users: np.ndarray, seed: int = SEED_BOOTSTRAP, n_boot: int = N_BOOT) -> tuple[float, float, list[float]]:
    rng = np.random.default_rng(seed)
    uniq = np.asarray(sorted(set(users[sel & (n > 0)].tolist())))
    if len(uniq) == 0:
        return float("nan"), float("nan"), []
    idx = {u: np.flatnonzero(sel & (n > 0) & (users == u)) for u in uniq}
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([idx[u] for u in draw])
        vals.append(_pooled_ratio(err[ix], const[ix], n[ix], np.ones(len(ix), dtype=bool)))
    vals = np.asarray(vals)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)), []


def _per_user(err, const, n, sel, users) -> dict:
    out = {}
    for u in sorted(set(users.tolist())):
        s = sel & (n > 0) & (users == u)
        if s.any():
            out[u] = {"ratio": _pooled_ratio(err, const, n, s), "mae": float((err[s] * n[s]).sum() / n[s].sum()),
                      "mae_const": float((const[s] * n[s]).sum() / n[s].sum()), "n_windows": int(s.sum())}
    return out


def _block(err, const, n, users, sel, label: str) -> dict:
    ratio = _pooled_ratio(err, const, n, sel)
    lo, hi, _ = _bootstrap_users(err, const, n, sel, users)
    per_user = _per_user(err, const, n, sel, users)
    ratios = [v["ratio"] for v in per_user.values() if np.isfinite(v["ratio"])]
    s = sel & (n > 0)
    return {"label": label, "n_windows": int(s.sum()), "n_users": len(per_user), "mae": float((err[s] * n[s]).sum() / max(n[s].sum(), 1)),
            "mae_const": float((const[s] * n[s]).sum() / max(n[s].sum(), 1)), "ratio": ratio, "ratio_ci95_users": [lo, hi],
            "median_user_ratio": float(np.median(ratios)) if ratios else float("nan"), "per_user": per_user}


def misaligned_pairs(meta: list[dict], target: np.ndarray, mask: np.ndarray, *, min_gap_s: float = RULE["misaligned_min_gap_s"],
                     drop_quantile: float = RULE["misaligned_drop_quantile"], seed: int = SEED_WINDOWS) -> tuple[np.ndarray, dict]:
    """Per ogni finestra i, una finestra j dello STESSO utente, della STESSA fase e dello STESSO lato: prima scelta la stessa registrazione a
    >= `min_gap_s`; altrimenti (fallback) un'ALTRA SESSIONE dello stesso utente, sempre stessa fase e stesso lato; -1 se non c'e' (la finestra si
    esclude e si conta: i vincoli non si rilassano). Seme fisso, scelta deterministica nell'ordine delle finestre. Le coppie con distanza
    standardizzata dei target sotto il `drop_quantile` si scartano (quasi identiche: il controllo non cambierebbe il movimento).
    Correzione del 27/10/2026 (review di GPT): prima il fallback era «registrazione diversa», e poteva prendere la mano opposta della stessa
    sessione allo stesso istante (nessun disallineamento temporale; con la standardizzazione per lato, coordinate diverse)."""
    rng = np.random.default_rng(seed)
    n = len(meta)
    users = np.asarray([m["user"] for m in meta])
    stages = np.asarray([m["stage"] for m in meta])
    sides = np.asarray([m["side"] for m in meta])
    sessions = np.asarray([m["session"] for m in meta])
    recs = np.asarray([m["recording"] for m in meta])
    starts = np.asarray([m["start"] for m in meta], dtype=np.int64)
    fs = np.asarray([m["fs"] for m in meta], dtype=np.float64)
    partner = np.full(n, -1, dtype=np.int64)
    how = np.zeros(n, dtype=np.int8)  # 0 nessun candidato, 1 stessa registrazione a >= min_gap_s, 2 altra sessione
    idx = np.arange(n)
    for i in range(n):
        same = np.flatnonzero((users == users[i]) & (stages == stages[i]) & (sides == sides[i]) & (idx != i))
        if same.size == 0:
            continue
        far = same[(recs[same] == recs[i]) & (np.abs(starts[same] - starts[i]) >= min_gap_s * fs[i])]
        if far.size:
            pool, how[i] = far, 1
        else:
            other = same[sessions[same] != sessions[i]]
            if not other.size:
                continue
            pool, how[i] = other, 2
        partner[i] = int(rng.choice(pool))
    ok = partner >= 0
    dist = np.full(n, np.nan)
    for i in np.flatnonzero(ok):
        j = partner[i]
        both = mask[i] & mask[j]
        if both.any():
            dist[i] = float(np.abs(target[i] - target[j])[both].mean())
    finite = np.isfinite(dist)
    thr = float(np.nanquantile(dist[finite], drop_quantile)) if finite.any() else float("nan")
    keep = finite & (dist >= thr) if finite.any() else finite
    partner[~keep] = -1
    info = {"n_windows": int(n), "n_with_partner": int(ok.sum()), "n_kept": int(keep.sum()), "distance_threshold": thr, "min_gap_s": min_gap_s,
            "n_same_recording": int((how == 1).sum()), "n_other_session": int((how == 2).sum()), "n_no_candidate": int((how == 0).sum()),
            "n_dropped_near_identical": int((ok & ~keep).sum()),
            "n_kept_same_recording": int((keep & (how == 1)).sum()), "n_kept_other_session": int((keep & (how == 2)).sum()),
            "constraints": "stesso utente, stessa fase, stesso lato; prima la stessa registrazione a >= min_gap_s, poi un'altra sessione; "
                           "senza candidati la finestra e' esclusa (vincoli mai rilassati)"}
    return partner, info


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Ridge con intercetta: ritorna (W (d, J), b (J,))."""
    mx, my = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mx, y - my
    d = xc.shape[1]
    w = np.linalg.solve(xc.T @ xc + alpha * np.eye(d), xc.T @ yc)
    return w, my - mx @ w


def _instants(data: dict, per_window: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Istanti validi campionati per finestra: (X (N, d), Y (N, J), std (N, J), utente (N,))."""
    rng = np.random.default_rng([seed, 7])
    xs, ys, sds, us = [], [], [], []
    users = [m["user"] for m in data["meta"]]
    for i in range(data["lat"].shape[0]):
        valid = np.flatnonzero(data["mask"][i].any(axis=1))
        if valid.size == 0:
            continue
        pick = valid if per_window is None or valid.size <= per_window else np.sort(rng.choice(valid, size=per_window, replace=False))
        xs.append(data["lat"][i, pick].astype(np.float32))
        ys.append(data["target"][i, pick])
        sds.append(np.repeat(data["std"][i][None, :], len(pick), axis=0))
        us.extend([users[i]] * len(pick))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(sds), np.asarray(us)


def ridge_accessibility(train: dict, val: dict, *, alphas=RULE["ridge_alphas"], per_window: int = RULE["instants_per_window"],
                        seed: int = SEED_WINDOWS) -> dict:
    """E3: ridge dai latenti alla posa, alpha scelto fra gli utenti del train (2 meta' per utente), letto su tutti gli istanti validi di val con
    le metriche di E1 (per finestra, bootstrap sugli utenti)."""
    xt, yt, _, ut = _instants(train, per_window, seed)
    uniq = sorted(set(ut.tolist()))
    half = set(uniq[::2])
    a_mask = np.asarray([u in half for u in ut])
    best, best_err = None, np.inf
    for alpha in alphas:
        errs = []
        for fit_mask in (a_mask, ~a_mask):
            if fit_mask.sum() < 10 or (~fit_mask).sum() < 10:
                continue
            w, b = ridge_fit(xt[fit_mask], yt[fit_mask], alpha)
            errs.append(float(np.abs(xt[~fit_mask] @ w + b - yt[~fit_mask]).mean()))
        err = float(np.mean(errs)) if errs else np.inf
        if err < best_err:
            best, best_err = alpha, err
    best = best if best is not None else alphas[0]
    w, b = ridge_fit(xt, yt, best)
    pred = np.einsum("npd,dj->npj", val["lat"].astype(np.float32), w) + b
    return {"alpha": best, "inner_mae_std": best_err, "n_train_instants": int(xt.shape[0]), "pred": pred}


def report(val: dict, train: dict | None = None, *, rule: dict = RULE) -> dict:
    """Il rapporto di E1-E3 su `val` (e `train` per E3). Le regole congelate sono applicate e riportate; nessun'altra."""
    meta = val["meta"]
    users = np.asarray([m["user"] for m in meta])
    n_win = val["lat"].shape[0]
    pv = val["pose_var"]
    has = np.isfinite(pv)
    median_var = float(np.nanmedian(pv)) if has.any() else float("nan")
    dynamic = has & (pv >= median_var)
    out = {"rule": {k: (list(v) if isinstance(v, tuple) else v) for k, v in rule.items()}, "n_windows": int(n_win), "n_users": int(len(set(users))),
           "n_recordings": val["n_recordings"], "config": val["config"], "median_pose_var": median_var,
           "n_dynamic": int(dynamic.sum()), "has_head": val["has_head"]}
    if val["has_head"]:
        err, const, n = _window_errors(val["pred"], val["target"], val["mask"], val["std"])
        e1 = {"dynamic": _block(err, const, n, users, dynamic, "finestre dinamiche (regola)"),
              "static": _block(err, const, n, users, has & ~dynamic, "finestre poco dinamiche"),
              "all": _block(err, const, n, users, np.ones(n_win, dtype=bool), "tutte")}
        per_joint = (np.abs(val["pred"] - val["target"]) * val["std"][:, None, :] * val["mask"]).sum(axis=(0, 1)) / np.maximum(val["mask"].sum(axis=(0, 1)), 1)
        per_joint_const = (np.abs(val["target"]) * val["std"][:, None, :] * val["mask"]).sum(axis=(0, 1)) / np.maximum(val["mask"].sum(axis=(0, 1)), 1)
        e1["per_joint_mae"] = per_joint.tolist()
        e1["per_joint_mae_const"] = per_joint_const.tolist()
        d = e1["dynamic"]
        e1["learned"] = bool(np.isfinite(d["ratio_ci95_users"][1]) and d["ratio_ci95_users"][1] < rule["learned_ci_upper"]
                             and d["median_user_ratio"] < rule["learned_median_user"])
        e1["learned_substantially"] = bool(np.isfinite(d["ratio"]) and d["ratio"] < 0.8)
        out["E1"] = e1
        partner, info = misaligned_pairs(meta, val["target"], val["mask"], min_gap_s=rule["misaligned_min_gap_s"],
                                         drop_quantile=rule["misaligned_drop_quantile"])
        ok = partner >= 0
        j = np.maximum(partner, 0)
        # coordinate coerenti: le due finestre della coppia devono avere le stesse statistiche di standardizzazione (stesso lato => uguali;
        # un disaccordo si esclude e si conta, non si converte)
        std_mismatch = ok & ~np.all(np.isclose(val["std"], val["std"][j]), axis=1)
        partner[std_mismatch] = -1
        ok = partner >= 0
        j = np.maximum(partner, 0)
        info["n_excluded_std_mismatch"] = int(std_mismatch.sum())
        info["n_kept"] = int(ok.sum())
        if ok.any():
            tj = val["target"][j]
            mj = val["mask"][j] & val["mask"]  # supporto comune della coppia: intersezione delle maschere valide
            err_m, const_m, n_m = _window_errors(val["pred"], tj, mj, val["std"])  # disallineato sul supporto comune
            err_a, _, n_a = _window_errors(val["pred"], val["target"], mj, val["std"])  # allineato sullo STESSO supporto
            n_m = np.where(ok, n_m, 0)
            n_a = np.where(ok, n_a, 0)
            sel = dynamic & ok
            blk = _block(err_m, const_m, n_m, users, sel, "disallineato, finestre dinamiche")  # gate: formula e soglia invariate
            # gap MAE_disallineato - MAE_allineato sullo stesso supporto: stesse finestre (dinamiche, con coppia conservata), stessa
            # intersezione delle maschere, stessi pesi (elementi validi), stessa aggregazione per utente (review GPT del 27/10/2026)
            recs_all = np.asarray([m["recording"] for m in meta])
            s = sel & (n_m > 0)
            per_user = {}
            for u in sorted(set(users[s].tolist())):
                su = s & (users == u)
                w = n_m[su]
                ma, mm = float((err_a[su] * w).sum() / w.sum()), float((err_m[su] * w).sum() / w.sum())
                per_user[u] = {"mae_aligned": ma, "mae_misaligned": mm, "gap": mm - ma, "n_windows": int(su.sum()), "n_valid_elements": int(w.sum()),
                               "n_same_recording": int((recs_all[su] == recs_all[j[su]]).sum()),
                               "n_other_session": int((recs_all[su] != recs_all[j[su]]).sum())}
            w = n_m[s]
            ma_all = float((err_a[s] * w).sum() / w.sum()) if w.sum() else float("nan")
            mm_all = float((err_m[s] * w).sum() / w.sum()) if w.sum() else float("nan")
            gaps = [v["gap"] for v in per_user.values()]
            paired = {"label": "stesso supporto: finestre dinamiche con coppia conservata, intersezione delle maschere valide della finestra e della partner",
                      "n_windows": int(s.sum()), "n_users": len(per_user), "n_valid_elements": int(w.sum()),
                      "n_same_recording": int((recs_all[s] == recs_all[j[s]]).sum()), "n_other_session": int((recs_all[s] != recs_all[j[s]]).sum()),
                      "mae_aligned": ma_all, "mae_misaligned": mm_all, "gap_mae": mm_all - ma_all,
                      "median_user_gap": float(np.median(gaps)) if gaps else float("nan"), "per_user": per_user}
            out["E2"] = {**blk, "pairing": info, "paired_support": paired, "gap_mae_per_user": {u: v["gap"] for u, v in per_user.items()},
                         "misaligned_ok": bool(np.isfinite(blk["ratio_ci95_users"][0]) and blk["ratio_ci95_users"][0] > rule["misaligned_ci_lower"])}
        else:
            out["E2"] = {"pairing": info, "misaligned_ok": None, "note": "nessuna coppia"}
    if train is not None:
        rr = ridge_accessibility(train, val, alphas=tuple(rule["ridge_alphas"]), per_window=rule["instants_per_window"])
        err, const, n = _window_errors(rr["pred"], val["target"], val["mask"], val["std"])
        out["E3"] = {"alpha": rr["alpha"], "inner_mae_std": rr["inner_mae_std"], "n_train_instants": rr["n_train_instants"],
                     "dynamic": _block(err, const, n, users, dynamic, "ridge dai latenti, finestre dinamiche"),
                     "all": _block(err, const, n, users, np.ones(n_win, dtype=bool), "ridge dai latenti, tutte")}
        out["E3"]["accessible"] = bool(np.isfinite(out["E3"]["dynamic"]["ratio"]) and out["E3"]["dynamic"]["ratio"] < 0.9)
    return out
