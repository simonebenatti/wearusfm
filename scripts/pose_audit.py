"""Audit della catena dei bersagli di posa su registrazioni VERE di emg2pose e ridge da feature grezze dell'EMG alla posa (controllo 1 della
diagnosi di GPT del 27/10/2026, `docs/reviews/r3_dbg_diagnosis_20261027.md`; correzioni R1-R4 della sua review `pose_audit_e198a87_review_20261027.md`).
CPU, 0 GPU-ora; uscite fuori dal repo. Nessuna soglia nuova: controlli di uguaglianza con stato esplicito e numeri descrittivi.

Per ogni registrazione scelta (utenti e registrazioni spalmati, seme fisso), ogni controllo ha stato `verificato` / `fallito` / `non_verificato`:
- sidecar: utente, stem, lato, `source_member`, frequenza e asse del tempo uguali fra il sidecar dell'EMG e quello della posa, e coerenti col CSV;
- EMG grezzo (dal tar) contro EMG processato (int16 / scala) sui campioni delle finestre campionate, con tolleranza della quantizzazione: un EMG
  sostituito o spostato fallisce anche con sidecar plausibili; lettura diretta (`read_window`) contro lettura da blocco (`read_window_block`);
- posa: blocchi e validita' ricalcolati dagli angoli grezzi con la regola M, uguali all'albero della posa;
- oracolo delle patch sulle finestre vere del loader: regola completa dell'ingest reimplementata (zeri dilatati di M, righe non finite non valide e
  non dilatate, blocco valido se tutti i 10 campioni lo sono, media dei blocchi col centro nella patch, statistiche del LATO della finestra);
- supporto: blocchi validi per patch (su 5), quota di patch valide per finestra, movimento della posa su coppie di blocchi ADIACENTI entrambi validi
  (nessun salto attraverso i buchi) e, a parte, su tutte le coppie adiacenti.
La copertura (controlli attesi / verificati / falliti / non verificati) e' nel rapporto: senza letture del grezzo l'audit e' PARZIALE, mai «ok».

Ridge da log-RMS per canale per patch (contesto +-K patch) alla posa standardizzata, per utente: finestre ordinate per tempo assoluto (inizio
della registrazione dal CSV + inizio della finestra); fit sulle prime 60%, separazione 10% (almeno 1 finestra), test sulle ultime 30%; alpha
scelto con uno split interno al fit, per finestre e con separazione, normalizzazione stimata sul solo fit interno; costante = media del fit;
acquisizioni simultanee (stessa sessione, fase e istante, lato diverso) a cavallo dello split contate e dichiarate.
Controllo disallineato come in E2 (`pose_eval.misaligned_pairs`: stesso utente, fase e lato; stessa registrazione a >= 8 s, altrimenti altra
sessione) sulle finestre di test: MAE allineata, disallineata e gap sullo STESSO supporto; senza coppie il controllo e' «non verificabile».
Scansione del ritardo ESPLORATIVA: bersagli spostati di k patch sul supporto comune a tutti i k (stesse righe di feature), MAE assolute e N per
lag; il lag «operativo» si sceglie dentro il fit e si valuta una volta sul test. Nessun verdetto «minimo a zero = allineato».

Uso: python3 scripts/pose_audit.py --csv <csv> --processed-root <emg> --pose-root <posa> [--tar <emg2pose_dataset.tar>] --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wearusfm.data import pose_targets as PT  # noqa: E402
from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.harness import pose_eval as PE  # noqa: E402
from wearusfm.ingest import emg2pose as E2P  # noqa: E402
from wearusfm.ingest import emg2pose_pose as PP  # noqa: E402

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
EPS = 1e-6
VERIFICATO, FALLITO, NON_VERIFICATO = "verificato", "fallito", "non_verificato"
CHECKS = ("sidecar", "emg_grezzo", "lettura_blocco", "posa_ricostruzione", "oracolo")


# --- oracolo e controlli deterministici -------------------------------------------------------------------------------------------------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(np.int8)
    e = np.flatnonzero(np.diff(m))
    return [(int(a), int(b)) for a, b in zip(e[0::2], e[1::2])]


def oracle_patch_targets(angles: np.ndarray, start: int, n_patches: int, patch_samples: int, margin: int, mean, std):
    """Regola completa dell'ingest reimplementata dagli indici grezzi (nessuna funzione di aggregazione dell'ingest): riga non valida se tutta a
    zero (|a| <= ZERO_ATOL), se non finita, o entro `margin` campioni (inclusi) da una riga tutta a zero (i non finiti NON si dilatano); blocco
    di 10 valido se tutti i campioni lo sono; bersaglio della patch = media dei blocchi validi col centro (10 j + 5) in [inizio, fine) della patch,
    standardizzato con (mean, std) del lato della finestra. Ritorna (target (P, 20), maschera (P,), blocchi validi per patch (P,))."""
    n = angles.shape[0]
    bounds = start + np.ceil(np.arange(n_patches + 1) * patch_samples - 1e-9).astype(np.int64)
    lo, hi = max(0, int(bounds[0]) - margin - 10), min(n, int(bounds[-1]) + margin + 10)
    a = np.asarray(angles[lo:hi], dtype=np.float64)
    zero = np.all(np.abs(a) <= PP.ZERO_ATOL, axis=1)
    nonfinite = ~np.all(np.isfinite(a), axis=1)
    bad = zero | nonfinite
    for r0, r1 in _runs(zero):
        bad[max(0, r0 - margin):min(hi - lo, r1 + margin)] = True
    tgt, msk, n_valid = np.zeros((n_patches, PT.N_JOINTS), np.float32), np.zeros(n_patches, bool), np.zeros(n_patches, np.int64)
    mean, std = np.asarray(mean, np.float64), np.asarray(std, np.float64)
    for p in range(n_patches):
        vals = []
        for j in range(max(0, int(bounds[p]) // 10 - 1), min(n // 10, int(bounds[p + 1]) // 10 + 2)):
            c = 10 * j + 5
            if bounds[p] <= c < bounds[p + 1] and 10 * j >= lo and 10 * j + 10 <= hi and not bad[10 * j - lo:10 * j + 10 - lo].any():
                vals.append(a[10 * j - lo:10 * j + 10 - lo].mean(axis=0))
        if vals:
            tgt[p] = (np.mean(vals, axis=0) - mean) / std
            msk[p] = True
            n_valid[p] = len(vals)
    return tgt, msk, n_valid


def movement_adjacent(blocks: np.ndarray, valid: np.ndarray) -> tuple[float | None, float | None, int]:
    """Movimento della posa come |differenza prima| fra blocchi ADIACENTI (indici originali): (su coppie entrambe valide, su tutte le coppie,
    numero di coppie valide). Un buco nella maschera non produce mai una coppia: il salto attraverso il buco non e' movimento osservato."""
    b = np.asarray(blocks, np.float64)
    if b.shape[0] < 2:
        return None, None, 0
    d = np.abs(np.diff(b, axis=0)).mean(axis=1)
    both = np.asarray(valid[:-1], bool) & np.asarray(valid[1:], bool)
    return (float(d[both].mean()) if both.any() else None), float(d.mean()), int(both.sum())


def dynamic_patches(target: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Patch «dinamiche»: valide con la precedente valida e |differenza prima| media sui giunti >= mediana fra tali coppie."""
    p = valid.shape[0]
    both = np.zeros(p, bool)
    both[1:] = valid[1:] & valid[:-1]
    dyn = np.zeros(p, bool)
    if both.sum() >= 2:
        d = np.abs(np.diff(target, axis=0)).mean(axis=1)  # d[p-1] = |t[p] - t[p-1]|
        idx = np.flatnonzero(both)
        thr = np.median(d[idx - 1])
        dyn[idx[d[idx - 1] >= thr]] = True
    return dyn


def log_rms_features(x: np.ndarray, n_patches: int, patch_samples: int, context: int) -> np.ndarray:
    """x (C, T) -> (P, C x (2 K + 1)): log-RMS per canale per patch, con le patch vicine (bordi replicati)."""
    c = x.shape[0]
    xp = x[:, :n_patches * patch_samples].reshape(c, n_patches, patch_samples)
    f = np.log(np.sqrt((xp.astype(np.float64) ** 2).mean(axis=2)) + EPS).T
    if context <= 0:
        return f
    idx = np.arange(n_patches)
    return np.concatenate([f[np.clip(idx + k, 0, n_patches - 1)] for k in range(-context, context + 1)], axis=1)


def emg_checks(view, raw_emg: np.ndarray | None, wins: list, cfg) -> dict:
    """R1: EMG grezzo contro processato (int16 / scala) sui campioni delle finestre, tolleranza = mezzo passo di quantizzazione; lettura diretta
    contro lettura da blocco sulle stesse finestre (uguali al float32)."""
    out = {"emg_grezzo": NON_VERIFICATO, "lettura_blocco": NON_VERIFICATO, "n_finestre": len(wins)}
    if not wins:
        out["motivo"] = "nessuna finestra"
        return out
    scale = float(np.asarray(view.scale).max()) if view.scale is not None else 1.0
    if raw_emg is not None:
        bad, worst = 0, 0.0
        for w in wins:
            proc = view.read(w.trial, w.start, w.stop)  # (C, n) = int16 / scala
            raw = np.asarray(raw_emg[w.start:w.stop], np.float32).T
            if raw.shape != proc.shape:
                bad += 1
                continue
            err = float(np.abs(proc.astype(np.float64) - raw.astype(np.float64)).max())
            worst = max(worst, err)
            if err > 0.5 / scale * (1 + 1e-3) + 1e-6 * float(np.abs(raw).max()):
                bad += 1
        out.update({"emg_grezzo": FALLITO if bad else VERIFICATO, "finestre_emg_fallite": bad, "errore_massimo": worst,
                    "tolleranza": 0.5 / scale})
    else:
        out["motivo"] = "nessun EMG grezzo letto dal tar"
    ok_block, worst_b = True, 0.0
    for w in wins:
        m = int(round(cfg.filter_margin_s * view.fs)) if cfg.filter_band_hz is not None else 0
        a, b = max(w.span[0], w.start - m - 100), min(w.span[1], w.stop + m + 100)
        block = L.Block(row={}, view=view, cand=(w.trial, w.span[0], w.span[1], w.start, 0, np.asarray([0])), ks=np.asarray([0]),
                        data_lo=a, data_hi=b, uses_left=1)
        direct = L.read_window(view, w.trial, w.start, w.stop, w.span, cfg)
        via_block = L.read_window_block(block, w, cfg)
        err = float(np.abs(direct - via_block).max()) if direct.shape == via_block.shape else float("inf")
        worst_b = max(worst_b, err)
        ok_block &= err <= 1e-5 * max(1.0, float(np.abs(direct).max()))
    out.update({"lettura_blocco": VERIFICATO if ok_block else FALLITO, "errore_massimo_blocco": worst_b})
    return out


def sidecar_checks(meta_emg: dict, meta_pose: dict, rec: dict, d_pose: Path) -> dict:
    row = meta_emg.get("csv_row") or {}
    checks = {"utente": meta_pose.get("user") == rec["user"] == row.get("user", rec["user"]),
              "stem": meta_pose.get("stem") == rec["stem"] and d_pose.name == rec["stem"].replace("@", "_") or d_pose.name == rec["stem"],
              "lato": (meta_pose.get("side") or "").lower() == rec["side"].lower() == (row.get("side") or "").lower(),
              "source_member": bool(meta_emg.get("source_member")) and meta_emg.get("source_member") == meta_pose.get("source_member"),
              "fs": float(meta_emg.get("native_fs_hz", 0)) == PT.NATIVE_FS_HZ == float(meta_pose.get("native_fs_hz", 0)),
              "asse_del_tempo": meta_emg.get("time_axis") is not None and meta_emg.get("time_axis") == meta_pose.get("time_axis")}
    return {"dettaglio": checks, "stato": VERIFICATO if all(checks.values()) else FALLITO}


# --- ridge ---------------------------------------------------------------------------------------------------------------------------------

def _ridge(x, y, alpha):
    mx, my = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mx, y - my
    w = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)
    return w, my - mx @ w


def _stack(sel, shift=0, common=0):
    """Righe valide delle finestre `sel` con i bersagli spostati di `shift` patch; con `common` > 0 solo le righe p in [common, P - common) valide
    per OGNI spostamento in [-common, common] (supporto comune alla scansione)."""
    xs, ys, ds = [], [], []
    for w in sel:
        p = w["feat"].shape[0]
        if common:
            rows = np.arange(common, p - common)
            ok = np.ones(rows.shape[0], bool)
            for k in range(-common, common + 1):
                ok &= w["valid"][rows + k]
            rows = rows[ok]
            xs.append(w["feat"][rows])
            ys.append(w["target"][rows + shift])
            ds.append(w["dyn"][rows + shift])
        else:
            fi, ti = (slice(0, p - shift), slice(shift, p)) if shift >= 0 else (slice(-shift, p), slice(0, p + shift))
            v = w["valid"][ti]
            xs.append(w["feat"][fi][v])
            ys.append(w["target"][ti][v])
            ds.append(w["dyn"][ti][v])
    return (np.concatenate(xs) if xs else np.zeros((0, 1))), (np.concatenate(ys) if ys else np.zeros((0, PT.N_JOINTS))), \
        (np.concatenate(ds) if ds else np.zeros(0, bool))


def _split_windows(ws: list, frac_fit=0.6, frac_gap=0.1):
    n = len(ws)
    n_fit, n_gap = max(2, int(round(frac_fit * n))), max(1, int(round(frac_gap * n)))
    return ws[:n_fit], ws[n_fit:n_fit + n_gap], ws[n_fit + n_gap:]


def _choose_alpha(fit: list, shift=0, common=0) -> tuple[float, dict]:
    """Split interno al fit per finestre (prime 60%, separazione, resto), normalizzazione stimata sul solo fit interno."""
    inner_fit, _, inner_hold = _split_windows(fit)
    if len(inner_fit) < 2 or len(inner_hold) < 1:
        return ALPHAS[1], {"nota": "fit interno troppo piccolo: alpha predefinito", "alpha": ALPHAS[1]}
    xf, yf, _ = _stack(inner_fit, shift, common)
    xh, yh, _ = _stack(inner_hold, shift, common)
    if xf.shape[0] < 20 or xh.shape[0] == 0:
        return ALPHAS[1], {"nota": "troppo poche patch nel fit interno: alpha predefinito", "alpha": ALPHAS[1]}
    mu, sd = xf.mean(axis=0), xf.std(axis=0) + EPS
    errs = {}
    for alpha in ALPHAS:
        w, b = _ridge((xf - mu) / sd, yf, alpha)
        errs[str(alpha)] = float(np.abs(((xh - mu) / sd) @ w + b - yh).mean())
    best = float(min(errs, key=errs.get))
    return best, {"alpha": best, "mae_interna_per_alpha": errs, "n_fit_interno": len(inner_fit), "n_holdout_interno": len(inner_hold)}


def ridge_per_user(windows: list[dict], *, lags: int, seed: int, window_s: float, min_gap_s: float = 8.0) -> dict:
    out = {}
    for user in sorted({w["user"] for w in windows}):
        ws = sorted((w for w in windows if w["user"] == user), key=lambda w: (w["t_abs"], w["side"]))
        n = len(ws)
        fit, gap, test = _split_windows(ws)
        res = {"n_windows": n, "n_fit": len(fit), "n_gap": len(gap), "n_test": len(test),
               "cronologia": "tempo assoluto dal CSV (inizio della registrazione) + inizio della finestra" if all(w["t_abs_from_csv"] for w in ws)
               else "ordine lessicografico (sessione, stem, inizio): il CSV non da' l'inizio assoluto per tutte le registrazioni"}
        # acquisizioni simultanee a cavallo dello split (stessa sessione e fase, stesso istante, lato diverso)
        sim = 0
        for a in fit:
            for b in test:
                if a["session"] == b["session"] and a["stage"] == b["stage"] and a["side"] != b["side"] and abs(a["t_abs"] - b["t_abs"]) < window_s:
                    sim += 1
        res["simultanee_fit_test"] = sim
        if len(fit) < 2 or len(test) < 1:
            res["nota"] = "troppo poche finestre per fit / separazione / test"
            out[user] = res
            continue
        alpha, alpha_info = _choose_alpha(fit)
        res["alpha"] = alpha_info
        xf, yf, _ = _stack(fit)
        xt, yt, dt = _stack(test)
        if xf.shape[0] < 20 or xt.shape[0] == 0:
            res["nota"] = "troppo poche patch valide"
            out[user] = res
            continue
        mu, sd = xf.mean(axis=0), xf.std(axis=0) + EPS
        w, b = _ridge((xf - mu) / sd, yf, alpha)
        const = yf.mean(axis=0)
        pred = ((xt - mu) / sd) @ w + b
        mae, mae_c = float(np.abs(pred - yt).mean()), float(np.abs(yt - const).mean())
        res.update({"n_fit_patches": int(xf.shape[0]), "n_test_patches": int(xt.shape[0]), "mae": mae, "mae_const": mae_c,
                    "ratio_all": mae / mae_c if mae_c > 0 else None, "n_dynamic": int(dt.sum()),
                    "mae_dynamic": float(np.abs(pred[dt] - yt[dt]).mean()) if dt.any() else None,
                    "mae_const_dynamic": float(np.abs(yt[dt] - const).mean()) if dt.any() else None})
        res["ratio_dynamic"] = (res["mae_dynamic"] / res["mae_const_dynamic"]) if dt.any() and res["mae_const_dynamic"] else None
        # disallineato come in E2, sulle finestre di test: stesso utente, fase e lato; stessa registrazione a >= 8 s, altrimenti altra sessione
        meta = [{"user": t["user"], "session": t["session"], "stage": t["stage"], "side": t["side"], "recording": t["recording"],
                 "start": t["start"], "fs": t["fs"]} for t in test]
        p_min = min(t["feat"].shape[0] for t in test)
        tgt = np.stack([t["target"][:p_min] for t in test])
        msk = np.stack([np.repeat(t["valid"][:p_min, None], PT.N_JOINTS, axis=1) for t in test])
        partner, info = PE.misaligned_pairs(meta, tgt, msk, min_gap_s=min_gap_s)
        pa, pm, pc = [], [], []
        for i, j in enumerate(partner):
            if j < 0:
                continue
            both = test[i]["valid"][:p_min] & test[j]["valid"][:p_min]
            if both.any():
                pr = ((test[i]["feat"][:p_min][both] - mu) / sd) @ w + b
                pa.append(np.abs(pr - test[i]["target"][:p_min][both]).mean(axis=1))
                pm.append(np.abs(pr - test[j]["target"][:p_min][both]).mean(axis=1))
                pc.append(np.abs(test[j]["target"][:p_min][both] - const).mean(axis=1))
        if pa:
            pa, pm, pc = np.concatenate(pa), np.concatenate(pm), np.concatenate(pc)
            res["disallineato"] = {"stato": VERIFICATO, "n_coppie": int((partner >= 0).sum()), "n_patch": int(pa.shape[0]),
                                   "mae_allineata": float(pa.mean()), "mae_disallineata": float(pm.mean()), "gap": float(pm.mean() - pa.mean()),
                                   "mae_const_sui_bersagli_disallineati": float(pc.mean()), "appaiamento": info}
        else:
            res["disallineato"] = {"stato": "non_verificabile", "motivo": "nessuna coppia ammissibile fra le finestre di test", "appaiamento": info}
        # scansione del ritardo, esplorativa: supporto comune a tutti i k, stesse righe di feature; MAE assolute e N; lag operativo dal fit
        if lags > 0:
            xfc, _, _ = _stack(fit, 0, lags)
            xtc, _, _ = _stack(test, 0, lags)
            if xfc.shape[0] >= 20 and xtc.shape[0] > 0:
                muc, sdc = xfc.mean(axis=0), xfc.std(axis=0) + EPS
                scan = {}
                for k in range(-lags, lags + 1):
                    _, yk, _ = _stack(fit, k, lags)
                    _, ytk, _ = _stack(test, k, lags)
                    wk, bk = _ridge((xfc - muc) / sdc, yk, alpha)
                    pk = ((xtc - muc) / sdc) @ wk + bk
                    scan[str(k)] = {"mae": float(np.abs(pk - ytk).mean()), "mae_const": float(np.abs(ytk - yk.mean(axis=0)).mean()), "n": int(ytk.shape[0])}
                # lag operativo: scelto dentro il fit (split interno), valutato una volta sul test
                inner_fit, _, inner_hold = _split_windows(fit)
                best_k, best_e = 0, np.inf
                if len(inner_fit) >= 2 and len(inner_hold) >= 1:
                    xi, _, _ = _stack(inner_fit, 0, lags)
                    xh, _, _ = _stack(inner_hold, 0, lags)
                    if xi.shape[0] >= 20 and xh.shape[0] > 0:
                        mi, si = xi.mean(axis=0), xi.std(axis=0) + EPS
                        for k in range(-lags, lags + 1):
                            _, yi, _ = _stack(inner_fit, k, lags)
                            _, yh, _ = _stack(inner_hold, k, lags)
                            wk, bk = _ridge((xi - mi) / si, yi, alpha)
                            e = float(np.abs(((xh - mi) / si) @ wk + bk - yh).mean())
                            if e < best_e:
                                best_k, best_e = k, e
                res["scansione_ritardo"] = {"esplorativa": True, "lags": lags, "n_righe_fit": int(xfc.shape[0]), "n_righe_test": int(xtc.shape[0]),
                                            "per_lag": scan, "lag_operativo_dal_fit": best_k, "mae_test_al_lag_operativo": scan[str(best_k)]["mae"],
                                            "mae_test_a_0": scan["0"]["mae"],
                                            "nota": "nessun verdetto di allineamento: con contesto > 0 piu' lag sono rappresentabili dalle stesse feature"}
        out[user] = res
    return out


# --- main ------------------------------------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--processed-root", type=Path, required=True)
    ap.add_argument("--pose-root", type=Path, required=True)
    ap.add_argument("--tar", type=Path, default=None, help="emg2pose_dataset.tar (non compresso): EMG e posa grezzi; senza, i controlli sul grezzo restano non verificati")
    ap.add_argument("--tar-index-budget-s", type=float, default=1200.0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-users", type=int, default=10)
    ap.add_argument("--n-recordings", type=int, default=20)
    ap.add_argument("--n-windows", type=int, default=20)
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--margin", type=int, default=280, help="M firmato il 27/10/2026")
    ap.add_argument("--context", type=int, default=2)
    ap.add_argument("--lags", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    from wearusfm.training import run as R

    cfg = R.sanity_config().loader
    patch_samples = int(round(cfg.patch_ms / 1000.0 * PT.NATIVE_FS_HZ))
    recs = PE.recordings_for(args.csv, args.processed_root, args.split, n_users=args.n_users, n_recordings=args.n_recordings, seed=args.seed)
    store = PT.PoseStore(args.pose_root)
    print(f"{len(recs)} registrazioni dello split {args.split!r}, {len({r['user'] for r in recs})} utenti; patch {patch_samples} campioni; M = {args.margin}", flush=True)
    members, tar_note, tf = {}, None, None
    if args.tar is not None:
        wanted = {}
        for r in recs:
            mp = PT.pose_dir(args.pose_root, Path(r["path"])) / PT.META_FILE
            if mp.exists():
                wanted[json.loads(mp.read_text()).get("source_member")] = r["stem"]
        tf = tarfile.open(args.tar, "r:")
        t1 = time.time()
        while wanted and time.time() - t1 < args.tar_index_budget_s:
            info = tf.next()
            if info is None:
                break
            if info.name in wanted:
                members[wanted.pop(info.name)] = info
        tar_note = {"found": len(members), "missing": len(wanted), "index_seconds": time.time() - t1}
        print(f"tar: {tar_note}", flush=True)
    per_rec, windows = [], []
    support_hist = np.zeros(6, dtype=np.int64)
    for ri, r in enumerate(recs):
        view = L.SessionView.open(Path(r["path"]), cfg.split_at_gaps)
        d_pose = PT.pose_dir(args.pose_root, Path(r["path"]))
        meta_pose = json.loads((d_pose / PT.META_FILE).read_text()) if (d_pose / PT.META_FILE).exists() else {}
        meta_emg = json.loads((Path(r["path"]) / "metadata.json").read_text())
        blocks, valid = store.entry(view)
        mean, std = store.norm(view)
        n_samples = int(view.arr.shape[0])
        rec_out = {"user": r["user"], "session": r["session"], "stage": r["stage"], "side": r["side"], "moving_hand": r["moving_hand"], "stem": r["stem"],
                   "n_samples": n_samples, "n_blocks": int(blocks.shape[0]), "frac_valid_blocks": float(np.mean(valid)),
                   "unprompted": PT.PoseStore.unprompted(view), "stati": {k: NON_VERIFICATO for k in CHECKS}}
        sc = sidecar_checks(meta_emg, meta_pose, r, d_pose)
        sc["dettaglio"]["n_campioni"] = meta_pose.get("n_samples") == n_samples
        sc["dettaglio"]["n_blocchi"] = blocks.shape[0] == n_samples // PT.BLOCK
        sc["dettaglio"]["margine"] = meta_pose.get("zero_margin_samples") == args.margin
        sc["stato"] = VERIFICATO if all(sc["dettaglio"].values()) else FALLITO
        rec_out["sidecar"] = sc["dettaglio"]
        rec_out["stati"]["sidecar"] = sc["stato"]
        angles = raw_emg = None
        if tf is not None and r["stem"] in members:
            angles = PP.read_pose(tf.extractfile(members[r["stem"]]))["angles"]
            raw_emg = E2P.read_recording(tf.extractfile(members[r["stem"]]))["emg"]
            bad = PP.invalid_rows(angles, args.margin)
            rb, rv = PP.block_reduce(angles, ~bad)
            ok_rec = rb.shape == blocks.shape and np.array_equal(rv, valid) and np.allclose(np.asarray(rb, np.float32), np.asarray(blocks, np.float32), atol=2e-3)
            rec_out["posa_ricostruzione"] = {"n_rows": int(angles.shape[0]), "frac_zero_rows": float(PP.zero_rows(angles).mean()),
                                             "frac_nonfinite_rows": float(PP.nonfinite_rows(angles).mean()), "frac_invalid_rows": float(bad.mean())}
            rec_out["stati"]["posa_ricostruzione"] = VERIFICATO if ok_rec else FALLITO
        rng = np.random.default_rng([args.seed, ri])
        try:
            scale = L.estimate_session_scale(view, cfg, rng)
        except ValueError as e:
            rec_out["nota"] = f"scala non calcolabile: {e}"
            per_rec.append(rec_out)
            continue
        wins = PE.fixed_windows(view, cfg, args.n_windows, args.window_s, args.seed)
        rec_out["n_windows"] = len(wins)
        ec = emg_checks(view, raw_emg, wins, cfg)
        rec_out["emg"] = ec
        rec_out["stati"]["emg_grezzo"], rec_out["stati"]["lettura_blocco"] = ec["emg_grezzo"], ec["lettura_blocco"]
        oracle_ok, n_or = True, 0
        mov_v, mov_a, fracs = [], [], []
        for w in wins:
            got = store.window({"dataset": "emg2pose"}, view, w, cfg.patch_ms)
            if got is None:
                continue
            t, m, sd = got
            v = m[:, 0]
            if angles is not None:
                o_t, o_m, n_v = oracle_patch_targets(angles, int(w.start), int(w.n_patches), patch_samples, args.margin, mean, std)
                oracle_ok &= bool(np.array_equal(v, o_m) and (not o_m.any() or np.allclose(t[o_m], o_t[o_m], atol=6e-3)))
                n_or += 1
                support_hist += np.bincount(np.clip(n_v, 0, 5), minlength=6)
            x = L.read_window(view, w.trial, w.start, w.stop, w.span, cfg).astype(np.float32) / np.float32(scale)
            feat = log_rms_features(x, int(w.n_patches), patch_samples, args.context)
            j0, j1 = int(w.start) // PT.BLOCK, int(w.stop) // PT.BLOCK
            mv, ma, _ = movement_adjacent(blocks[j0:j1], valid[j0:j1])
            if mv is not None:
                mov_v.append(mv)
            if ma is not None:
                mov_a.append(ma)
            fracs.append(float(v.mean()))
            t_abs = (r.get("start_s") if r.get("start_s") is not None else 0.0) + int(w.start) / float(view.fs)
            windows.append({"user": r["user"], "session": r["session"], "stage": r["stage"], "side": r["side"], "recording": ri, "start": int(w.start),
                            "fs": float(view.fs), "t_abs": t_abs, "t_abs_from_csv": r.get("start_s") is not None, "feat": feat, "target": t, "valid": v,
                            "dyn": dynamic_patches(t, v)})
        rec_out["stati"]["oracolo"] = (VERIFICATO if oracle_ok else FALLITO) if n_or else NON_VERIFICATO
        rec_out.update({"oracolo_finestre": n_or, "frac_valid_patches_mean": float(np.mean(fracs)) if fracs else None,
                        "movimento_adiacenti_validi": float(np.mean(mov_v)) if mov_v else None,
                        "movimento_adiacenti_tutti": float(np.mean(mov_a)) if mov_a else None})
        per_rec.append(rec_out)
        print(f"  [{ri + 1}/{len(recs)}] {r['stem']}: " + ", ".join(f"{k} {v}" for k, v in rec_out["stati"].items()) + f"; {len(wins)} finestre", flush=True)
    copertura = {k: {s: sum(1 for rr in per_rec if rr["stati"].get(k) == s) for s in (VERIFICATO, FALLITO, NON_VERIFICATO)} for k in CHECKS}
    any_failed = any(c[FALLITO] for c in copertura.values())
    all_verified = all(c[VERIFICATO] == len(per_rec) for c in copertura.values()) and per_rec
    stato = FALLITO if any_failed else (VERIFICATO if all_verified else "parziale")
    ridge = ridge_per_user(windows, lags=args.lags, seed=args.seed, window_s=args.window_s) if windows else {}
    ratios = [v["ratio_all"] for v in ridge.values() if v.get("ratio_all") is not None]
    out = {"params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "patch_samples": patch_samples, "n_recordings": len(recs),
           "n_users": len({r["user"] for r in recs}), "tar": tar_note, "stato_catena": stato, "copertura": copertura, "per_recording": per_rec,
           "sintesi": {"stato_catena": stato, "registrazioni": len(per_rec), "copertura": copertura,
                       "blocchi_validi_per_patch_istogramma_0_5": support_hist.tolist(),
                       "ridge_ratio_all_per_utente": {u: v.get("ratio_all") for u, v in ridge.items()},
                       "ridge_ratio_all_mediana_utenti": float(np.median(ratios)) if ratios else None,
                       "ridge_ratio_dynamic_mediana_utenti": float(np.median([v["ratio_dynamic"] for v in ridge.values() if v.get("ratio_dynamic") is not None])) if any(v.get("ratio_dynamic") is not None for v in ridge.values()) else None,
                       "disallineato": {u: v.get("disallineato", {}).get("stato") for u, v in ridge.items()},
                       "lag_operativo_per_utente": {u: v.get("scansione_ritardo", {}).get("lag_operativo_dal_fit") for u, v in ridge.items()},
                       "simultanee_fit_test": {u: v.get("simultanee_fit_test") for u, v in ridge.items()}},
           "ridge": ridge, "n_windows_total": len(windows), "elapsed_s": time.time() - t0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["sintesi"], indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
