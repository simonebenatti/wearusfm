"""Audit della catena dei bersagli di posa su registrazioni VERE di emg2pose e ridge da feature grezze dell'EMG alla posa (controllo 1 della
diagnosi di GPT del 27/10/2026, `docs/reviews/r3_dbg_diagnosis_20261027.md`, dopo il collaudo dbg di R3 62429050 non superato per `posa_discesa`).
CPU, 0 GPU-ora; uscite fuori dal repo. Nessuna soglia nuova: numeri descrittivi e controlli di uguaglianza.

Per ogni registrazione scelta (utenti e registrazioni spalmati, seme fisso):
- identita': `pose_meta.json` contro la riga del CSV e il sidecar dell'EMG (utente, stem, lato, campioni, blocchi, margine);
- ricostruzione dal grezzo (tar): blocchi e validita' ricalcolati dagli angoli con la regola M, confrontati con l'albero della posa (uguaglianza);
- oracolo delle patch: sulle finestre fisse del loader (griglia a 200 ms, patch da 25 ms) il bersaglio di `PoseStore.window` contro la media dei
  blocchi validi col centro nella patch, calcolata dagli indici grezzi (uguaglianza entro il float16);
- supporto: blocchi validi per patch (su 5), quota di patch valide per finestra, movimento (|differenza prima| della posa) prima e dopo la maschera.
Poi, per utente, ridge da log-RMS per canale per patch (con contesto di +-K patch) alla posa standardizzata: fit sulle prime finestre, un
intervallo di separazione, test sulle ultime (ordine temporale, mai patch mescolate); alpha scelto dentro il fit; costante = media del fit;
rapporto MAE / MAE_costante su tutte le patch valide e sulle dinamiche; disallineato sullo stesso supporto (bersagli di un'altra finestra di test);
scansione del ritardo (bersagli spostati di k patch): il k migliore DEVE essere 0 se EMG e posa sono allineati.

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
from wearusfm.ingest import emg2pose_pose as PP  # noqa: E402

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
EPS = 1e-6


def oracle_patch_targets(angles: np.ndarray, start: int, n_patches: int, patch_samples: int, margin: int, mean, std):
    """Dagli indici grezzi, senza le funzioni dell'ingest: non valido = riga tutta a zero o entro `margin` campioni (inclusi) da una; blocco di
    10 valido se tutti i campioni lo sono; bersaglio = media dei blocchi validi col centro (10 j + 5) in [inizio, fine) della patch."""
    n = angles.shape[0]
    zero = np.all(np.abs(angles) <= PP.ZERO_ATOL, axis=1)
    bad = zero.copy()
    for a, b in PP.runs_of(zero):
        bad[max(0, a - margin):min(n, b + margin)] = True
    bounds = start + np.ceil(np.arange(n_patches + 1) * patch_samples - 1e-9).astype(np.int64)
    tgt, msk, n_valid = np.zeros((n_patches, PT.N_JOINTS), np.float32), np.zeros(n_patches, bool), np.zeros(n_patches, np.int64)
    for p in range(n_patches):
        vals = []
        for j in range(max(0, bounds[p] // 10 - 1), min(n // 10, bounds[p + 1] // 10 + 2)):
            c = 10 * j + 5
            if bounds[p] <= c < bounds[p + 1] and not bad[10 * j:10 * j + 10].any():
                vals.append(angles[10 * j:10 * j + 10].astype(np.float64).mean(axis=0))
        if vals:
            tgt[p] = (np.mean(vals, axis=0) - mean) / std
            msk[p] = True
            n_valid[p] = len(vals)
    return tgt, msk, n_valid


def log_rms_features(x: np.ndarray, n_patches: int, patch_samples: int, context: int) -> np.ndarray:
    """x (C, T) -> (P, C x (2 K + 1)): log-RMS per canale per patch, con le patch vicine (bordi replicati)."""
    c = x.shape[0]
    xp = x[:, :n_patches * patch_samples].reshape(c, n_patches, patch_samples)
    f = np.log(np.sqrt((xp.astype(np.float64) ** 2).mean(axis=2)) + EPS).T  # (P, C)
    if context <= 0:
        return f
    parts = []
    for k in range(-context, context + 1):
        idx = np.clip(np.arange(n_patches) + k, 0, n_patches - 1)
        parts.append(f[idx])
    return np.concatenate(parts, axis=1)


def _ridge(x, y, alpha):
    mx, my = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mx, y - my
    w = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)
    return w, my - mx @ w


def _mae_ratio(pred, y, const):
    """MAE del modello / MAE della costante, sugli stessi elementi."""
    den = float(np.abs(y - const).mean())
    return float(np.abs(pred - y).mean()) / den if den > 0 else float("nan")


def ridge_per_user(windows: list[dict], *, lags: int, seed: int) -> dict:
    """`windows`: dizionari con user, order (chiave temporale), feat (P, F), target (P, 20), valid (P,), dyn (P,) (patch dinamiche). Per utente:
    fit sulle prime 60% finestre, separazione 10% (almeno 1), test sulle ultime 30%; alpha dall'interno del fit (prima meta' -> seconda meta');
    rapporti sul test, disallineato sullo stesso supporto, scansione del ritardo."""
    out = {}
    rng = np.random.default_rng(seed)
    for user in sorted({w["user"] for w in windows}):
        ws = sorted((w for w in windows if w["user"] == user), key=lambda w: w["order"])
        n = len(ws)
        n_fit, n_gap = max(2, int(round(0.6 * n))), max(1, int(round(0.1 * n)))
        fit, test = ws[:n_fit], ws[n_fit + n_gap:]
        if len(fit) < 2 or len(test) < 1:
            out[user] = {"n_windows": n, "note": "troppo poche finestre per fit / separazione / test"}
            continue

        def stack(sel, shift=0):
            xs, ys, ds = [], [], []
            for w in sel:
                p = w["feat"].shape[0]
                if shift >= 0:
                    fi, ti = slice(0, p - shift), slice(shift, p)
                else:
                    fi, ti = slice(-shift, p), slice(0, p + shift)
                v = w["valid"][ti]
                xs.append(w["feat"][fi][v])
                ys.append(w["target"][ti][v])
                ds.append(w["dyn"][ti][v])
            return np.concatenate(xs), np.concatenate(ys), np.concatenate(ds)

        xf, yf, _ = stack(fit)
        if xf.shape[0] < 20:
            out[user] = {"n_windows": n, "note": "troppo poche patch valide nel fit"}
            continue
        mu, sd = xf.mean(axis=0), xf.std(axis=0) + EPS
        half = xf.shape[0] // 2
        best, best_err = ALPHAS[0], np.inf
        for alpha in ALPHAS:
            w, b = _ridge((xf[:half] - mu) / sd, yf[:half], alpha)
            err = float(np.abs(((xf[half:] - mu) / sd) @ w + b - yf[half:]).mean())
            if err < best_err:
                best, best_err = alpha, err
        w, b = _ridge((xf - mu) / sd, yf, best)
        const = yf.mean(axis=0)
        xt, yt, dt = stack(test)
        if xt.shape[0] == 0:
            out[user] = {"n_windows": n, "note": "nessuna patch valida nel test"}
            continue
        pred = ((xt - mu) / sd) @ w + b
        res = {"n_windows": n, "n_fit": len(fit), "n_gap": n_gap, "n_test": len(test), "alpha": best, "n_fit_patches": int(xf.shape[0]),
               "n_test_patches": int(xt.shape[0]), "ratio_all": _mae_ratio(pred, yt, const),
               "ratio_dynamic": _mae_ratio(pred[dt], yt[dt], const) if dt.any() else float("nan"), "n_dynamic": int(dt.sum()),
               "mae_const": float(np.abs(yt - const).mean())}
        # disallineato sullo stesso supporto: predizioni di una finestra di test contro i bersagli di un'altra, sulle patch valide in entrambe
        if len(test) >= 2:
            perm = rng.permutation(len(test))
            if np.any(perm == np.arange(len(test))):
                perm = np.roll(np.arange(len(test)), 1)
            pm, ym, cm = [], [], []
            for i, j in enumerate(perm):
                a, bw = test[i], test[j]
                p = min(a["feat"].shape[0], bw["feat"].shape[0])
                both = a["valid"][:p] & bw["valid"][:p]
                if both.any():
                    pm.append(((a["feat"][:p][both] - mu) / sd) @ w + b)
                    ym.append(bw["target"][:p][both])
                    cm.append(a["target"][:p][both])
            if pm:
                pm, ym, cm = np.concatenate(pm), np.concatenate(ym), np.concatenate(cm)
                res["misaligned"] = {"ratio_vs_const": _mae_ratio(pm, ym, const), "ratio_aligned_same_support": _mae_ratio(pm, cm, const),
                                     "n_patches": int(pm.shape[0])}
        # scansione del ritardo: bersagli spostati di k patch (stesso alpha, stesso fit / test)
        scan = {}
        for k in range(-lags, lags + 1):
            xk, yk, _ = stack(fit, k)
            xtk, ytk, _ = stack(test, k)
            if xk.shape[0] < 20 or xtk.shape[0] == 0:
                continue
            wk, bk = _ridge((xk - mu) / sd, yk, best)
            scan[str(k)] = _mae_ratio(((xtk - mu) / sd) @ wk + bk, ytk, yk.mean(axis=0))
        if scan:
            best_k = min(scan, key=scan.get)
            res["lag_scan"] = {"ratio_by_lag": scan, "best_lag": int(best_k), "ratio_at_0": scan.get("0"), "ratio_at_best": scan[best_k]}
        out[user] = res
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--processed-root", type=Path, required=True)
    ap.add_argument("--pose-root", type=Path, required=True)
    ap.add_argument("--tar", type=Path, default=None, help="emg2pose_dataset.tar (non compresso): ricostruzione dal grezzo; senza, saltata e detto")
    ap.add_argument("--tar-index-budget-s", type=float, default=1200.0, help="tempo massimo per trovare i membri nel tar (indice sequenziale)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-users", type=int, default=10)
    ap.add_argument("--n-recordings", type=int, default=20)
    ap.add_argument("--n-windows", type=int, default=20)
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--margin", type=int, default=280, help="M firmato il 27/10/2026")
    ap.add_argument("--context", type=int, default=2, help="patch vicine nelle feature (+-K)")
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
    mean, std = store.mean_all, store.std_all
    print(f"{len(recs)} registrazioni dello split {args.split!r}, {len({r['user'] for r in recs})} utenti; patch {patch_samples} campioni; M = {args.margin}", flush=True)
    members = {}
    tar_note = None
    tf = None
    if args.tar is not None:
        wanted = {}
        for r in recs:
            meta_path = PT.pose_dir(args.pose_root, Path(r["path"])) / PT.META_FILE
            if meta_path.exists():
                wanted[json.loads(meta_path.read_text()).get("source_member")] = r["stem"]
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
        d = PT.pose_dir(args.pose_root, Path(r["path"]))
        meta = json.loads((d / PT.META_FILE).read_text()) if (d / PT.META_FILE).exists() else {}
        blocks, valid = store.entry(view)
        n_samples = int(view.arr.shape[0])
        side_emg = ((view.meta.get("csv_row") or {}).get("side") or "").lower()
        ident = {"user": meta.get("user") == r["user"], "stem": meta.get("stem") == r["stem"], "side_csv": (meta.get("side") or "").lower() == r["side"].lower(),
                 "side_emg": side_emg == r["side"].lower(), "n_samples": meta.get("n_samples") == n_samples, "n_blocks": blocks.shape[0] == n_samples // PT.BLOCK,
                 "margin": meta.get("zero_margin_samples") == args.margin}
        rec_out = {"user": r["user"], "session": r["session"], "stage": r["stage"], "side": r["side"], "moving_hand": r["moving_hand"], "stem": r["stem"],
                   "n_samples": n_samples, "n_blocks": int(blocks.shape[0]), "frac_valid_blocks": float(np.mean(valid)), "identita": ident,
                   "identita_ok": all(ident.values()), "unprompted": PT.PoseStore.unprompted(view)}
        angles = None
        if tf is not None and r["stem"] in members:
            angles = PP.read_pose(tf.extractfile(members[r["stem"]]))["angles"]
            bad = PP.invalid_rows(angles, args.margin)
            rb, rv = PP.block_reduce(angles, ~bad)
            rec_out["ricostruzione"] = {"n_rows": int(angles.shape[0]), "valid_equal": bool(np.array_equal(rv, valid)),
                                        "blocks_close": bool(rb.shape == blocks.shape and np.allclose(np.asarray(rb, np.float32), np.asarray(blocks, np.float32), atol=2e-3)),
                                        "frac_zero_rows": float(PP.zero_rows(angles).mean()), "frac_invalid_rows": float(bad.mean())}
            rec_out["ricostruzione_ok"] = rec_out["ricostruzione"]["valid_equal"] and rec_out["ricostruzione"]["blocks_close"]
        rng = np.random.default_rng([args.seed, ri])
        try:
            scale = L.estimate_session_scale(view, cfg, rng)
        except ValueError as e:
            rec_out["note"] = f"scala non calcolabile: {e}"
            per_rec.append(rec_out)
            continue
        wins = PE.fixed_windows(view, cfg, args.n_windows, args.window_s, args.seed)
        oracle_ok, n_or = True, 0
        mov_all, mov_valid, fracs = [], [], []
        for w in wins:
            got = store.window({"dataset": "emg2pose"}, view, w, cfg.patch_ms)
            if got is None:
                continue
            t, m, sd = got
            if angles is not None:
                o_t, o_m, n_v = oracle_patch_targets(angles, int(w.start), int(w.n_patches), patch_samples, args.margin, mean, std)
                ok = np.array_equal(m[:, 0], o_m) and (not o_m.any() or np.allclose(t[o_m], o_t[o_m], atol=6e-3))
                oracle_ok &= bool(ok)
                n_or += 1
                support_hist += np.bincount(np.clip(n_v, 0, 5), minlength=6)
            x = L.read_window(view, w.trial, w.start, w.stop, w.span, cfg).astype(np.float32) / np.float32(scale)
            if x.ndim == 2 and x.shape[0] == w.stop - w.start and x.shape[1] != w.stop - w.start:
                x = x.T
            feat = log_rms_features(x, int(w.n_patches), patch_samples, args.context)
            v = m[:, 0]
            dyn = np.zeros(int(w.n_patches), dtype=bool)
            if v.sum() >= 2:
                dtv = np.abs(np.diff(t[v], axis=0)).mean(axis=1)
                dyn[np.flatnonzero(v)[1:][dtv >= np.median(dtv)]] = True
            # movimento della posa (|differenza prima| fra blocchi) prima e dopo la maschera, sui blocchi della finestra
            j0, j1 = int(w.start) // PT.BLOCK, int(w.stop) // PT.BLOCK
            bw = np.asarray(blocks[j0:j1], np.float32)
            vw = valid[j0:j1]
            if bw.shape[0] > 1:
                mov_all.append(float(np.abs(np.diff(bw, axis=0)).mean()))
                if vw.sum() > 1:
                    mov_valid.append(float(np.abs(np.diff(bw[vw], axis=0)).mean()))
            fracs.append(float(v.mean()))
            windows.append({"user": r["user"], "order": (r["session"], r["stem"], int(w.start)), "feat": feat, "target": t, "valid": v, "dyn": dyn})
        rec_out.update({"n_windows": len(wins), "oracolo_finestre": n_or, "oracolo_ok": bool(oracle_ok) if n_or else None,
                        "frac_valid_patches_mean": float(np.mean(fracs)) if fracs else None,
                        "movimento_blocchi_tutti": float(np.mean(mov_all)) if mov_all else None, "movimento_blocchi_validi": float(np.mean(mov_valid)) if mov_valid else None})
        per_rec.append(rec_out)
        print(f"  [{ri + 1}/{len(recs)}] {r['stem']}: identita' {rec_out['identita_ok']}, ricostruzione {rec_out.get('ricostruzione_ok')}, oracolo {rec_out['oracolo_ok']}, "
              f"{len(wins)} finestre, patch valide {rec_out['frac_valid_patches_mean']}", flush=True)
    ridge = ridge_per_user(windows, lags=args.lags, seed=args.seed) if windows else {}
    ratios = [v["ratio_all"] for v in ridge.values() if "ratio_all" in v]
    dyn_ratios = [v["ratio_dynamic"] for v in ridge.values() if "ratio_dynamic" in v]
    mis = [v["misaligned"]["ratio_vs_const"] for v in ridge.values() if "misaligned" in v]
    lags_best = [v["lag_scan"]["best_lag"] for v in ridge.values() if "lag_scan" in v]
    out = {"params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "patch_samples": patch_samples, "n_recordings": len(recs),
           "n_users": len({r["user"] for r in recs}), "tar": tar_note, "per_recording": per_rec,
           "sintesi": {"identita_ok_tutte": all(r["identita_ok"] for r in per_rec), "ricostruzione_ok_tutte": all(r.get("ricostruzione_ok", True) for r in per_rec),
                       "n_ricostruite": sum(1 for r in per_rec if "ricostruzione" in r), "oracolo_ok_tutte": all(r.get("oracolo_ok") in (True, None) for r in per_rec),
                       "n_finestre_oracolo": int(sum(r.get("oracolo_finestre", 0) for r in per_rec)),
                       "blocchi_validi_per_patch_istogramma_0_5": support_hist.tolist(),
                       "ridge_ratio_all_mediana_utenti": float(np.median(ratios)) if ratios else None, "ridge_ratio_all_per_utente": {u: v.get("ratio_all") for u, v in ridge.items()},
                       "ridge_ratio_dynamic_mediana_utenti": float(np.nanmedian(dyn_ratios)) if dyn_ratios else None,
                       "misaligned_ratio_vs_const_mediana": float(np.median(mis)) if mis else None,
                       "lag_migliore_per_utente": lags_best, "lag_zero_per_tutti": bool(lags_best) and all(k == 0 for k in lags_best)},
           "ridge": ridge, "n_windows_total": len(windows), "elapsed_s": time.time() - t0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["sintesi"], indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
