#!/usr/bin/env python3
"""Confronto dei target candidati per le ancore in frequenza (protocollo e soglie congelati in `docs/decisioni.md` il 04/10/2026, «Ancore in
frequenza: confronto dei target candidati»). Due fasi:

- `extract` (GPU): sessioni di pretraining di emg2qwerty di 40 utenti (30 di addestramento e 10 di test per utente, seme 0), finestre da 4 s del
  dataloader (filtro, scala, maschere D10, codici RVQ); per (canale, blocco da 200 ms): uscite del decoder del teacher sulle patch visibili (V),
  uscite del decoder dello studente sulle patch nascoste (H, solo blocchi nascosti per intero su quel canale), le stesse di un modello
  all'inizializzazione (R), e i target candidati (16 bande, momenti, forma attuale, codice RVQ e il suo vettore, feature continue di NeuroRVQ ramo 0).
- `probe` (CPU): sonde lineari e verdetto secondo la regola congelata.

  python3 scripts/anchor_candidates.py extract --manifest M --root R1 --root R2 --scales S --rvq-codes C --checkpoint CKPT \\
      --repo-dir $SCRATCH/external/NeuroRVQ_926e770 --tokenizer-checkpoint $WORK/models/NeuroRVQ_EMG_tokenizer_v1.pt --out units.npz
  python3 scripts/anchor_candidates.py probe --units units.npz --out report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.data import rvq_codes as RC  # noqa: E402
from wearusfm.harness import anchor_candidates as A  # noqa: E402

PER = 8  # patch da 25 ms per blocco da 200 ms


def _block_bounds(w: int, fs: float, patch_s: float = 0.025) -> tuple[int, int]:
    return int(math.ceil(PER * w * patch_s * fs - 1e-9)), int(math.ceil(PER * (w + 1) * patch_s * fs - 1e-9))


def extract(args) -> int:
    import torch

    from wearusfm.harness.fm_features import load_model
    from wearusfm.model.fm import WearUsFM
    from wearusfm.preprocessing.canonical_view import apply_scale, bandpass_resample
    from wearusfm.tokenizer_checks.neurorvq import SPATIAL_INDEX_FIXED, NeuroRVQRunner
    from wearusfm.training.jepa import hidden_queries

    t0 = time.time()
    dev = args.device
    index = L.ManifestIndex.load(args.manifest, args.root)
    rows = [r for r in index.rows if r["dataset"] == args.dataset and r["rvq"] == "on"]
    users = sorted({r["subject"] for r in rows})
    rng = np.random.default_rng(args.seed)
    chosen = list(rng.choice(users, size=min(args.n_users, len(users)), replace=False))
    train_users = set(chosen[: int(round(0.75 * len(chosen)))])
    scales = json.loads(args.scales.read_text())["scales"]
    cfg = L.LoaderConfig(**{**L.signed_config().__dict__, "rvq_codes_root": str(args.rvq_codes)})
    teacher = load_model(str(args.checkpoint), dev, "teacher")
    student = load_model(str(args.checkpoint), dev, "student")
    torch.manual_seed(0)
    fresh = WearUsFM(teacher.cfg).to(dev).eval()
    sd = torch.load(args.tokenizer_checkpoint, map_location="cpu", weights_only=True)
    codebook = sd[f"quantize_{args.branch + 1}.layers.0.embedding.weight"].numpy().astype(np.float32)  # (8192, 128)
    del sd
    runner = NeuroRVQRunner(args.repo_dir, args.tokenizer_checkpoint, device=dev)
    store = RC.RVQCodeStore(args.rvq_codes)
    out = {k: [] for k in ("user", "V", "R", "H", "h_mask", "c16", "mom", "ms", "cur", "code", "nrvq")}

    def decoder_full(model, inp, visible):
        enc = model.encode(inp, visible)
        valid = enc.patch_valid & inp.qc_valid[:, None]
        sel = valid if visible is None else (~enc.visible & valid)
        q_ch, q_t, q_off = hidden_queries(sel, inp.counts)
        o = model.decode(enc, q_ch, q_t, q_off).float()
        full = torch.zeros(sel.shape[0], sel.shape[1], o.shape[-1], device=o.device)
        full[q_ch, q_t] = o
        return full, sel

    nrvq_cache: dict = {}
    keep_rng = np.random.default_rng(args.seed + 1)  # sottocampionamento delle unita' (dimensione del file), indipendente dalle finestre
    for ui, user in enumerate(chosen):
        urows = [r for r in rows if r["subject"] == user]
        w = np.array([r["weight"] for r in urows], dtype=np.float64)
        loader = L.PretrainLoader(L.ManifestIndex(urows, w / w.sum(), index.roots), cfg, dict(scales))
        for _ in range(args.windows_per_user // args.batch):
            b = loader.batch(args.batch, rng)
            inp, visible, _ = L.to_model_inputs(b)
            inp.signals = [x.to(dev) for x in inp.signals]
            inp.codes = {k: v.to(dev) for k, v in inp.codes.items()}
            inp.sets = {k: v.to(dev) for k, v in inp.sets.items()}
            inp.qc_valid = inp.qc_valid.to(dev)
            with torch.no_grad():
                fv, _ = decoder_full(teacher, inp, None)
                fr, _ = decoder_full(fresh, inp, None)
                fh, sh = decoder_full(student, inp, visible.to(dev))
            fv, fr, fh, sh = fv.cpu().numpy(), fr.cpu().numpy(), fh.cpu().numpy(), sh.cpu().numpy()  # una sola copia dalla GPU per batch
            a = 0
            for s, (x, fs, cnt, row, win) in enumerate(zip(b.signals, b.fs, b.counts, b.rows, b.windows)):
                n_w = b.n_patches[s] // PER
                view = loader.view(row)
                entry = store.get(row)
                ti, anchor = RC.trial_of(view, win)
                k0 = int(round((win.start - anchor) / (0.2 * view.fs)))
                for wi in range(n_w):
                    lo, hi = _block_bounds(wi, fs)
                    blk = x[:, lo:hi]
                    sl = slice(PER * wi, PER * (wi + 1))
                    c16, mom, ms = A.band_logpower(blk, fs), A.spectral_moments(blk, fs), A.multiscale_logpower(x, fs, lo, hi)
                    cur = np.concatenate([A.log_rms(blk), A.band_shape(blk, fs)], axis=1)
                    codes = b.rvq_codes[s][:, wi]
                    # feature continue di NeuroRVQ: il blocco da 16 patch canoniche (dall'inizio della prova) che contiene questa patch,
                    # preprocessato da solo (vista canonica, scala di sessione dell'archivio dei codici, fattore di V2)
                    j = k0 + wi
                    key = (row["session"], row["subject"], ti, j // RC.BLOCK_PATCHES)
                    if key not in nrvq_cache and entry is not None and j < int(entry["trial_patches"][ti]):
                        c0 = (j // RC.BLOCK_PATCHES) * RC.BLOCK_PATCHES
                        c1 = min(c0 + RC.BLOCK_PATCHES, int(entry["trial_patches"][ti]))
                        na, nb = anchor + int(round(c0 * 0.2 * view.fs)), anchor + int(round(c1 * 0.2 * view.fs))
                        seg = view.read(win.trial, na, nb).T.astype(np.float64)  # (n, C) unita' fisiche
                        canon = apply_scale(bandpass_resample(seg, view.fs), float(entry["scale"]))[: (c1 - c0) * 200].T
                        xin = (canon.astype(np.float32) * np.float32(RC.V2_SCALE_FACTOR))
                        f = runner.features(xin[:, None, :], [SPATIAL_INDEX_FIXED])  # (4, C*(c1-c0), 128)
                        nrvq_cache[key] = (c0, f[args.branch].reshape(cnt, c1 - c0, 128))
                    nr = nrvq_cache.get(key)
                    for c in range(cnt):
                        if not view.qc_valid[c] or codes[c] < 0 or nr is None or keep_rng.random() > args.keep:
                            continue
                        out["user"].append(user)
                        out["V"].append(fv[a + c, sl].mean(axis=0))
                        out["R"].append(fr[a + c, sl].mean(axis=0))
                        hid = bool(sh[a + c, sl].all())
                        out["h_mask"].append(hid)
                        out["H"].append(fh[a + c, sl].mean(axis=0) if hid else np.zeros(fv.shape[-1], np.float32))
                        out["c16"].append(c16[c])
                        out["mom"].append(mom[c])
                        out["ms"].append(ms[c])
                        out["cur"].append(cur[c])
                        out["code"].append(int(codes[c]))
                        out["nrvq"].append(nr[1][c, j - nr[0]])
                a += cnt
        print(f"[{time.strftime('%H:%M:%S')}] utente {ui + 1}/{len(chosen)}: {len(out['user'])} unita', {time.time() - t0:.0f} s", flush=True)
    arr = {k: np.asarray(v) for k, v in out.items()}
    arr["train_user"] = np.array([u in train_users for u in out["user"]])
    np.savez(args.out, codebook=codebook, **{k: (v.astype(np.float16) if v.dtype == np.float32 and v.ndim == 2 else v) for k, v in arr.items()})
    print(f"salvato {args.out}: {len(out['user'])} unita', {int(arr['h_mask'].sum())} nascoste, {time.time() - t0:.0f} s", flush=True)
    return 0


def probe(args) -> int:
    z = np.load(args.units)
    rng = np.random.default_rng(args.seed)
    tr_all, te_all = np.flatnonzero(z["train_user"]), np.flatnonzero(~z["train_user"])
    hm = z["h_mask"]

    def pick(idx, n):
        return np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))

    tr, te = pick(tr_all, args.max_train), pick(te_all, args.max_test)
    trh, teh = pick(tr_all[hm[tr_all]], args.max_train), pick(te_all[hm[te_all]], args.max_test)
    codebook = z["codebook"].astype(np.float64)
    code = z["code"]
    groups = A.kmeans_labels(codebook, [codebook], k=64, seed=args.seed)[0]  # (a): gruppo di ogni codice
    c16 = np.nan_to_num(z["c16"].astype(np.float64))
    fam = A.kmeans_labels(c16[tr], [c16], k=64, seed=args.seed)[0]  # (f): famiglia spettrale, adattata sul train
    targets = {"a_rvq_group": ("cls", groups[code]), "b_rvq_vector": ("reg", codebook[code]), "c_16_bands": ("reg", c16),
               "d_moments": ("reg", z["mom"].astype(np.float64)), "g_multiscale": ("reg", np.nan_to_num(z["ms"].astype(np.float64))),
               "e_neurorvq_features": ("reg", z["nrvq"].astype(np.float64)),
               "f_spectral_family": ("cls", fam)}
    reps = {"V": (z["V"].astype(np.float32), tr, te), "R": (z["R"].astype(np.float32), tr, te), "B": (z["cur"].astype(np.float64), tr, te),
            "H": (z["H"].astype(np.float32), trh, teh)}
    report = {"n_units": int(len(code)), "n_train": int(len(tr)), "n_test": int(len(te)), "n_train_hidden": int(len(trh)), "n_test_hidden": int(len(teh)),
              "thresholds": {"r2_min": A.R2_MIN, "acc_gain_min": A.ACC_GAIN_MIN}, "results": {}}
    for name, (kind, y) in targets.items():
        res = {}
        for rep, (x, i_tr, i_te) in reps.items():
            if kind == "reg":
                res[rep] = {"r2": A.ridge_r2(x[i_tr], y[i_tr], x[i_te], y[i_te])}
            else:
                n_cls_tr = min(len(i_tr), args.max_train_cls)
                res[rep] = A.logistic_gain(x[i_tr[:n_cls_tr]], y[i_tr[:n_cls_tr]], x[i_te], y[i_te], args.seed)
            print(name, rep, res[rep], flush=True)
        h = res["H"]
        res["predictable"] = A.verdict(r2_h=h["r2"]) if kind == "reg" else A.verdict(gain_h=h["gain"])
        res["explained_by_current_anchors"] = res["B"].get("r2", res["B"].get("gain"))
        report["results"][name] = res
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: {"predictable": v["predictable"], "H": v["H"], "B": v["explained_by_current_anchors"]} for k, v in report["results"].items()},
                     indent=1), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--manifest", type=Path, required=True)
    e.add_argument("--root", type=Path, action="append", required=True)
    e.add_argument("--scales", type=Path, required=True)
    e.add_argument("--rvq-codes", type=Path, required=True)
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--repo-dir", type=Path, required=True)
    e.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--dataset", default="emg2qwerty")
    e.add_argument("--n-users", type=int, default=40)
    e.add_argument("--windows-per-user", type=int, default=50)
    e.add_argument("--batch", type=int, default=25)
    e.add_argument("--branch", type=int, default=0)
    e.add_argument("--keep", type=float, default=0.15, help="frazione delle unita' tenute (a caso, seme fisso)")
    e.add_argument("--device", default="cuda")
    e.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("probe")
    p.add_argument("--units", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-train", type=int, default=100_000)
    p.add_argument("--max-test", type=int, default=30_000)
    p.add_argument("--max-train-cls", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    return extract(args) if args.stage == "extract" else probe(args)


if __name__ == "__main__":
    sys.exit(main())
