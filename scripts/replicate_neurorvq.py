#!/usr/bin/env python3
"""Replica dei numeri pubblicati di NeuroRVQ-EMG su EPN-612 e UCI-EMG (passo 5; foglio `docs/foglio_replica_neurorvq.md`, firmato da Simone il
04/10/2026 con due correzioni: banda 20-400 Hz e codice di NeuroRVQ nel repo). GPU su Leonardo.

Come da foglio: il FM pubblico (`NeuroRVQ_EMG_foundation_model_v1.pt`) col codice di `third_party/neurorvq` NON modificato; preprocessing 20-400 Hz
(Butterworth di ordine 3, fase zero, taglio alto min(400, fs/2) - 0,5 Hz) alla frequenza nativa, poi ricampionamento a 1000 Hz (`resample_poly`),
nessuna normalizzazione; gli 8 canali sugli elettrodi c1..c8; testa del paper: embedding delle patch (4 rami) concatenati sui canali e mediati sul
tempo, poi LayerNorm e uno strato lineare; fine-tuning completo con Tab. 13 «Finetuning (EMG)»: batch 128, AdamW lr 1e-3, beta (0,9, 0,95), weight
decay 0,001, warmup lineare 5 epoche (0,01 -> 1), poi coseno fino a 1e-6, 100 epoche; entropia incrociata pesata per classe; bf16. Split per
soggetto 7:1:2 con i semi dati (3: 0, 1, 2); modello valutato = l'epoca con la miglior accuratezza di validazione (si riporta anche l'ultima).
Tolleranza (foglio): media sui 3 split entro [92,65; 96,65] per EPN-612 e [87,23; 91,63] per UCI-EMG.

  python3 scripts/replicate_neurorvq.py --dataset epn612 --root ".../trainingJSON" --root ".../testingJSON" \\
      --checkpoint $WORK/models/NeuroRVQ_EMG_foundation_model_v1.pt --seeds 0 1 2 --out report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "neurorvq"))
from wearusfm.harness import benchmarks as B  # noqa: E402
from wearusfm.harness.splits import masks_from_split, split_subjects  # noqa: E402

PUBLISHED = {"epn612": (94.65, 0.4), "uci_emg": (89.43, 1.1)}  # Tab. 1(c), accuratezza ± (fatto 24)
TOLERANCE = {"epn612": (92.65, 96.65), "uci_emg": (87.23, 91.63)}  # foglio firmato
TARGET_FS = 1000.0
BAND = (20.0, 400.0)


def preprocess_native(x: np.ndarray, fs: float) -> np.ndarray:
    """(N, T, C) -> (N, C, T) float32 filtrati 20-400 Hz alla frequenza nativa (come il loro codice d'esempio)."""
    from scipy import signal

    hi = min(BAND[1], fs / 2.0) - 0.5
    b, a = signal.butter(N=3, Wn=[BAND[0], hi], btype="bandpass", fs=fs)
    return signal.filtfilt(b, a, np.transpose(x, (0, 2, 1)), axis=-1).astype(np.float32)


def resample_1k(x: np.ndarray, fs: float) -> np.ndarray:
    """(..., T) alla frequenza nativa -> (..., T') a 1000 Hz, poi tagliato a un multiplo di 200 campioni (patch da 200 ms)."""
    from fractions import Fraction

    from scipy import signal

    if fs != TARGET_FS:
        fr = Fraction(TARGET_FS / fs).limit_denominator(1000)
        x = signal.resample_poly(x, fr.numerator, fr.denominator, axis=-1)
    n = (x.shape[-1] // 200) * 200
    return np.ascontiguousarray(x[..., :n], dtype=np.float32)


def lr_at(step: int, steps_per_epoch: int, epochs: int, lr: float, warmup_epochs: int = 5, start: float = 0.01, min_lr: float = 1e-6) -> float:
    """Warmup lineare da start*lr a lr in `warmup_epochs`, poi coseno fino a min_lr alla fine (Tab. 13)."""
    w = warmup_epochs * steps_per_epoch
    total = epochs * steps_per_epoch
    if step < w:
        return lr * (start + (1.0 - start) * step / max(1, w))
    p = (step - w) / max(1, total - w)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, p)))


def class_weights(y: np.ndarray, n_cls: int) -> np.ndarray:
    """Come `get_class_weights` del loro modulo: 1/frequenza, normalizzati a media 1."""
    cnt = np.bincount(y, minlength=n_cls).astype(np.float64)
    w = np.where(cnt > 0, 1.0 / np.maximum(cnt, 1), 0.0)
    return (w / w.sum() * (cnt > 0).sum()).astype(np.float32)


def build_model(checkpoint: Path, device: str):
    import torch
    import yaml
    from torch import nn

    from inference.modules.NeuroRVQ_EMG_FM_inference_modules import ch_names_global
    from NeuroRVQ_EMG.NeuroRVQ import NeuroRVQFM

    args = yaml.safe_load((ROOT / "third_party" / "neurorvq" / "flags" / "NeuroRVQ_EMG_v1.yml").read_text())
    args["n_global_electrodes"] = len(ch_names_global)
    fm = NeuroRVQFM(n_patches=args["n_patches"], patch_size=args["patch_size"], in_chans=args["in_chans_second_stage"],
                    out_chans=args["out_chans_second_stage"], num_classes=0, embed_dim=args["embed_dim_second_stage"],
                    depth=args["depth_second_stage"], num_heads=args["num_heads_second_stage"], mlp_ratio=args["mlp_ratio_second_stage"],
                    qkv_bias=args["qkv_bias_second_stage"], qk_norm=partial(nn.LayerNorm, eps=1e-6), drop_rate=args["drop_rate_second_stage"],
                    attn_drop_rate=args["attn_drop_rate_second_stage"], drop_path_rate=args["drop_path_rate_second_stage"],
                    init_values=args["init_values_second_stage"], init_scale=args["init_scale_second_stage"],
                    n_global_electrodes=args["n_global_electrodes"], use_as_encoder=True, vocab_size=args["n_code"], use_for_pretraining=False)
    missing, unexpected = fm.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)  # come il loro esempio
    # la testa del loro codice (appiattimento) non serve: con head e fc_norm identita' il forward restituisce gli embedding appiattiti
    fm.head, fm.fc_norm = nn.Identity(), nn.Identity()
    return fm.to(device), args, list(missing), list(unexpected)


class PaperHead:
    """Testa del paper (App. J.3): embedding (B, C*T, 4*D) ordinati canale per canale -> concatenati sui canali (B, T, C*4*D) -> media sul tempo
    -> LayerNorm -> lineare."""

    def __init__(self, n_ch: int, emb: int, n_cls: int, device: str):
        from torch import nn

        self.n_ch = n_ch
        self.mod = nn.Sequential(nn.LayerNorm(n_ch * emb), nn.Linear(n_ch * emb, n_cls)).to(device)

    def __call__(self, flat, n_time: int):
        b = flat.shape[0]
        x = flat.view(b, self.n_ch, n_time, -1).permute(0, 2, 1, 3).reshape(b, n_time, -1).mean(dim=1)
        return self.mod(x)


def to_device_1k(xn: np.ndarray, fs: float, device: str, chunk: int = 4096):
    """Tutte le finestre ricampionate a 1 kHz una volta sola, in float16 sul dispositivo (EPN-612: ~7 GB): niente ricampionamento a ogni batch
    (collaudo 59336240: 177 s per epoca col ricampionamento sulla CPU)."""
    import torch

    parts = [torch.from_numpy(resample_1k(xn[i:i + chunk], fs)).to(torch.float16).to(device) for i in range(0, len(xn), chunk)]
    return torch.cat(parts)


def run_seed(xr, y: np.ndarray, subjects: np.ndarray, seed: int, args, log) -> dict:
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, f1_score

    from inference.modules.NeuroRVQ_EMG_FM_inference_modules import ch_names_global, create_embedding_ix

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = args.device
    split = split_subjects(subjects, (0.7, 0.1, 0.2), seed=seed)
    masks = masks_from_split(subjects, split)
    fm, cfg, missing, unexpected = build_model(args.checkpoint, dev)
    n_ch = xr.shape[1]
    n_time = xr.shape[-1] // 200
    names = np.array([f"c{i + 1}".encode() for i in range(n_ch)])
    t_ix, s_ix = create_embedding_ix(n_time, cfg["n_patches"], names, ch_names_global)
    n_cls = int(y.max()) + 1
    head = PaperHead(n_ch, 4 * cfg["embed_dim_second_stage"], n_cls, dev)
    params = list(fm.parameters()) + list(head.mod.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.001)
    cw = torch.as_tensor(class_weights(y[masks["train"] | masks["val"]], n_cls), device=dev)  # come il loro modulo: train + validazione
    tr = np.flatnonzero(masks["train"])
    spe = int(math.ceil(len(tr) / args.batch))
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda"))

    def forward(idx, train: bool):
        x = xr[torch.as_tensor(idx, device=xr.device)].float().view(len(idx), n_ch, n_time, 200)
        with amp:
            flat, _ = fm(x, t_ix.to(dev), s_ix.to(dev))  # indici (1, N) come nel loro modulo di fine-tuning
            return head(flat.float(), n_time)

    def evaluate(which: str) -> tuple[float, float]:
        fm.eval()
        idx_all = np.flatnonzero(masks[which])
        preds = []
        with torch.no_grad():
            for i in range(0, len(idx_all), args.eval_batch):
                preds.append(forward(idx_all[i:i + args.eval_batch], False).argmax(dim=-1).cpu().numpy())
        p = np.concatenate(preds)
        return float(accuracy_score(y[idx_all], p)), float(f1_score(y[idx_all], p, average="macro"))

    rng = np.random.default_rng(seed)
    curve, step, t0 = [], 0, time.time()
    for ep in range(args.epochs):
        fm.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), args.batch):
            idx = np.sort(order[i:i + args.batch])
            for g in opt.param_groups:
                g["lr"] = lr_at(step, spe, args.epochs, args.lr)
            logits = forward(idx, True)
            loss = F.cross_entropy(logits.float(), torch.as_tensor(y[idx], device=dev), weight=cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        va = evaluate("val")
        improved = not curve or va[0] > max(r["val_acc"] for r in curve)
        # il test si valuta solo quando la validazione migliora (e all'ultima epoca): il modello scelto e' lo stesso, il costo no
        te = evaluate("test") if improved or ep == args.epochs - 1 else (None, None)
        curve.append({"epoch": ep + 1, "val_acc": va[0], "test_acc": te[0], "test_f1": te[1], "loss": float(loss.detach())})
        log(f"seme {seed} epoca {ep + 1}: val {va[0]:.4f}, test {te[0] if te[0] is None else round(te[0], 4)}, perdita "
            f"{float(loss.detach()):.4f}, {time.time() - t0:.0f} s")
    best = max(curve, key=lambda r: (r["val_acc"], -r["epoch"]))  # la prima epoca col massimo: e' quella in cui il test e' stato valutato
    return {"seed": seed, "subjects": {k: len(getattr(split, k)) for k in ("train", "val", "test")},
            "windows": {k: int(m.sum()) for k, m in masks.items()}, "missing_keys": missing, "unexpected_keys": unexpected,
            "best_epoch": best["epoch"], "test_acc_best_val": best["test_acc"], "test_f1_best_val": best["test_f1"],
            "test_acc_last": curve[-1]["test_acc"], "curve": curve}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["epn612", "uci_emg"], required=True)
    ap.add_argument("--root", type=Path, action="append", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-users", type=int, default=None, help="solo per il collaudo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    t0 = time.time()

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    if args.dataset == "epn612":
        w, lab, subj, fs = B.load_epn612(args.root, n_users=args.max_users)
    else:
        w, lab, subj, fs = B.load_uci_emg(args.root[0], n_subjects=args.max_users)
    classes = sorted(set(lab.tolist()))
    y = np.array([classes.index(v) for v in lab.tolist()], dtype=np.int64)
    log(f"{args.dataset}: {len(w)} finestre, {len(set(subj))} soggetti, {len(classes)} classi, fs {fs} Hz, {w.shape[1] / fs:.1f} s")
    xr = to_device_1k(preprocess_native(w, fs), fs, args.device)
    del w
    log(f"dati a 1 kHz sul dispositivo: {tuple(xr.shape)}, {xr.element_size() * xr.nelement() / 2**30:.1f} GiB")
    runs = [run_seed(xr, y, subj, s, args, log) for s in args.seeds]
    accs = np.array([r["test_acc_best_val"] for r in runs]) * 100
    lo, hi = TOLERANCE[args.dataset]
    report = {"dataset": args.dataset, "classes": classes, "published_acc": PUBLISHED[args.dataset], "tolerance": [lo, hi],
              "mean_test_acc": float(accs.mean()), "std_test_acc": float(accs.std(ddof=1)) if len(accs) > 1 else None,
              "within_tolerance": bool(lo <= accs.mean() <= hi), "mean_test_acc_last_epoch": float(np.mean([r["test_acc_last"] for r in runs]) * 100),
              "protocol": "docs/foglio_replica_neurorvq.md (firmato 04/10/2026: banda 20-400 Hz)", "runs": runs, "elapsed_s": time.time() - t0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    log(f"media sui semi {accs.mean():.2f} (pubblicato {PUBLISHED[args.dataset][0]}; tolleranza [{lo}; {hi}]): "
        f"{'DENTRO' if report['within_tolerance'] else 'FUORI'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
