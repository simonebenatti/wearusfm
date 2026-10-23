#!/usr/bin/env python3
"""Fuga del contenuto nascosto attraverso il preprocessing del loader (pilot dei rami `rami_pilota_v1`, PSD-2; punto 3 di GPT del 23/10/2026).

Il loader filtra il ritaglio (passa-banda Butterworth di ordine 4 e notch, `sosfiltfilt`, `pretraining_loader._filter`, con il margine del
filtro) PRIMA che il pilot azzeri i tratti nascosti (`branch_masking.masked_signal`): i campioni visibili vicini a un tratto portano quindi una
traccia filtrata del contenuto nascosto, che la prova del protocollo (segnale cambiato DOPO il filtro) non vede. Questa misura parte dal segnale
GREZZO: due segnali uguali fuori dai tratti e diversi dentro, filtrati e normalizzati come il loader, mascherati come il pilot; differenza
delle log-potenze dello studente sui frame-banda di contesto, per ramo, scala e distanza del supporto del frame dal tratto piu' vicino
(guardia). Per confronto, la distorsione dell'azzeramento (studente contro teacher sugli stessi frame) gia' accettata dal protocollo.

  python3 scripts/branch_mask_leak.py --out results/rami_pilota/fuga_preprocessing_<data>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.model import branch_masking as BM  # noqa: E402
from wearusfm.model import spectral_branches as SB  # noqa: E402
from wearusfm.model.fm import ModelInputs  # noqa: E402

FS = 2000.0  # emg2qwerty, il dataset del 2C e del pilot
CHANNELS, CROP_S, TRIALS = 8, 4.0, 6
GUARDS_MS = (0, 25, 50, 75, 100, 150, 200, 300, 400, 500)
# contenuto del tratto nascosto nel segnale grezzo (prima del filtro), in unita' della deviazione standard del grezzo: casi estremi apposta
VARIANTS = ("burst_x10", "linea_50Hz_x10", "linea_60Hz_x10", "tono_25Hz_x10", "rumore_indipendente_x3")


def _raw(rng, c, t):
    """Grezzo simile all'EMG: rumore bianco con modulazione lenta d'ampiezza (attivazioni)."""
    env = 1.0 + 0.8 * np.sin(2 * np.pi * rng.uniform(0.3, 1.5, (c, 1)) * np.arange(t) / FS + rng.uniform(0, 2 * np.pi, (c, 1)))
    return (rng.standard_normal((c, t)) * env).astype(np.float64)


def _hidden(variant, rng, c, n, sd):
    t = np.arange(n) / FS
    if variant == "burst_x10":
        return None  # il contenuto originale moltiplicato per 10 (sotto)
    if variant.startswith("linea_") or variant.startswith("tono_"):
        f0 = float(variant.split("_")[1].rstrip("Hz"))
        return 10 * sd * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi, (c, 1)))
    return 3 * sd * rng.standard_normal((c, n))


def _inputs(x):
    c = x.shape[0]
    return ModelInputs([torch.as_tensor(x, dtype=torch.float32)], [FS], {}, {}, [c], torch.ones(c, dtype=torch.bool),
                       band_limit=[np.full(c, 450.0)], edge_guard=[(0, 0)])


def _distance_ms(t, win_ms, hop_ms, radius, spans):
    """Per frame e banda: distanza in ms fra il supporto del frame e il tratto nascosto piu' vicino (0 se si toccano)."""
    n, starts = SB.frame_starts(t, FS, win_ms, hop_ms)
    r = np.asarray(radius, dtype=np.int64)
    lo, hi = starts[:, None] - r[None, :], starts[:, None] + n + r[None, :]
    gap = np.full(lo.shape, np.inf)
    for a, b in spans:
        gap = np.minimum(gap, np.maximum(np.maximum(a - hi, lo - b), 0))
    return gap / FS * 1000.0


def measure(seed: int = 20261023) -> dict:
    cfg = L.signed_config((20.0, 450.0))
    band, notch, margin = cfg.filter_band_hz, cfg.notch_hz, int(round(cfg.filter_margin_s * FS))
    torch.manual_seed(0)
    sbs = SB.SpectralBranches(SB.BRANCH_KINDS, out_dim=128, hidden=48)  # dimensioni del pilot (128 colonne, hidden 48); le statistiche non servono agli spettri
    rng = np.random.default_rng(seed)
    t = int(CROP_S * FS)
    rows = {v: {} for v in VARIANTS}
    zeroing = {}
    for trial in range(TRIALS):
        raw = _raw(rng, CHANNELS, t + 2 * margin)
        spans = BM.sample_spans(t, FS, (0, 0), rng)
        x_a = L._filter(raw, FS, band, notch)[:, margin:margin + t]
        scale = 1.4826 * np.median(np.abs(x_a)) + 1e-12  # una scala per sessione (`estimate_session_scale`): uguale per i due segnali
        inp_a = _inputs(x_a / scale)
        for variant in VARIANTS:
            raw_b = raw.copy()
            for a, b in spans:
                h = _hidden(variant, rng, CHANNELS, b - a, raw.std())
                raw_b[:, margin + a:margin + b] = raw_b[:, margin + a:margin + b] * 10 if h is None else h
            x_b = L._filter(raw_b, FS, band, notch)[:, margin:margin + t]
            assert np.array_equal(raw_b[:, :margin + spans[0][0]], raw[:, :margin + spans[0][0]])
            inp_b = _inputs(x_b / scale)
            for kind, branch in sbs.branches.items():
                side_a, side_b = BM.context_roles(branch, inp_a, [spans]), BM.context_roles(branch, inp_b, [spans])
                full = branch.spectra(inp_a) if variant == VARIANTS[0] else None
                for name, win_ms, hop_ms, nb in SB.SCALES:
                    sa, sb = side_a[name], side_b[name]
                    use = (sa["valid"] & sa["ctx"]).numpy()
                    dist = _distance_ms(t, win_ms, hop_ms, BM.branch_radius(branch, name, FS, nb), spans)[None].repeat(CHANNELS, 0)
                    delta = (sa["logp"] - sb["logp"]).abs().numpy()
                    key = f"{kind}_{name}"
                    acc = rows[variant].setdefault(key, {g: [] for g in GUARDS_MS})
                    for g in GUARDS_MS:
                        sel = use & (dist >= g)
                        acc[g].append((delta[sel], int(sel.sum()), int(sa["valid"].numpy().sum())))
                    if full is not None:
                        lp_t, _ = full[name]
                        zeroing.setdefault(key, []).append((sa["logp"] - lp_t).abs().numpy()[use])
    out = {"variants": {}, "zeroing_distortion": {}}
    for variant, per in rows.items():
        out["variants"][variant] = {}
        for key, acc in per.items():
            out["variants"][variant][key] = []
            for g in GUARDS_MS:
                d = np.concatenate([a for a, _, _ in acc[g]]) if acc[g] else np.zeros(0)
                n_ctx, n_valid = sum(n for _, n, _ in acc[g]), sum(v for _, _, v in acc[g])
                out["variants"][variant][key].append({
                    "guard_ms": g, "max": float(d.max()) if len(d) else None, "p99": float(np.quantile(d, 0.99)) if len(d) else None,
                    "median": float(np.median(d)) if len(d) else None, "context_frame_bands": n_ctx,
                    "context_fraction_of_valid": n_ctx / n_valid if n_valid else None})
    for key, parts in zeroing.items():
        d = np.concatenate(parts)
        out["zeroing_distortion"][key] = {"max": float(d.max()), "p99": float(np.quantile(d, 0.99))}
    out["setup"] = {"fs": FS, "channels": CHANNELS, "crop_s": CROP_S, "trials": TRIALS, "seed": seed, "filter_band_hz": list(band),
                    "notch_hz": list(notch), "filter_margin_s": cfg.filter_margin_s, "mask": {"span_ms": BM.MASK_SPAN_MS,
                                                                                               "fraction": BM.MASK_FRACTION,
                                                                                               "margin_ms": BM.MASK_MARGIN_MS},
                    "units": "differenza assoluta di log-potenza (nat) sui frame-banda di contesto dello studente"}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    res = measure()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    for variant, per in res["variants"].items():
        print(variant)
        for key, rows in per.items():
            print(f"  {key:13s} " + "  ".join(f"g{r['guard_ms']}:{r['max']:.1e}/{100 * r['context_fraction_of_valid']:.0f}%"
                                              if r["max"] is not None else f"g{r['guard_ms']}:-" for r in rows))
    print("distorsione dell'azzeramento:", {k: f"{v['max']:.1e}" for k, v in res["zeroing_distortion"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
