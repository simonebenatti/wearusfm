"""Misure CPU a corredo della review di Claude della proposta tokenizer di GPT (c052792; `docs/reviews/claude_review_tokenizer_20261102.md`).

Solo segnali sintetici e il generatore di maschere del repo: nessun dato, nessun checkpoint, nessun job. Tre misure:

1. rango effettivo del banco a kernel continui (64 uscite) sul suo supporto numerico, per frequenza di campionamento e finestra (25 e 100 ms a
   passo 25): quanti numeri indipendenti porta davvero un token, dopo l'anti-aliasing alla Nyquist di ciascuna fs;
2. punto di lavoro della GELU: deviazione per feature di `fe_gain * e(x)` (quello che entra nell'attivazione di T100-G) su rumore a banda
   20-450 Hz adattata alla Nyquist, come la calibrazione del guadagno; piu' la frazione dell'energia del kernel fuori dalla finestra nominale;
3. erosione delle query con `masked_loss_radius` 2 rispetto a tutte le patch nascoste (raggio 0, il B storico), per tipo di maschera, montaggio
   e durata, con le quote storiche di D10 e RVQ spento come in B.

Uso, dalla radice del repo: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/check_tokenizer_window_review.py [--parte banco|gelu|maschere|tutto]`.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wearusfm.model import channel_codes as CC  # noqa: E402
from wearusfm.model import frontend as FE  # noqa: E402
from wearusfm.model.patch_tokens import PatchTokenizer  # noqa: E402

torch.set_num_threads(1)
WINDOWS = (None, 100.0)
RATES = (200.0, 1000.0, 2000.0)


def banco() -> None:
    print("== 1. rango effettivo del banco (64 kernel) sul supporto numerico; soglie 1e-2 e 1e-3 del primo valore singolare ==")
    print("   energia del kernel fuori dalla finestra nominale e oltre 0,5 / 1,5 / 2,5 patch dal centro della patch (raggio 0 / 1 / 2)")
    for window in WINDOWS:
        fe = FE.ContinuousKernelFrontEnd(seed=0, **FE.window_kwargs(25.0, window)).double()
        for fs in RATES:
            reach = fe.context_s + fe.lead_s
            m = int(math.ceil(reach * fs - 1e-9))
            n_in = int(math.ceil(fe.patch_s * fs + 1e-9))
            rel = (np.arange(n_in + 2 * m) - m) / fs  # istanti relativi all'inizio della patch, come in `forward`
            with torch.no_grad():
                k = fe.kernels_at(torch.from_numpy(rel + fe.lead_s), fs)  # (64, n_istanti)
            s = torch.linalg.svdvals(k)
            r2, r3 = int((s > 1e-2 * s[0]).sum()), int((s > 1e-3 * s[0]).sum())
            fam = {}
            for name, sl in fe.family_slices().items():
                sf = torch.linalg.svdvals(k[sl])
                fam[name] = int((sf > 1e-2 * sf[0]).sum())
            t = torch.from_numpy(rel)
            nominal = (t >= -fe.lead_s - 1e-12) & (t < fe.patch_s + fe.lead_s - 1e-12)
            e_tot = (k ** 2).sum(1)
            out_nominal = float(1 - ((k[:, nominal] ** 2).sum(1) / e_tot).mean())
            far = []
            for r in (0.5, 1.5, 2.5):
                sel = (t + 0.5 / fs - fe.patch_s / 2).abs() > r * fe.patch_s
                far.append(float(((k[:, sel] ** 2).sum(1) / e_tot).mean()))
            print(f"finestra {window or 25:>4g} ms  fs {fs:>5g} Hz  campioni nominali {int(round(fe.window_s * fs)):>3d}  supporto {len(rel):>3d}  "
                  f"rango 1e-2 {r2:>2d}  1e-3 {r3:>2d}  per famiglia {fam}  fuori nominale {out_nominal:5.1%}  "
                  f"oltre 0,5/1,5/2,5 patch {far[0]:5.1%} {far[1]:5.1%} {far[2]:5.1%}")


def _band_noise(fs: float, seconds: float = 8.0, n_ch: int = 4, seed: int = 0) -> np.ndarray:
    x = np.random.default_rng(seed).normal(size=(n_ch, int(seconds * fs)))
    hi = min(450.0, 0.45 * fs)  # come il loader: passa-banda adattato alla Nyquist
    x = sps.sosfiltfilt(sps.butter(4, [20.0, hi], btype="bandpass", fs=fs, output="sos"), x, axis=-1)
    return x / x.std()


def gelu() -> None:
    print("== 2. punto di lavoro della GELU: deviazione per feature di fe_gain * e(x) su rumore a banda (20-450 Hz, adattata alla Nyquist) ==")
    for window in WINDOWS:
        tok = PatchTokenizer(16, patch_ms=25.0, **FE.window_kwargs(25.0, window))
        print(f"finestra {window or 25:g} ms: fe_gain = {float(tok.fe_gain):.4g}")
        for fs in RATES:
            x = torch.from_numpy(_band_noise(fs))
            with torch.no_grad():
                f = (tok.frontend(x[None], fs)[0] * tok.fe_gain)[:, 8:-8]  # (C, P, 64), senza i bordi
            sd = f.reshape(-1, 64).std(0)
            q = torch.quantile(sd, torch.tensor([0.1, 0.5, 0.9], dtype=sd.dtype)).tolist()
            fam = {name: (float(sd[sl].min()), float(sd[sl].median()), float(sd[sl].max())) for name, sl in tok.frontend.family_slices().items()}
            g = torch.nn.functional.gelu(f)
            nonlin = float((g - 0.5 * f).abs().mean() / f.abs().mean())  # 0 = GELU indistinguibile dalla retta 0,5 x
            print(f"   fs {fs:>5g}: std globale {float(f.std()):4.2f}; quantili 10/50/90 per feature {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}; "
                  + "; ".join(f"{n} min/med/max {a:.2f}/{b:.2f}/{c:.2f}" for n, (a, b, c) in fam.items()) + f"; |GELU(x)-0,5x|/|x| {nonlin:.2f}")


def maschere(n_draw: int = 200) -> None:
    from wearusfm.ingest import capgmyo, emg2pose, ninapro_std
    from wearusfm.ingest.common import montage_to_dict
    from wearusfm.harness import fm_features as FF
    from wearusfm.training import masking as M
    from wearusfm.training.jepa import core_hidden

    def layout(m):
        d = m if isinstance(m, dict) else montage_to_dict(m, [True] * m.n_channels)
        return CC.layout_from_montage(d), CC.anatomy_codes(d).compartment_weights.argmax(axis=1)

    montages = {"EPN myo8 (8 ch)": FF.myo8_montage("epn612", 200.0),
                "emg2pose (16 ch)": emg2pose.build_montage_metadata("u1", "s", "left"),
                "NinaPro DB2 (12 ch)": ninapro_std.build_montage_metadata(ninapro_std.DB2, 1, "right"),
                "CapgMyo (128 ch)": capgmyo.build_montage_metadata(1)}
    spec = M.MaskSpec.d10_proposal(0.5)
    kinds = (M.SHORT, M.MEDIUM, M.LONG, M.SPATIAL)
    print(f"== 3. query con raggio 2 rispetto alle patch nascoste (raggio 0): quote D10 {spec.share_short}/{spec.share_medium}/"
          f"{spec.share_long}/{spec.share_spatial}, ratio {spec.ratio}, RVQ spento, {n_draw} estrazioni per riga ==")
    print(f"{'montaggio':<22}{'patch':>6}{'nascoste':>10}{'query r2/nascoste':>19}{'corte':>8}{'medie':>8}{'lunghe':>8}{'spaz.':>8}{'finestre senza query':>22}")
    for name, m in montages.items():
        lay, comp = layout(m)
        for n_p in (40, 80, 160):
            rng = np.random.default_rng(0)
            tot_hidden = tot_core = empty = 0
            by_kind = {k: [0, 0] for k in kinds}
            for _ in range(n_draw):
                vis, kind = M.generate_mask(lay, comp, n_p, n_p, False, spec, rng)
                visible = torch.from_numpy(vis)
                hidden = ~visible
                core = core_hidden(hidden, visible, 2)
                tot_hidden += int(hidden.sum())
                tot_core += int(core.sum())
                empty += int(core.sum()) == 0
                for k in kinds:
                    sel = torch.from_numpy(kind == k)
                    by_kind[k][0] += int((sel & hidden).sum())
                    by_kind[k][1] += int((sel & core).sum())
            cell = lambda k: f"{by_kind[k][1] / by_kind[k][0]:6.0%}" if by_kind[k][0] else "   n/a"  # noqa: E731
            print(f"{name:<22}{n_p:>6}{tot_hidden / (n_draw * lay.n_channels * n_p):>10.1%}{tot_core / max(1, tot_hidden):>19.1%}"
                  f"{cell(M.SHORT):>8}{cell(M.MEDIUM):>8}{cell(M.LONG):>8}{cell(M.SPATIAL):>8}{empty / n_draw:>22.1%}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parte", choices=["banco", "gelu", "maschere", "tutto"], default="tutto")
    args = ap.parse_args(argv)
    if args.parte in ("banco", "tutto"):
        banco()
    if args.parte in ("gelu", "tutto"):
        gelu()
    if args.parte in ("maschere", "tutto"):
        maschere()
    return 0


if __name__ == "__main__":
    sys.exit(main())
