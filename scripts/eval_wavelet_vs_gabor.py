"""Gabor del pacchetto step 2/3 contro banco wavelet (Morlet a Q costante) di `model/spectral_branches.py`, stesse misure per tutti e due:
1. copertura della banda 20-450 Hz (energia per sinusoidi di ampiezza 1, somma sui filtri);
2. leggibilita' lineare dei 32 target dello Step 1 dalle log-energie medie nel tempo, su EMG sintetico a spettri casuali;
3. risoluzione temporale (scala veloce) su un burst di 20 ms;
4. costo (CPU, 1 thread) per una finestra da 16 canali, 4 s a 2 kHz.
Il Gabor e' quello del pacchetto `WearUsFM_Step2_Step3_Codice_e_Test.zip` (sha256 e994af3f...2453c, estratto fuori dal repo), non copiato qui.
Esito del 13/10/2026 sul Mac: `results/steps23/wavelet_vs_gabor_20261013.json`. Controllo software su EMG sintetico, non una prova su EMG
reale (le energie sintetiche sono normalizzate: R2 dell'energia non significativo).
Uso: python eval_wavelet_vs_gabor.py <worktree> <pacchetto wearusfm_steps23> <out.json>"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT, PKG, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PKG))
from emg_spectral.common import SignalSample  # noqa: E402
from emg_spectral.gabor import GaborFrontend  # noqa: E402
from wearusfm.model import anchor_targets as AT  # noqa: E402
from wearusfm.model.spectral_branches import PSDFrontend, WaveletFrontend  # noqa: E402
from wearusfm.model.spectral_step1 import global_spectral_targets  # noqa: E402

torch.set_num_threads(1)
FS = 2000.0
gabor, wavelet, psd = GaborFrontend(False), WaveletFrontend(False), PSDFrontend()


def run(front, x, fs=FS):
    """{scala: (C, P, B) potenza, validita'} per un frontend su x (C, T)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    c = x.shape[0]
    with torch.no_grad():
        if front is gabor:
            out = front(SignalSample(x, fs, min(450.0, fs / 2), torch.ones(c, dtype=torch.bool)))
            return {k: (v.power.numpy(), v.valid.numpy()) for k, v in out.items()}
        out = front(x, fs, np.full(c, min(450.0, fs / 2)), (0, 0))
        return {k: (p.numpy(), v.numpy()) for k, (p, v) in out.items()}


def mean_energy(front, x):
    out = run(front, x)
    return {k: np.where(v, p, np.nan) for k, (p, v) in out.items()}


# 1. copertura: sinusoidi di ampiezza 1 da 10 a 500 Hz, 2 s, energia media sui frame validi
freqs = np.arange(10.0, 500.0, 2.0)
t = np.arange(int(2.0 * FS)) / FS
resp = {"gabor": {"fast": [], "slow": []}, "wavelet": {"fast": [], "slow": []}}
for f in freqs:
    x = np.sin(2 * np.pi * f * t)[None, :]
    for name, front in (("gabor", gabor), ("wavelet", wavelet)):
        for k, e in mean_energy(front, x).items():
            resp[name][k].append(np.nanmean(e[0], axis=0))  # (B,)
coverage = {}
band = (freqs >= 20) & (freqs <= 450)
for name in resp:
    coverage[name] = {}
    for k in ("fast", "slow"):
        r = np.array(resp[name][k])  # (n_freq, B)
        total = r.sum(1)[band] / np.median(r.sum(1)[band])
        best = r.max(1)[band] / np.median(r.max(1)[band])
        coverage[name][k] = {
            "somma_min_su_mediana": float(total.min()), "somma_max_su_mediana": float(total.max()),
            "frazione_20_450_sotto_10pct_della_mediana": float((total < 0.1).mean()),
            "frazione_20_450_sotto_50pct_della_mediana": float((total < 0.5).mean()),
            "miglior_filtro_min_su_mediana": float(best.min()),
        }

# 2. leggibilita' dei 32 target: EMG sintetico, spettro di Shwedyk (f_l, f_h casuali) + picco stretto casuale, inviluppo lento casuale
rng = np.random.default_rng(0)


def synth(c=4, dur=4.0):
    n = int(dur * FS)
    f = np.fft.rfftfreq(n, 1 / FS)
    fl, fh = rng.uniform(20, 90), rng.uniform(50, 220)
    shape = fh ** 4 * f ** 2 / ((f ** 2 + fl ** 2) * (f ** 2 + fh ** 2) ** 2)
    pk, w = rng.uniform(30, 400), rng.uniform(3, 30)
    shape = shape / shape.max() + rng.uniform(0, 1.5) * np.exp(-0.5 * ((f - pk) / w) ** 2)
    shape[(f < 20) | (f > 450)] = 0
    xs = []
    for _ in range(c):
        spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) * np.sqrt(shape)
        x = np.fft.irfft(spec, n)
        env = 1 + 0.8 * np.sin(2 * np.pi * rng.uniform(0.2, 2) * np.arange(n) / FS + rng.uniform(0, 6.3))
        xs.append(x * env)
    x = np.stack(xs)
    return (x / x.std()).astype(np.float32)


def feats(front, x):
    out = []
    for k, e in mean_energy(front, x).items():
        out.append(np.log(np.nanmean(np.nanmean(e, axis=1), axis=0) + 1e-8))
    return np.concatenate(out)


N = 240
X = {"gabor": [], "wavelet": [], "psd": []}
Y, M = [], []
for i in range(N):
    x = synth()
    tg = AT.anchor_targets(x, FS, 450.0, multiscale=True)
    y, m = global_spectral_targets([tg], [x.shape[0]], np.ones(x.shape[0], bool))
    Y.append(y[0])
    M.append(m[0])
    for name, front in (("gabor", gabor), ("wavelet", wavelet), ("psd", psd)):
        X[name].append(feats(front, x))
Y, M = np.array(Y), np.array(M)


def ridge_r2(Xf, Yt, alphas=(1e-3, 1e-2, 1e-1, 1, 10)):
    """Train 160 / val 40 / test 40; alpha sulla validation; R2 per coordinata sul test."""
    tr, va, te = np.arange(160), np.arange(160, 200), np.arange(200, 240)
    mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-9
    Z = (Xf - mu) / sd
    ym = Yt[tr].mean(0)

    def fit(a):
        A = Z[tr]
        return np.linalg.solve(A.T @ A + a * np.eye(A.shape[1]), A.T @ (Yt[tr] - ym))

    def r2(W, idx):
        p = Z[idx] @ W + ym
        return 1 - ((Yt[idx] - p) ** 2).sum(0) / ((Yt[idx] - Yt[idx].mean(0)) ** 2).sum(0)

    best = max(alphas, key=lambda a: r2(fit(a), va).mean())
    return r2(fit(best), te), best


fams = {"forma_fast": slice(0, 6), "energia_fast": slice(6, 7), "forma_slow": slice(7, 31), "energia_slow": slice(31, 32)}
assert M.all()
readability = {}
for name in X:
    r2, a = ridge_r2(np.array(X[name]), Y)
    readability[name] = {f: float(np.mean(r2[s])) for f, s in fams.items()}
    readability[name]["alpha"] = a
    readability[name]["r2_min_coordinata_forma_fast"] = float(r2[0:6].min())

# 3. risoluzione temporale: burst di 20 ms a 60/120/250 Hz al centro di 2 s; ampiezza a meta' dell'energia nella scala veloce (frame da 25 ms)
loc = {}
for name, front in (("gabor", gabor), ("wavelet", wavelet)):
    loc[name] = {}
    for f in (60.0, 120.0, 250.0):
        x = np.zeros(int(2 * FS), np.float32)
        a, b = int(0.99 * FS), int(1.01 * FS)
        x[a:b] = np.sin(2 * np.pi * f * np.arange(b - a) / FS)
        p, v = run(front, x[None, :])["fast"]
        k = int(np.argmax(p[0].max(0)))
        e = p[0][:, k]
        loc[name][f"{int(f)}Hz"] = {"frame_sopra_meta_del_picco": int((e >= e.max() / 2).sum()),
                                    "frame_sopra_10pct": int((e >= e.max() / 10).sum())}

# 4. costo e supporto
x = torch.randn(16, int(4 * FS))
cost = {}
for name, front in (("gabor", gabor), ("wavelet", wavelet), ("psd", psd)):
    run(front, x)
    t0 = time.perf_counter()
    for _ in range(5):
        run(front, x)
    cost[name] = {"ms_per_finestra_16ch_4s_2kHz": 1e3 * (time.perf_counter() - t0) / 5}
    out = run(front, x)
    cost[name]["frazione_frame_validi"] = {k: float(v[0].mean()) for k, (p, v) in out.items()}
cost["wavelet"]["raggio_ms_slow_min_max"] = [1e3 * float(r) / FS for r in (wavelet.banks["slow"].radius(FS).min(), wavelet.banks["slow"].radius(FS).max())]
cost["wavelet"]["raggio_ms_fast_min_max"] = [1e3 * float(r) / FS for r in (wavelet.banks["fast"].radius(FS).min(), wavelet.banks["fast"].radius(FS).max())]
cost["gabor"]["raggio_ms_comune"] = 1e3 * gabor.banks["slow"].kernels(FS)[1] / FS

report = {"copertura": coverage, "leggibilita_target_R2_test": readability, "localizzazione_temporale_fast": loc, "costo": cost,
          "note": "EMG sintetico (spettro di Shwedyk piu' un picco casuale, inviluppo sinusoidale), 240 finestre 4 canali 4 s a 2 kHz; "
                  "ridge 160/40/40, alpha sulla validation. Controllo software, non una prova su EMG reale."}
OUT.write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
