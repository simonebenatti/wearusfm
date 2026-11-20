"""Controllo ESEGUITO: contesto temporale delle viste `pre` e `post` DENTRO il modello (perturbazione applicata direttamente a ModelInputs.signals,
cioe' dopo filtro e scala del loader). Modello piccolo a pesi casuali, tokenizer T100 con GELU come T100-G; fs 2 kHz, 16 canali come emg2pose
se il montaggio di prova lo consente, altrimenti 200 Hz e 8 canali."""
import sys, copy
SNAP = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/gpt_c15aac9"
sys.path.insert(0, SNAP + "/src")
import numpy as np
import torch
from wearusfm.harness import finetuning as F, fm_features as FF
from wearusfm.harness.backbone_depth import depth_readouts
from wearusfm.model.fm import FMConfig, WearUsFM, ModelInputs

cfg = FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=2, decoder_layers=1, muscle_dropout=.4,
               window_ms=100., tokenizer_activation="gelu", pose_head=True)
torch.manual_seed(42)
model = WearUsFM(cfg).eval()
for fs in (200., 2000.):
    n = int(4 * fs)
    rng = np.random.default_rng(1)
    w = rng.normal(size=(1, n, 8)).astype(np.float32)
    prepared = F.PreparedWindows(w, np.array(["a"]), fs, FF.myo8_montage("epn612", fs))
    inp = prepared.batch([0], "cpu")[0]
    spp = int(round(0.025 * fs))  # campioni per patch
    with torch.no_grad():
        base = depth_readouts(model, inp)["sequences"]
        inp2 = copy.copy(inp)
        sig = [x.clone() for x in inp.signals]
        sig[0][:, 80 * spp:81 * spp] += 5.0  # sola patch 80, tutti i canali
        inp2.signals = sig
        pert = depth_readouts(model, inp2)["sequences"]
    for name in ("pre", "post"):
        d = (pert[name] - base[name]).abs().amax(dim=(0, 2)).numpy()
        ch = np.nonzero(d > 0)[0]
        rel = np.nonzero(d > 1e-3 * d.max())[0]
        rel6 = np.nonzero(d > 1e-6 * d.max())[0]
        print(f"fs {fs:g} Hz, vista {name}: patch con variazione esattamente non nulla {ch.min() - 80}..{ch.max() - 80} ({len(ch)}); "
              f"> 1e-6 del massimo {rel6.min() - 80}..{rel6.max() - 80} ({len(rel6)}); > 1e-3 del massimo {rel.min() - 80}..{rel.max() - 80} ({len(rel)})")
