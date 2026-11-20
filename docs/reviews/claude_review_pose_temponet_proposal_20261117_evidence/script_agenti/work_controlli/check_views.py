"""Controllo ESEGUITO (modello piccolo a pesi casuali, segnale sintetico): che cosa sono le viste `pre` e `post` di
harness.backbone_depth.depth_readouts nello snapshot c15aac9, e quanto contesto temporale ha ciascuna."""
import sys
SNAP = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/gpt_c15aac9"
sys.path.insert(0, SNAP + "/src")
import numpy as np
import torch
from wearusfm.harness import finetuning as F, fm_features as FF
from wearusfm.harness.backbone_depth import depth_readouts
from wearusfm.model.fm import FMConfig, WearUsFM
from wearusfm.model.perceiver import PerceiverPooling, offsets_from_counts

cfg = FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=2, decoder_layers=1, muscle_dropout=.4,
               window_ms=100., tokenizer_activation="gelu", pose_head=True)
torch.manual_seed(42)
model = WearUsFM(cfg).eval()
print("model.pool e':", type(model.pool).__name__, "| e' PerceiverPooling:", isinstance(model.pool, PerceiverPooling))
fs, n = 200., 800  # 4 s -> 160 patch da 25 ms
rng = np.random.default_rng(1)
w = rng.normal(size=(1, n, 8)).astype(np.float32)
s = np.array(["a"])
montage = FF.myo8_montage("epn612", fs)

def views(x):
    prepared = F.PreparedWindows(x, s, fs, montage)
    inp = prepared.batch([0], "cpu")[0]
    with torch.no_grad():
        out = depth_readouts(model, inp)
        enc = model.encode(inp, None)
        # ricalcolo indipendente dell'uscita del Perceiver, con gli stessi passi di WearUsFM.encode
        tokens, patch_valid, time_valid = model.tokenizer(inp.signals, inp.fs)
        vis = patch_valid & inp.qc_valid[:, None]
        ids = model.identity(inp.codes, None)
        local = model.local(tokens + ids[:, None, :], inp.sets, vis)
        lat = model.pool(local, offsets_from_counts(inp.counts), vis)
        post = model.backbone(lat, enc.key_time_valid)
    return out, lat, post, enc

model.identity.eval()
out, lat, post, enc = views(w)
pre, pst = out["sequences"]["pre"], out["sequences"]["post"]
print("forme: uscita Perceiver", tuple(lat.shape), "(S, P, K, d) | pre", tuple(pre.shape), "| post", tuple(pst.shape))
print("pre == media sui K slot dell'uscita del Perceiver: scarto massimo", float((pre - lat.float().mean(2)).abs().max()))
print("post == media sui K slot dell'uscita del backbone (norma finale compresa): scarto massimo", float((pst - post.float().mean(2)).abs().max()),
      "| == enc.z.mean(2):", float((pst - enc.z.float().mean(2)).abs().max()))
print("testa di posa del modello sulla vista post: forma", tuple(model.pose_head(pst).shape))

# contesto temporale: si perturba il segnale nella sola patch 80 (campioni 400..404 a 200 Hz)
w2 = w.copy()
w2[0, 400:405, :] += 5.0
out2, _, _, _ = views(w2)
dpre = (out2["sequences"]["pre"] - pre).abs().amax(dim=(0, 2)).numpy()
dpost = (out2["sequences"]["post"] - pst).abs().amax(dim=(0, 2)).numpy()
tol = 1e-6
ch_pre = np.nonzero(dpre > tol)[0]
ch_post = np.nonzero(dpost > tol)[0]
print("patch perturbata: 80. Patch di `pre` che cambiano (> 1e-6):", ch_pre.min() - 80, "..", ch_pre.max() - 80, f"({len(ch_pre)} patch)")
print("patch di `post` che cambiano (> 1e-6):", ch_post.min() - 80, "..", ch_post.max() - 80, f"({len(ch_post)} patch su {len(dpost)})")
big = np.nonzero(dpre > 1e-3 * dpre.max())[0]
print("patch di `pre` con variazione > 0,1% del massimo:", big.min() - 80, "..", big.max() - 80, f"({len(big)} patch)")
