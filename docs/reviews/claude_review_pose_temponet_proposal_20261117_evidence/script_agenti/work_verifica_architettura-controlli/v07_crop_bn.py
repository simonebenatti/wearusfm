import sys
sys.path.insert(0, "/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-temponet/src")
import torch
from torch import nn
import torch.nn.functional as F
from wearusfm.model import temponet as TN
torch.set_default_dtype(torch.float64)
# TCN funzionale con padding a scelta: 'zeros' | 'replicate' | 'reflect' | 'circular' | 'none' (convoluzione valida)
def make_params(seed):
    g = torch.Generator().manual_seed(seed)
    shapes = [(64, 384, 1), (64, 64, 3), (64, 64, 3), (64, 64, 3), (32, 64, 1), (20, 32, 1)]
    return [(torch.randn(*s, generator=g) / (s[1] * s[2]) ** 0.5, torch.randn(s[0], generator=g) * 0.1) for s in shapes]
def tcn_f(x, P, mode):
    # x: (B, 384, T) gia' normalizzato
    dil = [1, 1, 2, 3, 1, 1]
    for i, (w, b) in enumerate(P):
        d = dil[i]; k = w.shape[2]; pad = d * (k - 1) // 2
        if pad and mode != "none":
            x = F.pad(x, (pad, pad), mode="constant" if mode == "zeros" else mode)
        x = F.conv1d(x, w, b, dilation=d)
        if i < 5: x = F.relu(x)
    return x
def l13_f(x, w, b, mode):
    if mode != "none":
        x = F.pad(x, (6, 6), mode="constant" if mode == "zeros" else mode)
    return F.conv1d(x, w, b)
g = torch.Generator().manual_seed(3)
x = F.layer_norm(torch.randn(2, 160, 384, generator=g), (384,)).transpose(1, 2).contiguous()
P = make_params(0)
w13, b13 = torch.randn(20, 384, 13, generator=g) / (384 * 13) ** 0.5, torch.randn(20, generator=g)
ref_t = tcn_f(x, P, "none")          # lunghezza 148 = centri 6..153
ref_l = l13_f(x, w13, b13, "none")
print("lunghezza uscita valida:", ref_t.shape[-1], ref_l.shape[-1])
for mode in ("zeros", "replicate", "reflect", "circular"):
    yt = tcn_f(x, P, mode); yl = l13_f(x, w13, b13, mode)
    print("padding %-9s: TCN centri 6..153 max|diff| = %.2e ; L13 = %.2e ; TCN bordi 0..5 max|diff vs zeros| = %.2e" % (
        mode, (yt[..., 6:154] - ref_t).abs().max(), (yl[..., 6:154] - ref_l).abs().max(),
        (yt[..., :6] - tcn_f(x, P, "zeros")[..., :6]).abs().max()))
# finestra con 100 patch reali: riempimento in due modi, quali centri cambiano
xa = x.clone(); xb = x.clone(); xb[..., 100:] = torch.randn(2, 384, 60, generator=g)
xa[..., 100:] = 0.0
da = (tcn_f(xa, P, "zeros") - tcn_f(xb, P, "zeros")).abs().amax((0, 1))
dl = (l13_f(xa, w13, b13, "zeros") - l13_f(xb, w13, b13, "zeros")).abs().amax((0, 1))
print("N=100: primo centro TCN dipendente dal riempimento:", int((da > 0).nonzero()[0]), "; L13:", int((dl > 0).nonzero()[0]))
# BatchNorm di TEMPONet: dipendenza fra campioni del batch
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
m = TN.TEMPONet(in_channels=384, window=160, out_dim=20, causal=False)
for mode in ("train", "eval"):
    getattr(m, mode)()
    xx = torch.randn(4, 384, 160, requires_grad=True)
    if mode == "train":
        torch.manual_seed(1)
    y = m(xx)
    gsum = torch.autograd.grad(y[0].sum(), xx)[0][1].abs().sum()
    print("TEMPONet %s: |d uscita campione0 / d ingresso campione1| = %.4g" % (mode, gsum))
