"""Prova sintetica del revisore (NON sui dati del progetto): la ricetta del paragrafo 4 riesce ad allenare le teste del paragrafo 3
quando nei dati c'e' davvero una struttura temporale non lineare? E i controlli L0/MLP/L13 la separano?
Dati finti: stato liscio s_t (8 dimensioni), feature 384 = m + A s_t + rumore, bersaglio (20 giunti) = parte lineare istantanea
+ parte lineare temporale (s_{t+3}-s_{t-3}) + parte non lineare istantanea (s_t^2-1) + parte non lineare temporale (s_{t-4}*s_{t+4}) + rumore.
Scala ridotta per stare in pochi minuti su una CPU: finestre da 40 patch (28 centri col crop6), 960 finestre di train (30 passi per epoca,
come un fold da ~950 finestre), 320 tenute fuori. Tutto il resto come nel protocollo: AdamW (0.9, 0.999), eps 1e-8, wd 0.01 solo su
matrici/kernel, clipping 1, batch 32, LR costante {1e-4, 1e-3}, 50 epoche, rimescolamento per epoca, ultimo strato a zero."""
import sys, time, json
sys.path.insert(0, "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_architettura")
import torch
from torch import nn
import torch.nn.functional as F
from heads import L0, MLP, TCN, L13, D, J, last_layer

T, R, NTR, NHE, CROP = 40, 8, 960, 320, 6
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 50


def smooth_state(n, gen):
    w = torch.randn(n, R, T + 48, generator=gen)
    k = torch.exp(-torch.arange(-12, 13).float() ** 2 / (2 * 4.0 ** 2)); k /= k.norm()
    s = F.conv1d(w.reshape(n * R, 1, -1), k[None, None]).reshape(n, R, -1)[:, :, 12:12 + T]  # (n,R,T), varianza 1
    return s.transpose(1, 2).contiguous()  # (n,T,R)


def make(n, gen, A, m, Ws, sigma_x=1.0):
    s = smooth_state(n, gen)
    x = m + s @ A.T + sigma_x * torch.randn(n, T, D, generator=gen)
    lin = s @ Ws[0].T
    dif = (torch.roll(s, -3, 1) - torch.roll(s, 3, 1)) @ Ws[1].T
    sq = (s ** 2 - 1) @ Ws[2].T
    prod = (torch.roll(s, 4, 1) * torch.roll(s, -4, 1)) @ Ws[3].T
    parts = [lin, dif, sq, prod]
    c = slice(CROP, T - CROP)  # solo i centri: roll non contamina (|shift| <= 4 < 6)
    parts = [p / p[:, c].std() for p in parts]
    y = sum(parts) + torch.randn(n, T, J, generator=gen)
    return x, y, parts


def loss_fn(pred, y):
    c = slice(CROP, T - CROP)
    return (pred[:, c] - y[:, c]).pow(2).mean(dim=(1, 2)).mean()


def groups(m):
    decay, no_decay = [], []
    for mod in m.modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d)):
            decay.append(mod.weight); no_decay.append(mod.bias)
        elif isinstance(mod, nn.LayerNorm):
            no_decay += [mod.weight, mod.bias]
    return [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def held_mse(m, x, y):
    m.eval()
    tot = 0.0
    for i in range(0, len(x), 64):
        p = m(x[i:i + 64])
        tot += float((p[:, CROP:T - CROP] - y[i:i + 64, CROP:T - CROP]).pow(2).mean(dim=(1, 2)).sum())
    m.train()
    return tot / len(x)


@torch.no_grad()
def init_stats(m, x):
    """Per ogni strato nascosto (dopo la ReLU): quota di attivazioni accese, quota di unita' mai accese, quota di unita' sempre accese."""
    m.eval()
    acts = []
    hooks = [mod.register_forward_hook(lambda mod, i, o: acts.append(o)) for mod in m.modules() if isinstance(mod, nn.ReLU)]
    m(x[:64])
    for h in hooks:
        h.remove()
    out = []
    for a in acts:
        a = a if a.shape[-1] in (96, 64, 32) and a.dim() == 3 and a.shape[1] == T else a.transpose(1, 2)  # -> (B,T,C)
        on = a > 0
        per_unit = on.float().mean(dim=(0, 1))
        var_pos = a.reshape(-1, a.shape[-1]).var(dim=0)
        msq = a.reshape(-1, a.shape[-1]).pow(2).mean(dim=0)
        out.append({"larghezza": a.shape[-1], "accese": round(float(on.float().mean()), 3), "mai_accese": round(float((per_unit == 0).float().mean()), 3),
                    "sempre_accese": round(float((per_unit == 1).float().mean()), 3),
                    "quota_variabile": round(float(var_pos.sum() / msq.sum().clamp_min(1e-30)), 3), "rms": float(msq.mean().sqrt())})
    m.train()
    return out


def fit(cls, lr, xtr, ytr, xhe, yhe, seed=0):
    torch.manual_seed(seed)
    m = cls().train()
    opt = torch.optim.AdamW(groups(m), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    shuffle = torch.Generator().manual_seed(1000 + seed)
    curve = []
    for ep in range(EPOCHS):
        perm = torch.randperm(len(xtr), generator=shuffle)
        for i in range(0, len(xtr), 32):
            ix = perm[i:i + 32]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(m(xtr[ix]), ytr[ix])
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        curve.append(held_mse(m, xhe, yhe))
    return curve, m


def main():
    results = {}
    for regime, offset_scale in (("senza componente comune", 0.0), ("componente comune dominante", 3.0)):
        gen = torch.Generator().manual_seed(20261117)
        A = torch.randn(D, R, generator=gen) / R ** 0.5
        m_vec = offset_scale * torch.randn(D, generator=gen)
        Ws = [torch.randn(J, R, generator=gen) for _ in range(4)]
        xtr, ytr, _ = make(NTR, gen, A, m_vec, Ws)
        xhe, yhe, parts = make(NHE, gen, A, m_vec, Ws)
        mu, sd = ytr[:, CROP:T - CROP].mean(dim=(0, 1)), ytr[:, CROP:T - CROP].std(dim=(0, 1))
        ytr, yhe = (ytr - mu) / sd, (yhe - mu) / sd
        c = slice(CROP, T - CROP)
        tot = sum(float(p[:, c].var()) for p in parts) + 1.0
        print(f"\n===== regime: {regime} =====")
        print("quota di varianza del bersaglio per parte (lineare ist., lineare temp., non lin. ist., non lin. temp., rumore):",
              [round(float(p[:, c].var()) / tot, 3) for p in parts] + [round(1.0 / tot, 3)], flush=True)
        results[regime] = {}
        for name, cls in (("TCN", TCN), ("MLP", MLP)):
            torch.manual_seed(0)
            st = init_stats(cls(), xtr)
            print(f"  attivazioni all'inizializzazione, {name} (seme 0, 64 finestre):")
            for i, s in enumerate(st, 1):
                print(f"    hidden {i} (larghezza {s['larghezza']}): accese {s['accese']}, unita' mai accese {s['mai_accese']}, sempre accese {s['sempre_accese']}, quota variabile fra posizioni {s['quota_variabile']}, rms {s['rms']:.4f}")
        for name, cls in (("L0", L0), ("MLP", MLP), ("L13", L13), ("TCN", TCN)):
            for lr in (1e-4, 1e-3):
                t0 = time.perf_counter()
                curve, _ = fit(cls, lr, xtr, ytr, xhe, yhe)
                best = min(range(len(curve)), key=lambda i: curve[i])
                results[regime][f"{name}@{lr}"] = curve
                print(f"  {name:4s} LR {lr:g}: MSE tenuta fuori epoca 1 = {curve[0]:.4f}, 10 = {curve[min(9, len(curve) - 1)]:.4f}, 25 = {curve[min(24, len(curve) - 1)]:.4f}, ultima ({len(curve)}) = {curve[-1]:.4f}; minimo {curve[best]:.4f} all'epoca {best + 1}  [{time.perf_counter() - t0:.0f} s]", flush=True)
    json.dump(results, open("/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_architettura/h_synthetic_trainability.json", "w"))


if __name__ == "__main__":
    main()
