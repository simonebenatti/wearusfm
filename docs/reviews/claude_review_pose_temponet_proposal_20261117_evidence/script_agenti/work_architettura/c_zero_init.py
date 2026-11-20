"""Dimensione 1, punto (c): ultimo strato a zero -> gradienti interni nulli al primo backward, non nulli dopo un passo di AdamW.
Ricetta del paragrafo 4: AdamW beta (0.9, 0.999), eps 1e-8, wd 0.01 sulle sole matrici/kernel, clipping norma 1, batch 32,
loss = media dell'errore quadratico standardizzato sugli elementi validi della finestra (centri del crop6), poi media sulle finestre."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/work_architettura")
import torch
from torch import nn
from heads import BUILD, D, J, P, last_layer


def loss_fn(pred, target, mask):
    c = slice(6, P - 6)
    err = (pred[:, c] - target[:, c]).pow(2) * mask[:, c]
    den = mask[:, c].sum(dim=(1, 2))
    ok = den > 0
    return (err.sum(dim=(1, 2))[ok] / den[ok]).mean()


def groups(m):
    decay, no_decay = [], []
    for mod in m.modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d)):
            decay.append(mod.weight); no_decay.append(mod.bias)
        elif isinstance(mod, nn.LayerNorm):
            no_decay += [mod.weight, mod.bias]
    return [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}]


def grad_report(m):
    last = last_layer(m)
    last_ids = {id(last.weight), id(last.bias)}
    inner = [(n, p) for n, p in m.named_parameters() if id(p) not in last_ids]
    inner_max = max((float(p.grad.abs().max()) for _, p in inner), default=0.0) if inner else 0.0
    inner_nonzero = sum(int((p.grad != 0).sum()) for _, p in inner)
    inner_total = sum(p.numel() for _, p in inner)
    finite = all(bool(torch.isfinite(p.grad).all()) for p in m.parameters())
    return {"interni_max_abs": inner_max, "interni_non_nulli": inner_nonzero, "interni_totale": inner_total,
            "ultimo_peso_max_abs": float(last.weight.grad.abs().max()), "ultimo_bias_max_abs": float(last.bias.grad.abs().max()),
            "ultimo_peso_non_nulli": int((last.weight.grad != 0).sum()), "ultimo_peso_totale": last.weight.numel(), "finiti": finite}


for lr in (1e-3, 1e-4):
    print(f"\n===== LR {lr} =====")
    for name in ("L0", "MLP", "L13", "TCN_post"):
        torch.manual_seed(0)
        m = BUILD[name]().train()
        g = torch.Generator().manual_seed(11)
        x = torch.randn(32, P, D, generator=g)
        y = torch.randn(32, P, J, generator=g)
        mask = torch.rand(32, P, J, generator=g) > 0.1
        opt = torch.optim.AdamW(groups(m), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        out0 = m(x)
        print(f"{name:9s} uscita iniziale tutta zero: {bool((out0 == 0).all())}; loss iniziale = {float(loss_fn(out0, y, mask)):.6f}")
        snap = {n: p.detach().clone() for n, p in m.named_parameters()}
        for step in (1, 2, 3):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(m(x), y, mask)
            loss.backward()
            total_norm = float(nn.utils.clip_grad_norm_(m.parameters(), 1.0))
            r = grad_report(m)
            print(f"          backward {step}: loss={float(loss):.6f} norma_grad={total_norm:.4f} | interni: max|g|={r['interni_max_abs']:.3e}, non nulli {r['interni_non_nulli']}/{r['interni_totale']} | ultimo strato: max|g_w|={r['ultimo_peso_max_abs']:.3e} (non nulli {r['ultimo_peso_non_nulli']}/{r['ultimo_peso_totale']}), max|g_b|={r['ultimo_bias_max_abs']:.3e} | finiti={r['finiti']}")
            opt.step()
            if step == 1:
                last = last_layer(m)
                moved_inner = max(float((p.detach() - snap[n]).abs().max()) for n, p in m.named_parameters() if p is not last.weight and p is not last.bias)
                print(f"          dopo il passo 1: max |peso ultimo strato| = {float(last.weight.abs().max()):.3e}, pesi ultimo strato non nulli {int((last.weight != 0).sum())}/{last.weight.numel()}; spostamento massimo dei parametri interni (solo weight decay) = {moved_inner:.3e}")
