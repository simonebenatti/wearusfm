"""Apprendibilita' della testa di posa su feature NOTE (controllo 0b della diagnosi di GPT del 27/10/2026, `docs/reviews/r3_dbg_diagnosis_20261027.md`,
dopo il collaudo dbg di R3 62429050 non superato per `posa_discesa`).

La testa vera (`LayerNorm(d) -> Linear(d, 20)` a zero, `model/pose_head.py`) viene allenata con AdamW (lr di picco 1e-4, weight decay 0,05, clip
1,0: come il trainer) su un problema costruito: feature isotrope x ~ N(0, I_d) per istante, bersaglio y = x W* + rumore con frazione spiegabile q
(varianza del segnale q, del rumore 1 - q, per giunto: varianza totale 1 come i target standardizzati), B finestre per passo di P istanti,
perdita = errore quadratico medio come `pose_terms`. Schedule: `dbg` = quello del collaudo (warmup lineare 1e-7 -> 1e-4 in 1.000 passi, plateau
fino al passo 1.040, decadimento lineare a ~0 al 1.300); `r3` = quello del training vero (warmup lineare in 1.962 passi, poi costante).
Misura la MSE su un insieme fisso. Serve a CALIBRARE il controllo (quanti passi servono a feature fisse con q dato): non prevede il dataset vero,
dove feature, correlazioni fra istanti e q non sono noti (gli istanti qui sono indipendenti: B_eff reale e' piu' basso).

Uso: python3 scripts/pose_head_synthetic.py [--steps-dbg 1300 --steps-r3 5000 --seeds 3 --out tabella.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

LR_PEAK, WEIGHT_DECAY, GRAD_CLIP = 1e-4, 0.05, 1.0
N_JOINTS = 20


def lr_at(step: int, schedule: str) -> float:
    """lr PRIMA dell'aggiornamento `step` (1-based), come nel trainer: warmup lineare da ~0; dbg: plateau 1.000-1.040 e decadimento lineare
    su 260 passi; r3: warmup 1.962 poi costante (il decadimento di R3 sta oltre i 49.000 passi)."""
    if schedule == "dbg":
        if step <= 1000:
            return LR_PEAK * step / 1000
        if step <= 1040:
            return LR_PEAK
        return LR_PEAK * max(0.0, 1.0 - (step - 1040) / 260)
    if schedule == "r3":
        return LR_PEAK * min(1.0, step / 1962)
    raise ValueError(schedule)


def simulate(q: float, B: int, schedule: str, seed: int, steps: int, *, d: int = 384, P: int = 10, eval_every: int = 500,
             n_eval: int = 256) -> dict:
    import torch

    from wearusfm.model.pose_head import PoseHead

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    w_star = torch.randn(d, N_JOINTS, generator=g)
    w_star = w_star / w_star.norm(dim=0, keepdim=True) * math.sqrt(max(q, 0.0))  # Var(x w) = q per giunto
    noise_sd = math.sqrt(max(1.0 - q, 0.0))

    def draw(n):
        x = torch.randn(n, P, d, generator=g)
        y = x @ w_star + noise_sd * torch.randn(n, P, N_JOINTS, generator=g)
        return x, y

    x_eval, y_eval = draw(n_eval)
    head = PoseHead(d)
    opt = torch.optim.AdamW(head.parameters(), lr=LR_PEAK, weight_decay=WEIGHT_DECAY)

    def mse_eval():
        with torch.no_grad():
            return float(((head(x_eval) - y_eval) ** 2).mean())

    curve = {0: mse_eval()}
    first_10 = None
    for step in range(1, steps + 1):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, schedule)
        x, y = draw(B)
        loss = ((head(x) - y) ** 2).mean(dim=(1, 2)).mean()  # media per finestra, poi fra le finestre (come pose_terms)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
        opt.step()
        if step % eval_every == 0 or step == steps:
            curve[step] = mse_eval()
            if first_10 is None and curve[step] <= 0.9 * curve[0]:
                first_10 = step
    return {"q": q, "B": B, "schedule": schedule, "seed": seed, "steps": steps, "mse": curve, "mse_0": curve[0], "mse_end": curve[steps],
            "ratio_end": curve[steps] / curve[0], "first_step_with_10pct": first_10, "lr_dose_peak_steps": sum(lr_at(s, schedule) for s in range(1, steps + 1)) / LR_PEAK,
            "weight_norm": float(head.out.weight.detach().norm()), "instants_per_step": B * P, "optimum_ratio_at_fixed_features": 1.0 - q}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps-dbg", type=int, default=1300)
    ap.add_argument("--steps-r3", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--qs", default="1.0,0.5,0.2,0.0")
    ap.add_argument("--bs", default="7,28")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    rows = []
    for q in (float(v) for v in args.qs.split(",")):
        for B in (int(v) for v in args.bs.split(",")):
            for schedule, steps in (("dbg", args.steps_dbg), ("r3", args.steps_r3)):
                for seed in range(args.seeds):
                    r = simulate(q, B, schedule, seed, steps)
                    rows.append(r)
                    print(f"q={q:.2f} B={B:2d} {schedule:3s} seme {seed}: MSE {r['mse_0']:.3f} -> {r['mse_end']:.3f} (x{r['ratio_end']:.3f}); "
                          f"10% al passo {r['first_step_with_10pct']}; dose lr {r['lr_dose_peak_steps']:.0f} passi di picco", flush=True)
    if args.out:
        args.out.write_text(json.dumps({"rule": {"lr_peak": LR_PEAK, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP}, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
