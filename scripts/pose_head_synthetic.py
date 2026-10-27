"""Apprendibilita' della testa di posa su feature NOTE (controllo 0b della diagnosi di GPT del 27/10/2026, `docs/reviews/r3_dbg_diagnosis_20261027.md`,
dopo il collaudo dbg di R3 62429050 non superato per `posa_discesa`; correzioni dalla review `pose_audit_e198a87_review_20261027.md`).

La testa vera (`LayerNorm(d) -> Linear(d, 20)` a zero, `model/pose_head.py`) viene allenata con AdamW (lr di picco 1e-4, weight decay 0,05, clip
1,0: come il trainer) su un problema costruito: feature x ~ N(0, I_d) per istante; il bersaglio e' lineare nelle feature NORMALIZZATE
(LayerNorm senza affine, cioe' quello che la testa vede): y = LN(x) W* + rumore, con frazione spiegabile q (varianza del segnale q, del rumore
1 - q per giunto: varianza totale 1 come i bersagli standardizzati), cosi' 1 - q e' l'ottimo esatto della testa a feature fisse. B finestre per
passo di P istanti indipendenti (istanti per passo = B x P: nel dataset vero gli istanti di una finestra sono correlati e il numero efficace non e'
misurato), perdita = errore quadratico medio come `pose_terms`. Schedule PRESI DAL TRAINER (`run._lr_at` su `with_wsd`): `dbg` = quello del collaudo
(total_steps = steps_at_d = 1.300: warmup 1.000, decadimento dal passo 1.042; dose 671 passi di picco); `r3` = quello del training vero (total 49.046,
warmup 1.962, poi costante per i primi 5.000 passi). Misura la MSE su un insieme fisso. Serve a CALIBRARE il controllo: non prevede il dataset vero
ne' identifica la causa nel training con encoder mobile e ancore.

Uso: python3 scripts/pose_head_synthetic.py [--steps-dbg 1300 --steps-r3 5000 --seeds 3 --qs 1.0,0.5,0.2,0.0 --bs 7,28 --ps 10,1 --out tabella.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

LR_PEAK, WEIGHT_DECAY, GRAD_CLIP = 1e-4, 0.05, 1.0
N_JOINTS = 20
DBG_TOTAL_STEPS, R3_TOTAL_STEPS, R3_STEPS_AT_D = 1300, 49046, 98094


@lru_cache(maxsize=None)
def schedule_cfg(schedule: str):
    """La RunConfig del trainer con lo schedule WSD del collaudo dbg o di R3 (stesso codice del training, `run.with_wsd` / `wsd_warmup_steps`)."""
    from wearusfm.training import run as R

    base = replace(R.sanity_config(), lr=LR_PEAK)
    if schedule == "dbg":
        return R.with_wsd(base, DBG_TOTAL_STEPS, warmup_steps=R.wsd_warmup_steps(DBG_TOTAL_STEPS))
    if schedule == "r3":
        return R.with_wsd(base, R3_TOTAL_STEPS, warmup_steps=R.wsd_warmup_steps(R3_STEPS_AT_D))
    raise ValueError(schedule)


def lr_at(step: int, schedule: str) -> float:
    """lr dell'aggiornamento `step` (1-based), esattamente come il trainer (`_lr_at(step - 1, cfg)`)."""
    from wearusfm.training import run as R

    return R._lr_at(step - 1, schedule_cfg(schedule))


def simulate(q: float, B: int, schedule: str, seed: int, steps: int, *, d: int = 384, P: int = 10, eval_every: int = 500, n_eval: int = 256) -> dict:
    import torch
    import torch.nn.functional as F

    from wearusfm.model.pose_head import PoseHead

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    w_star = torch.randn(d, N_JOINTS, generator=g)
    w_star = w_star / w_star.norm(dim=0, keepdim=True) * math.sqrt(max(q, 0.0))  # Var(LN(x) w) = q per giunto (LN(x) ha varianza ~1 per coordinata)
    noise_sd = math.sqrt(max(1.0 - q, 0.0))

    def draw(n):
        x = torch.randn(n, P, d, generator=g)
        y = F.layer_norm(x, (d,)) @ w_star + noise_sd * torch.randn(n, P, N_JOINTS, generator=g)
        return x, y

    x_eval, y_eval = draw(n_eval)
    head = PoseHead(d)
    opt = torch.optim.AdamW(head.parameters(), lr=LR_PEAK, weight_decay=WEIGHT_DECAY)

    def mse_eval():
        with torch.no_grad():
            return float(((head(x_eval) - y_eval) ** 2).mean())

    curve = {0: mse_eval()}
    first_10, dose = None, 0.0
    for step in range(1, steps + 1):
        lr = lr_at(step, schedule)
        dose += lr / LR_PEAK
        for grp in opt.param_groups:
            grp["lr"] = lr
        x, y = draw(B)
        loss = ((head(x) - y) ** 2).mean(dim=(1, 2)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
        opt.step()
        if step % eval_every == 0 or step == steps:
            curve[step] = mse_eval()
            if first_10 is None and curve[step] <= 0.9 * curve[0]:
                first_10 = step
    return {"q": q, "B": B, "P": P, "instants_per_step": B * P, "schedule": schedule, "seed": seed, "steps": steps, "mse": curve, "mse_0": curve[0],
            "mse_end": curve[steps], "ratio_end": curve[steps] / curve[0], "first_step_with_10pct": first_10, "lr_dose_peak_steps": dose,
            "weight_norm": float(head.out.weight.detach().norm()), "optimum_ratio_at_fixed_features": 1.0 - q}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps-dbg", type=int, default=DBG_TOTAL_STEPS)
    ap.add_argument("--steps-r3", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--qs", default="1.0,0.5,0.2,0.0")
    ap.add_argument("--bs", default="7,28")
    ap.add_argument("--ps", default="10,1")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    rows = []
    for q in (float(v) for v in args.qs.split(",")):
        for B in (int(v) for v in args.bs.split(",")):
            for P in (int(v) for v in args.ps.split(",")):
                for schedule, steps in (("dbg", args.steps_dbg), ("r3", args.steps_r3)):
                    for seed in range(args.seeds):
                        r = simulate(q, B, schedule, seed, steps, P=P)
                        rows.append(r)
                        print(f"q={q:.2f} B={B:2d} P={P:2d} {schedule:3s} seme {seed}: MSE {r['mse_0']:.3f} -> {r['mse_end']:.3f} (x{r['ratio_end']:.3f}); "
                              f"10% al passo {r['first_step_with_10pct']}; dose lr {r['lr_dose_peak_steps']:.0f}", flush=True)
    if args.out:
        args.out.write_text(json.dumps({"rule": {"lr_peak": LR_PEAK, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "teacher": "lineare in LayerNorm(x)",
                                                 "schedules": "dal trainer: run.with_wsd + run._lr_at"},
                                        "scenari_eseguiti": {"qs": args.qs, "bs": args.bs, "ps": args.ps, "seeds": args.seeds, "steps_dbg": args.steps_dbg, "steps_r3": args.steps_r3},
                                        "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
