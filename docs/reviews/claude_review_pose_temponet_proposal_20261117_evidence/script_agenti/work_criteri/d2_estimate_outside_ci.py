"""(d2) Quanto spesso la stima puntuale cade fuori dall'IC percentile 99% (caso in cui "beneficio" e "non sostenuto"
possono valere insieme). Simulazione con le ipotesi di c_simlib (pesi veri dei 15 utenti, 2.000 repliche, seme 20261102).
Uso: python d2_estimate_outside_ci.py <cartella con le ricevute>"""
import sys
import numpy as np
import c_simlib as S

W = sys.argv[1]
R = 20000
base = S.real_design(W)
n = len(base[0])
m = S.multiplicity(S.B_PROTOCOL, n, S.SEED_PROTOCOL)
print("mu sigma_u code | P(stima > limite sup.) | P(stima < limite inf.) | P(stima >= 2 e limite sup. < 2)")
for tails in ("normale", "t3"):
    for sigma_u in (0.7, 1.5, 3.0, 5.0):
        for mu in (2.0, 4.0):
            rng = np.random.default_rng([20261117, 22, ("normale", "t3").index(tails), int(sigma_u*10), int(mu*10)])
            nref, nwin = S.make_design("reale", rng, R, base)
            r = S.user_effects(rng, R, n, mu, sigma_u, nwin, tails=tails)
            res = S.intervals(nref*(1-r/100), nref, m, levels=(99.0,))
            est, lo, hi = res["est"], res["q"][0.5], res["q"][99.5]
            print(f"{mu:3.1f} {sigma_u:3.1f} {tails:7s} | {np.mean(est > hi):.5f} | {np.mean(est < lo):.5f} | {np.mean((est >= 2) & (hi < 2)):.5f}")
print(f"R={R} per scenario")
