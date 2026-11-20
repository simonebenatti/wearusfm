"""(c3) Copertura simultanea della famiglia di cinque contrasti con IC percentile 99% ciascuno (Bonferroni).
Cinque riferimenti contro la stessa TCN: i contrasti condividono il numeratore, quindi sono correlati (rho).
Simulazione, non dati. Disegno: i 15 utenti veri con i loro pesi."""
import sys, time
import numpy as np
import c_simlib as S
W = sys.argv[1]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
base = S.real_design(W)
n = len(base[0])
m = S.multiplicity(S.B_PROTOCOL, n, S.SEED_PROTOCOL)
t0 = time.time()
print("sigma_u  rho | copertura di un contrasto | copertura simultanea dei 5 (nominale 0,95) | limite di Bonferroni dal singolo | P(almeno un IC tutto sopra il vero) | P(almeno un IC tutto sotto il vero)")
for sigma_u in (0.7, 1.5, 3.0):
    for rho in (0.2, 0.5, 0.8):
        rng = np.random.default_rng([20261117, 3, int(sigma_u*10), int(rho*10)])
        mu = 2.0
        nb, nwin = S.make_design("reale", rng, R, base)
        sa, sb = sigma_u*np.sqrt(rho), sigma_u*np.sqrt(1-rho)
        a = S.user_effects(rng, R, n, mu, sa, nwin, sigma_w=10.0/np.sqrt(2))
        nt = nb*(1-a/100)
        ok = np.ones(R, bool); above = np.zeros(R, bool); below = np.zeros(R, bool); single = []
        for c in range(5):
            b = S.user_effects(rng, R, n, 0.0, sb, nwin, sigma_w=10.0/np.sqrt(2))
            res = S.intervals(nt, nb*(1-b/100), m, levels=(99.0,))
            lo, hi = res["q"][0.5], res["q"][99.5]
            truth = 100*(1-(1-mu/100)/1.0)
            cover = (lo <= truth) & (truth <= hi)
            single.append(cover.mean()); ok &= cover; above |= lo > truth; below |= hi < truth
        s = float(np.mean(single))
        print(f"  {sigma_u:3.1f}   {rho:3.1f} |        {s:.4f}             |            {ok.mean():.4f}                        |        {1-5*(1-s):.4f}             |          {above.mean():.4f}                 |          {below.mean():.4f}")
print(f"tempo {time.time()-t0:.0f} s; R={R}")
