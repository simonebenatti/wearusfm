"""(c2) Stessi dataset simulati: IC percentile 99% a 2.000 repliche (protocollo) contro alternative.
Serve a dire quanto costa la sottocopertura e se un rimedio semplice esiste. Simulazione, non dati."""
import sys, time
import numpy as np
from scipy import stats
import c_simlib as S
W = sys.argv[1]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
base = S.real_design(W)
n = len(base[0])
m2k = S.multiplicity(2000, n, S.SEED_PROTOCOL)
m20k = S.multiplicity(20000, n, S.SEED_PROTOCOL)
t0 = time.time()
print("valori teorici con pesi uguali e dati normali: copertura attesa del percentile 99% = P(|t_14| < 2,5758*sqrt(14/15)) =",
      round(1-2*stats.t.sf(stats.norm.ppf(0.995)*np.sqrt(14/15), 14), 4), "; del percentile 95% =",
      round(1-2*stats.t.sf(stats.norm.ppf(0.975)*np.sqrt(14/15), 14), 4))
print("livello nominale del percentile che darebbe 99% vero con 15 utenti uguali: ", round(100*(1-2*stats.norm.sf(stats.t.ppf(0.995, 14)*np.sqrt(15/14))), 3), "%")
def cov(lo, hi, mu): return float(np.mean((lo <= mu) & (mu <= hi)))
for design, tails, sigma_u in (("reale", "normale", 0.7), ("reale", "normale", 3.0), ("reale", "t3", 1.5), ("uguali", "normale", 1.5), ("sbilanciato", "normale", 1.5)):
    mu = 2.0
    rng = np.random.default_rng([20261117, 2, ("reale", "ricampionato", "uguali", "sbilanciato").index(design), ("normale", "t3").index(tails), int(sigma_u*10)])
    nref, nwin = S.make_design(design, rng, R, base)
    r = S.user_effects(rng, R, n, mu, sigma_u, nwin, tails=tails)
    nc = nref*(1-r/100)
    a = S.intervals(nc, nref, m2k, levels=(99.0,))
    b = S.intervals(nc, nref, m20k, levels=(99.0, 99.8))
    jl, jh, se = S.jackknife_t(nc, nref, 99.0)
    bl, bh = S.bca(nc, nref, m20k, 99.0)
    print(f"[{design}, code {tails}, sigma_u={sigma_u}, mu={mu}] copertura: percentile99 B=2000 {cov(a['q'][0.5], a['q'][99.5], mu):.4f} | percentile99 B=20000 {cov(b['q'][0.5], b['q'][99.5], mu):.4f} | "
          f"percentile nominale 99,8% B=20000 {cov(b['q'][0.1], b['q'][99.9], mu):.4f} | BCa99 B=20000 {cov(bl, bh, mu):.4f} | t di Student con errore standard jackknife (14 gdl) {cov(jl, jh, mu):.4f}")
    print(f"      semiampiezza media: percentile99 {((a['q'][99.5]-a['q'][0.5])/2).mean():.3f} | percentile 99,8% {((b['q'][99.9]-b['q'][0.1])/2).mean():.3f} | BCa {((bh-bl)/2).mean():.3f} | t-jackknife {((jh-jl)/2).mean():.3f}")
print(f"tempo {time.time()-t0:.0f} s; R={R}; errore standard Monte Carlo di una copertura: circa {np.sqrt(0.98*0.02/R):.4f}")
