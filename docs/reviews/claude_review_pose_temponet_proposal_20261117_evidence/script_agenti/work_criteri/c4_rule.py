"""(c4) Con quale probabilita' scatta ciascuna etichetta della regola del protocollo. Simulazione, non dati.
Disegno: i 15 utenti veri con i loro pesi; bootstrap 2.000 repliche, seme 20261102; IC percentile 99%.
mu = miglioramento relativo vero (medio sui semi della testa). sigma_u = eterogeneita' fra utenti.
sigma_seed = deviazione standard relativa (in %) di E_dyn fra fit della stessa testa con semi diversi: fattore comune a
tutti gli utenti, tre semi per braccio; NON entra nel bootstrap sugli utenti (l'IC e' condizionato ai tre fit).
riferimento 'rifit' = L0/MLP/L13 (tre semi anche li'); 'storico' = ridge o testa storica (una sola predizione)."""
import sys, time, json
import numpy as np
import c_simlib as S
W = sys.argv[1]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
base = S.real_design(W)
n = len(base[0])
m = S.multiplicity(S.B_PROTOCOL, n, S.SEED_PROTOCOL)
t0 = time.time()
rows = []
print("rifer.  sigma_seed sigma_u   mu | P(beneficio) P(deterior.) P(non sostenuto) P(inconclusivo) | P(stima>=2) P(stima>=2 ma L<=0) | sd totale della stima")
for ref in ("rifit", "storico"):
    for sigma_seed in (0.0, 0.5, 1.0):
        for sigma_u in (0.7, 1.5, 3.0, 5.0):
            for mu in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
                rng = np.random.default_rng([20261117, 4, ("rifit", "storico").index(ref), int(sigma_seed*10), int(sigma_u*10), int(mu*10)])
                nref, nwin = S.make_design("reale", rng, R, base)
                r = S.user_effects(rng, R, n, mu, sigma_u, nwin)
                f_t = 1+rng.normal(0, sigma_seed/100, (R, 3)).mean(1, keepdims=True)
                f_r = 1+rng.normal(0, sigma_seed/100, (R, 3)).mean(1, keepdims=True) if ref == "rifit" else 1.0
                nc = nref*(1-r/100)*f_t
                nr = nref*f_r
                res = S.intervals(nc, nr, m, levels=(99.0,))
                est, lo, hi = res["est"], res["q"][0.5], res["q"][99.5]
                lab = S.label(est, lo, hi)
                p = {k: float(np.mean(lab == k)) for k in "BDNI"}
                rows.append(dict(ref=ref, sigma_seed=sigma_seed, sigma_u=sigma_u, mu=mu, **p, p_est_ge2=float(np.mean(est >= 2)),
                                 p_est_ge2_L_le0=float(np.mean((est >= 2) & (lo <= 0))), sd_est=float(est.std(ddof=1))))
                print(f"{ref:7s}   {sigma_seed:3.1f}      {sigma_u:3.1f}   {mu:3.1f} |   {p['B']:.4f}      {p['D']:.4f}       {p['N']:.4f}          {p['I']:.4f}      |   {np.mean(est >= 2):.4f}        {np.mean((est >= 2) & (lo <= 0)):.4f}         |   {est.std(ddof=1):.3f}")
print(f"tempo {time.time()-t0:.0f} s; R={R}; errore standard Monte Carlo di una probabilita': al piu' {0.5/np.sqrt(R):.4f}")
json.dump(rows, open(f"{W}/c4_rule.json", "w"), indent=1)
