"""(c1) Copertura dell'IC percentile 99% con 15 utenti. Simulazione, non dati.
Per ogni scenario: R dataset; in ognuno 15 utenti con miglioramento relativo vero medio mu ed eterogeneita' sigma_u;
bootstrap sugli utenti con 2.000 repliche e seme 20261102 (come nel protocollo). Verita' = mu (effetti indipendenti dai pesi)."""
import sys, time, json
import numpy as np
import c_simlib as S
W = sys.argv[1]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
base = S.real_design(W)
n = len(base[0])
m_prot = S.multiplicity(S.B_PROTOCOL, n, S.SEED_PROTOCOL)
w = base[0]/base[0].sum()
print(f"utenti={n}; finestre dinamiche per utente min={int(base[2].min())} max={int(base[2].max())}; numero efficace di utenti (1/somma w^2)={1/np.sum(w**2):.2f}")
rows = []
t0 = time.time()
print("disegno       mu  sigma_u code     | copertura IC99  manca_sotto(L>mu) manca_sopra(U<mu) | copertura IC95 | sd_boot medio | semiampiezza media IC99")
for design in ("reale", "ricampionato", "uguali", "sbilanciato"):
    for tails in ("normale", "t3"):
        for sigma_u in (0.7, 1.5, 3.0, 5.0):
            for mu in (0.0, 2.0, 4.0):
                if tails == "t3" and design not in ("reale",):
                    continue
                if design == "sbilanciato" and sigma_u not in (1.5,):
                    continue
                rng = np.random.default_rng([20261117, ("reale", "ricampionato", "uguali", "sbilanciato").index(design), ("normale", "t3").index(tails), int(sigma_u*10), int(mu*10)])
                nref, nwin = S.make_design(design, rng, R, base)
                r = S.user_effects(rng, R, n, mu, sigma_u, nwin, tails=tails)
                ncand = nref*(1-r/100)
                res = S.intervals(ncand, nref, m_prot, levels=(99.0, 95.0), want_sd=True)
                lo, hi = res["q"][0.5], res["q"][99.5]
                lo95, hi95 = res["q"][2.5], res["q"][97.5]
                cov = np.mean((lo <= mu) & (mu <= hi)); ml = np.mean(lo > mu); mh = np.mean(hi < mu)
                cov95 = np.mean((lo95 <= mu) & (mu <= hi95))
                rows.append(dict(design=design, tails=tails, sigma_u=sigma_u, mu=mu, cov99=cov, miss_low=ml, miss_high=mh, cov95=cov95,
                                 sd_boot=float(res["sd"].mean()), half99=float(((hi-lo)/2).mean()), sd_est=float(res["est"].std(ddof=1)),
                                 bias=float(res["est"].mean()-mu)))
                print(f"{design:12s} {mu:4.1f} {sigma_u:5.1f}   {tails:8s} |   {cov:.4f}         {ml:.4f}            {mh:.4f}       |   {cov95:.4f}     |   {res['sd'].mean():.3f}       | {((hi-lo)/2).mean():.3f}   (sd vera della stima {res['est'].std(ddof=1):.3f}, distorsione {res['est'].mean()-mu:+.4f})")
print(f"tempo {time.time()-t0:.0f} s; R={R} dataset per scenario; errore standard Monte Carlo di una copertura vicina a 0,97: {np.sqrt(0.97*0.03/R):.4f}")
json.dump(rows, open(f"{W}/c1_coverage.json", "w"), indent=1)
