"""(c6) L'IC del protocollo e' condizionato ai tre fit. Quanto conta, e un bootstrap a due livelli (utenti e semi) lo ripara?
Simulazione, non dati. Verita' = mu, miglioramento relativo medio sui semi. Contrasto con un braccio rifittato (tre semi per braccio).
Bootstrap a due livelli: in ogni replica si ricampionano i 15 utenti e, per ciascun braccio, i 3 semi con reinserimento."""
import sys, time
import numpy as np
import c_simlib as S
W = sys.argv[1]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
base = S.real_design(W)
n = len(base[0])
B = S.B_PROTOCOL
m = S.multiplicity(B, n, S.SEED_PROTOCOL)
rs = np.random.default_rng(S.SEED_PROTOCOL+1)
def seedmult():
    d = rs.integers(0, 3, (B, 3)); mm = np.zeros((B, 3)); np.add.at(mm, (np.arange(B)[:, None], d), 1.0); return mm/3
sT, sR = seedmult(), seedmult()
t0 = time.time()
print("sigma_seed sigma_u  mu | solo utenti (protocollo): copertura IC99, P(B), P(N), P(I) | utenti+semi: copertura IC99, P(B), P(N), P(I) | semi concordi: P(B e i 3 semi >=2), P(N e i 3 semi con U<2)")
for sigma_seed in (0.0, 0.25, 0.5, 1.0):
    for sigma_u in (0.7, 1.5):
        for mu in (0.0, 1.0, 2.0, 3.0, 4.0):
            rng = np.random.default_rng([20261117, 6, int(sigma_seed*100), int(sigma_u*10), int(mu*10)])
            nref, nwin = S.make_design("reale", rng, R, base)
            r = S.user_effects(rng, R, n, mu, sigma_u, nwin)
            eT = 1+rng.normal(0, sigma_seed/100, (R, 3)); eR = 1+rng.normal(0, sigma_seed/100, (R, 3))
            nc0 = nref*(1-r/100)
            est = 100*(1-(nc0.sum(1)*eT.mean(1))/(nref.sum(1)*eR.mean(1)))
            lo1, hi1, lo2, hi2 = (np.empty(R) for _ in range(4))
            los, his = np.empty((R, 3)), np.empty((R, 3))
            ch = 2000
            for s in range(0, R, ch):
                sl = slice(s, s+ch)
                ratio = (nc0[sl] @ m.T)/(nref[sl] @ m.T)                       # (ch, B): solo utenti
                b1 = 100*(1-ratio*(eT[sl].mean(1)/eR[sl].mean(1))[:, None])
                lo1[sl], hi1[sl] = np.percentile(b1, [0.5, 99.5], axis=1)
                b2 = 100*(1-ratio*(eT[sl] @ sT.T)/(eR[sl] @ sR.T))             # utenti e semi
                lo2[sl], hi2[sl] = np.percentile(b2, [0.5, 99.5], axis=1)
                for k in range(3):                                             # ogni seme della TCN contro la media dei semi del riferimento
                    bk = 100*(1-ratio*(eT[sl, k]/eR[sl].mean(1))[:, None])
                    los[sl, k], his[sl, k] = np.percentile(bk, [0.5, 99.5], axis=1)
            est_k = 100*(1-(nc0.sum(1, keepdims=True)*eT)/(nref.sum(1, keepdims=True)*eR.mean(1, keepdims=True)))
            l1, l2 = S.label(est, lo1, hi1), S.label(est, lo2, hi2)
            stableB = (l1 == "B") & (est_k >= 2).all(1)
            stableN = (l1 == "N") & (his < 2).all(1)
            c1 = np.mean((lo1 <= mu) & (mu <= hi1)); c2 = np.mean((lo2 <= mu) & (mu <= hi2))
            print(f"   {sigma_seed:4.2f}     {sigma_u:3.1f}   {mu:3.1f} |   {c1:.4f}  {np.mean(l1 == 'B'):.4f}  {np.mean(l1 == 'N'):.4f}  {np.mean(l1 == 'I'):.4f}   |   {c2:.4f}  {np.mean(l2 == 'B'):.4f}  {np.mean(l2 == 'N'):.4f}  {np.mean(l2 == 'I'):.4f}   |   {stableB.mean():.4f}   {stableN.mean():.4f}")
print(f"tempo {time.time()-t0:.0f} s; R={R}")
