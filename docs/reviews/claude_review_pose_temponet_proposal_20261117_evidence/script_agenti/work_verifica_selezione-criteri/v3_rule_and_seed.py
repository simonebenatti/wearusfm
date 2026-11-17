# (1) Enumerazione mia delle etichette di PROT:95-98 su una griglia fine; (2) conto analitico (normale) dell'effetto del rumore fra semi.
import itertools, numpy as np
from scipy import stats as st
lab = lambda e, L, U: [n for n, c in (('B', e >= 2 and L > 0), ('D', e <= -2 and U < 0), ('NS', U < 2), ('I', U >= 2 and not (e >= 2 and L > 0))) if c]
grid = np.round(np.arange(-4, 4.01, 0.5), 2)
cnt = {}; none = 0; tot = 0; coherent = 0
for e, L, U in itertools.product(grid, grid, grid):
    if L > U: continue
    tot += 1; ls = tuple(lab(e, L, U)); cnt[ls] = cnt.get(ls, 0) + 1; none += (len(ls) == 0)
for k in sorted(cnt, key=lambda k: -cnt[k]): print('etichette', k, cnt[k])
print('combinazioni', tot, 'senza etichetta', none)
coh = {}
for e, L, U in itertools.product(grid, grid, grid):
    if L <= e <= U:
        ls = tuple(lab(e, L, U)); coh[ls] = coh.get(ls, 0) + 1
print('solo L<=stima<=U:', coh)
# (2) normale: s_b = dev. bootstrap fra utenti; s_s = errore di seme della stima media (punti percentuali)
z = st.norm.ppf(0.995)
print('\ns_b  sigma_seed  s_s(1-3)  cop99(1-3)  P(NS|vero 2%)(1-3)  s_s(4-5)  cop99(4-5)  P(NS|vero 2%)(4-5)')
for sb in (0.17, 0.25, 0.46):
    for ss in (0.0, 0.25, 0.5, 1.0):
        out = [sb, ss]
        for f in (np.sqrt(2 / 3), np.sqrt(1 / 3)):
            s = ss * f; tot_sd = np.hypot(sb, s)
            out += [s, 2 * st.norm.cdf(z * sb / tot_sd) - 1, st.norm.cdf(-z * sb / tot_sd)]
        print('  '.join(f'{v:.3f}' for v in out))
