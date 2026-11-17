# Ridge e ridge con ritardi nel mondo V, una sola direzione (allena fold 0 = utenti [::2], legge fold 1), alpha scelto sul fold letto
# (diagnostica di imparabilita', ottimista) e anche alpha fisso 1e5/1e6; oracolo = bersaglio senza rumore.
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[1])
import gen_v as G
import calib_v as Cv
torch.set_num_threads(1)
d = G.make(G.P0, n_val_users=1, wpu=48)
ids = np.arange(40); tr = np.isin(d['utr'], ids[::2]); he = np.isin(d['utr'], ids[1::2])
for lg in [(0,), tuple(range(-6, 7, 2)), tuple(range(-6, 7))]:
    sol = Cv.solver(Cv.stats(d['xtr'][tr], d['ytr'][tr], lg))
    rows = []
    for e in range(3, 8):
        W_, _, mx, my = sol(10.0 ** e)
        p = Cv.predict(d['xtr'][he], (W_, None, mx, my), lg)
        m = Cv.metrics(p, d['ytr'][he], d['dtr'][he]); rows.append((e, round(m['mse'], 4), round(m['E_dyn'], 4)))
    print('ritardi', lg if len(lg) < 8 else 'tutti -6..6', rows, flush=True)
yc = d['yctr'][he][:, Cv.SL]
print('oracolo fold 1:', {k: round(v, 4) for k, v in Cv.metrics(yc, d['ytr'][he], d['dtr'][he]).items()})
