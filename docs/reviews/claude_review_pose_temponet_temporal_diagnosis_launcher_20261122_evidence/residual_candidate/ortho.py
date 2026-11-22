"""Il residuo in-sample della base e' ortogonale a z? (giocattolo, fold 0). Quota lineare del residuo di fitting: R2 di una WLS del residuo su z."""
import numpy as np, torch
torch.set_num_threads(1)
W = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/residual_candidate/toy/"
d = np.load(W+"data_f0.npz"); x, fit, y = d["x"], d["fit"], d["y"].astype(np.float64)
D = 384
for kind in ("wls", "ridge"):
    r = y[fit][:, 6:-6]-d[kind+"_fit"].astype(np.float64)
    G = np.zeros((D+1, D+1)); B = np.zeros((D+1, 20)); ss = 0.0; mean = r.mean(axis=(0, 1))
    for s in range(0, len(fit), 32):
        z = torch.nn.functional.layer_norm(torch.from_numpy(x[fit[s:s+32]]).float(), (D,), eps=1e-5).numpy().astype(np.float64)[:, 6:-6].reshape(-1, D)
        z = np.concatenate([z, np.ones((len(z), 1))], 1); rr = r[s:s+32].reshape(-1, 20)
        G += z.T@z; B += z.T@rr; ss += ((rr-mean)**2).sum()
    w, V = np.linalg.eigh(G); keep = w > w.max()*1e-11
    beta = (V[:, keep]/w[keep])@V[:, keep].T@B
    explained = float((beta*B).sum()-len(fit)*148*(mean**2).sum())
    print(kind, "quota lineare (R2) del residuo di fitting: %.3e" % (explained/ss), "| varianza residuo %.5f" % (ss/(len(fit)*148*20)))
