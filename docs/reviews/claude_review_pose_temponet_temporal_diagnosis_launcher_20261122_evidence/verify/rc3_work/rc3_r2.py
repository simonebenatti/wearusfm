"""RC-3: quota lineare (R2) del residuo di fitting nel mio giocattolo, stessa metrica del revisore (regressione WLS del residuo su z con intercetta). Uso: rc3_r2.py"""
import sys, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("toy", "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/verify/rc3_work/rc3_toy_lib.py")
toy = importlib.util.module_from_spec(spec); spec.loader.exec_module(toy)
from pose_temponet import linear
x, fit, c, F, CROP = toy.x, toy.fit, toy.c, toy.F, toy.CROP
y = c["L"]+toy.A*c["T"]+toy.ST*c["epsilon"]; mix_y = {"L": 1., "T": toy.A, "epsilon": toy.ST}
targets = {}
for alpha in (0., 0.1, 1.):
    r = np.zeros_like(y); r[fit, CROP:-CROP] = y[fit][:, CROP:-CROP]-F.predict_mixture(fit, mix_y, alpha)[0]
    targets["r%g" % alpha] = r
F2 = linear.WindowedBasisFit(x, targets, toy.full, fit, block_windows=16)
pred, _ = F2.predictions(fit, 0.)
for k, r in targets.items():
    rr = r[fit][:, CROP:-CROP]; sst = ((rr-rr.mean(axis=(0, 1)))**2).sum(); sse = ((rr-pred[k])**2).sum()
    print("base alpha %s: quota lineare (R2) del residuo di fitting %.3e | varianza del residuo %.5f" % (k[1:], 1-sse/sst, sst/rr.size))
