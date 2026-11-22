"""SCETTICO TL-1 - GIOCATTOLO: contributo del ramo profondo = TCN meno gemello L0 (stesso flusso di shuffle), per epoca.
Uso: twin.py <testa L0|TCN_post> <lr> <seed> <epoche> [f32]   (punteggio = scala del cancello sugli esclusi, come toy_tl.run)"""
import sys, json
import numpy as np, torch
import toy_tl as K
from pose_temponet import heads, train as T
torch.backends.nnpack.set_flags(False)
head, lr, seed, epochs = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
x, fit, held, ystar, y, refs, a, thr = K.load()
data = T.TensorTrainingSet(torch.from_numpy(x[fit]), torch.from_numpy(y[fit]), torch.ones((len(fit), K.T_LEN-2*K.CROP, K.J), dtype=torch.bool), ["w%05d" % i for i in fit])
traj = T.Trajectory(lambda s, f: heads.make_head(head, seed=s), data, lr=lr, seed=seed, fold=0, device="cpu")
R = refs["selected"]["held"]; dyn = K.pose_var(ystar[held]) >= thr; margin = R["sse_cleanLS"]-R["sse_oracle"]
yc = y[held][:, K.CROP:-K.CROP]
def score():
    m = traj.model; mode = m.training; m.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(held), 64):
            out.append(m(torch.from_numpy(x[held[s:s+64]].astype(np.float32)))[:, K.CROP:-K.CROP].numpy().astype(np.float64))
    m.train(mode)
    s_n, ss, mse = K.edyn_stats(np.concatenate(out), yc, dyn)
    return 100*(R["sse_cleanLS"]-s_n)/margin, mse
EVERY = int(sys.argv[5]) if len(sys.argv) > 5 else 1
for ep in range(EVERY, epochs+1, EVERY):
    traj.train_to_step(ep*traj.steps_per_epoch)
    r, mse = score()
    print(json.dumps({"head": head, "lr": lr, "seed": seed, "epoch": ep, "rec_pct": round(r, 3), "mse": round(mse, 6)}), flush=True)
