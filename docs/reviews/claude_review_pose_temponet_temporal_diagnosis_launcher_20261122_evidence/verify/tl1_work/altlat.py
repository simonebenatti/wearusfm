"""SCETTICO TL-1 - GIOCATTOLO: il collasso delle ReLU dipende da come sono costruiti i latenti sintetici?
Uso: altlat.py <variante>   varianti: orig | seedB | nouser | common | common3 | white | k8
Generatore VERO (controls.generate_components, fold 0), a = 0,4 e sigma = 1,3 fissati (come la calibrazione reale), testa e ciclo VERI."""
import sys, json, time
import numpy as np, torch
from torch import nn
import toy_tl as K
from pose_temponet import controls, heads, train as T
torch.backends.nnpack.set_flags(False)
variant = sys.argv[1]
D, TL, J, C = K.D, K.T_LEN, K.J, K.CROP
N = K.N_USERS*K.WIN_PER_USER

def latents_custom(seed, Kc=32, pert=0.35, cu_scale=0.5, noise=0.15):
    rng = np.random.default_rng(seed)
    rho = np.linspace(0.60, 0.98, Kc); lam = 1.0/(1.0+np.arange(Kc)/4.0)
    A = rng.normal(size=(D, Kc))/np.sqrt(Kc)
    x = np.empty((N, TL, D), dtype=np.float16); users = []; i = 0
    for u in range(K.N_USERS):
        Au = A + pert*rng.normal(size=(D, Kc))/np.sqrt(Kc); cu = cu_scale*rng.normal(size=D)
        for w in range(K.WIN_PER_USER):
            s = np.empty((TL, Kc)); s[0] = rng.normal(size=Kc)
            innov = rng.normal(size=(TL, Kc))*np.sqrt(1-rho**2)
            for t in range(1, TL):
                s[t] = rho*s[t-1] + innov[t]
            s = s*lam + 0.3*rng.normal(size=Kc)*lam
            x[i] = (s@Au.T + cu + noise*rng.normal(size=(TL, D))).astype(np.float16); users.append("u%02d" % u); i += 1
    return x, users

ONLY = sys.argv[2] if len(sys.argv) > 2 else ""      # es. "0.003:1" = solo quel LR e quel seme
if variant in ("orig", "orig32"):
    x, users = K.make_latents()
elif variant == "seedB":
    x, users = K.make_latents(seed=54321)
elif variant == "k8":
    x, users = K.make_latents(seed=12345, K=8)
elif variant == "nouser":
    x, users = latents_custom(2468, pert=0.0, cu_scale=0.0)
elif variant in ("common", "common3"):
    x, users = K.make_latents()
    mvec = np.random.default_rng(999).normal(size=D)*(1.0 if variant == "common" else 3.0)
    for s in range(0, N, 64):
        x[s:s+64] = (x[s:s+64].astype(np.float32) + mvec.astype(np.float32)).astype(np.float16)
elif variant == "white":
    rng = np.random.default_rng(4242); x = np.empty((N, TL, D), dtype=np.float16)
    for s in range(0, N, 64):
        x[s:s+64] = rng.normal(size=(len(x[s:s+64]), TL, D)).astype(np.float16)
    users = ["u%02d" % (i//K.WIN_PER_USER) for i in range(N)]
else:
    raise SystemExit("variante sconosciuta")

fit = np.flatnonzero(np.array([int(u[1:]) < 20 for u in users])); held = np.flatnonzero(np.array([int(u[1:]) >= 20 for u in users]))
meta = [{"user": users[i], "window": i} for i in range(N)]
support = np.ones((N, TL-2*C, J), dtype=bool)
c = controls.generate_components(x, meta, support, fit, held, sorted(set(users)), outer_fold=0, block_windows=16)
a, sigma = 0.4, 1.3
if variant == "orig32":       # stesso arrotondamento del file npz dell'altro revisore (componenti in float32 prima della somma)
    c = dict(c); 
    for k in ("L", "T", "epsilon"):
        c[k] = c[k].astype(np.float32).astype(np.float64)
ystar = c["L"]+a*c["T"]; y = ystar+sigma*c["epsilon"]
z = K.ln64(x[fit[::8]])                                   # 115 finestre: descrittori dei latenti dopo LayerNorm
gm = z.mean(axis=(0, 1)); um = np.stack([z[i].mean(0) for i in range(len(z))])
desc = {"variant": variant, "var_L": float(c["L"][fit][:, C:-C].var()), "var_aT": float((a*c["T"])[fit][:, C:-C].var()),
        "var_sigma_eps": float((sigma*c["epsilon"])[fit][:, C:-C].var()), "generator_hidden_active": [round(float(v), 3) for v in c["hidden_active_fraction_fitting"]],
        "z_common_mode_fraction": float((gm**2).mean()), "z_window_mean_fraction": float((um**2).mean()),
        "z_within_window_fraction": float(((z-um[:, None, :])**2).mean())}
print(json.dumps(desc), flush=True)
xb = torch.from_numpy(x[fit[::8]].astype(np.float32))
data_x = torch.from_numpy(x[fit]); data_y = torch.from_numpy(y[fit]); sup = torch.ones((len(fit), TL-2*C, J), dtype=torch.bool); ids = ["w%05d" % i for i in fit]

def probe(m):
    out = {}
    for label, train_mode in (("eval", False), ("train", True)):
        mode = m.training; m.train(train_mode); acts = []
        hooks = [mod.register_forward_hook(lambda mod, i, o: acts.append(o)) for mod in m.deep if isinstance(mod, nn.ReLU)]
        state = torch.get_rng_state(); torch.manual_seed(13579)
        with torch.no_grad():
            h = m.norm(xb); deep = m.deep(h.transpose(1, 2)).transpose(1, 2)
        torch.set_rng_state(state)
        for hk in hooks: hk.remove()
        m.train(mode)
        out[label+"_active"] = [round(float((o > 0).float().mean()), 4) for o in acts]
        out[label+"_dead"] = [int(((o > 0).float().mean(dim=(0, 2)) < 0.001).sum()) for o in acts]
        out[label+"_never"] = [int(((o > 0).sum(dim=(0, 2)) == 0).sum()) for o in acts]
        if label == "eval":
            out["deep_out_rms"] = round(float(deep.pow(2).mean().sqrt()), 4)
    return out

for lr in (0.003, 0.01):
    for seed in (0, 1, 2):
        if ONLY and ONLY != "%s:%d" % (lr, seed):
            continue
        data = T.TensorTrainingSet(data_x, data_y, sup, ids)
        traj = T.Trajectory(lambda s, f: heads.make_head("TCN_post", seed=s), data, lr=lr, seed=seed, fold=0, device="cpu")
        t0 = time.time()
        for s in (0, 10, 29, 58, 116, 290):
            traj.train_to_step(s)
            print(json.dumps({"variant": variant, "lr": lr, "seed": seed, "step": s, "t": round(time.time()-t0, 1), **probe(traj.model)}), flush=True)
