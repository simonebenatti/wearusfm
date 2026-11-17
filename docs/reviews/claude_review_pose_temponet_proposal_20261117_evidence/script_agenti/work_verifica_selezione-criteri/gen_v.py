"""Mondo sintetico V del verificatore (scritto da zero, indipendente dagli script dei revisori).
Differenze dichiarate rispetto al mondo dei revisori:
- vantaggio temporale costruito con un RITARDO/ANTICIPO (h = tanh(1.5 d_{t-2}) per 4 fattori, d_{t+2} per altri 4), non una media mobile;
- livello del bersaglio in parte leggibile dalle feature (postura p_w) e scostamento d'utente piccolo, nessun livello di finestra
  non osservato separato;
- calibrazione sui numeri veri: ridge CV vicina a 0,656 e quota del livello nella MSE vicina al 55-66%.
Nessun dato del progetto."""
import numpy as np
T, D, J, CROP = 160, 384, 20, 6
KP, KD = 6, 8

def world(seed):
    r = np.random.default_rng(seed)
    return dict(M=r.normal(0, 1, D), Pp=r.normal(0, 1, (KP, D)), Pd=r.normal(0, 1, (KD, D)),
                Ap=r.normal(0, 1, (KP, J)) / np.sqrt(KP), Ad=r.normal(0, 1, (KD, J)) / np.sqrt(KD),
                tau=np.geomspace(3, 25, KD))

def ar1(r, n, L, tau):
    rho = np.exp(-1.0 / np.asarray(tau, float))
    out = np.empty((n, L, len(rho)))
    x = r.normal(0, 1, (n, len(rho)))
    for t in range(L):
        x = rho * x + np.sqrt(1 - rho ** 2) * r.normal(0, 1, x.shape)
        out[:, t] = x
    return out

def h_of(d, pad=2, mix=1.0):
    # d: (w, T+2*pad, KD); h_t = mix*[tanh(1.5 d_{t-2}) per k<4, d_{t+2} per k>=4] + (1-mix)*d_t
    a = np.tanh(1.5 * d[:, 0:T, :4])
    b = d[:, 2 * pad:2 * pad + T, 4:]
    return mix * np.concatenate([a, b], axis=2) + (1 - mix) * d[:, pad:pad + T]

def generate(w, seed, n_users, wpu, p):
    r = np.random.default_rng(seed)
    n = n_users * wpu
    x = np.empty((n, T, D), np.float16); y = np.empty((n, T, J), np.float32); yc = np.empty((n, T, J), np.float32)
    users = np.repeat(np.arange(n_users), wpu)
    pad = 2
    for u in range(n_users):
        g = r.normal(0, p['sig_g'], D); o = r.normal(0, p['sig_o'], J)
        pw = r.normal(0, 1, (wpu, KP))
        a = np.where(r.random(wpu) < 0.5, 1.0, p['a_static'])
        d = ar1(r, wpu, T + 2 * pad, w['tau'])
        dc = d[:, pad:pad + T]
        nw = r.normal(0, p['sig_n'], (wpu, D))
        z = (w['M'] + g)[None, None] + nw[:, None] + p['s_p'] * (pw @ w['Pp'])[:, None] \
            + p['s_d'] * a[:, None, None] * (dc @ w['Pd']) + r.normal(0, p['sig_eps'], (wpu, T, D))
        clean = p['t_p'] * (pw @ w['Ap'])[:, None] + p['t_d'] * a[:, None, None] * (h_of(d, mix=p.get('mix', 1.0)) @ w['Ad'])
        lw = r.normal(0, p['sig_lw'], (wpu, J))
        eta = ar1(r, wpu * J, T, [4.0]).reshape(wpu, J, T).transpose(0, 2, 1) * p['sig_eta'] + o[None, None] + lw[:, None]
        sl = slice(u * wpu, (u + 1) * wpu)
        x[sl] = z.astype(np.float16); y[sl] = (clean + eta).astype(np.float32); yc[sl] = clean.astype(np.float32)
    return x, y, yc, users

P0 = dict(sig_lw=0.5, sig_g=0.3, sig_o=0.5, sig_n=0.3, s_p=0.3, s_d=0.25, sig_eps=0.6, t_p=0.5, t_d=0.6, sig_eta=0.55, a_static=0.25)

def make(p=P0, wseed=7, n_train_users=40, n_val_users=15, wpu=48):
    w = world(wseed)
    xtr, ytr, yctr, utr = generate(w, 100 + wseed, n_train_users, wpu, p)
    xva, yva, ycva, uva = generate(w, 200 + wseed, n_val_users, wpu, p)
    m = ytr.reshape(-1, J).mean(0); s = ytr.reshape(-1, J).std(0)
    f = lambda a: ((a - m) / s).astype(np.float32)
    ytr, yctr, yva, ycva = f(ytr), f(yctr), f(yva), f(ycva)
    pv = lambda a: a.var(axis=1).mean(axis=1)
    thr = float(np.median(pv(yva)))
    return dict(xtr=xtr, ytr=ytr, yctr=yctr, utr=utr, xva=xva, yva=yva, ycva=ycva, uva=uva,
                dtr=pv(ytr) >= thr, dva=pv(yva) >= thr, thr=thr)
