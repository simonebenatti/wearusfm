import sys, json, numpy as np, torch
sys.path.insert(0, sys.argv[1])
import gen_v as G
torch.set_num_threads(1)
T, D, J, C = G.T, G.D, G.J, G.CROP
SL = slice(C, T - C)
def ln(x):
    return torch.nn.functional.layer_norm(torch.from_numpy(x.astype(np.float32)), (D,), eps=1e-5).double()
def stats(x, y, lags=(0,)):
    F = D * len(lags)
    sx = torch.zeros(F, dtype=torch.float64); sy = torch.zeros(J, dtype=torch.float64)
    xx = torch.zeros(F, F, dtype=torch.float64); xy = torch.zeros(F, J, dtype=torch.float64); n = 0
    for i in range(0, len(x), 64):
        a = ln(x[i:i+64]); b = torch.from_numpy(y[i:i+64, SL]).double().reshape(-1, J)
        a = torch.cat([a[:, C + l:T - C + l] for l in lags], dim=2).reshape(-1, F)
        sx += a.sum(0); sy += b.sum(0); xx += a.T @ a; xy += a.T @ b; n += len(a)
    return sx, sy, xx, xy, n
def solver(st):
    sx, sy, xx, xy, n = st
    mx, my = sx / n, sy / n
    g = xx - torch.outer(sx, sx) / n
    e, v = torch.linalg.eigh((g + g.T) / 2); e = e.clamp_min(0)
    cr = v.T @ (xy - torch.outer(sx, my))
    return lambda al: (v @ (cr / (e[:, None] + al)), None, mx, my)
def predict(x, sol, lags=(0,)):
    W_, _, mx, my = sol
    out = []
    for i in range(0, len(x), 64):
        a = ln(x[i:i+64]); a = torch.cat([a[:, C + l:T - C + l] for l in lags], dim=2)
        out.append(((a - mx) @ W_ + my).float().numpy())
    return np.concatenate(out)
def metrics(p, y, dyn):
    t = y[:, SL]
    mse = ((p - t) ** 2).mean(axis=(1, 2)).mean()
    cp = p - p.mean(1, keepdims=True); ct = t - t.mean(1, keepdims=True)
    cs = ((cp - ct) ** 2).sum(axis=(1, 2)); vs = (ct ** 2).sum(axis=(1, 2))
    lev = ((p.mean(1) - t.mean(1)) ** 2).sum(1).sum() * (T - 2 * C)
    tot = ((p - t) ** 2).sum(axis=(1, 2)).sum()
    return dict(mse=float(mse), E_dyn=float(cs[dyn].sum() / vs[dyn].sum()), E_all=float(cs.sum() / vs.sum()),
                level_share_all=float(lev / tot),
                level_share_dyn=float(((p.mean(1) - t.mean(1)) ** 2).sum(1)[dyn].sum() * (T - 2 * C) / ((p - t) ** 2).sum(axis=(1, 2))[dyn].sum()))
if __name__ == '__main__':
    over = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    p = dict(G.P0); p.update(over)
    d = G.make(p)
    ids = np.arange(40); folds = [np.isin(d['utr'], ids[::2]), np.isin(d['utr'], ids[1::2])]
    lags = [(0,), tuple(range(-6, 7, 2))]
    res = {}
    for lg in lags:
        sols = [solver(stats(d['xtr'][f], d['ytr'][f], lg)) for f in folds]
        rows = []
        for e in range(-2, 9):
            ps, ys, ds = [], [], []
            for k in range(2):
                ps.append(predict(d['xtr'][folds[1 - k]], (sols[k](10.0 ** e)[0], None, *sols[k](10.0 ** e)[2:]), lg))
            pa = np.concatenate(ps); ya = np.concatenate([d['ytr'][folds[1]], d['ytr'][folds[0]]])
            da = np.concatenate([d['dtr'][folds[1]], d['dtr'][folds[0]]])
            m = metrics(pa, ya, da); m['alpha'] = 10.0 ** e; rows.append(m)
        best = min(rows, key=lambda r: r['mse'])
        res[str(lg)] = dict(best=best, grid=[(r['alpha'], round(r['mse'], 4), round(r['E_dyn'], 4)) for r in rows])
        print('lags', lg, 'best', {k: round(v, 4) for k, v in best.items()}, flush=True)
        print('   grid', res[str(lg)]['grid'], flush=True)
    # oracolo: bersaglio pulito sulla CV (tetto dell'errore)
    yc = np.concatenate([d['yctr'][folds[1]], d['yctr'][folds[0]]])[:, SL]
    ya = np.concatenate([d['ytr'][folds[1]], d['ytr'][folds[0]]])
    da = np.concatenate([d['dtr'][folds[1]], d['dtr'][folds[0]]])
    print('oracolo (bersaglio senza rumore) sulla CV:', {k: round(v, 4) for k, v in metrics(yc, ya, da).items()})
    print('soglia', d['thr'], 'dinamiche train', int(d['dtr'].sum()), 'val', int(d['dva'].sum()))
