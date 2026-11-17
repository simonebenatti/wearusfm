"""Libreria della simulazione del punto (c). E' una SIMULAZIONE con ipotesi dichiarate, non i dati dell'esperimento.
Unita': utente. Statistica: miglioramento relativo 100*(1 - somma_u N_cand,u / somma_u N_ref,u), cioe' il rapporto fra somme
del protocollo (il denominatore comune della varianza del bersaglio si semplifica nel rapporto fra due E_dyn)."""
import json
import numpy as np
from scipy import stats

SEED_PROTOCOL, B_PROTOCOL = 20261102, 2000

def real_design(W):
    """Pesi veri dei 15 utenti val sulle finestre dinamiche (ricevuta LONG-FIT-G49046, ridge post): solo per dare alla
    simulazione utenti con peso diverso come nei dati; nessuna predizione nuova."""
    d = json.load(open(f"{W}/LONG-FIT-G49046-pose.json"))
    pu = d["cells"]["post_ridge"]["analysis"]["groups"]["dynamic"]["centered"]["per_user"]
    us = sorted(pu)
    return (np.array([pu[u]["sums"]["centered"] for u in us]), np.array([pu[u]["sums"]["vary"] for u in us]),
            np.array([pu[u]["n_windows"] for u in us], dtype=float))

def multiplicity(n_boot, n_users, seed):
    draws = np.random.default_rng(seed).integers(0, n_users, (n_boot, n_users))
    m = np.zeros((n_boot, n_users))
    np.add.at(m, (np.arange(n_boot)[:, None], draws), 1.0)
    return m

def make_design(kind, rng, R, base):
    """Restituisce N_ref (R,15) e numero di finestre (R,15)."""
    nref, _, nwin = base
    n = len(nref)
    if kind == "reale":            # i 15 utenti veri, fissi
        return np.broadcast_to(nref, (R, n)).copy(), np.broadcast_to(nwin, (R, n)).copy()
    if kind == "ricampionato":     # 15 utenti estratti con reinserimento dalla popolazione dei 15 veri
        ix = rng.integers(0, n, (R, n))
        return nref[ix], nwin[ix]
    if kind == "uguali":           # 15 utenti di peso uguale
        return np.ones((R, n)), np.full((R, n), 550.0)
    if kind == "sbilanciato":      # pesi lognormali (sd log 1): pochi utenti dominano
        w = rng.lognormal(0.0, 1.0, (R, n))
        return w, 550.0*w
    raise ValueError(kind)

def user_effects(rng, R, n, mu, sigma_u, nwin, sigma_w=10.0, tails="normale"):
    """Miglioramento relativo per utente in punti percentuali: mu + eterogeneita' fra utenti + rumore di stima entro utente
    (sigma_w/sqrt(numero di finestre))."""
    if tails == "normale":
        z = rng.standard_normal((R, n))
    else:                          # code pesanti: t con 3 gradi di liberta', varianza unitaria
        z = rng.standard_t(3, (R, n))/np.sqrt(3.0)
    return mu + sigma_u*z + sigma_w/np.sqrt(nwin)*rng.standard_normal((R, n))

def intervals(ncand, nref, m, levels=(99.0,), chunk=None, want_sd=False):
    """Stima puntuale e IC percentile per ogni dataset (riga). m: matrice di molteplicita' (B, n)."""
    R = len(ncand)
    est = 100*(1-ncand.sum(1)/nref.sum(1))
    qs = sorted({round(q, 6) for lv in levels for q in ((100-lv)/2, 100-(100-lv)/2)})
    out = np.empty((R, len(qs)))
    sd = np.empty(R)
    chunk = chunk or max(1, int(4e6//len(m)))
    for s in range(0, R, chunk):
        sl = slice(s, s+chunk)
        boot = 100*(1-(ncand[sl] @ m.T)/(nref[sl] @ m.T))
        out[sl] = np.percentile(boot, qs, axis=1).T
        if want_sd:
            sd[sl] = boot.std(axis=1, ddof=1)
    res = {"est": est, "q": {q: out[:, i] for i, q in enumerate(qs)}}
    if want_sd:
        res["sd"] = sd
    return res

def jackknife_t(ncand, nref, level=99.0):
    n = ncand.shape[1]
    est = 100*(1-ncand.sum(1)/nref.sum(1))
    loo = 100*(1-(ncand.sum(1, keepdims=True)-ncand)/(nref.sum(1, keepdims=True)-nref))
    se = np.sqrt((n-1)/n*((loo-loo.mean(1, keepdims=True))**2).sum(1))
    t = stats.t.ppf(1-(100-level)/200, n-1)
    return est-t*se, est+t*se, se

def bca(ncand, nref, m, level=99.0, chunk=None):
    R, n = ncand.shape
    est = 100*(1-ncand.sum(1)/nref.sum(1))
    loo = 100*(1-(ncand.sum(1, keepdims=True)-ncand)/(nref.sum(1, keepdims=True)-nref))
    dlt = loo.mean(1, keepdims=True)-loo
    acc = (dlt**3).sum(1)/(6*((dlt**2).sum(1))**1.5)
    za = stats.norm.ppf((100-level)/200)
    lo, hi = np.empty(R), np.empty(R)
    chunk = chunk or max(1, int(4e6//len(m)))
    for s in range(0, R, chunk):
        sl = slice(s, s+chunk)
        boot = np.sort(100*(1-(ncand[sl] @ m.T)/(nref[sl] @ m.T)), axis=1)
        B = boot.shape[1]
        frac = np.clip(((boot < est[sl, None]).sum(1)+0.5*(boot == est[sl, None]).sum(1))/B, 1/(2*B), 1-1/(2*B))
        z0 = stats.norm.ppf(frac)
        for arr, z in ((lo, za), (hi, -za)):
            a = stats.norm.cdf(z0+(z0+z)/(1-acc[sl]*(z0+z)))
            pos = np.clip(a*(B-1), 0, B-1)
            i0 = np.floor(pos).astype(int); i1 = np.minimum(i0+1, B-1); w = pos-i0
            rows = np.arange(boot.shape[0])
            arr[sl] = boot[rows, i0]*(1-w)+boot[rows, i1]*w
    return lo, hi

def label(est, lo, hi):
    """Regola del protocollo, applicata nell'ordine: beneficio, deterioramento, non sostenuto, inconclusivo."""
    out = np.full(est.shape, "I", dtype="<U1")
    out[hi < 2] = "N"
    out[(est <= -2) & (hi < 0)] = "D"
    out[(est >= 2) & (lo > 0)] = "B"
    return out
