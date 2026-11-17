"""(c5) Errore Monte Carlo dei limiti percentile 0,5% e 99,5% con 2.000 contro 20.000 repliche.
Tre insiemi di dati FISSI (due veri, dalle somme per utente archiviate; uno sintetico) e K semi indipendenti del bootstrap:
quanto cambiano i limiti da un seme all'altro. Il protocollo usa UN seme fisso (20261102): il suo limite e' un'estrazione
da questa distribuzione, con uno scarto ignoto rispetto al limite ideale (infinite repliche).
Uso: python c5_mc_error.py <cartella con le ricevute>"""
import json, sys, time
import numpy as np
from scipy import stats

W = sys.argv[1]
K2, K20 = 400, 400


def per_user(path, reader, group="dynamic"):
    d = json.load(open(path))
    pu = d["cells"][reader]["analysis"]["groups"][group]["centered"]["per_user"]
    us = sorted(pu)
    return us, np.array([pu[u]["sums"]["centered"] for u in us])


us, head49 = per_user(f"{W}/LONG-FIT-G49046-pose.json", "trained_head")
us2, ridge49 = per_user(f"{W}/LONG-FIT-G49046-pose.json", "post_ridge")
us3, ridge19 = per_user(f"{W}/LONG-FIT-G19618-pose.json", "post_ridge")
assert us == us2 == us3
n = len(us)
rng = np.random.default_rng(20261117)
r_syn = 2.0 + 1.5*rng.standard_normal(n)          # sintetico: effetto medio 2%, eterogeneita' 1,5 punti
datasets = {
    "vero: testa allenata vs ridge post, 49046": (head49, ridge49),
    "vero: dose ridge post 49046 vs 19618": (ridge49, ridge19),
    "sintetico: pesi veri, mu=2%, sigma_u=1,5": (ridge49*(1-r_syn/100), ridge49),
}


def limits(nc, nr, B, seed):
    draws = np.random.default_rng(seed).integers(0, n, (B, n))
    boot = 100*(1-nc[draws].sum(1)/nr[draws].sum(1))
    lo, hi = np.percentile(boot, [0.5, 99.5])
    return lo, hi, boot.std(ddof=1)


t0 = time.time()
print("insieme | stima | sd bootstrap | B | media limite inf. | sd MC limite inf. (in sd bootstrap) | media limite sup. | sd MC limite sup. (in sd bootstrap) | limite col seme del protocollo | escursione 2,5-97,5% del limite inf. fra semi")
for name, (nc, nr) in datasets.items():
    est = 100*(1-nc.sum()/nr.sum())
    for B, K in ((2000, K2), (20000, K20)):
        res = np.array([limits(nc, nr, B, 1000+k) for k in range(K)])
        prot = limits(nc, nr, B, 20261102)
        s = res[:, 2].mean()
        q = np.percentile(res[:, 0], [2.5, 97.5])
        print(f"{name} | {est:.4f} | {s:.4f} | {B} | {res[:,0].mean():.4f} | {res[:,0].std(ddof=1):.4f} ({res[:,0].std(ddof=1)/s:.3f}) | "
              f"{res[:,1].mean():.4f} | {res[:,1].std(ddof=1):.4f} ({res[:,1].std(ddof=1)/s:.3f}) | [{prot[0]:.4f}; {prot[1]:.4f}] | [{q[0]:.4f}; {q[1]:.4f}]")
p = 0.005
for B in (2000, 20000):
    se = np.sqrt(p*(1-p)/B)/stats.norm.pdf(stats.norm.ppf(p))
    print(f"teoria (distribuzione normale): errore standard del quantile 0,5% con B={B} = {se:.3f} sd bootstrap; repliche attese nella coda: {p*B:.0f}")
print(f"tempo {time.time()-t0:.0f} s; K={K2} semi per B=2000 e K={K20} per B=20000")
