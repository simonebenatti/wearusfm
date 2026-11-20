"""Riproduce E_dyn e IC95 della valutazione lunga dalle statistiche per utente della ricevuta LONG-FIT-G49046.
Solo numpy; nessun import del progetto. Stessa procedura di consolidation_long_pose._bootstrap_group/_contrast."""
import json, sys, hashlib
import numpy as np
W = sys.argv[1]
SEED, N_BOOT = 20261102, 2000
def load(step):
    d = json.load(open(f"{W}/LONG-FIT-G{step}-pose.json"))
    out = {}
    for reader, cell in d["cells"].items():
        out[reader] = cell["analysis"]["groups"]
    return d, out
d49, g49 = load(49046)
d19, g19 = load(19618)
print("dynamic_threshold nella ricevuta:", d49["cells"]["post_ridge"]["analysis"]["dynamic_threshold"])
users = sorted(g49["post_ridge"]["dynamic"]["centered"]["per_user"])
print("n utenti:", len(users))
def arr(groups, reader, group, key):
    pu = groups[reader][group]["centered"]["per_user"]
    assert sorted(pu) == users
    return np.array([pu[u]["sums"][key] for u in users])
def cnt(groups, reader, group, key):
    pu = groups[reader][group]["centered"]["per_user"]
    return np.array([pu[u][key] for u in users])
draws = np.random.default_rng(SEED).integers(0, len(users), (N_BOOT, len(users)))
for reader in ("post_ridge", "trained_head", "pre_ridge"):
    num, den = arr(g49, reader, "dynamic", "centered"), arr(g49, reader, "dynamic", "vary")
    point = num.sum()/den.sum()
    boot = num[draws].sum(1)/den[draws].sum(1)
    print(f"49046/{reader}: E_dyn={point!r} IC95={np.percentile(boot,[2.5,97.5]).tolist()}  n_windows={cnt(g49,reader,'dynamic','n_windows').sum()} n_valid={cnt(g49,reader,'dynamic','n_valid_elements').sum()}")
    print("   centered_mse_ratio nella ricevuta:", g49[reader]["dynamic"]["centered"]["centered_mse_ratio"])
# contrasto di dose: 49046 - 19618, ridge e testa
for reader in ("post_ridge", "trained_head"):
    n1, d1 = arr(g49, reader, "dynamic", "centered"), arr(g49, reader, "dynamic", "vary")
    n0, d0 = arr(g19, reader, "dynamic", "centered"), arr(g19, reader, "dynamic", "vary")
    assert np.array_equal(d1, d0)
    point = 100*(1-(n1.sum()/d1.sum())/(n0.sum()/d0.sum()))
    boot = 100*(1-(n1[draws].sum(1)/d1[draws].sum(1))/(n0[draws].sum(1)/d0[draws].sum(1)))
    print(f"dose 49046-19618 {reader}: rel={point:.6f}% IC95={np.percentile(boot,[2.5,97.5]).round(6).tolist()} sd_boot={boot.std(ddof=1):.4f}")
# contrasto fra lettori a 49046: testa allenata contro ridge post
n1, n0, den = arr(g49,"trained_head","dynamic","centered"), arr(g49,"post_ridge","dynamic","centered"), arr(g49,"post_ridge","dynamic","vary")
assert np.array_equal(den, arr(g49,"trained_head","dynamic","vary"))
point = 100*(1-n1.sum()/n0.sum())
boot = 100*(1-n1[draws].sum(1)/n0[draws].sum(1))
print(f"49046 testa allenata vs ridge post: rel={point:.6f}% IC95={np.percentile(boot,[2.5,97.5]).round(4).tolist()} IC99={np.percentile(boot,[0.5,99.5]).round(4).tolist()} sd_boot={boot.std(ddof=1):.4f}")
print("draws sha256 (int64 bytes):", hashlib.sha256(draws.tobytes()).hexdigest()[:16])
print()
print("Per utente, finestre dinamiche, 49046: n_windows, peso del denominatore, E_dyn ridge, E_dyn testa, miglioramento relativo testa vs ridge (%), dose ridge 49-19 (%)")
nw = cnt(g49,"post_ridge","dynamic","n_windows")
n19 = arr(g19,"post_ridge","dynamic","centered")
for i,u in enumerate(users):
    print(f"  {u} nwin={nw[i]:4d} peso={den[i]/den.sum():.4f} E_ridge={n0[i]/den[i]:.4f} E_testa={n1[i]/den[i]:.4f} rel_testa_vs_ridge={100*(1-n1[i]/n0[i]):+.3f} rel_dose_ridge={100*(1-n0[i]/n19[i]):+.3f}")
r = 100*(1-n1/n0)
print("rel testa vs ridge per utente: media semplice %.3f sd %.3f min %.3f max %.3f" % (r.mean(), r.std(ddof=1), r.min(), r.max()))
r14 = np.delete(r, users.index("ab228f34a5"))
print("  senza l'utente con 7 finestre: media %.3f sd %.3f" % (r14.mean(), r14.std(ddof=1)))
rd = 100*(1-n0/n19)
print("rel dose ridge per utente: media %.3f sd %.3f min %.3f max %.3f" % (rd.mean(), rd.std(ddof=1), rd.min(), rd.max()))
