"""Review Claude del pacchetto contrastivo db5ac41 (16/11/2026): quanto vale lungo il run il «10%» fissato all'inizializzazione.

La regola del protocollo (par. 4) fissa lambda = 0,10 * ||g_base|| / ||g_termine|| su batch a pesi iniziali. Qui prendo la norma
totale del gradiente prima del clipping dal log VERO del controllo 30M (stessa ricetta T100-G, 19.618 update) e calcolo il rapporto
rho_t = 0,10 * G_0 / G_t che si avrebbe SE la norma del termine contrastivo restasse quella dell'inizializzazione (ipotesi dichiarata:
nessun run contrastivo esiste). G_0 = media degli update 1..8. Limite: il log ha la norma su TUTTI i parametri, la calibrazione
usa i soli parametri condivisi. Con gradienti ortogonali, dopo il clipping alla base resta la quota 1/sqrt(1+rho^2) del passo.
"""
import json
from pathlib import Path
import numpy as np

log = Path.home() / "wearusfm_local/logs/review6_remote/control30m/metrics.jsonl"
g = np.array([json.loads(l)["grad_norm"] for l in open(log)], dtype=np.float64)
G0 = g[:8].mean()
rho = 0.10 * G0 / g
def seg(a, b):                      # update a..b a base 1, estremi inclusi
    return slice(a - 1, b)
print(f"update {len(g)}; G_0 (update 1-8) = {G0:.2f}")
print(f"{'tratto (update)':>20s} {'norma media':>12s} {'mediana':>9s} {'rho medio':>10s} {'quota base':>11s}")
for a, b in [(1, 8), (57, 64), (65, 128), (129, 256), (257, 512), (513, 1024), (1025, 1962), (1963, 5000), (5001, 10000), (10001, 19618), (19519, 19618)]:
    s = seg(a, b); r = rho[s]
    print(f"{a:>9d}-{b:<10d} {g[s].mean():12.2f} {np.median(g[s]):9.2f} {r.mean():10.3f} {(1 / np.sqrt(1 + r ** 2)).mean():11.3f}")
act = seg(513, len(g)); r = rho[act]
print(f"peso pieno (update 513-{len(g)}): rho medio {r.mean():.3f}, mediano {np.median(r):.3f}; rho > 0,5 nel {100 * (r > .5).mean():.1f}% degli update; rho > 1 nel {100 * (r > 1).mean():.1f}%; quota media della base {(1 / np.sqrt(1 + r ** 2)).mean():.3f}")
ramp = np.arange(257, 513); w = (ramp - 256) / 256.0
print(f"rampa (update 257-512, peso lineare): rho medio {(w * rho[256:512]).mean():.3f}")
print(f"rapporti: norma 257-512 / 1-8 = {g[256:512].mean() / G0:.3f}; ultimi 100 / 257-512 = {g[-100:].mean() / g[256:512].mean():.3f}; 1-8 / ultimi 100 = {G0 / g[-100:].mean():.2f}")
