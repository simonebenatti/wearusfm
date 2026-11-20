"""Conti aritmetici per la dimensione «ricetta». Nessun modello, nessun dato."""
import math
import numpy as np

print("== 1. Passi dei checkpoint della prima epoca: ceil(q*S) ==")
for S in (29, 30, 31, 59, 60):
    st = sorted({math.ceil(q * S) for q in (0.25, 0.5, 0.75, 1.0)})
    print(f"S={S}: {st}")
# limiti geometrici
N = 1907
print("S_full = ceil(1907/32) =", math.ceil(N / 32))
print("max S0+S1 = max_n ceil(n/32)+ceil((1907-n)/32) =", max(math.ceil(n / 32) + math.ceil((N - n) / 32) for n in range(1, N)))

print("\n== 2. Matrice: 6 LR (testo) contro 5 LR (raggiungibile) ==")
def matrix(L, E, s01=61, sfull=60):
    cv = 11 * L * 2
    upd = 11 * L * E * s01 + 15 * E * sfull
    evals = cv * (E + 3 + 1)
    return cv, upd, evals
for L, E in ((4, 150), (5, 300), (6, 300)):
    cv, upd, ev = matrix(L, E)
    nom = 11 * L * E * 60 + 15 * E * 60
    print(f"L={L} E={E}: traiettorie CV={cv}, totali={cv+15}, update limite={upd:,}, nominali (30+30)={nom:,}, eval held={ev:,}")
r = (1477800 - 1276500) / 1276500
print(f"eccesso relativo del limite a 6 LR sul limite a 5 LR: {100*r:.1f}%")
for ms in (4.1, 18.5):
    for upd in (1477800, 1276500):
        print(f"  {upd:,} update a {ms} ms: {upd*ms/1000/3600:.2f} GPUh; x2 = {2*upd*ms/1000/3600:.2f}")

print("\n== 3. Weight decay disaccoppiato: contrazione (1-lr*wd)^passi ==")
wd = 0.01
for lr, steps, nome in ((1e-3, 50 * 30, "orizzonte originale 50 ep x 30 passi, lr 1e-3"),
                        (1e-3, 150 * 30, "150 ep fold, lr 1e-3"),
                        (3e-3, 300 * 30, "300 ep fold, lr 3e-3"),
                        (1e-2, 300 * 30, "300 ep fold, lr 1e-2"),
                        (1e-2, 300 * 60, "rifit 300 ep x 60, lr 1e-2")):
    f = (1 - lr * wd) ** steps
    print(f"  {nome}: fattore {f:.4f} (contrazione {100*(1-f):.1f}%)")

print("\n== 4. S-2 sui numeri del mondo V (rapporto del verificatore, 32e5452) ==")
cases = {
    "TCN lr1e-3 a 150 ep: sel MSE ep93 E=0.7548, min E_dyn ep148 E=0.7370": (93, 0.7548, 148, 0.7370),
    "TCN lr1e-3 a 50 ep: sel MSE ep43 E=0.8077, min E_dyn ep50 E=0.8023": (43, 0.8077, 50, 0.8023),
    "L0 lr1e-4: sel ep17 E=0.7834, min E_dyn ep14 E=0.7821": (17, 0.7834, 14, 0.7821),
    "L13 lr1e-4: sel ep5, min E_dyn ep23 E=0.6111, costo 3,66%": (5, 0.6111 * 1.0366, 23, 0.6111),
}
for k, (es, Es, em, Em) in cases.items():
    d = abs(em - es); reg = (Es - Em) / Em
    print(f"  {k}: distanza {d} ep, regret {100*reg:.2f}% -> objective_tradeoff={d > 5 and reg >= 0.02}")
print("  trigger a 150 per TCN mondo V: min MSE ep93 in 146..150?", 146 <= 93 <= 150, "; min E_dyn ep148 in 146..150?", 146 <= 148 <= 150)
print("  trigger a 50 (vecchio orizzonte, ultime 5 = 46..50): MSE ep43?", 46 <= 43 <= 50, "; E_dyn ep50?", 46 <= 50 <= 50)

print("\n== 5. Soglia storica: valore float32? ==")
t = 0.2574231028556824
print("  float(np.float32(t)) == t:", float(np.float32(t)) == t)
