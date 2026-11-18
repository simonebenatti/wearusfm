import math
from fractions import Fraction as F
# Matrice e aggiornamenti (formula di P:102: 11*L*E*(S0+S1)+15*E*Sfull)
for L, E, s01, sf, nome in [(4,150,61,60,"iniziale limite"),(4,150,60,60,"iniziale nominale"),(6,300,61,60,"massimo 6 LR limite"),
                            (6,300,60,60,"massimo 6 LR nominale"),(5,300,61,60,"massimo 5 LR limite"),(5,300,60,60,"massimo 5 LR nominale")]:
    print(f"{nome:24s} {11*L*E*s01 + 15*E*sf:>10,d}")
print("traiettorie: iniziali", 5*4*2 + 6*4*2 + 15, " max6", 5*6*2 + 6*6*2 + 15, " max5", 5*5*2 + 6*5*2 + 15)
print("valutazioni held: 6 LR", 11*6*2*304, " 5 LR", 11*5*2*304)
print("S_full =", math.ceil(1907/32), " max ceil(n/32)+ceil((1907-n)/32) =", max(math.ceil(n/32)+math.ceil((1907-n)/32) for n in range(1,1907)))
print("quarti S=30:", [math.ceil(q*30) for q in (0.25,0.5,0.75,1)], " S=31:", [math.ceil(q*31) for q in (0.25,0.5,0.75,1)], " S=60:", [math.ceil(q*60) for q in (0.25,0.5,0.75,1)])
# Weight decay (fattore (1-lr*wd)^passi)
for lr, passi in [(1e-3,1500),(3e-3,9000),(1e-2,9000),(1e-2,18000),(3e-5,9000),(1e-4,9000)]:
    print(f"decay lr={lr:g} passi={passi}: fattore {(1-lr*0.01)**passi:.4f}")
# Soglia della testa e gain
head, ridge = 0.7786289843513977, 0.7859020004638595
thr = 0.98*head
print("0.98*testa =", repr(thr), " gain5 alla soglia =", repr(100*(head-thr)/head), " distanza dalla ridge % =", 100*(ridge-thr)/ridge, " 2% sotto ridge =", repr(0.98*ridge))
# Controesempi dei criteri
def g(ref, tcn): return 100*(ref-tcn)/ref
print("criteri-1: gain1", g(0.800,0.750), "gain4", g(ridge,0.750), "gain5", g(head,0.750), "gain3", g(0.790,0.750), "gain2", g(0.740,0.750), "variante gain2", g(0.731,0.750))
print("criteri-2: gain1", g(0.770,0.760), "gain4", g(ridge,0.760), "gain5", g(head,0.760), "L0 su ridge", g(ridge,0.770), "L0 su testa", g(head,0.770))
# S-2 nei numeri del mondo V (verify_selezione-criteri.txt)
for nome, sel_ep, sel_e, min_ep, min_e in [("TCN 1e-3 a 150", 93, 0.7548, 148, 0.7370), ("TCN 1e-3 a 50", 43, 0.8077, 50, 0.8023),
                                           ("L0 1e-4", 17, 0.7834, 14, 0.7821)]:
    print(f"S-2 {nome}: distanza {abs(min_ep-sel_ep)} regret {100*(sel_e-min_e)/min_e:.3f}%")
# Parametri TCN e MLP dalla specifica di P:28-30
g_tcn = (384*64+64) + 3*(64*64*3+64) + (64*32+32) + (32*20+20)
ln, aff = 2*384, 384*20+20
widths = [384,96,96,96,64,32,20]
g_mlp = sum(a*b+b for a,b in zip(widths[:-1], widths[1:]))
print("TCN g", g_tcn, "totale", g_tcn+ln+aff, " MLP g", g_mlp, "totale", g_mlp+ln+aff, " L0", ln+384*20+20, " L13", ln+384*20*13+20)
