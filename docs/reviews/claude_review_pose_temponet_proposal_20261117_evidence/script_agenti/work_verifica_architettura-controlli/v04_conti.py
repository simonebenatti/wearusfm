import json, math, sys
from fractions import Fraction as F
W = sys.argv[1]
j = json.load(open(W + "/RIC_49046_mine.json"))
grid = j["cells"]["post_ridge"]["fit"]["selection"]["grid"]
best = min(g["mse"] for g in grid)
print("griglia ridge post (MSE interno, +% sull'ottimo):")
for g in grid:
    print("  alpha %-9g mse %.6f  %+.2f%%" % (g["alpha"], g["mse"], 100 * (g["mse"] / best - 1)))
# weight decay disaccoppiato di AdamW: fattore (1 - lr*wd) per passo
for n_train in (1907, 954, 953):
    steps = math.ceil(n_train / 32)
    print("finestre %d -> passi/epoca %d -> 50 epoche %d passi" % (n_train, steps, 50 * steps))
for lr in (1e-3, 1e-4):
    for steps in (3000, 1500):
        print("  lr %g, %d passi: contrazione %.4f" % (lr, steps, (1 - lr * 0.01) ** steps))
ridge, head = 0.7859020004638595, 0.7786289843513977
imp = lambda ref, x: 100 * (ref - x) / ref
print("esempio controlli-1: L0=0.80 TCN=%.4f: c1 %.3f%% c4 %.3f%% c5 %.3f%%" % (0.98 * 0.80, imp(0.80, 0.784), imp(ridge, 0.784), imp(head, 0.784)))
print("esempio controlli-1: L0=0.81 TCN=0.7938: c1 %.3f%% c4 %.3f%% c5 %.3f%%" % (imp(0.81, 0.7938), imp(ridge, 0.7938), imp(head, 0.7938)))
print("esempio controlli-2: L0 .785 L13 .730 MLP .735 TCN .750: c1 %.3f c2 %.3f c3 %.3f c4 %.3f c5 %.3f" % (
    imp(0.785, 0.75), imp(0.735, 0.75), imp(0.73, 0.75), imp(ridge, 0.75), imp(head, 0.75)))
print("testa vs ridge 49046: %.4f%%" % imp(ridge, head))
# pesi TCN: centrali e laterali
w_central = 384 * 64 + 3 * 64 * 64 + 64 * 32 + 32 * 20
w_lat = 3 * 2 * 64 * 64
print("TCN pesi con sole prese centrali", w_central, "prese laterali", w_lat, "somma", w_central + w_lat)
print("MLP pesi", 384*96 + 96*96 + 96*96 + 96*64 + 64*32 + 32*20, "differenza MLP - TCNcentrale", 384*96 + 96*96 + 96*96 + 96*64 + 64*32 + 32*20 - w_central)
# endpoint per il confronto raw
print("endpoint fine-blocco 18548x148x5 =", 18548 * 148 * 5, "; per campione x50 =", 18548 * 148 * 50,
      "; lettura ufficiale 12468x1000 =", 12468 * 1000, "; rapporti %.4f e %.4f" % (18548*148*5/12468000, 18548*148*50/12468000))
# quota valida train
print("quota valida train 4813620/(1907*148*20) = %.4f%%" % (100 * 4813620 / (1907 * 148 * 20)))
# 0,23% del 30M
print("65204/28362013 = %.4f%% ; 65204/8468 = %.3f" % (100 * 65204 / 28362013, 65204 / 8468))
# Bonferroni e code
print("Bonferroni 5x1%% -> %.0f%% simultaneo; repliche per coda 2000*0.005 = %g" % (100 - 5 * 1, 2000 * 0.005))
