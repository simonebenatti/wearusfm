"""Ricalcolo dei numeri derivati della bozza (rilettura NUMERI, 17/11). Solo aritmetica sui valori delle fonti."""
import math
from fractions import Fraction as F

def rel(ref, x):  # convenzione del protocollo: 100*(E_ref - E_x)/E_ref
    return 100 * (ref - x) / ref

R, H = 0.7859020004638595, 0.7786289843513977
print("== C-1 (L0 0,81, TCN 0,7938)")
print("  c1 %.4f  c4 %.4f  c5 %.4f" % (rel(0.81, 0.7938), rel(R, 0.7938), rel(H, 0.7938)))
print("== C-2 (L0 0,785, L13 0,730, MLP 0,735, TCN 0,750)")
print("  c1 %.3f c2 %.3f c3 %.3f c4 %.3f c5 %.3f" % (rel(0.785, .75), rel(0.735, .75), rel(0.730, .75), rel(R, .75), rel(H, .75)))
print("== N3 soglia candidata")
print("  0,98*testa = %.6f ; 0,98*ridge = %.6f ; sotto la ridge %.3f%%" % (0.98 * H, 0.98 * R, rel(R, 0.98 * H)))
print("== D4/ testa vs ridge: %.4f%%" % rel(R, H))
print("== Parametri")
ln = 2 * 384; lin = 384 * 20 + 20
print("  ramo lineare 384->20 con bias:", lin, " L0 = LN + lineare:", ln + lin, " L13 pesi:", 384 * 20 * 13, "=13x", 384 * 20)
mlp_w = 384*96 + 96*96 + 96*96 + 96*64 + 64*32 + 32*20
tcn_w = 384*64 + 3*64*64*3 + 64*32 + 32*20
print("  pesi MLP", mlp_w, " pesi TCN", tcn_w, " MAC/finestra", tcn_w * 160)
print("== Weight decay (1-lr*0,01)^n")
for lr in (1e-3, 1e-4):
    for n in (3000, 1500):
        print("  lr %g n %d -> %.4f" % (lr, n, (1 - lr * 0.01) ** n))
print("== Ridge storica, MSE interna rispetto all'ottimo 0,656115")
for a, v in (("1e4", 0.691356), ("0,01", 0.769725), ("1e6", 0.670911)):
    print("  alpha %s: +%.2f%%" % (a, 100 * (v / 0.656115 - 1)))
print("== Byte e GiB")
GiB = 2 ** 30
pred = 18548 * 160 * 20 * 4
print("  una predizione val FP32 a 160 patch:", pred, " 15 ->", 15 * pred, "B =", round(15 * pred / GiB, 4), "GiB; a 148 patch:", round(15 * 18548 * 148 * 20 * 4 / GiB, 4))
cache = 474412856 + 4593977338
refs3 = 59367 + 418440754 + 234048726
print("  cache", cache, "=", round(cache / GiB, 4), "GiB; tre riferimenti pinnati", refs3, "=", round(refs3 / GiB, 4), "GiB; somma", round((cache + refs3) / GiB, 4), "GiB")
pose_refs = 257675076 + 2485295825
print("  riferimenti di posa su SCRATCH", pose_refs, "B =", round(pose_refs / 1e9, 3), "GB =", round(pose_refs / GiB, 3), "GiB; cache+pose_refs", round((cache + pose_refs) / GiB, 3), "GiB")
print("== Endpoint del confronto raw")
e = 18548 * 148 * 5
print("  ", e, " /12.468.000 =", round(e / 12468000, 4))
print("== Passi e GPU-ore")
def steps(n_lr, epochs, heads=5, folds=2, sel_steps=30, refit_fits=15, refit_steps=60):
    sel = heads * n_lr * folds * sel_steps * epochs
    ref = refit_fits * refit_steps * epochs
    return sel, ref, sel + ref
for n_lr, ep in ((2, 50), (4, 150), (3, 150), (4, 50), (2, 150)):
    s, r, t = steps(n_lr, ep)
    for ms, lab in ((18.5, "18,5 ms det. eager"), (14.6, "14,6 ms det. graph"), (4.1, "4,1 ms eager"), (1.1, "1,1 ms graph")):
        print("  LR %d epoche %3d: sel %6d rifit %6d tot %6d passi; x %s x2 = %.3f GPU-ore" % (n_lr, ep, s, r, t, lab, t * ms / 1000 * 2 / 3600))
print("  scala 315.000/75.000 =", 315000 / 75000, "; 0,77 x 4,2 =", round(0.77 * 4.2, 3), "; 0,693 (MPS) x 4,2 =", round(0.693 * 4.2, 3))
print("  scala ingenua 0,77 x 3 (epoche) x 2 (LR) =", round(0.77 * 6, 3))
print("== Livello B")
g19 = (3916 + 11334) * 4 / 3600
g49 = (7955.65 + 7145.29 + 7536.23) * 4 / 3600
print("  19.618: %.4f GPU-ore, x2 = %.3f; continuazione %.4f; un training a 49.046 %.2f, due %.2f; R3c 40,4 x2 = %.1f" % (g19, 2 * g19, g49, g19 + g49, 2 * (g19 + g49), 80.8))
print("  33,88 x 49046/19618 = %.2f" % (33.88 * 49046 / 19618))
print("== Mondo V (verificatore): L0 0,7834")
L0V = 0.7834
for lab, x in (("TCN 50 ep (0,8077)", 0.8077), ("TCN 150 ep (0,7548)", 0.7548), ("TCN kaiming (0,7629)", 0.7629),
               ("ridge ritardi -6,-4..+6 a 1e5 (0,6164)", 0.6164), ("ridge ritardi -6,-4..+6 a 1e6 (0,6753)", 0.6753),
               ("ridge tutti i ritardi -6..+6 a 1e6 (0,6519)", 0.6519), ("ridge istantanea (0,7945)", 0.7945), ("oracolo (0,5573)", 0.5573)):
    print("  %-45s %+.2f%%" % (lab, rel(L0V, x)))
print("  calibrazione V: ridge CV 0,7555 contro vera 0,6561 (%+.1f%%); quota livello V 60,2%% tutte / 57,5%% dinamiche, vere 65,9%% / 54,4%%" % (100 * (0.7555 / 0.6561 - 1)))
print("== Mondo temporale del revisore: L0 0,9357")
for lab, x in (("TCN (0,9848)", 0.9848), ("tetto (0,8539)", 0.8539)):
    print("  %-20s %+.2f%%" % (lab, rel(0.9357, x)))
print("  fold 0->1: L0 0,9413; ridge ritardi 0,9163 %+.2f%%; TCN 0,9782 %+.2f%%; TCN init che conserva il segnale 0,9766 / 0,9857: %+.2f%% / %+.2f%%" % (
    rel(0.9413, 0.9163), rel(0.9413, 0.9782), rel(0.9413, 0.9766), rel(0.9413, 0.9857)))
print("== A-1, mondo lineare del verificatore (MSE tenuta fuori, non E_dyn)")
print("  TCN 0,3205 contro L0 0,2500: %+.1f%%; TCN+ramo 0,2495: %+.2f%%" % (rel(0.25, 0.3205), rel(0.25, 0.2495)))
print("== S-2 / N4 (L13)")
print("  L13 V: E_dyn %.2f%%; MSE %+.2f%%; revisore E_dyn %.2f%%; MSE %+.1f%% / %+.1f%%" % (
    100 * (0.6335 / 0.6111 - 1), 100 * (0.7401 / 0.7107 - 1), 100 * (0.8901 / 0.7955 - 1), 100 * (0.9412 / 0.8420 - 1), 100 * (0.9692 / 0.8420 - 1)))
print("  distorsione sul primario V: L0 %.2f%% TCN %.2f%%" % (100 * (0.7834 / 0.7821 - 1), 100 * (0.8077 / 0.8023 - 1)))
print("== Segnale all'inizializzazione: rapporto di varianze 0,6-2,1%% -> rapporto di ampiezze %.1f-%.1f%%" % (100 * math.sqrt(0.0063), 100 * math.sqrt(0.0212)))
print("== Fold: ceil(953/32) =", math.ceil(953 / 32), " ceil(954/32) =", math.ceil(954 / 32), " ceil(1907/32) =", math.ceil(1907 / 32))
print("== Conteggi dei rilievi: 8+8+11+10+10+9 =", 8 + 8 + 11 + 10 + 10 + 9, "; nuovi 2+4+2 =", 2 + 4 + 2, "; attaccate 22+19+19 =", 22 + 19 + 19, "cadute 1+0+2 =", 3)
