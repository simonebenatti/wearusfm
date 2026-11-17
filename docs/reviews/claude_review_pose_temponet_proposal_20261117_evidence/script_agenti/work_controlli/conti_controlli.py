"""CALCOLATO (seconda stesura): conti usati nel rapporto della dimensione «Controlli e attribuzione».
Numeri d'ingresso letti dalla ricevuta storica (RIC) e dal protocollo c15aac9; nessun dato nuovo."""
import math
ridge, head, pre = 0.785902, 0.778629, 1.032128          # E_dyn dinamiche a 49.046 (RIC)
imp = lambda ref, x: 100 * (ref - x) / ref
print("testa storica contro ridge storica, 49.046: %.3f%%" % imp(ridge, head))
print("            a 19.618: %.3f%% ; a 39.237: %.3f%%" % (imp(0.838933, 0.832954), imp(0.809722, 0.805007)))
for L0 in (0.80, 0.81):
    t = 0.98 * L0
    print("L0 = %.2f -> TCN al limite del primario (stima 2%%) = %.4f; contro ridge %+.3f%%, contro testa %+.3f%%" % (L0, t, imp(ridge, t), imp(head, t)))
# esempio: TCN candidata secondo PROT:100 mentre un controllo e' migliore
L0, L13, tcn = 0.785, 0.730, 0.750
print("esempio L13 migliore: c1 %.2f%%, c3 %.2f%%, c4 %.2f%%, c5 %.2f%%" % (imp(L0, tcn), imp(L13, tcn), imp(ridge, tcn), imp(head, tcn)))
L0, mlp, tcn = 0.785, 0.735, 0.750
print("esempio MLP migliore: c1 %.2f%%, c2 %.2f%%, c4 %.2f%%, c5 %.2f%%" % (imp(L0, tcn), imp(mlp, tcn), imp(ridge, tcn), imp(head, tcn)))
# passi di ottimizzazione e weight decay disaccoppiato (AdamW: p <- p * (1 - lr*wd) a ogni passo)
steps_full = math.ceil(1907 / 32)
print("passi per epoca, 40 utenti:", steps_full, "; al massimo", 50 * steps_full)
for lr in (1e-4, 1e-3):
    print("LR %g: contrazione cumulata del solo weight decay in %d passi = %.4f" % (lr, 50 * steps_full, (1 - lr * 0.01) ** (50 * steps_full)))
# pesi: TCN con le sole prese centrali, prese laterali, MLP
central = 384 * 64 + 3 * 64 * 64 + 64 * 32 + 32 * 20
lateral = 3 * 2 * 64 * 64
mlp_w = 384 * 96 + 96 * 96 + 96 * 96 + 96 * 64 + 64 * 32 + 32 * 20
print("TCN prese centrali", central, "; prese laterali", lateral, "; somma", central + lateral, "; MLP", mlp_w)
print("L13 pesi", 13 * 384 * 20, "= %.2f volte la TCN (parametri 100628/65204 = %.3f)" % (13 * 384 * 20 / 64128, 100628 / 65204))
print("L13 pesi / ridge pesi =", 13 * 384 * 20 / (384 * 20))
# elementi validi del train nel crop6
print("quota valida train: %.4f" % (4813620 / (1907 * 148 * 20)))
# endpoint del confronto raw
fm_blocks = 18548 * 148 * 5
print("endpoint a fine blocco sul supporto FM:", fm_blocks, "; per campione:", fm_blocks * 10, "; lettura ufficiale val:", 12468 * 1000,
      "; rapporti %.2f e %.2f" % (fm_blocks / (12468 * 1000), fm_blocks * 10 / (12468 * 1000)))
print("lag ridge sintesi 8eba5e4: %.4f%%" % imp(0.859512, 0.855321))
