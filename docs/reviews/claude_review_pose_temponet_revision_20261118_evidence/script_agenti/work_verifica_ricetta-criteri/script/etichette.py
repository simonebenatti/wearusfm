# Regola delle quattro etichette come in J:85-102 (prima condizione vera), su una griglia; nessun dato.
from fractions import Fraction as Fr
def label(g, lo, hi):
    if g >= 2 and lo > 0: return 1
    if g <= -2 and hi < 0: return 2
    if hi < 2: return 3
    return 4
vals = [Fr(i, 4) for i in range(-24, 25)]
cnt, n = {1: 0, 2: 0, 3: 0, 4: 0}, 0
for g in vals:
    for lo in vals:
        for hi in vals:
            if lo <= hi:
                cnt[label(g, lo, hi)] += 1; n += 1
print("terne", n, "conteggi per etichetta", cnt)
for g, lo, hi in [(1.5, 0.5, 1.9), (1.5, 0.5, 2.5), (-1.5, -1.9, -0.8), (0, -1, 1), (-2.6, -5, 0.3), (2.05, 0.3, 1.98), (-2.74, -4, -1.5)]:
    print((g, lo, hi), "->", label(g, lo, hi))
