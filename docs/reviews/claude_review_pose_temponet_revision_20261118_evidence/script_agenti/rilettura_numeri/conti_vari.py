# Conti indipendenti (aritmetica pura) sui numeri della bozza.
import math, datetime
from math import ceil
LN = 2 * 384
ramo = 384 * 20 + 20
L0 = LN + 384 * 20 + 20
L13 = LN + 384 * 20 * 13 + 20
mlp_w = [384, 96, 96, 96, 64, 32, 20]
mlp_W = sum(a * b for a, b in zip(mlp_w[:-1], mlp_w[1:])); mlp_b = sum(mlp_w[1:])
MLP = LN + mlp_W + mlp_b + ramo
tcn_layers = [(384, 64, 1), (64, 64, 3), (64, 64, 3), (64, 64, 3), (64, 32, 1), (32, 20, 1)]
tcn_W = sum(i * o * k for i, o, k in tcn_layers); tcn_b = sum(o for _, o, _ in tcn_layers)
TCN = LN + tcn_W + tcn_b + ramo
print('L0', L0, 'L13', L13, 'MLP', MLP, 'TCN', TCN, 'ramo', ramo)
print('pesi di trasformazione MLP', mlp_W + 384 * 20, 'TCN', tcn_W + 384 * 20)
print('MAC per finestra160', (tcn_W + 384 * 20) * 160, ' diff bias', mlp_b - tcn_b)
rf = 1 + 2 * (1 * 1 + 1 * 2 + 1 * 3)
print('campo ricettivo', rf, ' centri crop6', 160 - 12)
# matrice
def upd(L, E, S01, Sf): return 11 * L * E * S01 + 15 * E * Sf
print('aggiornamenti nominali', upd(4, 150, 60, 60), upd(6, 300, 60, 60), ' limiti', upd(4, 150, 61, 60), upd(6, 300, 61, 60))
print('5 LR', upd(5, 300, 61, 60), upd(5, 300, 60, 60))
print('traiettorie', 5*4*2 + 2*3*4*2 + 15, 5*6*2 + 2*3*6*2 + 15, ' 5 LR', 11*5*2 + 15)
print('valutazioni held max', 132 * (300 + 3 + 1), ' 5 LR', 110 * 304, ' forward fitting', 11 * 2 + 15)
print('ceil(1907/32)', ceil(1907 / 32), ' max S0+S1', ceil(1907/32) + 1)
# risorse
print('GPUh', 0 + 0 + 0 + 0.5 + 20 + 1 + 0 + 0 + 2.5, ' coreh CPU', 0.5 + 4 + 64 + 0 + 0 + 0 + 8 + 0.5 + 3,
      ' ore locali', 0.5 + 4 + 64 + 4 + 160 + 8 + 8 + 0.5 + 23, ' 24*8+80', 24 * 8 + 80)
print('imputazione %.5f%%' % (100 * 272 / 691863), ' residuo', 691863 - 272)
print('03:57:00+180s =', datetime.timedelta(hours=3, minutes=57) + datetime.timedelta(seconds=180))
print('R 4993^2 float64 MiB', 4993**2 * 8 / 2**20)
print('weight decay (1-1e-2*0.01)^9000 =', (1 - 1e-2 * 0.01) ** 9000, ' (1-3e-3*0.01)^9000 =', (1 - 3e-3*0.01) ** 9000)
print('12/11 + 40 giorni =', datetime.date(2026, 11, 12) + datetime.timedelta(days=40))
print('17,3%:', 0.7697254997148791 / 0.6561153689170307 - 1)
print('OB-3: 0,735*1,0366 =', 0.735 * 1.0366, ' gain3 a 0,7619:', 100 * (0.7619 - 0.750) / 0.7619, ' a 0,735:', 100 * (0.735 - 0.750) / 0.735)
print('BL-3: 17*(1e-4/3e-5) =', 17 * (1e-4 / 3e-5), ' (0,7269-0,7268)/0,7269 =', 100 * (0.7269 - 0.7268) / 0.7269, '%')
print('ricetta-2: 3,66% e 11,9% -> arrotondati', round(3.66, 1), round(11.9))
print('criteri-9: diff pre_ridge', 1.0321280032044746 - 1.0321280032044744)
print('ranghi', 384, 13 * 383 + 1, 'su', 385, 13 * 384 + 1)
print('dropout seed f=0,s=0..2', [200000 + 100 * 0 + s for s in range(3)])
print('quota TCN dei passi', (4 * 6 * 300 * 61 + 6 * 300 * 60) / 1477800)
