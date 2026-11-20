# Replica di TEMPONet su NinaPro DB8: sintesi

- Grandezza primaria (MAE test, EMA, fold1/fold2): **8.795 gradi**, riferimento 6.89 (differenza +1.905), tolleranza 0.30: **non replicata**.
- Per seme: 8.754, 8.877, 8.755 (dev. std. 0.071).
- Dev. std. sui 60 valori soggetto x DoA: 2.21 (pubblicata 2,08, IWASI 2023).
- Accuratezza a 10 e 15 gradi: 0.691 e 0.822 (pubblicate 0,814 e 0,900).
- Alpha dell'EMA: media 0.931, dev. std. 0.030 (pubblicate 0,862 e 0,044).
- Senza EMA (fold): 9.356. Modello `full` (A1+A2; semi [0], 12 unita'): 7.778 con EMA, 8.241 senza; fold con il solo seme 0: 8.754.
- Sotto la LSTM di Koch et al. (7,04): no.

| Soggetto | MAE (gradi) |
|---|---|
| 1 | 9.573 |
| 2 | 7.572 |
| 3 | 8.592 |
| 4 | 9.142 |
| 5 | 8.310 |
| 6 | 8.284 |
| 7 | 8.216 |
| 8 | 9.108 |
| 9 | 9.589 |
| 10 | 8.414 |
| 11 | 7.531 |
| 12 | 11.215 |

| DoA | MAE (gradi) |
|---|---|
| rotazione pollice | 11.394 |
| flessione pollice | 6.466 |
| flessione indice | 8.293 |
| flessione medio | 9.479 |
| flessione anulare+mignolo | 8.344 |
