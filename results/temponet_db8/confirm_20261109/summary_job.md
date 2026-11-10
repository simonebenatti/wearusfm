# Replica di TEMPONet su NinaPro DB8: sintesi

- Grandezza primaria (MAE test, EMA, modello full): **7.585 gradi**, riferimento 6.89 (differenza +0.695), tolleranza 0.30: **non replicata**.
- Per seme: 8.517, 8.547, 8.463 (dev. std. 0.042).
- Dev. std. sui 60 valori soggetto x DoA: 2.20 (pubblicata 2,08, IWASI 2023).
- Accuratezza a 10 e 15 gradi: 0.704 e 0.847 (pubblicate 0,814 e 0,900).
- Alpha dell'EMA: media 0.928, dev. std. 0.028 (pubblicate 0,862 e 0,044).
- Senza EMA (fold): 9.010. Modello `full` (A1+A2; semi [0, 1, 2], 36 unita'): 7.585 con EMA, 7.990 senza; fold con il solo seme 0: 8.517.
- Sotto la LSTM di Koch et al. (7,04): no.

| Soggetto | MAE (gradi) |
|---|---|
| 1 | 8.943 |
| 2 | 7.453 |
| 3 | 8.513 |
| 4 | 8.716 |
| 5 | 8.336 |
| 6 | 7.900 |
| 7 | 8.294 |
| 8 | 8.987 |
| 9 | 8.945 |
| 10 | 7.694 |
| 11 (amputato) | 7.395 |
| 12 (amputato) | 10.934 |

Normodotati e amputati separati (descrittivo, nessun verdetto per gruppo; il criterio resta sui 12 soggetti):

| Gruppo | Soggetti | MAE fold con EMA | senza EMA | `full` | acc. 10 gradi | acc. 15 gradi | alpha medio |
|---|---|---|---|---|---|---|---|
| normodotati | 1-10 | 8.378 | 8.852 | 7.394 | 0.710 | 0.854 | 0.924 |
| amputati | 11-12 | 9.165 | 9.803 | 8.540 | 0.672 | 0.815 | 0.950 |

| DoA | MAE (gradi) |
|---|---|
| rotazione pollice | 11.099 |
| flessione pollice | 6.270 |
| flessione indice | 7.847 |
| flessione medio | 9.298 |
| flessione anulare+mignolo | 8.032 |
