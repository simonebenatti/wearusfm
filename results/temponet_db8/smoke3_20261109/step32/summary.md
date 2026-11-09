# Replica di TEMPONet su NinaPro DB8: sintesi

- Grandezza primaria (MAE test, EMA, fold1/fold2): **9.729 gradi**, riferimento 6.89 (differenza +2.839), tolleranza 0.30: **incompleta**.
- Per seme: 9.729.
- Dev. std. sui 60 valori soggetto x DoA: 1.51 (pubblicata 2,08, IWASI 2023).
- Accuratezza a 10 e 15 gradi: 0.681 e 0.750 (pubblicate 0,814 e 0,900).
- Alpha dell'EMA: media 0.936, dev. std. 0.021 (pubblicate 0,862 e 0,044).
- Senza EMA (fold): 10.273. Modello `full` (A1+A2): 8.837 con EMA, 9.419 senza.
- Sotto la LSTM di Koch et al. (7,04): no.

| Soggetto | MAE (gradi) |
|---|---|
| 1 | 9.729 |

| DoA | MAE (gradi) |
|---|---|
| rotazione pollice | 10.886 |
| flessione pollice | 7.066 |
| flessione indice | 9.954 |
| flessione medio | 11.382 |
| flessione anulare+mignolo | 9.357 |
