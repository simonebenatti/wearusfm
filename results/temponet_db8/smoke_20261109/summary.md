# Replica di TEMPONet su NinaPro DB8: sintesi

- Grandezza primaria (MAE test, EMA, fold1/fold2): **23.033 gradi**, riferimento 6.89 (differenza +16.143), tolleranza 0.30: **incompleta**.
- Per seme: 23.033.
- Dev. std. sui 60 valori soggetto x DoA: 5.92 (pubblicata 2,08, IWASI 2023).
- Accuratezza a 10 e 15 gradi: 0.032 e 0.316 (pubblicate 0,814 e 0,900).
- Alpha dell'EMA: media 0.710, dev. std. 0.295 (pubblicate 0,862 e 0,044).
- Senza EMA (fold): 23.034. Modello `full` (A1+A2): 19.984 con EMA, 19.997 senza.
- Sotto la LSTM di Koch et al. (7,04): no.

| Soggetto | MAE (gradi) |
|---|---|
| 1 | 21.715 |
| 2 | 25.632 |
| 3 | 23.614 |
| 4 | 21.171 |

| DoA | MAE (gradi) |
|---|---|
| rotazione pollice | 14.900 |
| flessione pollice | 24.553 |
| flessione indice | 27.712 |
| flessione medio | 25.434 |
| flessione anulare+mignolo | 22.565 |
