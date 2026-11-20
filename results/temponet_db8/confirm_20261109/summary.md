# Replica di TEMPONet su NinaPro DB8: sintesi

- Grandezza primaria (MAE test, EMA, modello full): **7.585 gradi**, riferimento 6.89 (differenza +0.695), tolleranza 0.30: **non replicata**.
- `full` per seme: 7.651, 7.507, 7.597 (dev. std. 0.073); senza EMA 7.990.
- `full`, dev. std. sui 60 valori soggetto x DoA: 2.05 (pubblicata 2,08, IWASI 2023).
- `full`, accuratezza a 10 e 15 gradi: 0.743 e 0.889 (pubblicate 0,814 e 0,900).
- `full`, alpha dell'EMA (dalle predizioni fuori fold): media 0.929, dev. std. 0.027 (pubblicate 0,862 e 0,044).
- Fold (secondaria): 8.509 con EMA (per seme 8.517, 8.547, 8.463; fold1 9.043, fold2 7.975), 9.010 senza; accuratezza 0.704 e 0.847; alpha medio 0.928.
- Sotto la LSTM di Koch et al. (7,04): no.

| Soggetto | `full` (gradi) | fold (gradi) |
|---|---|---|
| 1 | 7.882 | 8.943 |
| 2 | 6.688 | 7.453 |
| 3 | 7.524 | 8.513 |
| 4 | 7.985 | 8.716 |
| 5 | 7.587 | 8.336 |
| 6 | 7.061 | 7.900 |
| 7 | 7.024 | 8.294 |
| 8 | 7.684 | 8.987 |
| 9 | 7.510 | 8.945 |
| 10 | 6.995 | 7.694 |
| 11 (amputato) | 6.574 | 7.395 |
| 12 (amputato) | 10.506 | 10.934 |

Normodotati e amputati separati (descrittivo, nessun verdetto per gruppo; il criterio resta sui 12 soggetti):

| Gruppo | Soggetti | `full` | `full` acc. 10 / 15 | fold con EMA | fold senza EMA | fold acc. 10 / 15 | fold alpha medio |
|---|---|---|---|---|---|---|---|
| normodotati | 1-10 | 7.394 | 0.752 / 0.898 | 8.378 | 8.852 | 0.710 / 0.854 | 0.924 |
| amputati | 11-12 | 8.540 | 0.697 / 0.844 | 9.165 | 9.803 | 0.672 / 0.815 | 0.950 |

| DoA | `full` (gradi) | fold (gradi) |
|---|---|---|
| rotazione pollice | 9.970 | 11.099 |
| flessione pollice | 5.701 | 6.270 |
| flessione indice | 6.825 | 7.847 |
| flessione medio | 8.164 | 9.298 |
| flessione anulare+mignolo | 7.265 | 8.032 |
