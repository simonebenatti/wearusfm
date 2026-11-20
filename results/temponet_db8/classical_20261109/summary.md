# Baseline classiche su NinaPro DB8 (passo 1 della diagnosi di TEMPONet)

- Cancello (SVR, riferimento 7,28 ± 0,30): **indizio H1** (fold 8.279, full 7.366).

| Modello | fold con EMA | senza EMA | full con EMA | acc. 10 (fold) | acc. 15 (fold) | alpha (fold) | pubblicato |
|---|---|---|---|---|---|---|---|
| SVR | 8.279 | 8.944 | 7.366 | 0.733 | 0.859 | 0.715 | 7.28 (0.795, 0.883) |
| MLP | 7.691 | 8.720 | 6.824 | 0.778 | 0.894 | 0.896 | 7.14 (0.799, 0.889) |

| Gruppo | SVR fold | SVR full | MLP fold | MLP full |
|---|---|---|---|---|
| normodotati | 8.215 | 7.232 | 7.517 | 6.581 |
| amputati | 8.598 | 8.032 | 8.561 | 8.038 |

| Soggetto | SVR fold | SVR full | MLP fold | MLP full |
|---|---|---|---|---|
| 1 | 9.213 | 8.065 | 8.279 | 6.896 |
| 2 | 7.288 | 6.429 | 6.570 | 5.801 |
| 3 | 8.487 | 7.848 | 7.702 | 7.264 |
| 4 | 8.319 | 7.139 | 7.635 | 6.696 |
| 5 | 8.576 | 7.863 | 8.195 | 7.236 |
| 6 | 7.715 | 6.977 | 7.092 | 6.144 |
| 7 | 7.628 | 6.218 | 6.995 | 5.824 |
| 8 | 8.905 | 7.603 | 8.201 | 6.932 |
| 9 | 8.662 | 7.742 | 7.809 | 7.340 |
| 10 | 7.354 | 6.439 | 6.694 | 5.676 |
| 11 (amputato) | 6.778 | 6.295 | 6.836 | 6.142 |
| 12 (amputato) | 10.419 | 9.769 | 10.286 | 9.933 |

C scelti (fold): {'0.1': 0, '1.0': 2, '10.0': 21, '100.0': 97}; MLP arrivati a convergenza: 0.00.
