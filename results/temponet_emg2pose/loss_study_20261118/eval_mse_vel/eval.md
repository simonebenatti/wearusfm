# TEMPONet su emg2pose: valutazione

Validazione (15 utenti, 12468 finestre ufficiali): MAE con EMA 16.161 gradi, grezza 16.249, predittore costante 18.191 (media fra utenti).
**Cancello**: 16.161 / 18.191 = 0.888, soglia < 0.9: **superato**.

| Condizione | utenti | MAE con EMA (media ± dev. std. fra utenti) | grezza | costante | pesata per istanti | lettura | vemg2pose / NeuroPose / SensingDynamics |
|---|---|---|---|---|---|---|---|
| user | 20 | 15.64 ± 1.72 | 15.74 | 17.51 | 15.39 | sopra SensingDynamics | 12.2 / 13.2 / 15.5 |
| stage | 158 | 18.53 ± 1.63 | 18.66 | 19.73 | 18.40 | fra NeuroPose e SensingDynamics | 15.2 / 17.2 / 18.8 |
| user_stage | 20 | 18.45 ± 1.79 | 18.57 | 19.58 | 18.29 | fra NeuroPose e SensingDynamics | 15.8 / 17.5 / 18.7 |
