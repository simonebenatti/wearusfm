# TEMPONet su emg2pose: valutazione

Validazione (15 utenti, 12468 finestre ufficiali): MAE con EMA 15.607 gradi, grezza 15.777, predittore costante 18.191 (media fra utenti).
**Cancello**: 15.607 / 18.191 = 0.858, soglia < 0.9: **superato**.

| Condizione | utenti | MAE con EMA (media ± dev. std. fra utenti) | grezza | costante | pesata per istanti | lettura | vemg2pose / NeuroPose / SensingDynamics |
|---|---|---|---|---|---|---|---|
| user | 20 | 15.04 ± 1.67 | 15.25 | 17.51 | 14.82 | fra NeuroPose e SensingDynamics | 12.2 / 13.2 / 15.5 |
| stage | 158 | 18.11 ± 1.62 | 18.36 | 19.73 | 17.99 | fra NeuroPose e SensingDynamics | 15.2 / 17.2 / 18.8 |
| user_stage | 20 | 18.14 ± 1.78 | 18.38 | 19.58 | 17.97 | fra NeuroPose e SensingDynamics | 15.8 / 17.5 / 18.7 |
