# TEMPONet su emg2pose: valutazione

Validazione (15 utenti, 12468 finestre ufficiali): MAE con EMA 15.682 gradi, grezza 15.858, predittore costante 18.191 (media fra utenti).
**Cancello**: 15.682 / 18.191 = 0.862, soglia < 0.9: **superato**.

| Condizione | utenti | MAE con EMA (media ± dev. std. fra utenti) | grezza | costante | pesata per istanti | lettura | vemg2pose / NeuroPose / SensingDynamics |
|---|---|---|---|---|---|---|---|
| user | 20 | 15.05 ± 1.68 | 15.26 | 17.51 | 14.82 | fra NeuroPose e SensingDynamics | 12.2 / 13.2 / 15.5 |
| stage | 158 | 18.15 ± 1.62 | 18.39 | 19.73 | 18.02 | fra NeuroPose e SensingDynamics | 15.2 / 17.2 / 18.8 |
| user_stage | 20 | 18.15 ± 1.78 | 18.38 | 19.58 | 17.98 | fra NeuroPose e SensingDynamics | 15.8 / 17.5 / 18.7 |
