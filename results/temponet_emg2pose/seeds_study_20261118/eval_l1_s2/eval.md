# TEMPONet su emg2pose: valutazione

Validazione (15 utenti, 12468 finestre ufficiali): MAE con EMA 15.141 gradi, grezza 15.403, predittore costante 18.191 (media fra utenti).
**Cancello**: 15.141 / 18.191 = 0.832, soglia < 0.9: **superato**.

| Condizione | utenti | MAE con EMA (media ± dev. std. fra utenti) | grezza | costante | pesata per istanti | lettura | vemg2pose / NeuroPose / SensingDynamics |
|---|---|---|---|---|---|---|---|
| user | 20 | 14.59 ± 1.57 | 14.89 | 17.51 | 14.39 | fra NeuroPose e SensingDynamics | 12.2 / 13.2 / 15.5 |
| stage | 158 | 17.71 ± 1.59 | 18.08 | 19.73 | 17.61 | fra NeuroPose e SensingDynamics | 15.2 / 17.2 / 18.8 |
| user_stage | 20 | 17.62 ± 1.81 | 17.98 | 19.58 | 17.45 | fra NeuroPose e SensingDynamics | 15.8 / 17.5 / 18.7 |
