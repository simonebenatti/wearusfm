# TEMPONet su emg2pose: valutazione

Validazione (15 utenti, 12468 finestre ufficiali): MAE con EMA 15.065 gradi, grezza 15.332, predittore costante 18.191 (media fra utenti).
**Cancello**: 15.065 / 18.191 = 0.828, soglia < 0.9: **superato**.

| Condizione | utenti | MAE con EMA (media ± dev. std. fra utenti) | grezza | costante | pesata per istanti | lettura | vemg2pose / NeuroPose / SensingDynamics |
|---|---|---|---|---|---|---|---|
| user | 20 | 14.48 ± 1.55 | 14.79 | 17.51 | 14.29 | fra NeuroPose e SensingDynamics | 12.2 / 13.2 / 15.5 |
| stage | 158 | 17.57 ± 1.59 | 17.94 | 19.73 | 17.47 | fra NeuroPose e SensingDynamics | 15.2 / 17.2 / 18.8 |
| user_stage | 20 | 17.57 ± 1.74 | 17.93 | 19.58 | 17.40 | fra NeuroPose e SensingDynamics | 15.8 / 17.5 / 18.7 |
