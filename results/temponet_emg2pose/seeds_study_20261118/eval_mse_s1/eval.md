# TEMPONet su emg2pose: valutazione

Validazione (15 utenti, 12468 finestre ufficiali): MAE con EMA 15.601 gradi, grezza 15.784, predittore costante 18.191 (media fra utenti).
**Cancello**: 15.601 / 18.191 = 0.858, soglia < 0.9: **superato**.

| Condizione | utenti | MAE con EMA (media ± dev. std. fra utenti) | grezza | costante | pesata per istanti | lettura | vemg2pose / NeuroPose / SensingDynamics |
|---|---|---|---|---|---|---|---|
| user | 20 | 15.02 ± 1.70 | 15.24 | 17.51 | 14.81 | fra NeuroPose e SensingDynamics | 12.2 / 13.2 / 15.5 |
| stage | 158 | 18.20 ± 1.63 | 18.46 | 19.73 | 18.08 | fra NeuroPose e SensingDynamics | 15.2 / 17.2 / 18.8 |
| user_stage | 20 | 18.19 ± 1.77 | 18.43 | 19.58 | 18.03 | fra NeuroPose e SensingDynamics | 15.8 / 17.5 / 18.7 |
