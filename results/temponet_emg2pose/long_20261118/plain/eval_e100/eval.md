# TEMPONet su emg2pose: valutazione

Validazione (15 utenti, 12468 finestre ufficiali): MAE con EMA 15.124 gradi, grezza 15.385, predittore costante 18.191 (media fra utenti).
**Cancello**: 15.124 / 18.191 = 0.831, soglia < 0.9: **superato**.

| Condizione | utenti | MAE con EMA (media ± dev. std. fra utenti) | grezza | costante | pesata per istanti | lettura | vemg2pose / NeuroPose / SensingDynamics |
|---|---|---|---|---|---|---|---|
| user | 20 | 14.51 ± 1.59 | 14.82 | 17.51 | 14.32 | fra NeuroPose e SensingDynamics | 12.2 / 13.2 / 15.5 |
| stage | 158 | 17.60 ± 1.59 | 17.98 | 19.73 | 17.50 | fra NeuroPose e SensingDynamics | 15.2 / 17.2 / 18.8 |
| user_stage | 20 | 17.53 ± 1.77 | 17.90 | 19.58 | 17.37 | fra NeuroPose e SensingDynamics | 15.8 / 17.5 / 18.7 |
