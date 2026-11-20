# TEMPONet su emg2pose: valutazione

Validazione (15 utenti, 12468 finestre ufficiali): MAE con EMA 15.139 gradi, grezza 15.406, predittore costante 18.191 (media fra utenti).
**Cancello**: 15.139 / 18.191 = 0.832, soglia < 0.9: **superato**.

| Condizione | utenti | MAE con EMA (media ± dev. std. fra utenti) | grezza | costante | pesata per istanti | lettura | vemg2pose / NeuroPose / SensingDynamics |
|---|---|---|---|---|---|---|---|
| user | 20 | 14.56 ± 1.62 | 14.88 | 17.51 | 14.36 | fra NeuroPose e SensingDynamics | 12.2 / 13.2 / 15.5 |
| stage | 158 | 17.69 ± 1.59 | 18.08 | 19.73 | 17.59 | fra NeuroPose e SensingDynamics | 15.2 / 17.2 / 18.8 |
| user_stage | 20 | 17.60 ± 1.82 | 17.97 | 19.58 | 17.44 | fra NeuroPose e SensingDynamics | 15.8 / 17.5 / 18.7 |
