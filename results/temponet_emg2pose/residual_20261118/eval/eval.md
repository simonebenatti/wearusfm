# TEMPONet su emg2pose: valutazione

Validazione (15 utenti, 12468 finestre ufficiali): MAE con EMA 15.083 gradi, grezza 15.341, predittore costante 18.191 (media fra utenti).
**Cancello**: 15.083 / 18.191 = 0.829, soglia < 0.9: **superato**.

| Condizione | utenti | MAE con EMA (media ± dev. std. fra utenti) | grezza | costante | pesata per istanti | lettura | vemg2pose / NeuroPose / SensingDynamics |
|---|---|---|---|---|---|---|---|
| user | 20 | 14.56 ± 1.57 | 14.86 | 17.51 | 14.36 | fra NeuroPose e SensingDynamics | 12.2 / 13.2 / 15.5 |
| stage | 158 | 17.62 ± 1.59 | 17.98 | 19.73 | 17.52 | fra NeuroPose e SensingDynamics | 15.2 / 17.2 / 18.8 |
| user_stage | 20 | 17.58 ± 1.86 | 17.93 | 19.58 | 17.41 | fra NeuroPose e SensingDynamics | 15.8 / 17.5 / 18.7 |
