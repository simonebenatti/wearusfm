# TEMPONet su emg2pose: valutazione

Validazione (15 utenti, 12468 finestre ufficiali): MAE con EMA 15.171 gradi, grezza 15.433, predittore costante 18.191 (media fra utenti).
**Cancello**: 15.171 / 18.191 = 0.834, soglia < 0.9: **superato**.

| Condizione | utenti | MAE con EMA (media ± dev. std. fra utenti) | grezza | costante | pesata per istanti | lettura | vemg2pose / NeuroPose / SensingDynamics |
|---|---|---|---|---|---|---|---|
| user | 20 | 14.57 ± 1.59 | 14.88 | 17.51 | 14.38 | fra NeuroPose e SensingDynamics | 12.2 / 13.2 / 15.5 |
| stage | 158 | 17.66 ± 1.59 | 18.04 | 19.73 | 17.56 | fra NeuroPose e SensingDynamics | 15.2 / 17.2 / 18.8 |
| user_stage | 20 | 17.69 ± 1.72 | 18.05 | 19.58 | 17.53 | fra NeuroPose e SensingDynamics | 15.8 / 17.5 / 18.7 |
