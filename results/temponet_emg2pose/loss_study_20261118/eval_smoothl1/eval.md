# TEMPONet su emg2pose: valutazione

Validazione (15 utenti, 12468 finestre ufficiali): MAE con EMA 15.278 gradi, grezza 15.490, predittore costante 18.191 (media fra utenti).
**Cancello**: 15.278 / 18.191 = 0.840, soglia < 0.9: **superato**.

| Condizione | utenti | MAE con EMA (media ± dev. std. fra utenti) | grezza | costante | pesata per istanti | lettura | vemg2pose / NeuroPose / SensingDynamics |
|---|---|---|---|---|---|---|---|
| user | 20 | 14.75 ± 1.61 | 14.99 | 17.51 | 14.53 | fra NeuroPose e SensingDynamics | 12.2 / 13.2 / 15.5 |
| stage | 158 | 17.73 ± 1.60 | 18.03 | 19.73 | 17.63 | fra NeuroPose e SensingDynamics | 15.2 / 17.2 / 18.8 |
| user_stage | 20 | 17.67 ± 1.85 | 17.95 | 19.58 | 17.50 | fra NeuroPose e SensingDynamics | 15.8 / 17.5 / 18.7 |
