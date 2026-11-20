# TEMPONet su emg2pose: valutazione

Validazione (15 utenti, 12468 finestre ufficiali): MAE con EMA 15.208 gradi, grezza 15.473, predittore costante 18.191 (media fra utenti).
**Cancello**: 15.208 / 18.191 = 0.836, soglia < 0.9: **superato**.

| Condizione | utenti | MAE con EMA (media ± dev. std. fra utenti) | grezza | costante | pesata per istanti | lettura | vemg2pose / NeuroPose / SensingDynamics |
|---|---|---|---|---|---|---|---|
| user | 20 | 14.60 ± 1.60 | 14.92 | 17.51 | 14.40 | fra NeuroPose e SensingDynamics | 12.2 / 13.2 / 15.5 |
| stage | 158 | 17.76 ± 1.59 | 18.14 | 19.73 | 17.65 | fra NeuroPose e SensingDynamics | 15.2 / 17.2 / 18.8 |
| user_stage | 20 | 17.74 ± 1.74 | 18.12 | 19.58 | 17.59 | fra NeuroPose e SensingDynamics | 15.8 / 17.5 / 18.7 |
