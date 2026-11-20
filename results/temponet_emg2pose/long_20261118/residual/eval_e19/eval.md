# TEMPONet su emg2pose: valutazione

Validazione (15 utenti, 12468 finestre ufficiali): MAE con EMA 15.059 gradi, grezza 15.318, predittore costante 18.191 (media fra utenti).
**Cancello**: 15.059 / 18.191 = 0.828, soglia < 0.9: **superato**.

| Condizione | utenti | MAE con EMA (media ± dev. std. fra utenti) | grezza | costante | pesata per istanti | lettura | vemg2pose / NeuroPose / SensingDynamics |
|---|---|---|---|---|---|---|---|
| user | 20 | 14.50 ± 1.59 | 14.80 | 17.51 | 14.30 | fra NeuroPose e SensingDynamics | 12.2 / 13.2 / 15.5 |
| stage | 158 | 17.58 ± 1.61 | 17.96 | 19.73 | 17.48 | fra NeuroPose e SensingDynamics | 15.2 / 17.2 / 18.8 |
| user_stage | 20 | 17.49 ± 1.80 | 17.85 | 19.58 | 17.33 | fra vemg2pose e NeuroPose | 15.8 / 17.5 / 18.7 |
