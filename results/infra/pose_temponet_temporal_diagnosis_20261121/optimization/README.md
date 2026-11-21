# Diagnostica delle curve del controllo temporale

Analisi dei soli JSON archiviati. Nessun training, invio, lettura del cluster o punteggio della posa vera.

## Il risultato non dipende dalla scelta MSE invece di E_dyn

| Pannello interno | Clean-LS | Oracolo | Limite per recuperare 50% | TCN selezionata | Recupero |
|---|---:|---:|---:|---:|---:|
| fold0 | 0.710431 | 0.653451 | 0.681941 | 0.696657 | 24.17% |
| fold1 | 0.749920 | 0.691103 | 0.720512 | 0.738818 | 18.88% |
| pooled | 0.728479 | 0.670659 | 0.699569 | 0.715926 | 21.71% |

Il minimo E_dyn fra **tutti** gli LR e checkpoint salvati è 0.715826246; recupera 21.88% del margine. Quindi nessun checkpoint registrato raggiunge il gate pooled. Il costo della selezione MSE è solo 0.000099975 di E_dyn.

## Curve della TCN temporale

| LR | Orizzonte | Epoca min MSE | Min MSE | E_dyn lì | Epoca min E_dyn | Min E_dyn | E_dyn finale |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0001 | 300 | 181 | 1.818162 | 0.725644 | 201 | 0.725580 | 0.728150 |
| 0.0003 | 150 | 64 | 1.817431 | 0.725382 | 64 | 0.725382 | 0.731910 |
| 0.001 | 150 | 31 | 1.813964 | 0.722991 | 28 | 0.722228 | 0.751496 |
| 0.003 | 150 | 31 | 1.800145 | 0.715926 | 27 | 0.715826 | 0.741556 |
| 0.01 | 150 | 93 | 1.849272 | 0.737384 | 111 | 0.737066 | 0.737611 |

La selezione è LR 0,003, epoca 31; il minimo dinamico è all'epoca 27. Il LR 0,01 è stato provato e il LR 0,0001 ha già raggiunto 300 epoche. Non è il vecchio problema delle sole 50 epoche.
Alla fine delle 150 epoche del LR selezionato, E_dyn è peggiore del 3.58% e MSE del 2.89%. Questo sconsiglia una semplice continuazione; non dimostra che un altro schedule non possa aiutare.

## Fitting e utenti interni esclusi, al checkpoint scelto

| Testa | Fold | MSE fitting | MSE esclusi | E_dyn fitting | E_dyn esclusi | E_dyn esclusi contro target pulito |
|---|---|---:|---:|---:|---:|---:|
| temporal_L0 | 0 | 1.767544 | 1.835392 | 0.708789 | 0.718154 | 0.189922 |
| temporal_L0 | 1 | 1.779941 | 1.836917 | 0.748496 | 0.756284 | 0.211195 |
| temporal_MLP | 0 | 1.726989 | 1.833087 | 0.688138 | 0.717406 | 0.185055 |
| temporal_MLP | 1 | 1.745353 | 1.836623 | 0.734149 | 0.756187 | 0.210741 |
| temporal_TCN_post | 0 | 1.688067 | 1.797752 | 0.661873 | 0.696657 | 0.127856 |
| temporal_TCN_post | 1 | 1.709501 | 1.802574 | 0.710583 | 0.738818 | 0.154431 |

La TCN migliora E_dyn del 2.67% rispetto a L0 sulle finestre escluse. Quindi non è una testa che non impara: non recupera abbastanza del segnale temporale impiantato.
Il divario fitting/esclusi e il peggioramento tardivo sono compatibili con un problema di generalizzazione, rumore e complessità del lettore. Non bastano per attribuire causalmente il fallimento a overfitting: mancano curve fitting e errori sul target pulito nel fitting.

## Un indizio indipendente dal training neurale

| Fold | Lettore in forma chiusa, fit su y | E_dyn fitting | E_dyn esclusi |
|---|---|---:|---:|
| fold0 | L0 | 0.705764 | 0.721315 |
| fold0 | L13 | 0.667239 | 0.759182 |
| fold1 | L0 | 0.745870 | 0.759555 |
| fold1 | L13 | 0.708586 | 0.790777 |

L13 riduce l'errore sul fitting ma peggiora sugli utenti interni esclusi anche con soluzione in forma chiusa. È evidenza di un problema statistico del controllo con contesto, non una prova che la TCN abbia raggiunto il proprio ottimo.

## Limiti e conseguenze

- CV selection seed is 0; fold model initialization is shared, trajectories differ by fold.
- Saved full curves are pooled across two internal held folds, not per-fold curves.
- Fitting diagnostics exist only for the selected checkpoint and score the noisy fitted target.
- Thus no train/validation trajectory comparison or clean fitting error can be reconstructed.
- Absent optimizer alarms certify only the prespecified edge checks, not an optimum.

1. Non abbassare il gate e non cambiare la selezione dopo questi risultati. La ricetta corrente resta non ammessa; nessun risultato negativo sul compito reale o sull'encoder.
2. Non proporre soltanto più epoche: lungo le traiettorie osservate il minimo è già interno. Un nuovo schedule o una nuova inizializzazione richiederebbero un emendamento e un controllo, non una garanzia di soluzione.
3. Prima di nuovi tentativi, distinguere ottimizzazione/capacità e generalizzazione con lo stesso controllo temporale: errore pulito e rumoroso su fitting ed esclusi; traiettorie per fold, contributo del ramo lineare e di quello temporale; un controllo senza rumore, a architettura invariata. Sono proposte, non operazioni eseguite.
4. Se la ricetta apprende il bersaglio pulito ma non generalizza sotto il rumore prescritto, valutare regolarizzazione e dimensione/struttura del lettore o più finestre indipendenti, mantenendo il gate e congelando la scelta prima del test reale. Se non apprende neppure il pulito, indagare architettura/ottimizzazione/supporto del bersaglio. La scelta dipende anche dalla costruzione del controllo, oggetto di un audit separato.

Riproduzione: `python3 results/infra/pose_temponet_temporal_diagnosis_20261121/optimization/analyze.py`. Gli hash degli input e tutti i numeri non arrotondati sono in `analysis.json`.
