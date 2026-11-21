# Verifica indipendente del controllo temporale

`analyze.py` legge esclusivamente i sei JSON locali elencati nel risultato e ne
registra gli hash. Non esegue fit, inferenza, letture di cluster o modifiche ai
sorgenti scientifici. Si rigenera dalla radice del repository con:

```bash
python3 results/infra/pose_temponet_temporal_diagnosis_20261121/control/analyze.py
```

Il recupero del margine temporale sui fold esclusi dal fitting è 24,17%, 18,88%,
21,71% aggregato: il gate del 50% è fallito correttamente. Il recupero sui dati
di fitting è invece 86,12%, 69,25%, 77,64% aggregato. Quest'ultimo calcolo usa
le statistiche del checkpoint selezionato, realmente rivalutato dal runtime
(`train.py`, `selected_fitting_diagnostics`), e ricostruisce l'oracolo di fitting
da baseline, punteggio L13 e recupero L13 già archiviati. Non è una seconda
valutazione indipendente degli array. Il fitting può beneficiare della
memorizzazione del rumore: questi valori non dimostrano l'apprendimento della
funzione pulita.

Sul bersaglio pulito nei fold esclusi, la TCN riduce E_dyn da 0,1990 della L0 a
0,1392, circa il 30%. La calibrazione mantiene il margine dell'oracolo intorno
all'8% fra inner e outer, ma non certifica l'apprendibilità a campione finito.
Il rapporto nominale delle varianze di rumore e componente T (10,5625) non è
una misura del rapporto segnale/rumore dinamico: non include covarianze,
centratura temporale e realizzazione finita del rumore.

La diagnosi successiva proposta resta separata dall'ammissione: prima
inferenza dei checkpoint esistenti su y* e y per fitting e fold escluso; poi,
se autorizzata e preparata, due soli fit su y* con LR 0,003 e 150 epoche fisse,
senza griglia o estensioni. Selezionare sulla MSE originale rispetto a y
mantiene invariata la regola e isola il cambiamento del bersaglio di training.
La selezione resta però sensibile alla realizzazione del rumore; vanno quindi
conservate tutte le curve pulite e rumorose e il minimo della MSE pulita come
descrittivo, senza sostituirlo al checkpoint prescritto. L'eventuale passaggio
del gate con etichette pulite non ammette la ricetta originale. Nessuna nuova
soglia numerica sui punteggi puliti viene introdotta qui.
