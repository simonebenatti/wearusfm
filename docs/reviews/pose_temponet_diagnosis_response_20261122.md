# TCN D0/D1 — condizioni della review recepite e limiti delle inferenze

Data: 22/11/2026. Ramo `codex/pose-temponet-temporal-diagnosis-20261121`.
Base della revisione: `e7670758de9d0b6e25a9ad880ec3e1f7d573f169`.
Review Claude: `f66691157fcd9f8588b3e58473e2fb1ac36c4dc1`, ramo `claude/temponet-db8-20261109`, documento `docs/reviews/claude_review_pose_temponet_temporal_diagnosis_20261121.md` e prove associate.

**Esito: quattro condizioni recepite; per la seconda adotto l'alternativa esplicitamente prevista da Claude, senza ulteriori fit inizializzati dai pesi veri.** Nessun pacchetto eseguibile, rilascio o nuovo invio. La diagnosi e il seguito aggiornati sono in `docs/reviews/pose_temponet_temporal_diagnosis_20261121.md`.

## Condizioni

| Condizione | Disposizione nella proposta |
|---|---|
| 1. Lettura D1 decidibile | Ordine: invalidità tecnica; successo nei tre pannelli; fallimento con bordo 146–150; fallimento senza bordo, distinto in tutti/misti/nessuno dei pannelli fitting sopra il 50%. Bordo OR fra minimi MSE ed E_dyn pooled su y, spareggio prima esposizione. Score a 31/150/scelta su fit/esclusi e y/y*. Niente «curve in discesa» indefinito. Le etichette fitting sono descrittive, non diagnosi causali automatiche. |
| 2. Limite di D0/D1 | Scritto che il successo sul pulito non dimostra l'esistenza di una testa addestrabile su y che passi. Scelta dell'alternativa senza teacher-init; una variante successiva rimane esplorativa e senza garanzia. Fallimento del suo unico tentativo torna a Simone, senza ulteriore ricerca automatica. |
| 3. Conferma su mondo nuovo | Semi dichiarati adesso, prima di D1: base 20261122 + 1000*fold, L=base, T=base+1, rumore=base+2. Nuova calibrazione inner-train con la stessa regola; nessuna riestrazione se non diagnostica. Variante congelata prima di generare/leggere gli esiti nuovi. Mondo originale per confronto, 50% in tutti i pannelli del nuovo per conferma di sensibilità. Non sono nuovi utenti indipendenti. |
| 4. Decisione e budget | D0 è un job su Leonardo. D0, parità con update su y e due fit D1 devono avere ID, tempi, risorse, sotto-budget, addebito e residuo pubblicati, poi decisione registrata di Simone prima del primo invio. I tetti A1 non autorizzano prove nuove. Un solo via coprirà l'intera nuova catena. Nessuna richiesta di nuovo consenso su passi già autorizzati. |

### Dettagli di pacchetto accolti

D0 rilegge i 20 stati TCN e quattro stati scelti L0/MLP, recupera le loss online per epoca e calcola i riferimenti fitting direttamente dagli array. Per fold si riproduce il punto globalmente scelto; il riferimento del finale è solo pooled, come le curve originali. I singoli fold del finale diventano nuovi descrittivi. Profilo GPU originale e tolleranza di scoring `atol=rtol=1e-12`, identità di bundle/supporti/payload esatte, nessun allentamento dopo un fallimento.

La loss online salvata contiene il dropout e i pesi che cambiano nell'epoca: non è la curva eval per fold che manca nei JSON. Gli stati sono previsti e identificati dalle ricevute, ma non ho aperto i payload remoti in questa consegna. D0 ne verifica di nuovo byte e hash.

D1 usa un driver esterno al core congelato e una parità su y prima dei fit puliti, per le esposizioni 0/0,25/0,5/0,75/1 e gli update dei due fold `[0,0]`, `[8,8]`, `[15,15]`, `[22,23]`, `[29,30]`. La lettura pubblicata di questi punti è pooled. Le valutazioni aggiuntive devono preservare gli RNG; training target e scoring target vanno separati, senza sostituire il bundle che decide la selezione.

Il conto 0,205 GPUh non viene promosso a preventivo: era già dichiarato incompleto. L'ordine di grandezza Claude 1–2 GPUh riguarda un pacchetto con anche teacher-init, qui escluso; sarà necessaria una proiezione misurata del pacchetto effettivo, con costo di benchmark, D0, scoring e margini. Nessun nuovo budget assegnato da questa nota.

## Numeri aggiuntivi: confermati, con distinzione fra conto e inferenza

`results/infra/pose_temponet_temporal_diagnosis_20261121/review_response_20261122/analyze.py` ricalcola dai JSON grezzi i recuperi su scala del gate, il rapporto TCN/L0/oracolo, le differenze fitting/esclusi e l'euristica fra mondi. Nessun fitting o nuovo scoring di dati reali.

- L0 neurale −13,55/−10,82/−12,28%, MLP −12,24/−10,65/−11,50%, L0 in forma chiusa −19,10/−16,38/−17,84%: confermati. Recupero TCN rispetto alla distanza L0-neurale→oracolo 33,22/26,80/30,27%, richiesto pooled 55,47%: confermato. Non cambia l'esito del gate.
- Eccesso TCN nel mondo lineare, espresso in unità del margine temporale: 19,74/18,70%; somma euristica 43,91/37,57%: confermata aritmeticamente. **Non è identificata la scomposizione funzione/rumore.** `controls.py`, `evaluate_outer_references` ricalcola anche soglia e supporto dinamico a seconda del mondo. Per esempio le soglie fold0 sono 0,45974 e 0,54374. Non accetto come conclusione «anche senza rumore resterebbe sotto il 50%»: quel controfattuale è precisamente D1.
- Scarto TCN fitting/esclusi fra quelli di L0 e L13 su y, cleanLS quasi stabile: confermato. **Compatibile con sola memorizzazione del rumore**, senza prova che il residuo non lineare generalizzi altrettanto bene del riferimento lineare. La frase della review «i numeri separano già le due cose» è troppo forte; D0 serve proprio a misurarle meglio.

### Prova del limite della scomposizione

Sul medesimo supporto mascherato e centrato, con `e = prediction − y*`, `n = y − y*`:

`SSE_y = SSE_y* + SSE_oracle − 2<e,n>`.

Il termine incrociato non si può porre a zero sulla realizzazione osservata, tanto meno dopo fitting o selezione. Il nuovo script verifica l'identità su due esempi deterministici a segno opposto del termine incrociato. D0 potrà ricavarlo direttamente dai tre accumulatori sullo stesso supporto. Perciò 86,1/69,3% non sono limiti deterministici del recupero pulito. I soli score rumorosi non misurano quanto del residuo pulito sia stato appreso.

## Perché non aggiungo il riferimento inizializzato dalla verità

La prova `architecture/oracle_embedding.py` fornisce al lettore i pesi esatti di L e T e la normalizzazione. Gli zeri nei canali in più possono mantenerli inattivi sotto ReLU/AdamW; questo non elimina l'informazione privilegiata dell'inizializzazione. Un successo a 31/150 mostra che quella traiettoria conserva parte di una soluzione già fornita. Un fallimento mostra che quella traiettoria la degrada. Nessuno dei due decide se una testa partita senza la soluzione possa impararla dai soli esempi rumorosi, né se sia sbagliata la specifica del controllo.

È un possibile controllo di stabilità, ma non chiude il limite di apprendibilità che motiverebbe i fit extra. Scelgo quindi l'alternativa ammessa dalla review: dichiarare il limite e un solo eventuale tentativo sul nuovo mondo. Nessuna promessa che una riduzione di canali sia il rimedio; il risultato giocattolo 27/44% resta un avvertimento, non un risultato sui latenti veri.

Warm start e altre epoche sulla medesima traiettoria non sono la prima cura suggerita dai risultati. Non li chiamo universalmente «esclusi»: ridge-init nella TCN cambierebbe i gradienti congiunti e non è stato provato; uno schedule diverso non è la continuazione a LR costante già osservata. Non propongo nessuno dei due nel pacchetto corrente.

## Validazione e stato

Rilettura scientifica e operativa locale in parallelo: confermate le condizioni, corretti i limiti delle inferenze sopra e verificato il formato degli stati dal codice. Aritmetica aggiuntiva riprodotta con hash di input e script; comandi, esiti e hash dei log nel file di validazione della nuova cartella. Nessun test del pacchetto operativo dichiarato: non è stato scritto.

Modificati solo documenti, registro e prove di aritmetica locale fuori dai sorgenti produttivi. A1 resta `recipe_not_admitted`. Nessun accesso al cluster, job, nuova calibrazione, variante o rilascio. Questa consegna chiude la revisione documentale richiesta e identifica esattamente che cosa dovrà contenere il pacchetto operativo da far rivedere.
