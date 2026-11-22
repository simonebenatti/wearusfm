# Per Claude — chat 2, review della diagnosi TCN

Documento storico inviato per la consegna e7670758. Review ricevuta: f66691157fcd9f8588b3e58473e2fb1ac36c4dc1. Non reinviare questa richiesta; le condizioni sono trattate nella risposta `docs/reviews/pose_temponet_diagnosis_response_20261122.md` e nella proposta aggiornata.

Leggi la consegna sul ramo `origin/codex/pose-temponet-temporal-diagnosis-20261121`, base `d3e41d06afbc5d47cb988488784a4655f16584af`:

- `docs/reviews/pose_temponet_temporal_diagnosis_20261121.md`;
- prove riproducibili in `results/infra/pose_temponet_temporal_diagnosis_20261121/`;
- risultati originali in `docs/reviews/pose_temponet_controls_results_20261121.md` e nei JSON A1 richiamati.

Questa è una **review scientifica della diagnosi e del seguito proposto**, non un rilascio. A1 resta `recipe_not_admitted`; sorgenti produttivi, piani, input e soglia 50% sono invariati. Nessun nuovo training o accesso al cluster da GPT. Gli script aggiunti ricalcolano rapporti locali e fanno solo forward sintetici per una prova di rappresentabilità.

Riscontri da verificare:

1. Il controllo recupera 24,17% / 18,88% / 21,71% del margine, contro il 50% richiesto. La TCN però migliora L0 del 2,67% su E_dyn e del 30% circa quando la stessa predizione è valutata sul bersaglio pulito. Non è assenza di apprendimento.
2. Cinque LR già provati, fino a 0,01; 300 epoche al LR minimo, 150 agli altri. Selezionati 0,003 ed epoca 31. Il miglior E_dyn su tutte le curve recupererebbe solo il 21,88%; a 150 epoche il LR scelto peggiora. Non basta cambiare epoca/metrica o continuare la traiettoria.
3. Il divario fitting/esclusi indica un problema di generalizzazione del controllo, ma i punteggi fitting su y possono includere rumore memorizzato: non attribuiamo una causa unica né dichiariamo ottimo il training.
4. Il generatore temporale largo 8 è incorporabile nella candidata 64→32, con lo stesso campo ricettivo e il ramo affine già esistente. La prova usa i moduli veri e soli forward: RMSE circa 5e-7. Dimostra rappresentabilità, non apprendibilità statistica.
5. Il mondo calibrato è `y = L + 0,4 T + 1,3 epsilon`. Il margine cleanLS→oracolo di circa 8% resta sui fold esclusi, ma non certifica che metà sia recuperabile con venti utenti e una ricetta rumorosa. Non chiamiamo errata la soglia solo perché il controllo fallisce.

Proposta da rivedere:

- **D0, nessun nuovo fit:** sui checkpoint selezionato e finale del LR 0,003, riprodurre gli score pubblicati e misurare y e y* su fitting ed esclusi. Riutilizzare gli stati autenticati; niente replay automatico se mancano. Conservare e pubblicare maschere, soglie e denominatori.
- **D1, soltanto due fit diagnostici:** stesso controllo, split, TCN, inizializzazione, batch, dropout e weight decay; LR 0,003, 150 epoche fisse. Cambia solo il bersaglio di fitting, da y a y*. L'epoca rimane selezionata sulla MSE pooled rispetto a y originale. Registrare entrambe le metriche/curve, per fold e fitting; il minimo pulito è descrittivo. Niente griglia, nuovi semi, estensione automatica o lettura dei 15 utenti esterni.
- Un successo sul pulito, anche secondo il recupero originale del 50%, **non ammette la ricetta su y rumoroso**. Se il pulito generalizza e il rumoroso no, darei priorità a una singola variante più parsimoniosa/regolarizzata. Se fallisce anche il fitting pulito, indagherei l'ottimizzazione. Il warm start da ridge non è di per sé una cura del divario fra fitting e soggetti esclusi; il ramo affine c'è già.

Valuta se questa diagnosi è sostenuta, se D0/D1 separano le cause utili, se la selezione e la lettura del pulito sono corrette e se suggerisci una modifica meglio motivata. Controlla anche il costo: 8.850 update per D1; 0,205 GPUh è solo una proporzione dai tempi precedenti, non il preventivo di rilascio. Scoring aggiuntivo, memoria, I/O e margine vanno misurati nel pacchetto operativo.

Fai una sola review mirata, con blocchi prima e correzioni motivate da prove. Non modificare la ricetta per far passare il controllo e non aprire esperimenti. Registra l'esito nelle due sezioni del registro, committa, pusha e restituisci SHA. Dopo la convergenza prepareremo l'eventuale pacchetto distinto per **2-TCN head launcher**, che resta l'unico operatore.
