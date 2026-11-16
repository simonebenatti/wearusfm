# Regressione WearUsFM: evidenze e prossime domande

16/11/2026, GPT coordinamento. Sintesi richiesta da Simone, dopo la discussione su teste temporali e contrastiva. Riletti i risultati pubblicati nelle diverse linee; `git fetch origin` completato. Le nuove evidenze scientifiche qui incluse arrivano al 12/11. Non è un controllo live dei job del 16/11. Nessun job, fit, training o modifica ai protocolli esistenti.

**La regressione resta debole, ma il backbone impara movimento: la ricetta lunga migliora sia la testa già allenata sia una ridge nuova sulle rappresentazioni congelate. Il log-quadratico non aggiunge un beneficio sulla posa. TEMPONet funziona come baseline autonoma su EMG grezzo, ma non è ancora stata provata come testa del FM. La contrastiva è ancora una proposta e quella scritta il 07/11 riguarda soprattutto la classificazione.**

## Che cosa abbiamo fatto

| Prova | Risultato osservato | Conclusione sostenuta |
|---|---|---|
| Audit dei bersagli, ingest, identità e finestre | 100/100 controlli aggregati nei 20 recording esaminati; oracoli e maschere verificati | Nessun difetto dimostrato nella catena campionata. Non certifica tutto il corpus né un allineamento fisiologico perfetto; il vecchio «allineamento 10/10» era una lettura da correggere. |
| Lettore temporale lineare sul vecchio B | Ridge con t−100 ms, t, t+100 ms: E_dyn 0,859512→0,855321, circa −0,49%; varianza predetta diminuisce | Un poco di informazione aggiuntiva leggibile, nessun recupero dell'ampiezza. Non era una testa temporale non lineare. |
| Informazione oltre la media degli slot | Sette contrasti fissi aggiunti alla media completa: E_dyn B 0,859512→0,859406, −0,0123% relativo | Nessuna soluzione pratica con questa vista e lettore. Non esclude slot completi, pooling appreso o lettori non lineari. |
| Tokenizer 25 ms lineare → 100 ms + GELU, a 19.618 passi | E_dyn 0,851975→0,838933, −1,53% relativo; EPN frozen 81,601→82,577% | Beneficio reale ma piccolo sulla posa; risultato EPN favorevole. Cambiano insieme finestra e attivazione nel contrasto combinato. |
| Ricetta lunga 30M, T100-G, 19.618→49.046 | E_dyn ridge 0,839→0,786; testa 0,833→0,779. Riduzioni relative 6,32% e 6,52%, entrambi gli IC favorevoli e oltre il criterio pratico del 2% | Migliora anche il movimento, non solo la postura. Il confronto include il decadimento del learning rate; non isola il solo numero di passi. |
| Scaling 30M→300M a 5.000 passi | E_dyn 0,899816→0,899931, delta +0,000115 [−0,003387; +0,003778] | Nessun beneficio sulla posa a questa dose. Non è una bocciatura dello scaling: 300M a 19.618 è allenato secondo la consegna dell'11/11, ma manca un report scientifico pubblicato della valutazione. |
| Log-quadratico aggiunto a T100-L, a 19.618 | E_dyn 0,843363→0,844088, delta +0,000726 [−0,001190; +0,002724] | Nessun vantaggio dinamico aggiuntivo; leggermente peggiore di GELU. EPN invece +1,531 pp sul lineare, senza superiorità dimostrata sulla GELU. |
| TEMPONet autonoma da EMG grezzo | Modello da 71.412 parametri, 128 ms, 20 angoli, 158 utenti train; risultati sotto | Baseline supervisionata utile. Non è la TEMPONet sui latenti né un nuovo pretraining del FM. |

I valori del vecchio B usano student/crop4, quelli nuovi del tokenizer e del lungo teacher/crop6. Le righe sono confronti interni alle rispettive prove: non formano una graduatoria unica fra tutti i checkpoint.

## Il risultato più informativo: il lungo e i due livelli del modello

| Finestre dinamiche | Ridge post 19k | Ridge post 39k | Ridge post 49k | Testa allenata 19k | Testa allenata 49k |
|---|---:|---:|---:|---:|---:|
| E_dyn, minore è meglio | 0,839 | 0,810 | 0,786 | 0,833 | 0,779 |
| Correlazione centrata | 0,401 | 0,436 | 0,463 | 0,410 | 0,471 |
| Varianza predetta / bersaglio | 16,9% | 19,9% | 22,1% | 19,2% | 23,9% |

La riduzione relativa primaria è 6,32% [5,87; 6,83] per la ridge e 6,52% [5,92; 7,15] per la testa. Il controllo allineato/disallineato E2 è coerente: il gap cresce perché migliora la predizione allineata. Prima del backbone la ridge rimane debole: E_dyn 1,035→1,032, correlazione 0,074→0,079. A 49k il lettore post-backbone raggiunge 0,786 e 0,463. Il backbone rende la dinamica molto più accessibile a questo lettore; non abbiamo dimostrato che prima del backbone sia assente per qualsiasi decoder.

E_dyn è errore quadratico delle traiettorie centrate diviso per la varianza centrata dei bersagli: 1 corrisponde a nessuna variazione temporale predetta. Non è un'accuracy. Il 22–24% di varianza predetta corrisponde a circa 47–49% in deviazione standard aggregata, non a un'ampiezza del 22–24%. Aumentare artificialmente l'ampiezza non risolve una correlazione intorno a 0,47.

Quindi la diagnosi «il FM non impara movimento» è superata; «il FM ricostruisce bene il movimento» non è sostenuta. Sono 15 utenti esclusi dal training ma già usati come sviluppo, un seme, IC per soggetto condizionati ai checkpoint/fit. La posa è già supervisionata nel pretraining: questo non è trasferimento a una regressione nuova.

## TEMPONet: ciò che è stato davvero provato dopo il 6 novembre

Il ramo Claude `1049c45`, conclusioni firmate il 12/11, contiene una TEMPONet allenata da zero sul segnale, con loss L1 e protocollo emg2pose distinto da quello del FM.

| Test ufficiale | MAE TEMPONet con filtro EMA | Uscita grezza | Costante |
|---|---:|---:|---:|
| Utenti nuovi | 14,55° | 14,86° | 17,51° |
| Fasi nuove | 17,67° | 18,06° | 19,73° |
| Utenti e fasi nuovi | 17,60° | 17,97° | 19,58° |

Qui EMA è un filtro delle predizioni, non il teacher del FM. Un solo seme; guadagno rispetto alla costante di circa 17% e 10%/10%. La MAE migliore della costante non dimostra da sola buona ricostruzione delle variazioni: servono le metriche dinamiche sullo stesso supporto. Non confrontare direttamente questi gradi con E_dyn o con le MAE dei sottoinsiemi FM. Il test ufficiale è già stato osservato, e non torna intatto per selezionare nuove varianti.

La replica TEMPONet su NinaPro DB8 si è invece chiusa a 7,585° contro 6,89° pubblicati, fuori dalla tolleranza fissata. Nella stessa catena una MLP su Waveform Length fa 6,82° (descrittivo). DB8 è un confronto entro soggetto, non la prova su utenti nuovi di emg2pose. Queste evidenze non autorizzano a dare TEMPONet per superiore in ogni condizione. Le correzioni specifiche DB8 non dimostrano un difetto della vecchia pipeline FM.

## Cosa funziona e cosa resta aperto

- **Funziona:** training stabile della posa oltre il collaudo breve; maggiore accessibilità della dinamica dopo il backbone; miglioramento pratico del lungo; classificazione EPN buona con adattamento completo (93,89% contro 93,06% scratch, +0,83 pp). Quest'ultimo è il sistema encoder+testa e non una dimostrazione sulla regressione.
- **Non risolto:** inseguimento temporale ancora attenuato; generalizzazione fra fasi; valore del pretraining per una regressione indipendente; confronto regressivo a protocollo comune FM frozen/finetuned, stessa architettura scratch e baseline TEMPONet raw.
- **Non identificato causalmente:** quanto dipenda da lettura troppo semplice, rappresentazione, obiettivo, dati o dose. Ridge simile alla testa esclude soltanto un facile recupero con quel lettore/vista; non identifica il limite di tutta l'informazione. Non abbiamo evidenza che il dataset sia intrinsecamente troppo piccolo o l'EMG imprevedibile.
- **Limite della loss:** MSE sugli angoli non assegna un peso separato al movimento. Le predizioni attenuate sono compatibili con incertezza e informazione letta incompleta; non sono da sole prova che la MSE sia difettosa o che basti cambiarne la distanza.

## Le due proposte di Simone, rese distinte e verificabili

**Teste di regressione.** La TEMPONet sui latenti è ancora una domanda aperta. Primo controllo economico possibile: stessa sequenza dell'encoder congelato e stesso supporto, testa attuale/ridge, piccola MLP per istante e testa temporale compatta. Si distinguono così non linearità e contesto. Una prova separata può sostituire la testa di posa nel pretraining mantenendo inizializzazione dell'encoder, dati, loss, dose e schedule: valutare sia le teste proprie sia lettori comuni rifittati. Un esito negativo a encoder congelato non esclude un beneficio dell'addestramento congiunto. Una testa più potente può anche assorbire parte del compito senza migliorare la trasferibilità: non basta la loss di train. Il fine-tuning regressivo con controllo scratch resta una terza domanda distinta.

**Contrastiva.** La proposta del 07/11 (`df36272`) non è implementata né lanciata nelle evidenze pubblicate. Usa T100-G, InfoNCE tra vista mascherata e completa, proiettore e code per dataset, ma media slot e tempo: endpoint primari P1/P2, posa come guardia di non deterioramento. Non è già una proposta mirata al movimento. Se la priorità attuale è la regressione, va formulata prima una domanda temporale: positivi riferiti agli stessi istanti/eventi, trasformazioni compatibili con posa, ampiezza e ordine, controllo dei falsi negativi fra istanti vicini. Sono scelte da progettare e rivedere, non benefici acquisiti. Tenere fissa la testa quando si misura la nuova loss; tenere fissa la loss quando si misura la nuova testa. Nessuna autorizzazione o nuovo esperimento deriva da questa sintesi.

Raccomandazione scientifica: dare priorità alla diagnosi del lettore temporale e al confronto regressivo su supporto comune; preparare in parallelo la domanda contrastiva pertinente al movimento. La replica completa del pretraining con testa temporale resta motivata ma distinta dalla diagnosi congelata. Non prolungare automaticamente il log-quadratico come rimedio alla regressione.

## Fonti e verifica

Le fonti sono pin e documenti nei diversi worktree, senza merge dei rami altrui. Percorsi assoluti, commit e hash dei file letti sono in `results/infra/regression_synthesis_20261116/sources.json`.

- Coordinamento `ec5b309`: audit `pose_scientific_assessment_20261028`, temporale/slot del 02/11, posa tokenizer del 04/11, sintesi scaling/FT del 06/11.
- Consolidamento `cfa6462`: `docs/reviews/consolidation_long_results_20261107.md` e appendici numeriche.
- Logcontrast `df36272`: `docs/logsquare_scientific_readout_20261107.md`, `docs/contrastive_proposal_20261107.md`.
- TEMPONet `1049c45`: `docs/temponet_db8_replication.md`, `docs/temponet_emg2pose_plan.md`, `results/temponet_emg2pose/eval_20261111/eval.md`.
- Scaling GPT `6f73209`: `docs/reviews/gpt_scaling_scientific_final_20261106.md`; Claude operativo `2e79815`, decisioni/registro sull'endpoint di training e sulla valutazione ancora non pubblicata.

Tre riletture indipendenti hanno controllato rispettivamente lungo/scaling, log-quadratico/contrastiva e TEMPONet. Nessun nuovo bootstrap, misura remota o confronto sperimentale; i valori sono trascritti dai report già verificati e i limiti dei protocolli restano distinti.
