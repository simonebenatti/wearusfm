# TCN: diagnosi del controllo temporale e proposta per Claude 2

Stato: **review scientifica Claude ricevuta; condizioni recepite il 22/11/2026, pacchetto operativo da preparare, nessun rilascio**.
Review: `f66691157fcd9f8588b3e58473e2fb1ac36c4dc1` su `claude/temponet-db8-20261109`. La review non autorizza nuove allocazioni; risposta punto per punto in `docs/reviews/pose_temponet_diagnosis_response_20261122.md`.
Base delle prove: `d3e41d06afbc5d47cb988488784a4655f16584af`.
Ramo della consegna: `codex/pose-temponet-temporal-diagnosis-20261121`.
La campagna A1 resta `recipe_not_admitted`. Operatore unico: **2-TCN head launcher**.

## 1. Conclusione

La TCN apprende parte del segnale del controllo temporale, ma non raggiunge la sensibilità richiesta dal controllo. Il gate negativo è calcolato correttamente. Non emerge un errore di supporto temporale o una funzione sintetica non rappresentabile dalla testa. L'indizio principale è il divario fra fitting e soggetti esclusi dal fitting, insieme al peggioramento tardivo. È compatibile con stima difficile sotto rumore e generalizzazione insufficiente; non dimostra né un ottimo di training né una causa unica.

Non propongo di abbassare il 50%, scegliere un checkpoint con un'altra metrica, aumentare semplicemente le epoche o cambiare il FM. Propongo prima una lettura aggiuntiva dei checkpoint esistenti e due soli fit diagnostici sullo stesso bersaglio pulito. La scelta di un'eventuale ricetta diversa viene dopo questa separazione delle cause.

## 2. Provenienza e perimetro della verifica

- Codice del controllo: `a1ae9607681c1a181034d7f72770537054af419d`; checkout `edebfb6c290cf661989c5cff45916bc6a93e607e`.
- Source SHA256: `9aaf1c8b6a050a524dfb60af65df9406313d4fd0e690dab1097bf11df2ec67ea`.
- Piano controls: SHA256 `b334ece870bc85b804708f92e4c240f4999750ef8d089ca7df1bd040d2ee04e3`.
- Gate archiviato: `results/infra/pose_temponet_execution_20261121_A1/launcher/positive_controls_gate_20261121.json`; hash interno `9e91c8a237cc305f6e8c47bacff170536e4b5f0c1c1599f542e078c22e133858`.
- Selezione temporale TCN: job `65035631`, `COMPLETED 0:0`, 4.427 s. Gli altri cinque controlli sono anch'essi conclusi; la verifica operativa completa è in `docs/reviews/pose_temponet_controls_results_20261121.md`.

Qui si rileggono codice e JSON già archiviati sul Mac, si ricalcolano rapporti e curve e si esegue un controllo costruttivo di soli forward su dati sintetici piccoli. Nessun accesso a Leonardo, nuovo addestramento, rilettura dei checkpoint remoti o inferenza sui 15 utenti esterni. Nei controlli, `held` indica metà dei **40 utenti di train**; non è il pannello esterno dei 15 utenti. Gli hash degli input dell'analisi sono negli artefatti riproducibili.

## 3. Che cosa non passa

Il gate misura la frazione del vantaggio dell'oracolo recuperata dalla testa: `(E_cleanLS − E_TCN) / (E_cleanLS − E_oracle)`. L'oracolo conosce la funzione sintetica esatta; cleanLS è il lettore lineare fittato sul bersaglio pulito, valutato sul bersaglio rumoroso.

| Pannello | cleanLS | Oracolo | TCN | Recupero | Limite E_dyn per il 50% |
|---|---:|---:|---:|---:|---:|
| fold 0 | 0,710431 | 0,653451 | 0,696657 | 24,17% | 0,681941 |
| fold 1 | 0,749920 | 0,691103 | 0,738818 | 18,88% | 0,720512 |
| pooled | 0,728479 | 0,670659 | 0,715926 | 21,71% | 0,699569 |

Tutti e tre devono raggiungere il 50%; nessuno lo raggiunge. Sul controllo lineare, invece, tutte le teste recuperano almeno il 97,46% del vantaggio di riferimento, contro il 90% richiesto.

### Precisazioni dopo la review Claude del 22/11

Il riferimento cleanLS è stimato sul pulito; le teste A1 imparano dal rumoroso. Sulla scala del gate, 0% = cleanLS e 100% = oracolo:

| Lettore allenato su y | fold 0 | fold 1 | pooled |
|---|---:|---:|---:|
| L0 neurale | −13,55% | −10,82% | −12,28% |
| MLP | −12,24% | −10,65% | −11,50% |
| L0 in forma chiusa | −19,10% | −16,38% | −17,84% |
| TCN | +24,17% | +18,88% | +21,71% |

Partendo invece da L0 neurale, la TCN chiude il 33,22% / 26,80% / 30,27% della distanza dall'oracolo; il gate richiederebbe il 55,47% pooled su questa scala. L'asimmetria è dichiarata dal protocollo e non invalida il gate.

Nel controllo lineare la TCN scelta a 28 epoche ha un eccesso rispetto all'oracolo pari al 19,74% / 18,70% del margine del mondo temporale. Sommarlo al recupero temporale dà circa 43,91% / 37,57%: riportiamo il conto di Claude **solo come euristica fra mondi**, non come misura di «funzione appresa meno prezzo del rumore». Cambiano anche funzione, soglia/selezione delle finestre dinamiche, denominatori, sigma (1,35/1,4 contro 1,3), traiettoria e checkpoint. Non deduciamo che senza rumore la TCN resterebbe sotto il 50%; D1 verifica proprio un intervento sul target mantenendo il mondo originale.

Il divario fitting→esclusi della TCN (+0,03478 / +0,02824) è fra quello di L0 in forma chiusa su y (+0,01555 / +0,01368) e L13 (+0,09194 / +0,08219). cleanLS varia solo +0,00178 / +0,00029. È compatibile anche con sola memorizzazione del rumore; non dimostra che questa sia l'unica causa. Il buon trasferimento di una proiezione lineare non prova il buon trasferimento del residuo non lineare della TCN. D0 misura direttamente l'errore pulito sul fitting; non usiamo il recupero rumoroso 86,1%/69,3% come suo limite deterministico.

La TCN temporale migliora E_dyn rispetto a L0 neurale del **2,67%**. Sul bersaglio pulito dei soggetti esclusi, il suo E_dyn è **0,139236**, contro **0,199032** di L0: circa il 30% di errore in meno. È apprendimento utile nel mondo sintetico, non un risultato sulla posa reale. Non attribuiamo tutto questo vantaggio all'uso causale dei campioni vicini: parte di T può essere accessibile da un singolo istante; questi numeri non sono un'ablazione del contesto.

### Epoche e selezione non spiegano il fallimento

Sono già stati completati cinque LR, da 0,0001 a 0,01. Il più piccolo arriva a 300 epoche, gli altri a 150. La selezione sceglie LR 0,003, epoca 31. Il miglior E_dyn fra **tutti** i punti delle curve registrate è 0,715826, all'epoca 27 dello stesso LR: recupererebbe solo il 21,88%. Cambiare il criterio MSE in E_dyn non salverebbe il controllo.

A 150 epoche, LR 0,003 dà E_dyn 0,741556, peggiore del 3,58% rispetto al selezionato. Non è più il problema delle sole 50 epoche della prima proposta. Nessun allarme di bordo sulla TCN significa soltanto che le regole di bordo non scattano, non che l'ottimo sia dimostrato.

### Il divario fra fitting e soggetti esclusi

Al checkpoint selezionato, E_dyn sul fitting è 0,661873 / 0,710583; sui soggetti interni esclusi è 0,696657 / 0,738818. Ricostruendo algebricamente i riferimenti del fitting dai diagnostici archiviati, il recupero apparente è 86,12% / 69,25% contro 24,17% / 18,88% fuori fitting. Il primo numero può includere memorizzazione del rumore: **non è la misura del segnale pulito appreso**.

Anche L13 in forma chiusa, fittata su y, peggiora dal fitting agli esclusi: 0,667239 → 0,759182 e 0,708586 → 0,790777. Una soluzione del problema lineare non elimina quindi il problema statistico del contesto. Questo non prova che la TCN sia ottimizzata al meglio.

Mancano nei JSON l'errore pulito sul fitting e le curve complete di scoring eval separate fitting/fold. Le curve di scoring archiviate sono pooled. Gli stati binari conservano però `epoch_losses` per fold: loss online in modalità train, con dropout e pesi che cambiano entro l'epoca; non sono le curve eval mancanti. D0 le recupera. Il seme della selezione è 0; nessuna inferenza sui tre semi dei rifit, che non sono stati eseguiti.

## 4. Architettura e costruzione del controllo

La candidata ha già un ramo affine diretto, sulla stessa LayerNorm di L0. Il ramo non lineare ha 64 canali, poi 32, con dilatazioni 1/2/3 e campo ricettivo di 13 patch. Il generatore sintetico usa la stessa struttura temporale, larga 8. Possiamo incorporare quei pesi nei primi otto canali della candidata, azzerare gli altri e assorbire la normalizzazione nel layer di uscita e nel ramo affine. Il segnale pulito è dunque rappresentabile a inference. La verifica di soli forward riproduce il generatore con RMSE circa 5e-7, massimo errore 6,4e-6, su otto finestre sintetiche; gli scarti sono compatibili con il calcolo FP32.

Questa è una prova di capacità funzionale, **non di apprendibilità da esempi rumorosi**. Usa i pesi noti del generatore solo per la prova, non per inizializzare una candidata da ammettere. Dropout è spento a inference, come nello scoring produttivo. L'uscita inizialmente a zero azzera i gradienti dei nascosti al primo update, non per tutto il training; nessun bug di gradiente permanente è dimostrato.

Entrambe le calibrazioni reali hanno scelto `y* = L + 0,4 T`, `y = y* + 1,3 epsilon`, con L e T normalizzati sul fitting e rumore AR(1), rho = exp(-1/4). Non si deve leggere il rapporto 0,4/1,3 come SNR esatto: covarianze e metrica dinamica contano. Il vantaggio cleanLS→oracolo resta circa l'8% dalla calibrazione interna ai fold esclusi, quindi non emerge un crollo grossolano del margine disponibile.

La calibrazione garantisce quel margine e un livello di errore realistico; **non garantisce che metà del margine dell'oracolo sia stimabile con circa 920/934 finestre di 20 utenti per fit**. È una lacuna della garanzia offerta dalla calibrazione, compatibile con il fallimento; non è dimostrato che il campione finito ne sia la causa. Né l'oracolo né la prova costruttiva risolvono il problema del campione finito. La ricetta ha già dropout 0,1 e AdamW weight_decay 0,01: non manca completamente la regolarizzazione.

## 5. Seguito diagnostico: condizioni recepite prima dei fit

### D0 — un job di sola inferenza su Leonardo

D0 non addestra, ma è un job con ID, tempo, risorse e costo propri: checkpoint e componenti originali stanno su Leonardo. Legge `calibration.pt` e `selection_states.pt` autenticati; ricostruisce bundle, componenti, soglie e supporti e ne richiede l'identità con gli archiviati prima dello scoring. Non rigenera rumore, non fa replay del training e non rinfresca input SCRATCH.

Si leggono i **20 stati TCN**: cinque LR × migliore/finale × due fold. I finali del LR 0,0001 sono a 300 epoche, gli altri a 150. In aggiunta, si leggono i quattro stati globalmente scelti di L0 e MLP temporali come autocontrollo. Per la TCN al LR 0,003: stato scelto a 31 (899/930 update), finale a 150 (4.350/4.500 update). I punti delle curve non sono tutti checkpoint salvati: non si promette una rivalutazione per fold del minimo dinamico a 27.

Per ogni stato: MSE, E_dyn, numeratori, denominatori e supporto, contro **y e y***, su fitting ed esclusi, per fold e pooled. Si conserva la soglia dinamica fissata da y*. Si leggono anche le loss online di ogni epoca dagli stati finali, identificandole come train-mode, senza presentarle come valutazione dello stesso checkpoint su tutto il fitting.

Parità prima dei nuovi score: per fold e pooled sullo stato globalmente scelto, dove esistono riferimenti; per lo stato finale del LR 0,003 solo pooled al punto 150. Gli altri stati si confrontano con i punti pooled corrispondenti delle curve originali. I nuovi score per fold del finale non hanno un riferimento per fold archiviato. Profilo CUDA FP32, TF32 spento, determinismo e contesto gpu8 congelato; score float64 come nel valutatore originale. Metriche già pubblicate: `abs(new-old) <= 1e-12 + 1e-12*abs(old)`; hash di bundle, target, supporto, payload e contatori esatti. Nessun allentamento automatico se la parità fallisce.

Oracolo e cleanLS sul fitting vengono calcolati **dagli array originali**, non ricostruiti dal recupero L13. Sul medesimo supporto dinamico si pubblicano errore pulito, errore rumoroso e oracolo. Per residui centrati e mascherati `e = prediction − y*`, `n = y − y*`, vale `SSE_y = SSE_y* + SSE_oracle − 2<e,n>`. Il termine incrociato si ricava dai tre score senza un'altra inferenza. Evita di assumere che il rumore abbia prodotto un errore additivo indipendente. Errore sul fitting sotto l'oracolo implica allineamento della predizione con la realizzazione del rumore su quel campione; da solo non quantifica una soglia di apprendibilità universale.

### D1 — due fit sul pulito e un controllo di parità del driver

Un fit per fold, stessa TCN_post, feature, popolazione, inizializzazione seme 0, ordine dei batch/RNG, supporto, dropout 0,1, weight decay 0,01, numerica e loss MSE. Unica modifica del fitting: **y*** anziché y. LR **0,003**, **150 epoche fisse**, batch 32. Niente griglia nuova, seme aggiuntivo o estensione a 300. La scelta del LR deriva dal mondo già osservato: diagnostico esplorativo, non conferma indipendente.

Il driver deve vivere fuori dall'albero congelato, importando i moduli originali senza modificarli. Identità separate per core congelato e driver nuovo. Prima dei fit puliti, lo stesso driver su **y originale** deve riprodurre la prima epoca del LR 0,003: punti 0, 0,25, 0,5, 0,75 e 1, score pooled archiviati, init/batch/RNG/contatori del percorso originale ove verificabili. Tolleranza sugli score e profilo numerico identici a D0; identità strutturali esatte. Il controllo comporta update e va incluso nei costi; non è un nuovo candidato. Mancata parità ferma D1, senza aggiustamenti della tolleranza.

Epoca scelta: minimo della **MSE pooled contro y originale** degli esclusi, esposizione positiva, spareggio alla prima esposizione, stesso calendario congelato. Le curve pulite e il loro minimo sono descrittivi. Si salvano letture e stati alle esposizioni **31, 150 e scelta**, per fold e pooled, fitting ed esclusi, contro y e y*, oltre alle curve e alle loss train-mode. Se due esposizioni coincidono si conserva un'identità unica, non si conta una replica.

A1 e D1 alla stessa esposizione 31, valutati sul medesimo y, sono numericamente confrontabili; la differenza condizionata a questi dati/semi riflette l'intervento sul target di fitting. Non è una scomposizione universale dell'errore. D1 può usare informazione pulita privilegiata: un successo non ammette l'allenamento su y. Nessuna nuova lettura dei 15 utenti esterni.

### Regola di lettura D1, in ordine e senza sovrapposizioni

Per ciascun pannello `p = fold0, fold1, pooled`, sia `R_held,p` il recupero originale su y al punto scelto. Sia `R_fit,p` la **stessa formula**, con cleanLS e oracolo ricalcolati sul fitting, sempre valutati su y e sul supporto fitting. Non è il recupero calcolato sul target pulito; quest'ultimo resta descrittivo con il proprio denominatore.

Anche il fitting si legge al medesimo punto scelto, senza selezionare un'altra epoca. Il pooled somma gli accumulatori dei fold prima di calcolare i rapporti, non media le percentuali.

Il flag `late` è vero se **almeno uno** dei minimi pooled MSE o E_dyn contro y cade nelle epoche intere 146–150. Si esclude l'esposizione zero; a parità si prende la prima esposizione. Il minimo pulito non attiva il flag. `late` è lo stesso controllo di bordo per le due metriche del protocollo; «curve ancora in discesa» non è un criterio aggiuntivo.

| Priorità | Condizione, dopo parità e supporti validi | Esito diagnostico |
|---|---|---|
| 0 | Artefatti non autentici, parità fallita, riferimenti non finiti o margine cleanLS−oracolo non positivo | Non valutabile; fermarsi alla fase corrispondente. |
| 1 | `R_held,p >= 0,50` in tutti e tre i pannelli | **Sensibilità recuperata addestrando sul pulito**; eventuale `late` si annota, non cambia l'esito. Non ammette la ricetta rumorosa. |
| 2 | Almeno un `R_held,p < 0,50` e `late` vero | **Non risolto entro 150 epoche**. Nessuna estensione automatica e nessuna causa assegnata. |
| 3a | Held sotto soglia, `late` falso, tutti i `R_fit,p >= 0,50` | **Recupero raggiunto sul fitting, non in tutti i pannelli esclusi**. Indizio descrittivo, non prova causale di trasferimento mancato fra utenti. |
| 3b | Held sotto soglia, `late` falso, tutti i `R_fit,p < 0,50` | **Recupero insufficiente anche sul fitting**. Non equivale a difetto dell'ottimizzatore: dropout, penalizzazione e LR scelto sul rumoroso possono contribuire. |
| 3c | Held sotto soglia, `late` falso, fitting misto fra pannelli | **Esito eterogeneo**, pubblicare i tre pannelli senza assegnare una causa unica. |

I confronti usano numeri non arrotondati. Tutte le righe portano anche gli score puliti e rumorosi a 31/150/scelta. Non si usa «scarso» o «molto inferiore» per decidere l'esito. Le etichette non dichiarano ottimo, impossibilità statistica o promozione della testa. La distinzione fra errore pulito appreso e adattamento al rumore è una lettura dei pannelli e del termine incrociato, non una nuova soglia aggiustata dopo i risultati.

## 6. Limite di D0/D1 e unica variante eventuale

**Condizione 2 di Claude: scelta dell'alternativa esplicita senza il braccio teacher-init.** Un successo di D1 dimostra sensibilità del lettore quando riceve etichette pulite. Non dimostra che esista una testa addestrabile su y rumoroso con questo campione e capace di raggiungere il 50%. L'eventuale unica variante sarà una scelta esplorativa motivata, senza tale garanzia.

Partire dai pesi veri misurerebbe quanto una traiettoria preserva una soluzione fornita in anticipo; sarebbe un controllo di stabilità utile per un'altra domanda. Il successo non dimostra che la soluzione sia apprendibile dai soli dati; il fallimento non dimostra che nessuna testa o ricetta possa trovarla. Dropout spento non elimina gli altri fattori (LR, penalizzazione, durata, campione). Per questo non aggiungo ora quei fit e non uso «sotto il 50% dal teacher ⇒ specifica sbagliata» come decisione.

Se D1 recupera sensibilità, una regolarizzazione diversa o una parametrizzazione più parsimoniosa sono possibili leve, non rimedi dimostrati. «Più parsimoniosa» non significa automaticamente meno canali: il giocattolo di Claude mostra che una testa larga 8 può faticare anche sul pulito. Conserviamo questa cautela senza trasferire quantitativamente gli esiti del giocattolo ai latenti veri. Un cambio di capacità richiederebbe di rivedere comparatori MLP e attribuzione dei contrasti. Il ramo affine esiste; altre epoche sulla medesima traiettoria e warm start non sono la prima scelta motivata dai dati, ma non sono universalmente confutati come famiglie di interventi.

### Conferma di una futura variante: un mondo nuovo, un tentativo

La variante, la ricetta e la sua regola di selezione devono essere fissate **prima di generare o leggere il nuovo mondo**. Il mondo originale resta per confronto con A1; il nuovo verifica robustezza rispetto a un'altra funzione e realizzazione del rumore, non rappresenta nuovi utenti o un test indipendente della generalizzazione clinica.

Semi fissati qui prima di D1: `base_seed = 20261122 + 1000*outer_fold`; L usa base, T base+1, epsilon base+2, con l'attuale derivazione per metadati/giunto. Fold0: **20261122 / 20261123 / 20261124**; fold1: **20262122 / 20262123 / 20262124**. Stessi utenti, split, famiglia del generatore, griglie e algoritmo di calibrazione solo inner-train; nuova calibrazione autenticata, nessuna scelta fra semi. Non si genera ora il mondo.

Il 50% dovrà essere raggiunto nei due fold e nel pooled del nuovo mondo, con controlli lineari e altri gate del protocollo ancora validi. I parametri del candidato non si modificano dopo aver letto gli esiti nuovi, neppure se passa sul vecchio e fallisce sul nuovo. Se la calibrazione nuova non è diagnostica, non si estrae un altro seme. Un unico tentativo di variante: fallimento o esito non valutabile torna a Simone per decidere fra arresto e revisione motivata della specifica; nessun'altra ricerca automatica. L'eventuale protocollo operativo preciserà la separazione fra selezione interna consentita e conferma, senza pretendere un test su dati completamente nuovi.

Nuova ricetta e identità per il futuro candidato; nessun riuso implicito di ricevute dopo un cambio source_identity. La compatibilità di input immutati deve essere esplicita e revisionata. Nessuna nuova loss, encoder o finestra prevista da D0/D1.

## 7. Risorse, decisione e pacchetto operativo

Questa revisione è documentale, con aritmetica locale: zero GPU-ore e nessun job. **D0, parità del driver e D1 sono nuove allocazioni**, non comprese automaticamente nei tetti della matrice A1. Il pacchetto deve riportare per ciascuna: ID logico, dipendenze, dispositivo/profilo, memoria, walltime, costo GPU e core/ore locali, sotto-budget complessivo, addebito e residuo. La decisione di Simone su questo perimetro va registrata prima del primo invio; una volta data copre tutta la catena approvata e non si richiede di nuovo ai singoli passi.

La vecchia proporzione 150/900 × 4.427 s = 0,205 GPUh era un conto incompleto, non un tetto; non la uso come previsione aggiornata. D1 resta 8.850 update, ma i nuovi score fitting/esclusi e pulito/rumoroso, D0, parità, letture e memoria devono entrare nel preventivo. Claude propone **1–2 GPUh come ordine di grandezza non misurato del suo pacchetto completo**, che include anche il riferimento teacher-init qui non adottato. Non trasferiamo quella cifra come misura o budget esatto del pacchetto scelto; la misura e la proiezione prudenziale saranno pubblicate prima del rilascio, includendo il costo di ogni eventuale benchmark preliminare.

Consuntivo A1 archiviato invariato: 3,681111 GPUh / 31,975000 coreh, 16 allocazioni. Il suo residuo non è una nuova decisione scientifica. Questa review non è l'autorizzazione dei diagnostici: non creiamo ora ID di deploy o prenotazioni. Il pacchetto operativo sarà distinto, revisionato e consegnato al solo **2-TCN head launcher**. La manutenzione del 23–25 novembre va ricontrollata dall'operatore prima della pianificazione e non giustifica salti dei gate.

## 8. Riproducibilità e risposta alla review

Prove in `results/infra/pose_temponet_temporal_diagnosis_20261121/`: `optimization/` ricalcola curve e score dai JSON; `control/` verifica calibrazione e gate; `architecture/` dimostra l'incorporamento del generatore con soli forward. Gli script non addestrano né inviano job. Comandi e hash degli output della verifica locale sono in `local_validation.json` nella stessa directory.

La review Claude f66691157fcd9f8588b3e58473e2fb1ac36c4dc1 conferma i riscontri e chiede quattro condizioni; questa revisione le recepisce con l'alternativa esplicitamente ammessa alla condizione 2. La risposta analitica, le precisazioni alle inferenze e la verifica dei numeri aggiuntivi sono in `docs/reviews/pose_temponet_diagnosis_response_20261122.md` e `results/infra/pose_temponet_temporal_diagnosis_20261121/review_response_20261122/`. Nessuna nuova review del codice è dichiarata: il codice operativo non è ancora implementato.
