# TCN: diagnosi del controllo temporale e proposta per Claude 2

Stato: **REVIEW_REQUIRED — analisi conclusa, seguito diagnostico proposto, nessun rilascio**.
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

La TCN temporale migliora E_dyn rispetto a L0 neurale del **2,67%**. Sul bersaglio pulito dei soggetti esclusi, il suo E_dyn è **0,139236**, contro **0,199032** di L0: circa il 30% di errore in meno. È apprendimento utile nel mondo sintetico, non un risultato sulla posa reale. Non attribuiamo tutto questo vantaggio all'uso causale dei campioni vicini: parte di T può essere accessibile da un singolo istante; questi numeri non sono un'ablazione del contesto.

### Epoche e selezione non spiegano il fallimento

Sono già stati completati cinque LR, da 0,0001 a 0,01. Il più piccolo arriva a 300 epoche, gli altri a 150. La selezione sceglie LR 0,003, epoca 31. Il miglior E_dyn fra **tutti** i checkpoint registrati è 0,715826, all'epoca 27 dello stesso LR: recupererebbe solo il 21,88%. Cambiare il criterio MSE in E_dyn non salverebbe il controllo.

A 150 epoche, LR 0,003 dà E_dyn 0,741556, peggiore del 3,58% rispetto al selezionato. Non è più il problema delle sole 50 epoche della prima proposta. Nessun allarme di bordo sulla TCN significa soltanto che le regole di bordo non scattano, non che l'ottimo sia dimostrato.

### Il divario fra fitting e soggetti esclusi

Al checkpoint selezionato, E_dyn sul fitting è 0,661873 / 0,710583; sui soggetti interni esclusi è 0,696657 / 0,738818. Ricostruendo algebricamente i riferimenti del fitting dai diagnostici archiviati, il recupero apparente è 86,12% / 69,25% contro 24,17% / 18,88% fuori fitting. Il primo numero può includere memorizzazione del rumore: **non è la misura del segnale pulito appreso**.

Anche L13 in forma chiusa, fittata su y, peggiora dal fitting agli esclusi: 0,667239 → 0,759182 e 0,708586 → 0,790777. Una soluzione del problema lineare non elimina quindi il problema statistico del contesto. Questo non prova che la TCN sia ottimizzata al meglio.

Mancano nel JSON l'errore pulito sul fitting e le curve complete separate fitting/fold. Le curve archiviate sono pooled; il seme della selezione è 0. Non ricaviamo intervalli o conclusioni sui tre semi dei rifit, che non sono stati eseguiti.

## 4. Architettura e costruzione del controllo

La candidata ha già un ramo affine diretto, sulla stessa LayerNorm di L0. Il ramo non lineare ha 64 canali, poi 32, con dilatazioni 1/2/3 e campo ricettivo di 13 patch. Il generatore sintetico usa la stessa struttura temporale, larga 8. Possiamo incorporare quei pesi nei primi otto canali della candidata, azzerare gli altri e assorbire la normalizzazione nel layer di uscita e nel ramo affine. Il segnale pulito è dunque rappresentabile a inference. La verifica di soli forward riproduce il generatore con RMSE circa 5e-7, massimo errore 6,4e-6, su otto finestre sintetiche; gli scarti sono compatibili con il calcolo FP32.

Questa è una prova di capacità funzionale, **non di apprendibilità da esempi rumorosi**. Usa i pesi noti del generatore solo per la prova, non per inizializzare una candidata da ammettere. Dropout è spento a inference, come nello scoring produttivo. L'uscita inizialmente a zero azzera i gradienti dei nascosti al primo update, non per tutto il training; nessun bug di gradiente permanente è dimostrato.

Entrambe le calibrazioni reali hanno scelto `y* = L + 0,4 T`, `y = y* + 1,3 epsilon`, con L e T normalizzati sul fitting e rumore AR(1), rho = exp(-1/4). Non si deve leggere il rapporto 0,4/1,3 come SNR esatto: covarianze e metrica dinamica contano. Il vantaggio cleanLS→oracolo resta circa l'8% dalla calibrazione interna ai fold esclusi, quindi non emerge un crollo grossolano del margine disponibile.

La calibrazione garantisce quel margine e un livello di errore realistico; **non garantisce che metà del margine dell'oracolo sia stimabile con circa 920/934 finestre di 20 utenti per fit**. È una lacuna della garanzia offerta dalla calibrazione, compatibile con il fallimento; non è dimostrato che il campione finito ne sia la causa. Né l'oracolo né la prova costruttiva risolvono il problema del campione finito. La ricetta ha già dropout 0,1 e AdamW weight_decay 0,01: non manca completamente la regolarizzazione.

## 5. Seguito minimo proposto, da rivedere prima di implementare

### D0 — completare la diagnosi dai checkpoint, senza nuovi fit

Il launcher rilegge con hash verificati gli stati già archiviati del LR 0,003: selezionato e finale a 150 epoche, per i due fold. `candidate_states` conserva `best_states` e `current_states`: update attesi 899/930 per il selezionato a 31, 4.350/4.500 per il finale. Prima riproduce gli score rumorosi pubblicati, rispettivamente del selezionato e del punto 150; poi misura MSE ed E_dyn contro **y e y*** su fitting e soggetti interni esclusi. Conserva soglia dinamica, maschere, ritaglio e aggregazione originali, senza rigenerare rumore o componenti. Per entrambi i bersagli pubblica i denominatori: E_dyn pulito e rumoroso non sono direttamente intercambiabili. Se uno stato non è disponibile o non è autenticabile, lo dichiara; nessun replay del training automatico.

Serve soprattutto l'errore pulito sul fitting. Non inventiamo le curve mancanti: due checkpoint non sono una traiettoria completa. Questa lettura e i fit D1 sono solo proposti, non già eseguiti.

### D1 — due fit diagnostici sul bersaglio pulito

Un fit per fold, stessa TCN_post, feature, popolazione, inizializzazione seme 0, ordine dei batch/RNG iniziale, supporto, dropout, weight decay, numerica e loss MSE. Unica modifica del training: bersaglio **y*** anziché y. LR fissato a **0,003**, orizzonte **150 epoche**, batch 32; niente nuova griglia, altri semi o estensione automatica a 300. Il LR deriva dalla diagnosi appena letta: questa è un'analisi esplorativa dichiarata, non una nuova conferma indipendente.

Per isolare il cambiamento del bersaglio di fitting, l'epoca resta scelta sulla **MSE pooled contro y originale** dei fold interni esclusi, con identico calendario di valutazione e spareggi. Registrare anche curve pulite, MSE ed E_dyn, fitting e fold esclusi separati; il minimo dinamico/pulito resta descrittivo. Soglia e finestre dinamiche restano quelle già fissate da y*. Salvare lo stato scelto e il finale. Nessun uso dei 15 utenti esterni o della posa vera per scegliere la ricetta.

Ripetere il calcolo del recupero originale del 50% in tutti i pannelli come **sensibilità diagnostica**; anche se raggiunto, non ammette il training su y rumoroso. Il confronto usa lo stesso y originale, riferimenti e supporto. Pubblicare inoltre errore pulito su fitting ed esclusi, confrontato con la predizione cleanLS fittata su y* e valutata ora su y* (nuovo score descrittivo, non confonderlo con il cleanLS della tabella). Non introdurre una soglia di “ottimo pulito” dopo aver visto le curve. Se la lettura resta ambigua o a bordo a 150, esito diagnostico inconclusivo, senza estensioni automatiche.

| Lettura dei diagnostici | Ipotesi favorita e conseguenza |
|---|---|
| Migliore ricostruzione pulita anche sugli esclusi e recupero originale ≥50% nei tre pannelli, quando si addestra su y* | La funzione è apprendibile con questi utenti se si rimuove il rumore del training. Priorità a controllo della complessità/regolarizzazione nella ricetta su y. Non è prova di ottimo né ammissione. |
| Errore pulito molto inferiore sul fitting che sugli esclusi, anche addestrando su y* | Generalizzazione fra utenti della funzione sintetica resta difficile. Valutare lettore più semplice e adeguatezza del controllo; non attribuire al FM un limite di informazione. |
| Anche il fitting pulito resta scarso rispetto al riferimento pulito, o curve ancora in discesa/bordo | Ottimizzazione/implementazione da indagare con target deterministico; non concludere che il campo ricettivo sia insufficiente. |
| Quadri discordanti o miglioramenti insufficienti a separarli | Diagnosi inconclusiva; nessun rilascio e nessuna scelta automatica della variante. |

Le righe qualitative non sono nuovi gate numerici. Il solo riferimento quantitativo di sensibilità resta il recupero originale, espresso in modo identico; la lettura causale conserva i limiti del singolo seme e del confronto esplorativo.

## 6. Quale modifica preferirei dopo la diagnosi

Se D1 separa chiaramente il caso pulito dal rumoroso, preferisco **una TCN più parsimoniosa o una regolarizzazione diversa**, scelta e congelata una sola volta prima di ripetere il controllo originale. Non farei una ricerca aperta di varianti fino al passaggio. La dimensione del generatore non va copiata per costruire una testa specializzata a superare il test; il candidato deve avere un motivo valido anche per leggere la posa vera. Un cambio di capacità richiede di riesaminare i comparatori MLP e l'attribuzione dei contrasti; non è una modifica gratuita del solo nome TCN.

Non sceglierei come prima cura un'altra aggiunta lineare: il ramo esiste. Inizializzarlo da ridge fitting-only è tecnicamente possibile, ma cambia l'ottimizzazione e non elimina di per sé il divario fitting/esclusi. Con LayerNorm e affine aggiornabili, inoltre, non è un residuo rispetto a una baseline fissa; congelarli sarebbe un'altra ricetta da dichiarare. Un cambio di schedule resta possibile se il diagnostico pulito mostra un problema di ottimizzazione, non è già giustificato dalle curve rumorose.

Se una variante verrà proposta, nuova ricetta e ricevute, confronto con A1 conservato, controlli lineare e temporale e soglia 50% invariati. Nessun riuso implicito di ricevute dopo un cambio source_identity; l'eventuale compatibilità degli input immutati dovrà essere esplicita e revisionata. Nessuna nuova loss, encoder o finestra in questa proposta.

## 7. Costi e consegna operativa

Analisi attuale: zero GPU-ore e nessun job. Il run temporale originale ha esplorato 900 epoche per fold e consumato 4.427 s. D1 richiede 29+30 = **59 update per epoca** sui due fold, **8.850 update** a 150 epoche. La semplice proporzione 150/900 dà circa **738 s = 0,205 GPU-ore**, ma **non è un preventivo eseguibile**: mancano il costo delle quattro letture fitting/esclusi × pulito/rumoroso, D0, I/O, memoria e margine. Non si promette un tetto di 0,205 GPU-ore. La nuova implementazione deve misurare questi costi e pubblicare allocazioni e budget prima del rilascio.

Il consuntivo A1 rimane 3,681111 GPUh e 31,975000 coreh in 16 allocazioni, come archiviato dall'operatore. Il residuo del budget A1 non trasforma il gate scientifico negativo in un via a prove diverse. La review richiesta qui è sul disegno diagnostico, poi serve il pacchetto operativo distinto. Non si creano ora ID o deploy. La manutenzione Leonardo del 23–25 novembre resta un vincolo da controllare nel pacchetto dell'operatore, non una ragione per saltare verifiche.

## 8. Riproducibilità e domande per Claude

Prove in `results/infra/pose_temponet_temporal_diagnosis_20261121/`: `optimization/` ricalcola curve e score dai JSON; `control/` verifica calibrazione e gate; `architecture/` dimostra l'incorporamento del generatore con soli forward. Gli script non addestrano né inviano job. Comandi e hash degli output della verifica locale sono in `local_validation.json` nella stessa directory.

Chiedo a Claude 2 una sola review del delta scientifico: confermare o confutare la diagnosi; verificare che D0 e i due D1 separino davvero i casi utili; controllare selezione, denominatori e limiti dell'oracolo; valutare la priorità a parsimonia/regolarizzazione rispetto a warm start o altre epoche. Un disaccordo va sostenuto da codice o numeri, non da un ulteriore controllo generico. Nessuna implementazione, job o modifica della campagna A1 richiesta con questo documento.

Rilettura critica locale del disegno completata: nessun blocco; verificata dal codice la presenza prevista degli stati selezionato/finale e delle componenti sintetiche originali. Accolte le precisazioni sui contatori degli update e sulla distinzione fra limite della calibrazione e causa dimostrata. Non sostituisce la review richiesta a Claude.
