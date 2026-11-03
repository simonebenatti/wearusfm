# Piano scientifico dopo sonde, PSD-3, PCA, temporale e contrasti fra slot — 02/11/2026

**Sono conclusi PCA, lettore temporale e punto 6 media/contrasti fra slot.** Tutti i risultati sono verificati dal critico. Il nuovo controllo trova informazione utile oltre la media compressa, ma aggiungerla alla media completa migliora B solo dello **0,0123% relativo**, senza recupero dell’ampiezza. Nessun job residuo; catena punto 6: **1,155556 GPU-ore e 10,254444 core-ore**.

**Nuova priorità scelta da Simone, dopo la chiusura del punto 6:** tokenizer da **100 ms con passo25 e overlap75**, mantenendo l'anti-aliasing, seguito da una variante temporale non lineare apprendibile. Codice e configurazioni dei tre bracci sono preparati:25ms lineare,100ms lineare,100ms con GELU. Aggiornamento03/11: collaudo concluso, tutti e tre i training a5.000 completati, continuazioni19.618 già in coda. Le tre valutazioni teacher5k63399428/63399479/63399484 sono RUNNING da checkpoint verificati. [Protocollo e verifiche previste](tokenizer_window_nonlinear_protocol_20261102.md). Questo approfondimento precede le nuove loss7a/7b; non riapre i lettori conclusi.

**03/11, review Claude letta e verificata:** corretto il raggio delle query a0 esplicito con JEPA spento; gain storico conservato, EPN-612 diventa una domanda principale accanto alla posa.94test passati dopo la correzione. Il costo delle valutazioni citato da Claude includeva due encoder, ma il margine per tre training completi rimane stretto. [Decisioni e rettifiche](reviews/risposta_claude_tokenizer_20261103.md). Successivamente Simone ha autorizzato extrabudget e campagna completa; lo stato attuale è nella tabella.

Il percorso resta: **verificare la misura → capire come leggere le rappresentazioni → scegliere se cambiare testa, rappresentazione o obiettivo**. Il piano iniziale è conservato nel commit `63ba143`; questa nota ne aggiorna lo stato, senza modificare il [protocollo congelato](pose_temporal_pca_protocol_20261102.md) o i criteri precedenti.

## Cosa facciamo adesso

**Punto 6a/6b chiuso; 6c non attivato automaticamente.** Il [risultato completo](reviews/pose_slots_results_20261102.md) e la [review indipendente](reviews/pose_slots_results_critic_20261102.md) concordano: la vista con sette contrasti fissi non identifica la media come causa principale della dinamica attenuata. Non è stata modificata la testa del FM né il suo training.

| Priorità / stato | Attività | Risultato | Decisione successiva |
|---|---|---|---|
| **Chiusi** | Sonde, PSD-3 nella ricetta provata, PCA appaiata e lettore temporale minimo. | Convergenza verificata; pipeline128 penalizzante; contesto temporale con beneficio piccolo su B. | Conservare risultati e limiti; non rifare queste prove. |
| **6a: concluso** | Controllo media/contrasti con quattro viste, piano e matrici congelati, supporti e fold comuni. | 54 test prima dei lanci, parità della media verificata su tutte le finestre. | Disegno eseguito senza modifiche dopo i risultati. |
| **6b: concluso** | Estrarre feature dai checkpoint congelati e addestrare soltanto le ridge. | S384 migliora M48 ma peggiora M384. S720 migliora B di appena 0,000106 nel rapporto di errore; nessun recupero dell’ampiezza. | Non basta per motivare una modifica del pooling/FM. Chiudere la diagnosi nella vista provata. |
| **6c: condizionale, non avviato** | Eventuale modifica testa/aggregazione/FM. | Il temporale e i sette contrasti aggiungono poco oltre la media completa di B. | Non proporre una modifica architetturale sulla base di questi soli risultati. |
| **6d: tokenizer100, training5k completato** | Nuovo confronto25/100ms, passo25; stessi dati, loss e supporto delle query. | Geometria100/75 e anti-alias già verificati; nessun nuovo risultato di prestazione. | Valutazioni teacher5k RUNNING; continuazioni19618 già in coda. Il precedente150ms non dimostra che100ms migliori. |
| **6e: tokenizer non lineare, training5k completato** | Filtri100ms apprendibili → GELU → proiezione, prima del Perceiver. | Opzione implementata; stesso numero di parametri del100ms lineare. | Confronto separato100GELU−100lineare, senza cambiare loss; prima lettura5k, endpoint19618 autorizzato e già in coda. |
| **Successive al tokenizer: 7a/7b** | Migliorare la dinamica di posa oppure il trasferimento generale. | La supervisione di posa migliora la posa ma non ha mostrato un beneficio generale nelle classificazioni osservate. | Per 7a una sola ipotesi dinamica accanto agli angoli; per 7b definire prima una regressione di trasferimento diversa dalla posa. Nessun lancio automatico. |
| **Facoltativo, non prioritario** | Rifit che isoli la sola PCA. | L’effetto della pipeline compressa è già misurato. | Solo se l’attribuzione alla sola PCA serve alla decisione successiva. |

Il punto 6 ha richiesto inferenza e fit di piccoli lettori, **nessun nuovo pretraining**. La vista conserva sette contrasti fissi proiettati, 336 valori FP32 per patch, circa 8,19 GiB per i due modelli. Non è la rappresentazione completa dei 64 slot: la conclusione rimane limitata alla proiezione e ai lettori provati.

## Quadro organico aggiornato

Nei risultati storici P1/P2 sono classificazione interna dal **teacher**, la posa è regressione dallo **student**. La nuova campagna tokenizer valuta il **teacher per entrambe**, con pannello student separato. I risultati non sono intercambiabili.

| Passo e stato: quello che è stato fatto | Spiegazione | Risultato | Conclusione e seguito |
|---|---|---|---|
| **0. B a 19.618 passi — concluso** | Più addestramento con posa; 15 utenti esclusi dal train e confronto R3c a pari passo. | Dalla testa B5.000 a B19.618: errore dinamico **0,927 → 0,867**, correlazione **0,270 → 0,366**, varianza predetta **6,3% → 15,0%** del target. | La posa migliora fuori dal train, ma resta attenuata. Non dimostra utilità generale del FM o trasferimento ad altre regressioni. |
| **1. Controllo delle sonde — concluso** | Replica a 3.000 iterazioni e confronto simmetrico a 12.000 per verificare la convergenza. | Replica esatta, **0 fit al limite su 280**. B−R3c: P1 **−0,553 pp** [−1,105; −0,017], P2 **−1,172 pp** [−2,956; +0,651]. | La convergenza non spiega il mancato beneficio di B. Non occorre rifare la tranche. |
| **2. PSD-3 sulle viste mediate nel tempo — concluso** | B/R3c × pool/decoder × logistica/piccola MLP, encoder congelato, PCA128 sul train. | MLP perde **0,49–0,75 pp in P1**; IC aggregati P2 includono zero. Nessun vantaggio aggregato chiaro di decoder su pool. **56,75 core-ore, zero GPU**. | Questa ricetta non risolve il problema; non esaurisce lettori non lineari, slot interi o tempo. Il confronto MLP−logistica sulle stesse viste compresse resta valido. |
| **3. Confronto PCA — analisi appaiata conclusa; ablazione isolata non eseguita** | Quattro OOF recuperati da $WORK, pairing verificato e 2.000 bootstrap per soggetto, senza nuovi fit. | Pipeline128−768: B **−0,497 pp P1 / −2,859 P2**; R3c **−0,453 / −1,699**. Tutti i relativi IC sotto zero. Penalizzazione aggiuntiva di B su P2 **−1,160 pp** [−1,740; −0,660]. | La pipeline compressa penalizza entrambi, soprattutto B in P2. Cambiano anche precisione/thread: non è l'effetto isolato della sola PCA. B resta sotto R3c nelle stime puntuali anche senza PCA. |
| **4. Feature temporali — disponibili e validate** | Riutilizzate le cache student B/R3c19618, 160×384 per finestra: 40 utenti train e 15 val. | Quattro cache, SHA/sidecar/metadati e supporti verificati. **Nessuna nuova estrazione GPU**. | Consentono la prima prova temporale. La media dei64 slot è già avvenuta: per studiare gli slot interi servirebbero altre feature da checkpoint congelati. |
| **5. Lettore temporale minimo — concluso** | Ridge istantanea contro contesto **t−100ms, t, t+100ms**, senza PCA, supporto comune e selezione sui soli utenti train. | B errore dinamico **0,859512 → 0,855321**, delta **−0,004191** [−0,006329; −0,001939], beneficio12/15 utenti. R3c **0,986567 → 0,961495**. B varianza predetta **14,0% → 10,8%**, MAE leggermente peggiore. | Piccolo beneficio B, circa **0,49% relativo**, senza recupero dell'ampiezza. B conserva vantaggio di posa su R3c. Non basta per accusare il pooling, cambiare il FM o iniziare una ricerca estesa di teste/lag sulla stessa val. |
| **6a. Controllo media/slot — concluso** | Quattro lettori: M384, M48, M48+contrasti336 (S384), M384+contrasti336 (S720); matrici fisse senza selezione sui bersagli. | Tutti i controlli superati; M384 replica esattamente l’istantaneo precedente, inclusi CV e coefficienti. | Riferimento stabile; confronto a pari larghezza più due controlli incrementali. |
| **6b. Estrazione e lettori — conclusi** | Due estrazioni parallele, quattro viste per B/R3c student19618 congelati, selezione sul solo train. | B: M384 **0,859512**, M48 **0,882629**, S384 **0,879413**, S720 **0,859406**. S384−M48 **−0,003216** [−0,003807; −0,002523]; S720−M384 **−0,000106** [−0,000159; −0,000048]. | I contrasti aiutano rispetto alla media compressa, ma non compensano la compressione. Con tutta la media, beneficio B **0,0123% relativo**, varianza predetta ancora circa **14%**. Catena chiusa: **1,155556 GPU-ore / 10,254444 core-ore**. |
| **6c. Modifica testa/pooling/FM — non avviata** | Intervenire solo su una precisa ipotesi con confronto separato. | Nessun recupero sostanziale della dinamica da temporale minimo o sette contrasti oltre la media completa. | Questi risultati non motivano da soli un cambio del FM. Non escludono slot completi o lettori diversi; evitare una ricerca estesa sulla stessa val. |
| **6d. Finestra del tokenizer — collaudo tecnico concluso** | 100ms nominali, passo25, overlap75; confronto con25ms e raggio comune0 esplicito con JEPA0, dopo la review Claude. | Collaudo63370721 riuscito:2.048 finestre comuni/braccio,0,221GPUh;100lineare costa circa+0,64% per update. Tre training5k COMPLETED0:0,12,973333GPUh totali; continuazioni19618 PENDING/Priority alle11:49UTC. Estrattori teacher5k63399428/63399479/63399484 RUNNING; nessun esito scientifico ancora. | Verificare posa ed EPN-612 separatamente a pari dati/dose. Il riferimento25ms conserva la ricetta B; replica tecnica distinta dalla variabilità tra semi. |
| **6e. Non linearità del tokenizer — autorizzata, implementata** | GELU sulle64 risposte temporali, dopo il gain e prima della proiezione. Kernel e proiezione apprendibili come prima. | GELU attiva e gradienti verificati nel collaudo reale;100GELU circa+1,53% per update contro25lineare. Braccio100GELU completato a5k insieme ai due controlli; efficacia ancora da misurare nelle valutazioni comuni. | Confrontare100GELU con100lineare. Una non linearità locale è una nuova ipotesi, distinta dalla MLP sui vecchi readout mediati. |
| **7a. Obiettivo posa dinamica — da scegliere** | Possibile termine sulle variazioni/velocità accanto agli angoli assoluti, o diversa ponderazione della posa. | B combina già ancore0,2 e posa1; JEPA spento. Nessun nuovo braccio su queste modifiche. | Una sola ipotesi a pari dose, conservando il vincolo sul livello assoluto. Non aspettarsi che la sola loss sulla testa recuperi informazione assente dalla sua vista. |
| **7b. Obiettivo FM generale — alternativa di ricerca** | Contrastiva con ancore ed eventuale posa, con positivi/negativi/pesi motivati dal trasferimento desiderato. | Non sperimentato; la posa migliora senza beneficio generale dimostrato nelle classificazioni osservate. | Prima definire una regressione di trasferimento diversa dalla posa supervisionata. Non sommare insieme contrastiva, dinamica e cambio architettura senza un confronto interpretabile. |

L'errore dinamico è SSE delle variazioni centrate/varianza del target: più basso è meglio. La nuova ridge usa un supporto centrale comune; **i numeri della riga5 non sono direttamente confrontabili con la testa su tutta la finestra della riga0**. IC esplorativi condizionati ai checkpoint disponibili, senza variabilità fra pretraining indipendenti. Gli utenti val erano già osservati; nessun nuovo test intatto e nessuna prova online.

## Cosa cambia nella diagnosi

La misura è affidabile: convergenza delle sonde e identità dei dati non spiegano il quadro. La pipeline PCA128 riduce le prestazioni e amplifica parte del divario P2, ma non crea da sola lo svantaggio puntuale di B nella classificazione. Il primo lettore temporale migliora poco la posa di B; il movimento rimane attenuato, con varianza predetta persino inferiore.

Dunque **non abbiamo dimostrato né che il FM vada rifatto, né che basti cambiare la testa**. La perdita della pipeline compressa di classificazione è misurata. Nella regressione, i contrasti aggiungono informazione oltre la media compressa ma quasi nulla a B oltre quella completa: non emerge la media come causa principale del movimento attenuato nella vista provata. La supervisione di posa continua a produrre un beneficio specifico sulla posa, non un trasferimento generale dimostrato.

## Ordine consigliato ora

1. Considerare **chiusi** controllo sonde, PSD-3 nella ricetta provata, bootstrap PCA, primo temporale e diagnosi media/contrasti 6a/6b.
2. **Non attivare automaticamente 6c.** Non estendere ora la ricerca di proiezioni/teste sulla stessa val: i controlli già eseguiti non indicano un recupero sostanziale che motivi cambiare pooling o FM.
3. **Seguire la nuova priorità tokenizer6d/6e approvata da Simone.** Prima verifica tecnica e preventivo, poi25lineare/100lineare/100GELU a pari dose, con obiettivo B invariato.5.000 passi sono screening,19.618 è l’endpoint autorizzato; nessun giudizio definitivo negativo dal solo screening.
4. Lasciare le nuove loss **7a/7b** dopo questo controllo. Per la posa resta plausibile un termine dinamico accanto agli angoli; per il FM generale serve una regressione di trasferimento diversa dalla posa supervisionata. Non introdurre insieme loss nuove e tokenizer nuovo.

Un rifit che isoli la sola PCA resta facoltativo: il riferimento storico è **26,77 core-ore**, da affinare con timing, non un preventivo già approvato. Le rettifiche Claude `d93e22e` concordano con questa lettura; i quattro OOF indicati su $WORK sono già stati recuperati e usati. Nessun merge del ramo Claude è implicito nella review di quei documenti.

La lista storica PSD-1..4 resta distinta: PSD-1 chiuso; PSD-2 originario riorientato, non completato integralmente; PSD-3 dispone ora delle prove sui lettori, senza esaurire fusione spaziale/temporale; PSD-4 a esposizione controllata resta aperto. Nessun rilancio automatico dei quattro punti.

## Risultati e tracciabilità

- [B19618: posa e trasferimento](reviews/b19618_results_20261029.md).
- Consegna Claude al commit `f69741d` e rettifiche `d93e22e`: sonde/PSD-3 letti con `git show`, conservando provenienze originali.
- [Pipeline PCA128/pool768: risultati appaiati](reviews/pca_existing_results_20261102.md), [JSON](../results/infra/pca_20261102/paired_report.json).
- [Temporale: risultati e limiti](reviews/pose_temporal_results_20261102.md), [JSON](../results/infra/pose_temporal_20261102/summary.json), [esecuzione e costi](reviews/pose_temporal_execution_20261102.md).
- Codice scientifico congelato `440af70`,35 test CPU superati e review del critico prima dei lanci; risultati verificati indipendentemente dopo l'esecuzione.

L'eccezione operativa di Simone autorizzava GPT a sottomettere direttamente **questa catena** attraverso submitter/ledger. Tutti quattro i job sono conclusi, incluso il confronto63228748. La regola generale di semplicione rimane per altre campagne; nessuna nuova attività remota è disposta da questa nota.

Aggiornamento operativo punto6: Simone ha esteso esplicitamente la deroga diretta a questa catena.
La nota precedente sull'eccezione limitata al temporale resta lo stato storico prima di questo via.

- [Punto 6: risultati media/contrasti](reviews/pose_slots_results_20261102.md), [sintesi JSON](../results/infra/pose_slots_20261102/summary.json), [review indipendente](reviews/pose_slots_results_critic_20261102.md), [costi e job](reviews/pose_slots_execution_20261102.md).
- Report finale **63252509 COMPLETED 0:0**, 415 secondi; tutti i sei job conclusi. Automazione `completa-wearusfm-punto-6` eliminata alla chiusura della catena.

Aggiornamento03/11: [collaudo e piano esecutivo completo](tokenizer_training_execution_20261103.md). Tutti e tre i bracci nuovi fino a19.618, extrabudget autorizzato; nessun risultato intermedio selezionerà il solo vincitore.

[Valutazioni posa5k: piano effettivo, pin e risorse](tokenizer_evaluation_5000_execution_20261103.md).
