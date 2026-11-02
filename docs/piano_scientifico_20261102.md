# Piano scientifico dopo sonde, PSD-3, PCA e lettore temporale — 02/11/2026

**I due controlli approvati sono conclusi.** Il confronto appaiato PCA è stato eseguito sugli output esistenti; il lettore temporale ha completato tutte le quattro celle senza riaddestrare il FM. Risultati verificati dal critico, costo della catena temporale **0,554 core-ore e zero GPU**. Nessun job residuo della catena.

Il percorso resta: **verificare la misura → capire come leggere le rappresentazioni → scegliere se cambiare testa, rappresentazione o obiettivo**. Il piano iniziale è conservato nel commit `63ba143`; questa nota ne aggiorna lo stato, senza modificare il [protocollo congelato](pose_temporal_pca_protocol_20261102.md) o i criteri precedenti.

## Quadro organico aggiornato

P1/P2 sono classificazione interna dal **teacher**; la posa è regressione dallo **student**. I risultati non sono intercambiabili.

| Passo e stato: quello che è stato fatto | Spiegazione | Risultato | Conclusione e seguito |
|---|---|---|---|
| **0. B a 19.618 passi — concluso** | Più addestramento con posa; 15 utenti esclusi dal train e confronto R3c a pari passo. | Dalla testa B5.000 a B19.618: errore dinamico **0,927 → 0,867**, correlazione **0,270 → 0,366**, varianza predetta **6,3% → 15,0%** del target. | La posa migliora fuori dal train, ma resta attenuata. Non dimostra utilità generale del FM o trasferimento ad altre regressioni. |
| **1. Controllo delle sonde — concluso** | Replica a 3.000 iterazioni e confronto simmetrico a 12.000 per verificare la convergenza. | Replica esatta, **0 fit al limite su 280**. B−R3c: P1 **−0,553 pp** [−1,105; −0,017], P2 **−1,172 pp** [−2,956; +0,651]. | La convergenza non spiega il mancato beneficio di B. Non occorre rifare la tranche. |
| **2. PSD-3 sulle viste mediate nel tempo — concluso** | B/R3c × pool/decoder × logistica/piccola MLP, encoder congelato, PCA128 sul train. | MLP perde **0,49–0,75 pp in P1**; IC aggregati P2 includono zero. Nessun vantaggio aggregato chiaro di decoder su pool. **56,75 core-ore, zero GPU**. | Questa ricetta non risolve il problema; non esaurisce lettori non lineari, slot interi o tempo. Il confronto MLP−logistica sulle stesse viste compresse resta valido. |
| **3. Confronto PCA — analisi appaiata conclusa; ablazione isolata non eseguita** | Quattro OOF recuperati da $WORK, pairing verificato e 2.000 bootstrap per soggetto, senza nuovi fit. | Pipeline128−768: B **−0,497 pp P1 / −2,859 P2**; R3c **−0,453 / −1,699**. Tutti i relativi IC sotto zero. Penalizzazione aggiuntiva di B su P2 **−1,160 pp** [−1,740; −0,660]. | La pipeline compressa penalizza entrambi, soprattutto B in P2. Cambiano anche precisione/thread: non è l'effetto isolato della sola PCA. B resta sotto R3c nelle stime puntuali anche senza PCA. |
| **4. Feature temporali — disponibili e validate** | Riutilizzate le cache student B/R3c19618, 160×384 per finestra: 40 utenti train e 15 val. | Quattro cache, SHA/sidecar/metadati e supporti verificati. **Nessuna nuova estrazione GPU**. | Consentono la prima prova temporale. La media dei64 slot è già avvenuta: per studiare gli slot interi servirebbero altre feature da checkpoint congelati. |
| **5. Lettore temporale minimo — concluso** | Ridge istantanea contro contesto **t−100ms, t, t+100ms**, senza PCA, supporto comune e selezione sui soli utenti train. | B errore dinamico **0,859512 → 0,855321**, delta **−0,004191** [−0,006329; −0,001939], beneficio12/15 utenti. R3c **0,986567 → 0,961495**. B varianza predetta **14,0% → 10,8%**, MAE leggermente peggiore. | Piccolo beneficio B, circa **0,49% relativo**, senza recupero dell'ampiezza. B conserva vantaggio di posa su R3c. Non basta per accusare il pooling, cambiare il FM o iniziare una ricerca estesa di teste/lag sulla stessa val. |
| **6. Modifica testa/pooling/FM — eventuale** | Un intervento deve rispondere a un limite identificato e avere un confronto specifico. | Il temporale aiuta poco B; PSD-3 pool/decoder cambia anche strato e preprocessing. Nessun confronto causale media contro slot interi. | Se la domanda è il pooling, il controllo successivo deve conservare gli slot e controllare dimensionalità/costo, prima di modificare il FM. Nessun lancio incluso automaticamente. |
| **7a. Obiettivo posa dinamica — da scegliere** | Possibile termine sulle variazioni/velocità accanto agli angoli assoluti, o diversa ponderazione della posa. | B combina già ancore0,2 e posa1; JEPA spento. Nessun nuovo braccio su queste modifiche. | Una sola ipotesi a pari dose, conservando il vincolo sul livello assoluto. Non aspettarsi che la sola loss sulla testa recuperi informazione assente dalla sua vista. |
| **7b. Obiettivo FM generale — alternativa di ricerca** | Contrastiva con ancore ed eventuale posa, con positivi/negativi/pesi motivati dal trasferimento desiderato. | Non sperimentato; la posa migliora senza beneficio generale dimostrato nelle classificazioni osservate. | Prima definire una regressione di trasferimento diversa dalla posa supervisionata. Non sommare insieme contrastiva, dinamica e cambio architettura senza un confronto interpretabile. |

L'errore dinamico è SSE delle variazioni centrate/varianza del target: più basso è meglio. La nuova ridge usa un supporto centrale comune; **i numeri della riga5 non sono direttamente confrontabili con la testa su tutta la finestra della riga0**. IC esplorativi condizionati ai checkpoint disponibili, senza variabilità fra pretraining indipendenti. Gli utenti val erano già osservati; nessun nuovo test intatto e nessuna prova online.

## Cosa cambia nella diagnosi

La misura è affidabile: convergenza delle sonde e identità dei dati non spiegano il quadro. La pipeline PCA128 riduce le prestazioni e amplifica parte del divario P2, ma non crea da sola lo svantaggio puntuale di B nella classificazione. Il primo lettore temporale migliora poco la posa di B; il movimento rimane attenuato, con varianza predetta persino inferiore.

Dunque **non abbiamo dimostrato né che il FM vada rifatto, né che basti cambiare la testa**. Abbiamo delimitato due limiti delle letture eseguite: feature compresse nella classificazione e slot già mediati nella regressione. La supervisione di posa continua a produrre un beneficio specifico sulla posa, non un trasferimento generale dimostrato.

## Ordine consigliato ora

1. Considerare **chiusi** controllo sonde, PSD-3 nella ricetta provata, bootstrap PCA e primo temporale. Non rilanciare automaticamente MLP o training del FM.
2. Scegliere la domanda del prossimo esperimento: capire l'effetto del pooling, migliorare la posa dinamica o migliorare il trasferimento generale. Il risultato temporale non decide da solo fra queste domande.
3. Se vogliamo attribuire il limite alla media, preparare un controllo con **slot conservati da encoder congelato**, includendo un confronto a capacità/dimensione controllata. Sarebbe nuova inferenza e fit limitato al lettore, non nuovo pretraining; costo e protocollo da definire prima di qualsiasi lancio.
4. Se la priorità diventa una nuova loss, fissare una modifica e una verifica di trasferimento anche in regressione. La contrastiva è una candidata dentro questo ramo, non un passaggio obbligatorio dopo tutti gli altri.

Un rifit che isoli la sola PCA resta facoltativo: il riferimento storico è **26,77 core-ore**, da affinare con timing, non un preventivo già approvato. Le rettifiche Claude `d93e22e` concordano con questa lettura; i quattro OOF indicati su $WORK sono già stati recuperati e usati. Nessun merge del ramo Claude è implicito nella review di quei documenti.

La lista storica PSD-1..4 resta distinta: PSD-1 chiuso; PSD-2 originario riorientato, non completato integralmente; PSD-3 dispone ora delle prove sui lettori, senza esaurire fusione spaziale/temporale; PSD-4 a esposizione controllata resta aperto. Nessun rilancio automatico dei quattro punti.

## Risultati e tracciabilità

- [B19618: posa e trasferimento](reviews/b19618_results_20261029.md).
- Consegna Claude al commit `f69741d` e rettifiche `d93e22e`: sonde/PSD-3 letti con `git show`, conservando provenienze originali.
- [Pipeline PCA128/pool768: risultati appaiati](reviews/pca_existing_results_20261102.md), [JSON](../results/infra/pca_20261102/paired_report.json).
- [Temporale: risultati e limiti](reviews/pose_temporal_results_20261102.md), [JSON](../results/infra/pose_temporal_20261102/summary.json), [esecuzione e costi](reviews/pose_temporal_execution_20261102.md).
- Codice scientifico congelato `440af70`,35 test CPU superati e review del critico prima dei lanci; risultati verificati indipendentemente dopo l'esecuzione.

L'eccezione operativa di Simone autorizzava GPT a sottomettere direttamente **questa catena** attraverso submitter/ledger. Tutti quattro i job sono conclusi, incluso il confronto63228748. La regola generale di semplicione rimane per altre campagne; nessuna nuova attività remota è disposta da questa nota.
