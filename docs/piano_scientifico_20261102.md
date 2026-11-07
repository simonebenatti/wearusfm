# Piano scientifico dopo sonde, PSD-3, PCA, temporale e contrasti fra slot — 02/11/2026

**Seguito proposto il 07/11, ancora da revisionare:** [contrastiva su T100-G](contrastive_proposal_20261107.md),
due bracci nuovi C0/C1 a pari inizializzazione e dose, cambia soltanto il peso
InfoNCE. P1/P2 primari e guardia sulla posa; controllo del peso sul train,
nessuna nuova loss dinamica o tokenizer. Stato PROPOSTA, nessun codice/job.

**Aggiornamento 07/11, linea T100-LQ conclusa:** ramo log-quadratico a 19.618 passi,
EPN **82,633%**, **+1,531 pp [1,013; 2,015]** contro T100-L, senza superiorità
sulla GELU (+0,055 pp [−0,512; +0,603]). Posa E_dyn **0,844088** contro **0,843363**
del lineare: nessun beneficio dinamico aggiuntivo a 5.000 o 19.618. P1/P2 senza
miglioramento aggregato dimostrato, NI 1 pp sostenuta contro i due lineari.
Tutti i job LQ conclusi e le letture verificate indipendentemente; nessun monitor
né estensione a 49k. [Valutazione completa](logsquare_scientific_readout_20261107.md).
Le sezioni datate precedenti conservano le proposte e gli stati di allora;
questo aggiornamento non attesta lo stato delle altre linee parallele.

**Sono conclusi PCA, lettore temporale e punto 6 media/contrasti fra slot.** Tutti i risultati sono verificati dal critico. Il nuovo controllo trova informazione utile oltre la media compressa, ma aggiungerla alla media completa migliora B solo dello **0,0123% relativo**, senza recupero dell’ampiezza. Nessun job residuo; catena punto 6: **1,155556 GPU-ore e 10,254444 core-ore**.

**Tokenizer 6d/6e concluso il 04/11, compresa la classificazione.** Tutti i training e le 12 estrazioni completi; posa e sonde revisionate indipendentemente, **420/420 fit convergenti**, nessun job residuo. EPN-612: **81,601 / 81,102 / 82,577%** per T25-L/T100-L/T100-G. La GELU su100ms aggiunge **+1,476 pp [0,828; 2,121]**; il combinato vs25lineare **+0,977 pp [0,399; 1,546]**. La sola finestra lunga non migliora la classificazione in modo dimostrato. P1/P2 aggregati senza IC di miglioramento tutto positivo; NI1pp di G vs25 sostenuta, ma P2 al margine. Posa: E_dyn **0,851975 / 0,843363 / 0,838933**, beneficio combinato **1,53% relativo**, sotto2% pratico; varianza predetta16,88% dei target. [Analisi completa](reviews/tokenizer_classification_results_20261104.md), [review indipendente](reviews/tokenizer_classification_critic_20261104.md), [posa endpoint](reviews/tokenizer_pose_19618_results_20261104.md). Sonde CPU41,871667coreh/101, zero GPU aggiuntive. Monitor spento; nessuna nuova loss o ramo log-quadratico lanciato.

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
| **6d: tokenizer100, concluso** | Confronto25/100ms, passo25, stessi dati/loss/supporti. | Posa E_dyn −1,01% relativo; EPN81,601→81,102%, delta−0,499pp con IC che include zero. | Finestra sola: beneficio piccolo sulla posa, non dimostrato in classificazione. |
| **6e: tokenizer non lineare, concluso** | Filtri100ms apprendibili → GELU → proiezione, prima del Perceiver. | GELU: EPN+1,476pp [0,828;2,121] vs100lineare; posa ulteriore−0,53%. | Candidato promettente su EPN, non soluzione generale. P1/P2 senza miglioramento aggregato dimostrato; NI di G sostenuta ma P2 vs25 al margine. |
| **6f: log-quadratico, concluso** | Aggiunta di 64 risposte log(0,001+x²), W_log inizialmente nullo, stessa base T100-L. | EPN +1,531 pp; posa E_dyn +0,000726 [−0,001190; +0,002724], nessun miglioramento; 140 nuovi fit convergenti. | Esito specifico EPN, non superiore a GELU. Non dare priorità a LQ49k per la posa; contrastiva successiva da proporre separatamente. |
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
| **6d. Finestra del tokenizer — conclusa** | 100ms nominali, passo25, overlap75; confronto con25ms a pari dose19618 e raggio0/JEPA0. | E_dyn0,851975→0,843363, **−1,01%**; EPN81,601→81,102%, delta−0,499pp [−1,074;+0,113]. | Beneficio posa sotto2% pratico; nessun beneficio classificazione dimostrato dalla sola finestra. NI1pp P1 sostenuta, P2 irrisolta. |
| **6e. Non linearità del tokenizer — conclusa** | GELU sulle64 risposte temporali dopo gain e prima della proiezione; kernel/proiezione apprendibili. | EPN82,577%, **+1,476pp** vs100L [0,828;2,121]; posa E_dyn0,838933, −0,53% vs100L. Combinato posa−1,53%, varianza16,88%. | Beneficio EPN e piccolo beneficio posa; P1/P2 miglioramento non dimostrato. NI1pp Gvs25 sostenuta, P2 al margine. Non risolve dinamica/trasferimento generale. |
| **6f. Ramo log-quadratico — concluso il 07/11** | Teacher congelato, posa ridge a 5.000/19.618 e classificazione pool768 a 19.618, controlli storici verificati. | E_dyn LQ 0,912850→0,844088, senza beneficio vs L alle due dosi; EPN 82,633%, +1,531 pp [1,013;2,015] vs L e +0,055 pp [−0,512;0,603] vs G. P1 −0,267 pp, P2 +1,036 pp vs L, IC includono zero. | Il ramo è appreso ma non recupera la dinamica; beneficio EPN, nessuna utilità generale dimostrata. NI 1 pp P1/P2 sostenuta vs L e T25-L. Training 17,065556 GPU-ore incluso tentativo fallito; CPU finale 13,143333 core-ore/51. Nessun LQ49k avviato. |
| **7a. Obiettivo posa dinamica — da scegliere** | Possibile termine sulle variazioni/velocità accanto agli angoli assoluti, o diversa ponderazione della posa. | B combina già ancore0,2 e posa1; JEPA spento. Nessun nuovo braccio su queste modifiche. | Una sola ipotesi a pari dose, conservando il vincolo sul livello assoluto. Non aspettarsi che la sola loss sulla testa recuperi informazione assente dalla sua vista. |
| **7b. Obiettivo FM generale — alternativa di ricerca** | Contrastiva con ancore ed eventuale posa, con positivi/negativi/pesi motivati dal trasferimento desiderato. | Non sperimentato; la posa migliora senza beneficio generale dimostrato nelle classificazioni osservate. | Prima definire una regressione di trasferimento diversa dalla posa supervisionata. Non sommare insieme contrastiva, dinamica e cambio architettura senza un confronto interpretabile. |

L'errore dinamico è SSE delle variazioni centrate/varianza del target: più basso è meglio. La nuova ridge usa un supporto centrale comune; **i numeri della riga5 non sono direttamente confrontabili con la testa su tutta la finestra della riga0**. IC esplorativi condizionati ai checkpoint disponibili, senza variabilità fra pretraining indipendenti. Gli utenti val erano già osservati; nessun nuovo test intatto e nessuna prova online.

## Cosa cambia nella diagnosi

La misura è affidabile: convergenza delle sonde e identità dei dati non spiegano il quadro. La pipeline PCA128 riduce le prestazioni e amplifica parte del divario P2, ma non crea da sola lo svantaggio puntuale di B nella classificazione. Il primo lettore temporale migliora poco la posa di B; il movimento rimane attenuato, con varianza predetta persino inferiore.

Dunque **non abbiamo dimostrato né che il FM vada rifatto, né che basti cambiare la testa**. La perdita della pipeline compressa di classificazione è misurata. Nella regressione, i contrasti aggiungono informazione oltre la media compressa ma quasi nulla a B oltre quella completa: non emerge la media come causa principale del movimento attenuato nella vista provata. La supervisione di posa continua a produrre un beneficio specifico sulla posa, non un trasferimento generale dimostrato.

Il controllo tokenizer aggiunge un risultato specifico: GELU su100ms migliora EPN rispetto ai due lineari, con IC appaiati sopra zero. Non attribuisce il guadagno alla sola frequenza200Hz, non cambia la conclusione storica B−R3c sulla supervisione di posa e non dimostra trasferimento generale. Manca il braccio25ms+GELU: l'interazione completa fra finestra e attivazione non è identificata.

**Controllo storico aggiunto il04/11, senza nuovi job:** OOF appaiate T25-L−B19618
con sonde convergenti: P1 **+0,510pp [0,090;0,926]**, P2 **−0,093pp
[−1,382;1,065]**, EPN **−0,236pp [−0,764;0,274]**. Stessa ricetta nominale e
stesso seme, pairing/280fit verificati. È uno scostamento osservato fra due run,
non un pavimento statistico o una soglia da sottrarre ai contrasti tokenizer.
[Analisi pubblicata aggiornata](reviews/tokenizer_classification_results_20261104.md).

## Ordine consigliato ora

1. Considerare **chiusi** controllo sonde, PSD-3 nella ricetta provata, bootstrap PCA, primo temporale e diagnosi media/contrasti 6a/6b.
2. **Non attivare automaticamente 6c.** Non estendere ora la ricerca di proiezioni/teste sulla stessa val: i controlli già eseguiti non indicano un recupero sostanziale che motivi cambiare pooling o FM.
3. **Considerare concluso anche tokenizer6d/6e.** Il quadro completo sostiene un beneficio di GELU su EPN e un piccolo beneficio sulla posa; non sostiene una soluzione generale. Il prossimo candidato già discusso rimane T100-L con risposte log-quadratiche affiancate, con protocollo/calibrazione/preventivo da preparare e confronto primario con T100-L. Nessun nuovo lancio da questa consegna.
4. Le nuove loss **7a/7b** restano alternative successive da scegliere, non conseguenze automatiche del controllo concluso. Per la posa resta plausibile un termine dinamico accanto agli angoli; per il FM generale serve una regressione di trasferimento diversa dalla posa supervisionata. Non introdurre insieme loss nuove e tokenizer nuovo.

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

[Fase CPU5k e richiesta di spegnere il monitor dopo i lanci](tokenizer_pose_cpu_5000_execution_20261103.md).

## Proposta del 04/11: seguito se emerge un beneficio utile del tokenizer

Discussione richiesta da Simone, prima del confronto finale di classificazione.
Questa sezione ordina le possibilità; non modifica il protocollo o i job correnti.
Non assume un vincitore unico fra posa a 2 kHz e classificazione a 200 Hz.

| Ordine / ruolo | Prova | Che cosa deve stabilire |
|---|---|---|
| Prossimo candidato già concordato | T100-L più 64 risposte log-quadratiche affiancate alle lineari. | Se questa trasformazione facilita l'apprendimento; confronto primario con T100-L, anche se GELU risultasse migliore. Stessa inizializzazione non addestrata della parte condivisa, nuovi pesi a zero, dati/loss/dose invariati e controlli train anche a 200 Hz. |
| Consolidamento della ricetta promettente | Candidato e riferimento pertinente fino a 49.046 passi, includendo il decay; repliche con semi indipendenti. | Se il beneficio persiste alla dose completa e oltre il singolo pretraining. Il confronto deve restare a pari dose; non è necessario estendere ogni variante esplorata. |
| Utilità sul compito finale | Encoder congelato e fine-tuning completo, con riferimento della stessa architettura da zero, su classificazione e una regressione distinta dalla posa supervisionata. | Se il pretraining migliora prestazioni o efficienza nelle etichette; split e budget downstream comparabili, test finale intatto. Gli utenti val attuali restano sviluppo. |
| Opzione energetica, condizionale | Filtraggio strutturato, modulo quadrato, aggregazione temporale e compressione; eventuali componenti apprendibili. | Domanda fisica diversa dal quadrato di singole risposte integrate. Un successo del ramo log-quadratico non rende questa prova obbligatoria; un nullo non la esclude. |
| Opzione posa, condizionale | Un termine dinamico (variazioni oppure velocità) accanto agli angoli assoluti. | Se spostare la supervisione sul movimento riduce l'attenuazione senza perdere la posa assoluta. Una ipotesi alla volta, con controllo a pari ricetta. |
| Opzione FM generale, condizionale | Termine contrastivo con ancore ed eventuale posa, positivi e negativi definiti per il trasferimento desiderato. | Se cambia utilmente la rappresentazione generale. Le invarianti imposte devono conservare l'informazione necessaria alla regressione. |
| PSD-4, dopo aver fissato frontend e valutazione | Quote del corpus/esposizione controllate. | Se specializzazione e trasferimento dipendono dalla composizione dei dati. Non coincide con valutare semplicemente più dataset; anticiparlo solo se il risultato indica una forte specificità per dataset/frequenza. |

Kernel e proiezione sono già apprendibili; GELU e log sono trasformazioni fisse.
Il candidato log-quadratico conserva il primario contro T100-L; il confronto con
un altro vincitore selezionato resta secondario. Non chiamarlo energia di banda.

Il probing separato dei due blocchi delle cache pool768 rimane una diagnostica
facoltativa a basso costo, se serve ad attribuire un effetto; non un passaggio
obbligatorio. Sonde, PSD-3 nella ricetta provata, PCA appaiata, temporale minimo e
contrasti fra slot non richiedono una ripetizione senza una nuova domanda precisa.

Review indipendente del subagent next_steps_review concorde su sequenza, appaiamento,
separazione delle alternative e limiti delle validazioni già osservate. Nessun nuovo
training, estrattore, fit o monitor avviato da questa proposta.

## Discussione del 04/11: la taglia 30M non è una garanzia o un veto allo scaling

Simone chiede come aumentare i parametri e se non convenga già provare300M.
Questa è una proposta di priorità, non un protocollo esecutivo o una sottomissione.
Rettifica del criterio storico H3: richiedere un beneficio replicabile a30M era una
scelta di allocazione del calcolo, non una prova che un risultato debole a30M
escluda300M. Le vecchie misure di rango nei primi passi non valutavano downstream300M.

Preset reali in src/wearusfm/training/run.py:LADDER; conteggio CPU del modello con
ricetta T100-G corrente, cambiando solo i valori del preset (student, decoder e teste;
esclusi teacher EMA duplicato e stati dell'ottimizzatore):

| Preset | Parametri | D | Teste | Encoder locale | Backbone | Decoder | K |
|---|---:|---:|---:|---:|---:|---:|---:|
|30M|28.362.013|384|6|2|8|2|64|
|100M|104.820.757|640|10|2|12|2|64|
|300M|292.054.847|896|14|3|18|2|64|

Si allarga la rappresentazione condivisa dal tokenizer al decoder e si approfondisce
soprattutto il backbone. Ogni testa conserva64dimensioni; non è il solo aumento del
numero di teste a produrre il maggior numero di pesi. Finestra, passo, K, dati e loss
si possono mantenere invariati nel confronto. Aumentare la taglia richiede un nuovo
training; il checkpoint30M non si riprende direttamente con forme dei tensori diverse.

**Proposta:** dopo il confronto corrente, inserire una prova mirata di taglia prima
di accumulare molte ulteriori varianti tutte a30M. Per la domanda specifica di Simone,
un confronto30M/300M è legittimo;100M è un punto intermedio utile, non obbligatorio.
Fissare una sola ricetta frontend/obiettivo, benchmark di memoria/throughput e
calibrazione train-only dell'ottimizzazione per taglia; confrontare esposizione ai
dati e checkpoint comuni, includendo una lettura dopo decay. Stesso costoGPU e stessa
esposizione sono domande diverse, da distinguere. Non concludere contro300M da un
solo checkpoint breve. Leggere validazione e compiti finali oltre alla loss di train.
La proposta non cancella la candidata log-quadratica concordata: scala e nuova
trasformazione vanno confrontate separatamente, senza cambiare entrambe insieme.

Il vecchio benchmark4A100/batch32 del15/10 dava0,707/1,390/2,769s perpasso e28,7GiB/GPU
per300M: fattibilità storica, non preventivo della ricetta tokenizer/posa attuale.
Il modello più grande era circa3,9volte più lento, non assumere costo10volte o tempi
attuali identici. Non serve riaprire il multinodo come prerequisito della discussione.

Riferimenti primari consultati: TimesFM, studio17/70/200M e curve rispetto al calcolo
(https://arxiv.org/html/2310.10688v4#S6.SS2); Scaling-laws for Large Time-series Models
(https://arxiv.org/abs/2405.13867), dipendenza congiunta da parametri/dati/calcolo.
Moirai2.0 riporta anche plateau con l'aumento dei parametri
(https://arxiv.org/abs/2511.11698). Questi lavori di forecasting non forniscono una
legge quantitativa o una garanzia per WearUsFM/EMG.

Audit indipendente scaling_architecture e review scaling_design_review concordi.
Nessuna nuova run100/300M, loss o automazione avviata dalla discussione.

## Proposta del 04/11 dopo la chiusura: tre linee parallele

Simone propone di separare cinque domande: ramo log-quadratico, schedule lungo
con due letture delle sonde, fine-tuning, contrastiva e test breve300M. La
struttura è sensata: dose/uso downstream, scala e nuove trasformazioni possono
procedere indipendentemente. Questa è la risposta scientifica alla proposta,
non ancora un piano di job congelato o un nuovo lancio.

| Linea | Prova e riferimento | Domanda | Dipendenza |
|---|---|---|---|
| Consolidamento | T100-G30M da19.618 a49.046, stesso schedule/obiettivo. | Più training e completamento del decay migliorano le rappresentazioni? | Checkpoint completo già disponibile; preparare continuazione e valutatori. |
| Uso downstream, in parallelo | Fine-tuning dal T100-G30M a19.618; encoder congelato e stessa architettura da zero come controlli. | Il pretraining aiuta quando adattiamo tutto il modello al compito? | Protocollo e implementazione downstream; non attende il training lungo. |
| Scala | T100-G300M da zero contro T100-G30M a dose/esposizione corrispondenti. | La maggiore taglia aiuta con il frontend attuale? | Benchmark memoria/throughput e controllo train-only dell'ottimizzazione. |
| Esplorazione, prima | T100-L30M più ramo log-quadratico contro T100-L30M. | La trasformazione affiancata migliora le rappresentazioni? | Calibrazione train anche200Hz, parità iniziale e protocollo già discusso. |
| Esplorazione, dopo | Una base fissata con/senza termine contrastivo. | Un diverso obiettivo migliora il trasferimento? | Disegno di positivi/negativi/augmentazioni/peso e controllo appaiato. |

### Schedule lungo e significato delle due letture

La configurazione corrente contiene già WSD totale49.046, warmup1.962 e decay
finale20%; non occorre inventare uno schedule nuovo o riavviare da zero. Ripresa
con stato AdamW, teacher, RNG, contatori e stream verificati, su cartella dedicata;
checkpoint nominato19.618 preservato e condiviso in sola lettura con il fine-tuning.

Simone ha chiarito **«Entrambi»**: due momenti del training e due livelli interni.
I due momenti proposti sono **39.237 prima del decay e49.046 dopo**,
mantenendo19.618 come riferimento già misurato. Le letture
sono descrittive della traiettoria: il secondo tratto cambia sia dose sia learning
rate, quindi non isola causalmente il decay. Usare sonde convergenti e posa sulle
stesse popolazioni/supporti, senza scegliere il checkpoint migliore sulla val.

Per i livelli, la scelta proposta dopo la rilettura del codice è: **uscita del
Perceiver immediatamente prima del backbone**, e **uscita finale del backbone**.
Entrambe hanno forma batch×patch×64×D (D384 nel30M): si può applicare la stessa
aggregazione e lo stesso supporto senza confondere profondità e cambio di larghezza.
La prima contiene già tokenizer ed encoder locale; non è il segnale grezzo o
l'uscita immediata del tokenizer. Il confronto misura accessibilità con questi
lettori, non dimostra perdita irreversibile di informazione se la sonda peggiora.
Il post include anche la LayerNorm finale del backbone: si valuta l'intero blocco,
non la sola attention. Aggregazione FP32 e identica maschera key_time_valid;
la larghezza è D896 se il protocollo viene applicato anche al300M. Review del
codice indipendente parallel_next_review concorde con queste precisazioni.

| Momento | Prima del backbone | Dopo il backbone |
|---|---|---|
|39.237, prima del decay|Sonda classificazione e ridge posa|Stessi lettori/supporti|
|49.046, dopo il decay|Sonda classificazione e ridge posa|Stessi lettori/supporti|

Quattro combinazioni per modello/endpoint, senza quattro training. Classificazione:
media su slot e tempo valido, D valori; regressione: media soltanto sugli slot,
sequenza patch×D preservata, stesso crop e target. Teacher congelato, split e
preprocessing train-only comuni. Il readout storico pool768 resta come riferimento
di continuità, non viene sostituito o reinterpretato: concatena backbone medio e
encoder locale medio, non i due lati del backbone. L'estrattore va implementato e
verificato contro il forward esistente, salvando entrambe le viste nella stessa
inferenza. Eventuali viste mancanti a19.618 si recuperano dal checkpoint congelato,
senza riaddestramento; calcolo e storage entrano nel preventivo delle valutazioni.

Continuare soloG risponde alla domanda sulla sua dose. Per affermare che il
vantaggio della GELU sul lineare persiste alla dose completa serve anche **T100-L
lungo**, con letture comuni. Questo controllo aggiuntivo va incluso esplicitamente
nel preventivo se vogliamo quella conclusione; non confrontare G49k con L19k.

### Fine-tuning: partire dal checkpoint disponibile

Tre regimi con stesso split downstream, quantità di etichette, architettura della
testa e pari opportunità di selezione degli iperparametri: encoder preaddestrato
congelato + testa; preaddestramento + fine-tuning completo; stessa architettura da
inizializzazione casuale + training supervisionato. LR del fine-tuning e del
training da zero non devono essere identici a scapito di uno dei due; il tuning
rimane interno al train/validation downstream. Eventuali repliche appaiate
downstream non sostituiscono semi indipendenti di pretraining.

EPN è un primo compito pratico, escluso dal pretraining. Per sostenere utilità
nella regressione occorre scegliere un compito diverso dalla posa supervisionata;
gli utenti posa esclusi da soli non rendono nuovo il task. Le popolazioni già
viste nelle sonde restano sviluppo, non diventano un test intatto cambiando
lettore. Il protocollo deve separare sviluppo e conferma prima di nuovi fit.

Preparazione e primo fine-tuning19.618 possono procedere durante la continuazione
del pretraining. Il fine-tuning dal checkpoint49.046 attende soltanto quel file.
Riutilizzabile l'infrastruttura di split/normalizzazione; non si dichiara già
pronto un trainer completo WearUsFM downstream sulla base del solo harness storico.

### Scala: test breve informativo, senza un verdetto prematuro

300M usa100ms+GELU, K64, stessi dati/miscele/batch globale/obiettivo del30M;
larghezza e profondità seguono il preset. Non riprende pesi30M con forme diverse.
Prima fattibilità e stabilità train-only; verificare LR/clipping/aggiornamento
relativo per taglia, senza selezionare l'ottimizzazione sugli score downstream.
Distinguere ricetta identica da calibrazione per taglia, con pari opportunità e
modifiche dichiarate. Stesso numero di GPU non implica stesso costo totale.

Screening scientifico a5.000 confrontato con **30M a5.000**, non con30M19.618.
La posa30M5k è già disponibile; eventuali sonde di classificazione5k richiedono
nuove estrazioni/fit anche del riferimento. Un nullo così breve non esclude un
vantaggio300M alla dose successiva. **19.618 è il primo punto già completo di
valutazioni30M** e il seguito naturale per una lettura più solida, da preventivare
separatamente dallo screening. La successiva ipotesi di dose piena non è implicita.

Il benchmark storico suggerisce circa4volte il tempo, non10, ma non è un preventivo
per la ricetta corrente. Conservare quattroGPU per run; il parallelismo riguarda
esperimenti indipendenti, subordinato alle risorse/quote della coda.

### Esplorazione: trasformazione e contrastiva separate

Il log-quadratico mantiene la base **T100-L**, inizializzazione non addestrata
comune e nuovi pesi a zero, come fissato prima dei risultati. Il confronto
secondario con il migliore dei lineari rimane esplorativo; G è un riferimento
descrittivo. Non innestare il ramo suG19k senza il controllo della stessa
continuazione: risponderebbe a una domanda diversa.

Aggiornamento successivo del04/11, suggerimento Claude riportato da Simone:
**T100-G entra anche come confronto secondario appaiato**, riusando cache/OOF,
senza spostare la baseT100-L o cancellareT25-L. Emendamento esplicitamente
successivo ai risultati lineare/GELU; [proposta aggiornata](reviews/tokenizer_logsquare_proposal_20261103.md).

La contrastiva può seguire cronologicamente il log senza dipendere dal suo
successo. Scegliere una base e confrontarla con/senza contrastiva a pari dose;
non attribuire alla loss un confronto che cambia contemporaneamente frontend.
Se si usa una base selezionata dopo il log, quella selezione resta esplorativa.
Un possibile obiettivo è ancore0,2 + posa1 + lambda*contrastiva, JEPA0, ma lambda,
normalizzazione e costruzione delle coppie non sono ancora decisi.

La contrastiva non va progettata solo per separare gesti: evitare di imporre
invarianza a variazioni di ampiezza o tempo che servono alla regressione; verificare
negativi semanticamente simili e scorciatoie di soggetto/dataset. **TS2Vec** è un
precedente di contrastiva temporale valutata anche su forecasting con regressione
lineare, non una prova di efficacia per la nostra posa o una ricetta da copiare
integralmente: [paper primario](https://arxiv.org/abs/2106.10466).

Review indipendente parallel_next_review concorde sulle tre linee e sui controlli.
Nessun nuovo job o automazione da questa discussione; i protocolli esecutivi
dovranno riportare risorse, cap, checkpoint, confronti e criteri prima dei risultati.
