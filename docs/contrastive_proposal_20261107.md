# Prima prova contrastiva WearUsFM — proposta del 07/11/2026

**Stato: PROPOSTA PER REVIEW CLAUDE. Nessuna implementazione, release, prenotazione o sottomissione autorizzata da questo documento.** Simone chiede di definire base e confronto e farli rivedere prima dei lanci. Dopo la convergenza scientifica serviranno implementazione verificata, collaudo preparatorio e preventivo misurato; nessun seguito implicito della campagna LQ.

## 1. Domanda e base scelta

**Su T100-G da 30M, aggiungere un solo termine contrastivo alla ricetta ancore + posa migliora la classificazione dalle rappresentazioni congelate, senza un peggioramento materialmente rilevante della posa?** È uno screening di rappresentazione, non ancora una dimostrazione di utilità generale del FM o di trasferimento a una nuova regressione.

Scelgo **T100-G: finestra 100 ms, passo 25 ms, overlap 75 ms, GELU**, senza ramo log-quadratico. Ha un beneficio EPN sul lineare e una posa leggermente migliore di LQ; LQ non ha mostrato superiorità sulla GELU. La scelta usa risultati di sviluppo già osservati ed è dichiaratamente adattiva. Non sosteniamo che GELU sia ottimale in generale.

Riferimento di ricetta: `results/infra/tokenizer_execution_20261103/19618/T100-G/config.json`, SHA256 `f9d0318462fd30a5592087ef72c63d3036a88ff1d07fc0dad8624e36253c2cfa`. È un riferimento di configurazione, **non un checkpoint da cui ripartire**. Prima del rilascio si verificheranno la versione integrata del trainer, i file scientifici e le eventuali differenze rispetto a questa configurazione, senza importare cambi dalle altre linee.

| Braccio nuovo | Tokenizer e architettura | Obiettivo | Ruolo |
| --- | --- | --- | --- |
| C0 | T100-G 30M + proiettore contrastivo presente | 0,2 L_ancore + L_posa; peso contrastivo 0 | Controllo effettivo della nuova implementazione |
| C1 | Identico a C0 | 0,2 L_ancore + L_posa + λ(t) L_NCE | Unico candidato |

Entrambi da **stessa inizializzazione non addestrata**, seme 0, stessi dati, ordine delle finestre, maschere, montaggi, batch globale 32, accumulo 1, quattro rank, optimizer, LR 1e−4, clipping 1 e teacher EMA 0,996. JEPA resta a peso zero. WSD totale 49.046 con warmup 1.962 e decay finale 20%; si osservano 5.000 e 19.618, non si ridisegna uno schedule corto. A 19.618 non si è ancora nel decay.

**Due nuovi training, non candidato contro il solo G storico.** Il controllo include proiettore, calcolo delle statistiche e gestione della memoria esattamente come C1; cambia solo il peso della loss. Il G storico resta un controllo descrittivo di compatibilità, non il riferimento primario o una stima della variabilità fra pretraining. La prova confronta pari dose/esposizione, non pari costo con il vecchio trainer.

## 2. Un solo obiettivo aggiunto: InfoNCE con teacher e memoria

Per ogni finestra la query proviene dallo **student con la maschera già prevista dalla ricetta**; il positivo dal **teacher EMA sulla stessa finestra non mascherata**, senza gradiente. Stessi campioni, scala di sessione, QC e montaggio per i due ingressi. Il dropout anatomico dello student resta quello storico; il teacher resta in eval. Questa asimmetria è dichiarata, non introdotta di nascosto.

Nessuna nuova trasformazione di ampiezza, time-warp, permutazione, inversione temporale, crop disallineato, rotazione dei canali o modifica della fase. Una vista può perdere informazione per la maschera: non affermiamo che preservi perfettamente la posa. È un'ipotesi di robustezza alla mascheratura già usata, con ancore e posa che restano attive.

**Punto letto:** `enc.z` dopo il backbone e prima del decoder. Media dei 64 slot e poi delle patch fisicamente esistenti con almeno un canale QC valido; stesso supporto temporale in query e positivo. Padding escluso; gli istanti interamente mascherati restano query calcolate dal backbone e sono inclusi nella media. Non si usa `enc.key_time_valid` dello student come filtro della media: escluderebbe proprio quegli istanti. Per una finestra senza alcuna chiave visibile valida la query NCE è invalida, senza alterare le altre loss.

Proiettore dedicato **384 → 384 → GELU → 128**, bias nei due lineari, nessuna BatchNorm o dropout aggiuntivo. Norma L2 soltanto sull'uscita contrastiva, FP32 con epsilon 1e−8; norma inferiore a 1e−8 o valori non finiti fermano il collaudo. La normalizzazione non si applica a `enc.z`, alla testa posa o alle sonde. Il teacher ha la copia EMA del proiettore. Il proiettore viene inizializzato con RNG separato dopo il modello, è identico nei due bracci e non cambia lo stato casuale del training di base.

Per query valida i:

`ell_i = −log[exp(q_i·k_i+/τ) / (exp(q_i·k_i+/τ) + Σ_j exp(q_i·k_j−/τ))]`, con **τ = 0,2**.

Prodotti, normalizzazione e logsumexp in FP32; chiavi e memoria detached. Loss asimmetrica: gradiente soltanto attraverso query e proiettore dello student. Niente simmetrizzazione o secondo student forward in questa prima prova.

**Perché una memoria:** il batch globale 32 offre al massimo 31 altre finestre, spesso correlate, e ancora meno dello stesso dataset. La coda disaccoppia il numero di negativi dalla memoria GPU del batch. Questo riprende il meccanismo di [MoCo](https://arxiv.org/abs/1911.05722), adattato qui al teacher e alle maschere esistenti; non è una replica del suo protocollo per immagini.

## 3. Negativi, duplicati e rischio di scorciatoie

Proposta unica da far rivedere:

- **FIFO di 1.024 chiavi per dataset**, massimo 256 aggiornamenti di anzianità. Nessuna memoria condivisa fra dataset diversi. Si usano le chiavi ammissibili della coda precedente più quelle del batch globale corrente, già raccolte dai quattro rank.
- Il positivo è esclusivamente la stessa finestra con lo stesso montaggio. Non si definiscono positivi da etichette di gesto, angoli, vicinanza di pose o identità dell'utente.
- Sono negativi candidati soltanto esempi dello **stesso dataset**. Nessuna esclusione sistematica dello stesso soggetto: altrimenti distinguere l'identità basterebbe ancora più facilmente.
- Identità fisica: dataset, subject, session, trial, start, stop, fs. Il montaggio non distingue due copie della stessa finestra ai fini dei negativi. Ogni query conserva il proprio positivo; tutte le altre copie fisiche sono escluse dal suo denominatore.
- Nella stessa registrazione e nello stesso trial si escludono intervalli sovrapposti o con distanza fra gli estremi inferiore a **4 secondi**, indipendentemente dal montaggio. Se sono della stessa sessione ma trial diversi senza coordinate temporali confrontabili, si esclude conservativamente la coppia. Identità o tempi mancanti sono un errore di metadati, non un invito a considerarli diversi.
- `trial=None` è lecito per una registrazione continua: dentro la stessa sessione indica un unico asse temporale confrontabile, non un'identità mancante. La regola a 4 secondi è una scelta prudenziale della proposta, non la prova che oltre quel margine due gesti siano diversi.
- Deduplicazione delle chiavi per identità fisica: chiave più recente, batch corrente prima della FIFO; pareggio per indice globale nel batch. Ordine deterministico rank/indice. Nessun hard-negative mining o selezione tramite la loss.
- Servono **almeno 32 negativi ammissibili per query**. Altrimenti il contributo contrastivo è zero e viene contato esplicitamente. Gli altri termini non saltano la finestra. `L_NCE = somma ell_i / 32` sul batch globale, anche quando alcune query sono invalide: non rinormalizzare sui soli casi facili/ammissibili.

La FIFO contiene soltanto chiavi di training e metadati, non segnali raw, validazione o test. Si aggiorna una volta per update riuscito, dopo il backward; le chiavi sono quelle prodotte dal teacher pre-update. Si applica poi l'EMA una volta come nel trainer. Ordine, anzianità, puntatori e contatori fanno parte del checkpoint; nessun aggiornamento della FIFO durante valutazione o ricomputazione per gradient checkpointing.

Il filtro per dataset elimina la soluzione basata **soltanto** sul separare dataset differenti. Non elimina scorciatoie di soggetto, sessione, montaggio o dispositivo all'interno del dataset. Finestre lontane possono contenere lo stesso gesto: rimangono falsi negativi non riconoscibili senza etichette. Pubblicare quota di negativi dello stesso soggetto/sessione/montaggio, età, conteggio, duplicati esclusi e copertura per dataset. L'obiettivo è discriminazione di istanze, non contrastiva supervisionata per gesto.

Il pooling globale non supervisiona esplicitamente l'ordine temporale. Questo primo candidato cerca un miglioramento nella classificazione mantenendo un vincolo sulla regressione; non promette di risolvere l'inseguimento del movimento.

## 4. Peso: una calibrazione sul train, poi fisso

Nessuna griglia di λ, nessuna selezione sulla validazione e nessun aggregatore apprendibile delle loss. **Una sola regola proposta:** scegliere λ* perché il gradiente contrastivo iniziale sull'encoder condiviso abbia norma pari al 10% del gradiente della loss di base, su un pannello train congelato.

Procedura preparatoria da implementare solo dopo review:

1. Modello e teacher allo stato iniziale, nessun optimizer step. Precaricare le FIFO passando soltanto il teacher sui primi 256 batch globali dello stream train congelato. Non leggere bersagli downstream per definire positivi/negativi.
2. Sui successivi 32 batch globali, stato dei pesi fisso, calcolare separatamente i gradienti completi **globali**, prima del clipping, di `0,2 L_ancore + L_posa` e di `L_NCE`. Parametri considerati: tokenizer, identity, local, pool, backbone; esclusi decoder, teste e proiettore. La FIFO continua ad avanzare deterministicamente, senza training.
3. `λ* = 0,10 × mediana_b(||g_base,b||₂ / ||g_NCE,b||₂)`. Tutti i 32 rapporti devono essere finiti e positivi; λ* fuori **[1e−4; 1]** comporta revisione, senza clamp, tentativi su altri pesi o ampliamenti automatici.
4. Pubblicare valore, distribuzioni, coseni e pin del pannello prima del training; poi **reset completo** di modello, optimizer, teacher, FIFO, contatori e RNG per entrambi i bracci.

Limite cruciale: la testa posa è inizializzata a zero, quindi all'inizio il suo gradiente sull'encoder è nullo. Questa calibrazione è principalmente relativa alle ancore; non garantisce un bilanciamento 10% durante il training o dopo AdamW/clipping. È una scelta iniziale conservativa, esplicitamente sottoposta a Claude. Non riadattare λ a metà run.

Training C1: primi 256 update `λ=0` per riempire la memoria; nei successivi 256 rampa lineare da 0 a λ*; dal completamento del passo 512, λ*. C0 mantiene λ=0. La testa del controllo è presente e calcolata nello stesso grafo, con gradiente nullo del termine; medesima manutenzione FIFO. Le curve di training mostreranno separatamente le loss, il rapporto dei gradienti e il clipping; una discesa di NCE non costituisce successo scientifico.

## 5. Collaudo e condizioni che possono fermare la proposta

**Prima dei training scientifici**, tre verifiche che non possono essere saltate:

1. **Audit della disponibilità dei negativi, solo metadati train.** Tracciare i primi 1.024 batch dello stream a quattro rank con le identità reali, applicando esattamente le regole FIFO. Sul tratto successivo al riempimento: almeno 90% di query con almeno 32 negativi complessive e 80% per dataset; ciascun dataset con quota positiva deve avere almeno 32 query osservate per essere valutabile. Questo è un gate necessario: non certifica ancora la validità QC/chiavi visibili della vista o la norma del proiettore. Il gate di copertura completa, con le stesse soglie, viene verificato nel collaudo con le viste reali. Il dataset poco osservato o con memoria troppo vecchia è un problema da discutere, non viene escluso automaticamente. Se necessario un tracciato più lungo o una diversa coda saranno emendamenti pubblicati prima del collaudo; non si modifica il mix dei dati per far passare il gate.
2. **Correttezza CPU/DDP.** Filtro dei negativi con casi duplicati/overlap/trial; padding e QC; niente gradiente al teacher; equivalenza a 1 e 4 processi della loss e dei gradienti a batch globale identico, anche con query invalide e un rank senza query ammissibili. Solo le chiavi detached vengono all-gather: numeratori locali e denominatore globale 32, compensazione della media DDP. Non usare un gradiente locale come stima globale. Stato delle FIFO identico fra rank; checkpoint/resume equivalente. Per questa prova accum_steps=1 è un vincolo esplicito, non supporto implicito a casi diversi.
3. **Collaudo CUDA della ricetta reale.** Calibrazione train, parità iniziale C0/C1 dei pesi condivisi e delle uscite di base, confronto λ0 con il percorso senza loss nuova, ripresa a quattro rank, finitezza, memoria e timing. Almeno 64 update appaiati in un segmento tecnico con NCE attiva dopo riempimento/rampa, senza leggere sonde o validation. Nessuna coda vuota viene spacciata per collaudo del termine. Stato e FIFO del collaudo si scartano.

Con curriculum variabile di maschere/dataset non si presume validità della copertura per sempre: loggare copertura cumulativa e per blocchi di 1.024 update; una violazione del gate dopo riempimento ferma la fase per diagnosi, senza cambiare la regola. I dati per dataset e il numero di query non valutabili restano visibili. La divisione per 32 preserva la scala della base ma **non** dà uguale peso NCE effettivo a dataset con copertura diversa.

Misure dei gradienti globali/coseni prima del clipping nel collaudo; nel training ai passi 256, 512, 1.962 e ai checkpoint, con modalità e costo fissati nel piano. Non inferire bilanciamento dall'ampiezza numerica delle loss. I controlli di collasso/NaN già presenti restano; non si introducono stop basati su un miglioramento intermedio delle metriche downstream.

## 6. Valutazione proposta e criterio di lettura

**Sonde sul teacher prima del proiettore contrastivo**, con ricette storiche: posa ridge istantanea media K64, e classificazione pool768 senza PCA, logistica convergente. Il proiettore viene ignorato nei readout e non modifica il pool768. Niente nuove teste, livelli o ricerca di pooling in questo confronto.

| Momento | Lettura | Ruolo |
| --- | --- | --- |
| 5.000, entrambi i bracci | Posa: E_dyn, correlazione, varianze, MAE, livello/dinamica | Descrittivo; non seleziona un vincitore o la continuazione |
| 19.618, entrambi | P1 e P2 con OOF e audit di tutti i fit | **Due endpoint principali della nuova domanda contrastiva** |
| 19.618, entrambi | Posa sulle stesse finestre/utenti, tutti i gruppi | Guardia contro il deterioramento della regressione; eventuale beneficio descrittivo |
| 19.618, entrambi | Cinque dataset separati, incluso EPN | Secondari; nessuna promozione basata soltanto sul dataset vincente |

Il primario è sempre **C1−C0 a pari passo**. La scelta di P1/P2 anziché il solo EPN cambia la domanda rispetto a LQ ed è fatta prima dei nuovi risultati. Pesi P1 8/13, 3/13, 2/13; P2 1/2, 1/2. Stessi fold e target, selezione dei lettori sul train interno; nessun refit dei lettori scelto in base all'esito del contrasto. Fit irrisolti invalidano la lettura interessata.

Proposta di screening a 19.618, da validare da Claude:

- Miglioramento utile in almeno uno fra P1/P2: **stima ≥ +0,5 pp e limite inferiore dell'IC appaiato al 97,5% > 0**. Due intervalli al 97,5% usano Bonferroni per i due endpoint principali. Pubblicare anche gli IC95 descrittivi di tutti i risultati.
- L'altro aggregato deve sostenere NI entro **1 pp**, limite inferiore dell'IC95 > −1 pp.
- Per la posa, `r = E_dyn(C1)/E_dyn(C0) − 1`: limite superiore dell'IC95 appaiato **< +2%**. È un margine di tolleranza proposto per questa domanda, non il vecchio criterio di miglioramento del 2% trasformato retroattivamente. Nessuna deroga perché EPN sale.
- Se manca una condizione: esito misto, irrisolto o sfavorevole, con numeri completi; nessun passaggio automatico a 49k, altra temperatura/peso, fine-tuning o nuova loss. Un successo motiva una proposta di conferma, non adozione immediata.

Bootstrap accoppiato per utente, **10.000 draw, seme 20261107**, sufficienti conservati; stessi draw fra bracci, ordine dataset alfabetico, convenzione classi mancanti invariata. Il numero maggiore serve a stabilizzare le code degli IC97,5, non aumenta il numero di soggetti. Intervallo del rapporto di posa r ricalcolato dentro ciascun draw. La correzione riguarda soltanto i due endpoint principali nuovi: non corregge tutte le esplorazioni precedenti.

Anche un risultato positivo resta condizionato a un solo pretraining per braccio e a dati di sviluppo già osservati. Non comprende variabilità fra semi o un test finale intatto. La posa è un obiettivo già supervisionato: preservarla **non dimostra trasferimento a una regressione nuova**. Quest'ultima e il fine-tuning controllato restano un seguito distinto, da definire se lo screening merita conferma.

## 7. Costo e ordine, senza lanci

Nel codice attuale `_jepa_compute` calcola già `student.encode(inp, visible)` e `teacher.encode(inp, None)`, anche a JEPA peso zero. La proposta riusa queste uscite: non richiede un secondo forward/backward dello student. Questo rende plausibile un sovraccosto contenuto, **da misurare**, non autorizza a dichiararlo nullo. Non si rimuove ora il decoder teacher calcolato storicamente: sarebbe un'altra modifica.

Il G storico richiese (3.916 + 11.334) secondi su quattro GPU, **16,9444 GPU-ore** per 19.618 passi. Due bracci fanno circa **33,89 GPU-ore prima del nuovo sovraccosto**; stessa dose, non la promessa di uguale costo. Il vecchio tetto 22 GPU-ore per braccio suggerisce un involucro preliminare di 44 GPU-ore per i due training, non un budget esecutivo approvato.

| Fase da preparare | Ordine di risorse proposto | Condizione prima di un lancio futuro |
| --- | --- | --- |
| Review scientifica e implementazione/test CPU | Lavoro locale | Convergenza sulla presente ricetta |
| Audit metadati train | Involucro 2 core-ore, zero GPU | Comando, dati e piano pubblicati; copertura valutabile |
| Collaudo/calibrazione e benchmark a 4 GPU | Involucro 2 GPU-ore, 16 core-ore | Test, piano e checkout congelati |
| Due training 5.000→19.618 | Riferimento 33,89 GPU-ore; vecchio tetto doppio 44 | Tempi/memoria reali, λ pinnato, nuovi tetti e review del rilascio |
| Otto estrazioni: posa×2 dosi + P1/P2 per ciascun braccio | Riferimento storico fino a 5,5 GPU-ore, benchmark separato fino a 1 GPU-ora | Endpoint completi, pin e costi effettivi d'inferenza |
| Ridge, sonde e confronti CPU | Involucro storico duplicato fino a 102 core-ore | Benchmark/gate e matrice concreta; confronto C1−C0 condiviso può ridurre il tetto |

Checkout e archiviazione vanno aggiunti nel preventivo concreto. Queste sono scale di costo, non prenotazioni: ogni tranche andrà pubblicata con stima osservata, tetto e accounting separati. Se il costo o la copertura non reggono, si torna alla proposta; nessun passaggio tacito a una ricetta differente. I due bracci sono indipendenti e potranno essere accodati in parallelo su due nodi da quattro GPU, senza modificare il batch o usare otto GPU per lo stesso modello.

Ordine: **review Claude della proposta → convergenza con Simone → implementazione e verifiche → collaudo preparatorio approvato → lettura tecnica e preventivo congelato → training e valutazioni autorizzati**. GPT resta operatore della linea 3 con submitter/ledger comuni; Claude revisore. Nessun monitor in background.

## 8. Fondamento e alternative lasciate fuori

- [SimCLR, Chen et al. 2020](https://proceedings.mlr.press/v119/chen20j.html) motiva la separazione fra rappresentazione e proiettore e mostra l'importanza delle viste e del batch. Non ne trasferiamo automaticamente le augmentazioni per immagini all'EMG.
- [MoCo, He et al. 2020](https://arxiv.org/abs/1911.05722) usa encoder a media mobile e coda per ottenere un dizionario che non sia limitato al batch. La nostra scelta di memoria per dataset, esclusioni temporali, λ e τ è una proposta WearUsFM, non un risultato garantito da quel lavoro.
- [TS2Vec, Yue et al. 2022](https://arxiv.org/abs/2106.10466) contrasta rappresentazioni temporali in contesti aumentati e valuta anche forecasting. Una loss temporale gerarchica sarebbe interessante per una domanda diversa, ma aggiungerebbe scelte su scale, allineamento e falsi negativi temporali: non viene mescolata alla prima prova globale.
- [TF-C, Zhang et al. 2022](https://arxiv.org/abs/2206.08496) studia coerenza tempo/frequenza. Introdurre anche un encoder/vista in frequenza confonderebbe il cambio di obiettivo con un nuovo frontend; resta fuori da questo confronto.

Nessuna ricerca di molte combinazioni. Se il controllo fallisce tecnicamente, prima si corregge quello; se la prova scientifica è negativa, si legge il risultato della ricetta specifica senza concludere che ogni contrastiva sia inutile.

## 9. Richiesta puntuale alla review

Claude dovrebbe pronunciarsi su: base G e due run nuovi; positivo masked/clean e livello letto; memoria per dataset ed esclusioni; rischio identità e falsa invarianza; λ da gradiente iniziale e suo limite dovuto alla posa-head zero; copertura/anzianità della coda; riduzioni DDP e ripresa; endpoint P1/P2 e margini; scala di costo. Richieste di modifica vanno motivate e convergono **prima** di implementazione/deploy. Il suo parere non è una sottomissione e non converte lo stato della proposta in APPROVED_FOR_TRAINING.
