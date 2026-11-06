# Claude — completa le sonde, poi esegui PSD-3

Per i prossimi due giorni ti affido la conduzione operativa dei due passaggi già
decisi: sonde e PSD-3. Porta a termine esecuzione, controlli, analisi e consegna
senza attendere una nuova attivazione di GPT o un secondo consenso per il piano
approvato. Le sottomissioni su Leonardo restano affidate a **semplicione**:
tu prepari e coordini, lui usa submitter e ledger comuni.

Questo prompt autorizza il seguito PSD-3, che nella consegna GPT era soltanto
preparato. Non autorizza nuovi training del FM, nuove loss o altri esperimenti.

## Riferimenti e revisione iniziale

Consegna PSD-3: **d4092b8b719e532b0f387c19d53688b0aa87a7c4**, disponibile su
`origin/codex/r3-long-collaudo-20261028`. Integra quel commit nel tuo worktree
preservando la storia; non intervenire su main o sui checkout di job attivi.
Leggi:

- `docs/psd3_readers_20261029.md` e `docs/reviews/psd3_preparation_20261029.md`;
- `results/infra/psd3_preparation_20261029/plan.json`;
- `scripts/psd3_readout.py`, `src/wearusfm/harness/psd3_{inputs,readers,compare}.py`
  e i relativi test CPU;
- la tua consegna sonde `22bf2cd` / `b98ece0` e i suoi seguiti;
- **`docs/reviews/sonde_convergenza_review_20261029.md`**, già presente nel ramo
  GPT: contiene tre difetti da correggere prima del deploy, con regressioni.

Chiudi R1 (parità completa), R2 (persistenza audit/fase) e R3 (ricetta verificata
prima dei fit), oppure documenta che sono già corretti nel nuovo SHA. Riesegui
le regressioni senza indebolirle. Per le correzioni e l'integrazione usa una
review indipendente del critico; durante questo passaggio temporaneo non serve
attendere un'altra attivazione di GPT. Rigenera piano/comandi sul codice finale.
Queste istruzioni non dichiarano già verificato un codice ancora da correggere.

## 1. Chiudi le sonde e decidi se PSD-3 richiede un adeguamento

Completa le fasi del protocollo sonde e pubblica separatamente: riproducibilità,
convergenza per candidato/refit, selezioni e punteggi, delta B−R3c e consuntivo.
Usa risultati, ricevute e ledger reali: non rilanciare job già eseguiti.

Applica queste regole prima dei risultati PSD-3:

1. **Replica/provenienza valide e confronto stabile:** esegui PSD-3 come preparata.
2. **Le sonde meglio convergenti cambiano punteggi o segno di B−R3c:** aggiorna
   la conclusione e procedi comunque. Il segno favorevole o sfavorevole a B non
   seleziona l'avvio né gli iperparametri di PSD-3.
3. **Restano warning a12000 iterazioni:** specifica se riguardano candidati o
   refit selezionati. Non aumentare automaticamente il budget PSD-3: le sonde
   storiche leggono768 feature, la nuova logistica128 componenti. Se necessario,
   prepara un controllo mirato sul solo train, con costo dichiarato prima del
   lancio. Solo se quel controllo dimostra un problema sulla nuova pipeline puoi
   aumentare max_iter, simmetricamente per B/R3c e pool/dec, mantenendo tol1e−4,
   solver e griglia C. Motiva la scelta con diagnostiche di ottimizzazione, non
   score held; congela prima della matrice un emendamento e aggiorna insieme
   codice, ricetta, piano e test. Nessuna ricerca aperta di iperparametri.
4. **Difetti di identità, leakage, pairing o riproducibilità:** correggi e verifica
   il perimetro coinvolto prima di proseguire. Conserva e identifica gli esiti
   invalidati. Non cambiare un hash soltanto per far passare una guardia. Se la
   soluzione richiede nuovi dati/checkpoint o una diversa domanda scientifica,
   segnala il problema concreto e la proposta invece di ampliare la campagna.

Scrivi nel registro il caso osservato e gli adeguamenti necessari. Correzioni
tecniche e adattamenti di risorse entro il budget vigente sono già autorizzati.

## 2. Verifica e lancia PSD-3

Mantieni il protocollo: **B/R3c a19618 × pool/dec × logistica/MLP**, cinque dataset
e quattro fold di soggetti. StandardScaler e PCA128 solo train, trasformazione
condivisa dai lettori; MLP128→32→classi,60epoche, semi0/1, griglie e selezione
validation fissate. Nessun ensemble. Totale20shard,80 trasformazioni,
560 fit logistici e480 fit MLP. Il CLI espone solo la ricetta congelata.

- Rivedi l'integrazione e riesegui i test pertinenti. Mantieni le guardie su
  etichette testuali EPN, split, hash, classi per fold, report/audit completi,
  provenienza di viste/righe e identità del sorgente prima/dopo. Correggi difetti
  dimostrati, committa/pusha e congela il codice prima dell'esecuzione.
- Verifica realmente le cache, anche con `inventory --verify-sha`. L'inventario
  GPT deriva dagli archivi e non attestava una nuova lettura dei NPZ remoti.
  Cache e checkpoint restano in sola lettura; nessuna nuova estrazione GPU.
- Prepara il launcher con CODE_DIR/SHA congelato, ambiente e thread numerici
  coerenti, directory nuove, ricevute e dipendenze. I comandi Python del
  protocollo non sono già job sottomessi. Nessun fit pesante sul login.
- Fai eseguire i quattro benchmark train-only previsti: B DB6/EPN ×pool/dec.
  Misura tempi e memoria. **86core-ore/0GPU-ore è un preventivo provvisorio**,
  non una garanzia che ogni shard finisca entro1h. La proiezione del benchmark
  non è un limite superiore per tutti i C, fold e modelli.
- Pubblica il preventivo definitivo dopo il timing, sottraendo dal ledger
  consumi e prenotazioni delle sonde. Puoi adeguare CPU/memoria/walltime entro il
  residuo, rispettando serial≤4h. Non ridurre celle, epoche o griglie per rientrare
  nel tempo; niente retry ciechi. Se il residuo non basta, quantifica il costo
  mancante prima di impegnarlo.
- Affida a semplicione matrice e rapporto completo, senza chiedere un secondo
  via. Conserva ID, SHA, dipendenze, log e costi. Comunica gli ID dopo il lancio.

Non modificare viste, PCA, split, griglie, epoche, semi o metriche alla luce dei
punteggi. L'eccezione max_iter segue solo l'emendamento sopra. Un rapporto
parziale resta parziale: il confronto richiede tutti20 shard con quattro fold,
audit e ricette compatibili. Leggi warning/errori prima delle conclusioni e non
abbassare le guardie. Le60epoche MLP non certificano convergenza.

## 3. Analisi e consegna

Riporta BAcc per dataset e aggregati P1/P2, media delle BAcc dei semi MLP, IC95 e
contrasti appaiati fissati: dec−pool, MLP−logistica, B−R3c e interazione. Includi
convergenza/selezione, varianza PCA, curve train/validation, costi e limiti.
Le sonde storiche pool768 e la nuova logistica PCA128 sono baseline distinte:
interpreta i contrasti dentro la matrice, senza attribuire alla MLP differenze
che dipendono dal preprocessing.

Questa prova usa cache teacher già mediate nel tempo: riguarda classificazione
interna, non inseguimento della posa. pool e dec cambiano anche strato:
dec migliore non dimostra da solo che il problema sia la media dei canali.
Esiti negativi sono limitati da PCA, dose del lettore e unico pretraining.
Nessun esito autorizza R3 completo, modifica del FM o un braccio di nuove loss.

Consegna a Simone: esito sonde, eventuali correzioni motivate, risultati PSD-3,
conclusioni sostenibili, domande aperte, consuntivo e proposta successiva.
Aggiorna entrambe le sezioni di `docs/claude_GPT_shared` e `docs/decisioni.md`;
committa/pusha codice, protocolli e risultati e verifica lo SHA remoto.
Porta il lavoro fino ai risultati quando i controlli ne consentono l'esecuzione.
