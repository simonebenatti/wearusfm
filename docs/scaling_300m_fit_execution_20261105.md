# Fit CPU scaling a5.000: rilascio misurato del5novembre2026

### 05/11/2026 — GPT scaling: UCI accettato, rilascio misurato dei fit

UCI63625951 COMPLETED0:0/29s,8CPU=0,064444coreh; tutti6C convergenti.
Report b97ef2cc, audit, log, sacct-P e accounting pubblicati con hash. Audit
riproducibile e verifica indipendente concordi:15report,90fit completi convergenti;
5fit del vecchio parziale restano storico separato. CPU cumulativa28,358889coreh.
Simone conferma di verificare UCI e avviare i fit senza nuovo consenso.

Piano misurato:14shard, copertura esatta60combinazioni cell/dataset/fold,
8CPU/30GiB, massimo2h58,205,866667coreh richieste (25h44m seriali); con margine
scheduler211,466667. Ridge posa8CPU/30GiB/60min, formula misurata660s.
Consumato più tutti i tetti residui255,758889<=310. Numerica aef, dispatcher
c2bd e criteri invariati. Pin/argv/risorse in cpu_fit_release_20261105.
Prima dei fit: commit/push verificati, deploy solo metadata e checkoutversionato.
Poi ridge e14fit in catena afterok, ordine per walltime crescente; si riempiono
solo slot liberi fino a10sottomessi nella QoSseriale. Prima delle tranche successive
si verificano risultati e accounting dei conclusi. Nessun retry o monitor inbackground.


### 05/11/2026 — GPT scaling: benchmark chiusi; qualità ancora da leggere

Il recupero UCI non introduce errori né nuovi criteri. Tre benchmarkNinaPro con
padreTIMEOUT restano esplicitamente riusati secondo Claude8; nessun costo cancellato.
Il packing è operativo, non selezione dei risultati: fold e griglia restano integri.
25h44m sono walltime richiesto prudenziale, non durata misurata né scadenza garantita;
la coda si aggiunge. Il limite10job richiede più tranche per15job totali (ridge+14fit).
Al primo errore/TIMEOUT si ferma la fase e si raccolgono evidenze; nessun nuovo training.


| ID | Stadio / vista | Dataset e fold | Minuti richiesti |
|---|---|---|---|
|SCALING-pose-ridge-20261105|pose-ridge tutte3|; fold |60|
|SCALING-probe-fit-01-20261105|probe-fit T100-G-300M-natural|ninapro_db2; fold 0|59|
|SCALING-probe-fit-02-20261105|probe-fit T100-G-30M-natural|ninapro_db3,ninapro_db6; fold 0,1,2,3|64|
|SCALING-probe-fit-03-20261105|probe-fit T100-G-300M-matched|epn612,uci_emg; fold 0,1,2,3|69|
|SCALING-probe-fit-04-20261105|probe-fit T100-G-30M-natural|epn612,uci_emg; fold 0,1,2,3|79|
|SCALING-probe-fit-05-20261105|probe-fit T100-G-300M-matched|ninapro_db3,ninapro_db6; fold 0,1,2,3|83|
|SCALING-probe-fit-06-20261105|probe-fit T100-G-300M-natural|epn612,uci_emg; fold 0|102|
|SCALING-probe-fit-07-20261105|probe-fit T100-G-300M-natural|epn612,uci_emg; fold 1|102|
|SCALING-probe-fit-08-20261105|probe-fit T100-G-300M-natural|epn612,uci_emg; fold 2|103|
|SCALING-probe-fit-09-20261105|probe-fit T100-G-300M-natural|epn612,uci_emg; fold 3|103|
|SCALING-probe-fit-10-20261105|probe-fit T100-G-300M-natural|ninapro_db3,ninapro_db6; fold 0,1|139|
|SCALING-probe-fit-11-20261105|probe-fit T100-G-300M-natural|ninapro_db3,ninapro_db6; fold 2,3|141|
|SCALING-probe-fit-12-20261105|probe-fit T100-G-30M-natural|ninapro_db2; fold 0,1,2,3|154|
|SCALING-probe-fit-13-20261105|probe-fit T100-G-300M-matched|ninapro_db2; fold 0,1,2,3|168|
|SCALING-probe-fit-14-20261105|probe-fit T100-G-300M-natural|ninapro_db2; fold 1,2,3|178|

Gli argv in submission_plan.json hanno tutti i pin effettivi, inclusi quelli dei15benchmark. Il checkout del budget conterrà questo piano e gli accounting della posa. Si pubblica separatamente il comando esatto con il commit metadata completo.
La prima dipendenza afterok è sul job ridge ancora presente; il checkout già verificato non viene usato come dipendenza per evitare MinJobAge=30s. Le dipendenze non sostituiscono le verifiche prima delle nuove tranche.

### 05/11/2026 — GPT scaling: checkout budget63626623 accettato

Piano fit ec221a6, comando esatto5e20d8c pubblicati suorigin; deploy metadata
verificato ec221a6, canonical main intatto. Job63626623 COMPLETED0:0/27s,
2CPU=0,015coreh, checkout metadata pulito e hash effettivi uguali ai pin.
Raccolti log originale con hash, sacct-P e accounting. Verificati27input remoti
(piani,15benchmark,report/accounting posa,budget), tutti15output ancora assenti,
runtimeaef pulito e coda seriale vuota. CPU consumata28,373889coreh.
Pronta prima tranche ridge+9fit, entro10sottomessi; altri5fit da inviare solo
quando liberi gli slot e verificati gli esiti precedenti. Tutti gli argv già pubblicati.

### 05/11/2026 — GPT scaling: correzione operativa prima dei fit

Review indipendente ha trovato --out posa trattato come file anziché directory:
corretto nel piano prima del commit ec221a6 e di ogni invio; nessunjob coinvolto.
Rieseguito parsing reale dei15comandi e controllati pin/risorse/copertura. Nessun
blocco residuo al rilascio. Le verifiche live dei lettori restano attive; afterok
arresta i successivi su errore ma non sostituisce la lettura prima delle nuove tranche.

### 05/11/2026 — GPT scaling: fit avviati, prima tranche di10job

Dopo9e1e24e pubblicato e SHAorigin verificato, sottomessi tramite submitter comune
ridge63626871 e9sonde63626931/63626956/63626966/63626971/63626975/63626978/
63627028/63627047/63627052. Ricevute copiate e pin/argv verificati; coda e stati
precedenti controllati prima di ogni invio. Alle21:48:58 ridgeRUNNING dal21:47:31,
9fitPENDINGDependency, catenaafterok corretta. Altri5fit10..14 ancora non inviati
per il cap10sottomessi seriali. Nessuna richiesta nuova di decisione scientifica;
Simone ribadisce di raggruppare le operazioni senza ripetere autorizzazioni.

### 05/11/2026 — GPT scaling: limiti di coda e permessi tecnici

Nessun gate runtime fallito al controllo21:48:58. Restano5sottomissioni, non14fit
giàinviati: il cap10 della QoS comprende la ridge. Nuove tranche solo dopo la
verifica dei conclusi e dei posti liberi. Le finestre di escalation sono il sandbox
dell'app, distinte dal consenso di Simone già acquisito; si raggruppano le operazioni
senza aggirare i controlli e senza attivare monitor inbackground.

### 05/11/2026 — GPT scaling: completamento dei soli cinque invii senza monitor AI

Simone chiarisce che vuole evitare il consumo di token durante l'attesa. I dieci
invii iniziali restano invariati. Preparato finish_submissions.py: avvio unico di
un processo locale indipendente, senza AI, limitato agli esatti cinque argv10..14
già pubblicati. Controlla ogni60s, massimo8h; raccoglie e verifica report/audit/hash/
accounting dei conclusi, rispetta10job seriali/8CPU e310coreh, poi termina appena
sono tutti15sottomessi. STOP senza retry su errore/ambiguità; prenotazione e lock
impediscono doppio avvio. Non osserva il completamento finale della campagna.

42test passati, anche10ricevute/ledger reali controllati. Nuova operazione solo
logistica: runtimeaef, dispatcher c2bd e piano0370cdf invariati. Solver unresolved
resta flag, nessuna nuova soglia; validazione OOF/scientifica sarà eseguita dagli
aggregatori congelati. Fonte e comando esatto in finish_execution.json pubblicati
prima dell'avvio. Mac deve restare non sospeso; schermo spento consentito.

### 05/11/2026 — GPT scaling: attesa senza token, distinta dalla review finale

Il processo locale chiude soltanto le cinque sottomissioni: non modifica budget,
protocollo, modelli o risultati e non riattiva automazioni della chat. LLM non resta
in polling. Un errore SSH ferma il processo senza ritentare invii; si conserva STOP.
Raccolta operativa usa hash OOF remoto senza scaricare dati; contenuto/score OOF
saranno verificati dall'aggregatore già rivisto. Limite8h o Mac sospeso possono
ritardare/interrompere i nuovi invii, senza fermare job già affidati a SLURM.

### 05/11/2026 — GPT scaling: processo locale senza AI avviato e verificato

Fonte ef8db2b suorigin con SHA verificato; review finale30b8b9b9 PASS,42test.
Avvio unico --start alle22:08:47CET, PID2321 verificato; limite8h, nessun restart.
Il processo ha già raccolto e accettato operativamente la ridge63626871:
COMPLETED0:0/289s,0,642222coreh. Report, log, accounting e hash dei6output
raccolti senza scaricare predizioni/coefficenti; nessuna interpretazione scientifica
prima del confronto previsto. CPU cumulativa29,016111coreh.

Snapshot22:08:48: fit01RUNNING,altri8fitPENDING; nella QoS anche checkout LONG63628160,
quindi10slot occupati e5fit ancora da inviare. Il processo li accoderà usando solo
argvpubblicati dopo i controlli; non serve tenere il modello in polling e non si
consumano token AI nell'attesa. Rimane un processo locale bounded, non automazione
Codex né monitor del completamento finale. Occorrono Mac nonsospeso e rete.

### 05/11/2026 — GPT scaling: controlli di processo e limiti della raccolta

--status in sandbox non poteva interrogare PID tramite os.kill(pid,0); la verifica
nel contesto tecnico autorizzato conferma processo presente e nessun risultatoSTOP.
Questo non ha toccato i job né comportato retry. CapQoS include l'altro checkout LONG;
nessun tentativo di superarlo. Stato/log mutabili locali esclusi dal versionamento
continuo; startup e snapshot immutabile pubblicati, ricevute/accettazioni nuove si
raccolgono nella prossima lettura ordinaria. La lettura scientifica della posa e le
verifiche OOF finali restano da completare; nessun verdetto30M/300M anticipato.

### 06/11/2026 — GPT scaling: ReviewClaude9 recepita, tutti i fit accodati

Acquisiti review b66a3e6 e due script di verifica come blob originali; conservate
verbatim quattro voci Claude nelle due sezioni. Verdetto CONFORME, nessun blocco.
Il processo locale ha chiuso all_15_submitted: ultimo invio fit14/63635524 alle
2026-11-06T00:14:56+01:00. Nessun STOP, rifiuto o duplicato. Audit locale indipendente:
15ricevute/ledger/argv corretti,5stdout nuovi riusciti;29hash su posa+fit01..04.
Nessun riavvio o modifica del processo congelato.

Lettura SLURM unica alle08:22:50CET: posa e12/14fit COMPLETED0:0, fit13/63634004
RUNNING dalle07:21:20, fit14/63635524 PENDINGDependency. Nessun job fallito.
CPU-only dei conclusi104,380556coreh; in corso ulteriori8,2coreh osservate,
non ancora consuntivo finale. Raw sacct/accounting in review9_followup_20261106.
I15invii sono completi; la lettura scientifica e aggregati/confronto restano successivi.

### 06/11/2026 — GPT scaling: N-4/N-5/N-6 e coordinamento della seriale

N-1/N-3 chiuse secondoClaude. N-4 è un difetto reale del controllo di transizione:
sacctCOMPLETED/squeueCOMPLETING va trattato come ancora in chiusura in un futuro
riuso, con regressione dedicata. Non osservato in questa esecuzione; non si altera
il file storico30b8b9b9 né si rilancia alcunjob. N-5: nessun rifiuto di questa catena;
il processo è terminato e non prende più slot. Alle08:22 restano2job seriali,
quindi8posti di sottomissione liberi; le8CPU sono ancora tutte occupate dal fit13.
Questa è l'informazione per il coordinamento con linea2, senza cambiare priorità.

N-6: prima dell'avvio esisteva review del critico interno (42test e pin/ricevute),
non reviewClaude; quella diClaude9 è successiva. Coperture e cronologia restano
distinte, nessuna approvazione retroattiva né nuova regola imposta a Simone.
Aggregati/confronto non sono coperti daClaude9 e richiedono i propri piani concreti
con pin dei risultati finali. Score non interpretati in questa verifica; nessuna
estensione scientifica, nuovi training, retry o pollingAI.
