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
