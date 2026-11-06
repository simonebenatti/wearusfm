# Verifica indipendente del budget di recupero scaling — 05/11/2026

Review locale, in sola lettura. Nessun SSH, job, score o modifica del repository. Letti review Claude7 e script dal commit `1c21047c77f956a1819fe09e8520d3a36dd7a4ae`, i benchmark già raccolti, lo stato e il freeze S8. Eseguito lo script di Claude direttamente dal contenuto Git: riprodotti148,134869coreh sui13report; scenari EPN; minimo16/12/10 shard ai limiti2/3/4ore nello scenario inferiore dichiarato. I559s di C100 restano inferenza del revisore dai tempi dei file, non misura nel report conclusivo.

## Esito

**Proposta310core-ore CPU-only per completare5000 coerente con il conto. Proposta350core-ore CPU-only fresh per la possibile19618 coerente con i tetti sotto.** Questi sono inviluppi di allocazione, non consumi attesi né certificati di fattibilità del fit: i due benchmark mancanti e quelli effettivi19618 devono ancora passare i gate. Nessun nuovo training o seguito19618 viene autorizzato da questo memo.

## Campagna5000: conto esatto

| Voce | Walltime richiesto / regola | Core-ore comprese di180s per job futuro |
|---|---|---:|
| Già consumato, inclusi checkout e TIMEOUT | Da accounting originale |22,027777778|
| Due benchmark di recupero | EPN120min + UCI10min,8CPU |18,133333333|
| Fit sonde | Somma richieste≤240coreh,≤15job,8CPU |246,000000000|
| Ridge posa |1job,8CPU,60min |8,400000000|
| Sei aggregati |6job,2CPU,10min |2,600000000|
| Confronto |1job,8CPU,30min |4,400000000|
| Due checkout documentali residui |2job,2CPU,5min |0,533333333|
| **Massimo pianificato con tolleranza scheduler** | |**302,094444444**|
| Riserva entro310 | |**7,905555556**|

La somma è302,094444, non302,194444. Il costo passato non va ricalcolato come richieste più180: usare il consumo effettivo22,027778, già comprensivo dei184s in più del TIMEOUT. La tolleranza180s è invece una riserva prospettica per ogni job futuro. Come dimostrato dal job fallito, possono esserci ulteriori pochi secondi di terminazione: non chiamare302,094444 un limite assoluto del sistema. Il tetto310 include margine per tale contabilità; l'accounting usa sempre tutti i secondi realmente osservati.

Dopo il recupero il cumulativo con due soli benchmark nuovi arriva a40,161111coreh con la tolleranza, prima di eventuali checkout nuovi: non39,4, che deriva dalle sole richieste. Il cap nominale precedente29,5 non è una prenotazione ancora viva: F5 si chiude mantenendo separati speso reale, futuri cap richiesti, riserva overtime, riserva non impegnata.

Regola fit prima di EPN: `max_shards=15`, `max_seconds_each=10800`,8CPU/30GiB, `sum(cpus*requested_seconds)/3600<=240`. La riserva di6coreh è15×8×180/3600, da sommare una sola volta al cap richiesto, non al tempo della formula o a ogni C. Il piano misurato deve comunque coprire esattamente3viste×5dataset×4fold e rispettare la formula originale con arrotondamento dichiarato. I15job non si possono inviare tutti insieme: rispettare il limite10submitted dell'utente, comprendendo gli eventuali job delle altre linee, e8CPU totali.

Nello scenario inferiore Claude3h richiede12shard; negli scenari più lenti EPN può richiedere4shard invece2, portando14. Questo motiva15/3h/240 come tetti ragionevoli, non dimostra a priori che EPN/UCI misurati vi rientreranno. Se non rientrano: stop prima dei fit, nessuna riduzione C/fold/max_iter/tolleranza e nessun aumento automatico. La riserva7,905556 non è spendibile per aggirare240,15o10800, né autorizza retry. Ogni nuovo errore/TIMEOUT applica lo stop concordato.

## S8: inviluppo fresh350, distinto dal recupero5000

Mantenere stesso limite fit15/3h/240richieste+6overtime, ma prendere nuovi benchmark train-only sulle cache reali19618. Non riusare i tempi5000 come misure19618 e non contare di nuovo22,027778 del recupero. Non ipotizzare risparmio di sonde del controllo prima di una validazione del riuso.

Tetti proposti per15benchmark, un processo/job per dataset, da fissare ora prima del nuovo EPN5000:

| Vista | DB2 | DB3 | DB6 | EPN612 | UCI | Minuti richiesti |
|---|---:|---:|---:|---:|---:|---:|
|30M naturale|45|20|20|30|10|125|
|300M matched|45|20|20|30|10|125|
|300M naturale|60|30|60|120|10|280|
|**Totale**||||||**530**|

Sono cap tecnici, non previsioni garantite per19618. Evidenza5000: i cap più stretti di20min coprono DB3/DB6 dei768dim (180–304s), quelli di30min coprono EPN768(495–525s), quelli di60min coprono DB2/DB6naturale(1554–1630s), EPNnaturale120min è il recupero rivisto. Il nuovo checkpoint può richiedere più iterazioni: vale stop, non un rialzo dopo gli score.

Benchmark sonde:8×530/60=70,666667coreh richieste;15×8×180/3600=6coreh overtime; totale76,666667.

| Voce fresh19618 | Core-ore con180s per job |
|---|---:|
| Tre checkout documentali,2CPU,5min ciascuno|0,800000000|
| Validazione posa,2CPU,30min|1,100000000|
| Tre benchmark posa,8CPU,10min ciascuno|5,200000000|
| Quindici benchmark sonde,8CPU,cap sopra|76,666666667|
| Fit sonde,≤15shard/3h/240richieste|246,000000000|
| Ridge posa,8CPU,60min|8,400000000|
| Sei aggregati,2CPU,10min|2,600000000|
| Confronto,8CPU,30min|4,400000000|
| **Totale pianificato**|**345,166666667**|
| **Riserva entro350**|**4,833333333**|

Il dispatcher5000 per recupero non deve già supportare19618. Il freeze futuro può pubblicare questi limiti senza un launcher eseguibile e conservare `continuation_authorized=false`, `execution_ready=false`, review obbligatoria dei futuri wrapper e dei pin reali. I riferimenti180/24/136 vanno sostituiti o marcati chiaramente superati; non lasciare due tabelle entrambe presentate come attive. Conservare storico/digestv1 e identificare la revisione tecnica prima del primo fit5000, senza retrodatazione.

## QoS e contabilità GPU del freeze

La documentazione locale v1 è errata nel sostenere che presenza di bprod in associazione e MaxWall bastino: mancaMinTRES. La review7 attesta bprod minimo65nodi/DenyOnLimit. Root comunica nuova lettura live con `normal` ammessa, nessunMinTRES/MaxWall, partizioneboost_usr_prod24h; è corretto predisporre il futuro piano con `--qos=normal`, mantenendo verifica prima di un ipotetico invio, una volta pubblicato il raw e il suo hash. Questo memo non ha interrogato Leonardo: non presentare la verifica live come mia.

La formula GPU nominale resta53.100s/59GPUh training +7GPUh estrazioni =66GPUh; CPU nominali associate528coreh. Con fresh350, il totale nominale diCPU è878coreh, non708. Va distinta questa contabilità per walltime richiesto dalla riserva prospettica overtimeCPU-only appena introdotta. Se si desidera un tetto assoluto anche per il costo dei jobGPU, verificare l'OverTimeLimit effettivo della loro partizione/QoS e indicare separatamente il margine di terminazione; non assumere che66 nominale includa già eventuale overtime. Nessun nuovo tempo utile di training è autorizzato per effetto della riserva.

## B1 e autorizzazioni

B1 è una richiesta di decisione registrata sui nuovi numeri, non un'autorizzazione a ignorare le istruzioni precedenti di Simone. Nel contesto esistono «puoi anche sforare il budget, non mi importa, ti autorizzo», autorizzazione alle sottomissioni e regola di non chiedere il via due volte. Il root deve registrarne testo e perimetro come base dell'emendamento310, distinguendo **autorizzazione generale già espressa** da **numero310 fissato ora da GPT**: non scrivere retroattivamente che Simone avesse pronunciato o approvato proprio310. La richiesta più recente è ripartire dopo convergenza Claude; il dispatcher nuovo deve comunque essere rivisto prima dell'uso. La review7 non lo copre ancora.

La frase storica di sforamento proviene dalla campagna tokenizer: se il root ritiene che il suo perimetro non copra chiaramente l'attuale inviluppo scaling, risolvere soltanto quella concreta ambiguità dopo avere pubblicato proposta completa e review, senza ripetere consensi già validi per letture, commit/push o recuperoCPU entro il vecchio totale. Per19618 non invocare le autorizzazioni generali come rilascio scientifico: il freeze resta condizionale e la decisione segueS8. Nessuna nuova richiesta di consenso è stata inviata da questo revisore.

## Aggiunta: verifica indipendente del dispatcher B2

Letti il diff completo tra launcher congelato e `scripts/slurm/scaling_readers_recovery.sbatch`, il launcher intero e i nuovi test. **Nessun problema bloccante trovato**: cambia soltanto confezionamento del benchmark, nuovi cap operativi e identità di log; il pin aef665ba è ora imposto esplicitamente. Il recupero è limitato300Mnaturale e a un solo EPN612/UCI per invio, con walltime esatto7200/600; fit15shard,10800s,240coreh nominali; copertura completa60combinazioni e guardie runtime/output/scheduler restano rigorose. Posa/aggregati/confronto e bootstrap numerico sono byte-identici al precedente fuori dalla allowlist.

I test verificano byte per byte la allowlist dal blob Git congelato, comandi dei figli, cap esatti e rifiuti oltre limite, identità del codice/plan, singolo figlio senza retry e gli altri stadi invariati. Non ho duplicato la riesecuzione42test già effettuata dall'autore; questa è review indipendente della lettura, internaGPT, non reviewClaude. Nessuna modifica ai file.

SHA256 launcher: `c2bd7979699b2d70150cdfad9d2a6eca657615551ba5073630417c8c46cf14d5`. Test: `5fba59bff8d89460faead328925a48cd2473f10039f094da1edab307e07893ab`.

Sul chiarimento successivo del root: distinguere66GPUh richieste da0,35GPUh di riserva di terminazione, e528CPUh richieste dai2,8CPUh associati alla medesima riserva, è aritmeticamente corretto. Il totale prudenziale con350CPU-only è880,8CPUh, mentre le richiesteGPU restano66 e il training utile53.100s. L'OverTimeLimit resta margine prudenziale, non limite assoluto di accounting. Il raw dello scheduler attuale e la rettifica QoS devono essere pubblicati dal root; questo memo non presenta letture remote proprie.

## Verifica finale dei documenti e JSON concreti

Letti `cpu_recovery_20261105/resource_policy.json`, `submission_plan.json`, `accepted_benchmarks.json`, `docs/scaling_300m_cpu_recovery_execution_20261105.md`, il JSON e documento S8v2. Verifica read-only conclusa: **coerenti, nessun rilievo tecnico bloccante alla consegna per reviewClaude**. B1 resta deliberatamente pendente, insieme alla review del dispatcher; nessun invio è rappresentato come eseguito o già rilasciato.

- Il conto310 coincide con302,094444+7,905556; fit240richieste+6overtime,15shard,10800s. Il conto350 coincide con345,166667+4,833333. Le formule e la futura decisione misurata dopo15report sono esplicite: non esiste un budgetfit empirico fittizio nei metadati.
- I soli benchmark sonde, già consumati21,768889 più recupero18,133333, raggiungono **39,902222coreh** nel caso prudenziale; l'intera CPU-only già spesa22,027778 più recupero raggiunge **40,161111coreh**, ancora sotto180. Questi numeri sono diversi perché il secondo comprende anche checkout/validazione/benchmarkposa pregressi. B1 impone decisione310 prima del primo fit, non dei due recuperi entro180. La documentazione mantiene tale separazione e i campi di decisione restanofalse.
- F1: verificati65pin locali ricorrenti sui13report (report, accounting, raw, ricevuta submitter, log), le righecomplete esatte nei log e i pathremoti correlati. Per i tre NinaPro300M resta parentTIMEOUT; righecomplete precedono la cancellazione nel log, `parent_success_required=false` e `timeout_parent_not_promoted_to_success=true` soltanto per quei tre. Non sono inventati hash per EPN/UCI mancanti; `matrix_complete=false`.
- I due argv sono coerenti con dispatcherhashc2bd7979…, piano d2cc7da1…, CODE_DIR/EXPECTED_SHAaef invariati,8CPU/30G,7200/600s. ID e directory nuovi, expected_report corretto; EPN prima e UCI soltanto dopo accettazione, nessuna dipendenza automatica o fit prematuro. Nessun nuovo checkout richiesto per i benchmark via stdin.
- S8v2 mantiene scientific policy immutata, continuation_authorized=false e vincola nuovi wrapper e benchmark19618. QoSnormal e nuoveevidenzescheduler sostituiscono correttamente bprod. Il JSONv1 è stato confrontato byte per byte con76a3126: invariato e pinnato da `supersedes`. Fonte vecchioenvelope conservata soltanto come storico, non tetto attivo. GPU66richieste/66,35conmargine eCPU878/880,8 sono distinti coerentemente.
- Il root deve ancora pubblicare e verificare loSHA; questa review interna non certifica il futuro commit né sostituisce Claude. Nessuna cache è stata riletta suLeonardo da me, nessuna funzione di scoring invocata, nessun file repo modificato.
