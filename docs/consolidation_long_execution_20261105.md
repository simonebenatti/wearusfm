# Valutazione del lungo T100-G — pacchetto per Claude

Stato: **handoff a Claude e uscite su SCRATCH, awaiting_claude_rereview**.
La terza review `24860d7447a8aa77fadf51f7ea62f074408e91e4` converge sul pacchetto
`51b4574` (R-1b chiuso, condizioni operative C-1/C-2). La presente modifica
minima di storage passa dalla quarta lettura prima del checkout.

Simone,05/11 sera: «da ora in poi questa parte la gestisci tu» e «sarai tu a
fare il deployment e a gestirlo». **Claude è l'unico owner operativo di EPN15
e della valutazione lunga**: deployment, invii, stato/log e raccolta. GPT
mantiene codice e responsabilità scientifica, verifica dopo senza bloccare.
Disposizione canonica in main `fe66e2d`, fonte registro Claude `3566c8d`.
Nessun altro job, deploy o controllo remoto da GPT durante questo handoff.
La nuova autorizzazione SCRATCH riguarda il lungo; FT15 in corso resta suWORK.

Consegna operativa in un solo file: [linea2, comandi del lungo e gestioneEPN15](handoff_line2_claude_20261105.md).

## Endpoint verificati

I tre segmenti63525803/63525805/63525809 sono COMPLETED0:0; ultimo termine
05/11 alle13:42:08CET. Tempi7994/7186/7575s, totale25,283333GPUh e
202,266667coreh, contro36/288 massimi. Tutte le ricevute complete riportano
la dose prevista e stopped=max_steps; niente STOP. Hash riletti sulle copie
persistenti in WORK, identici alle ricevute del runtime17dd455.

| Dose | Copia persistente | SHA256 |
|---:|---|---|
|19.618|checkpoint_step0019618.pt|a31c895061c1f273eaac2d796fc00cf76cffa4318b521a20be4d6ba65b63a056|
|39.237|checkpoint_predecay.pt|5369425f92096f5e4763da5137f1983a3809ef15433c4811736ce287a3eaa5e1|
|49.046|checkpoint_postdecay.pt|bfa6141fc2ed6505d2f582a46f0c04f36fd35f867e8cff2b11555b241947a490|

Tutti454.381.728byte. Anche30.000 è archiviato/verificato, ma è soltanto un
confine operativo. Prove in results/infra/consolidation_long_20261105/continuation.
Le estrazioni ricontrollano configurazione intera, dose, schedule, firma dati,
optimizer/RNG/stream, teacher/student finiti; il caricamento teacher deve
ripristinare esattamente tensori e gain. Hash e stato del modello ricontrollati
a fine estrazione. Le copie19k sono legate al persistent_source delle ricevute.

## Matrice scientifica

| Parte | Estrazione / lettore | Confronti |
|---|---|---|
|Classificazione|Tre dosi × pre D, post D, pool storico2D; stesso forward, crop/preprocessing/split storici|Dose49k−19k primario sul pool;39k−19k e49k−39k secondari; post−pre a ogni dose diagnostico|
|Posa ridge|Tre dosi × pre/post, sequenza mean_K FP16, LayerNorm FP32, crop6, CV2fold per utente e griglia già convergente|Dose49k−19k nel pannello post_ridge; pre_ridge secondario; post−pre diagnostico a pari lettore/D|
|Posa testa allenata|Predizione della testa già nel teacher, sul medesimo crop6|Dose49k−19k in un pannello primario separato; non si equipara questa testa alla ridge|

Classificazione: EPN/UCI/DB2/DB3/DB6, tutti i dataset e P1/P2, quattro fold,
C=1e−3…1e2, max_iter12000, scaler sul solo train, selezione sulla validation,
refit sul solo train. Pool2D = post-backbone + locale; non è pre+post.
Le nuove viste pre/post hanno384colonne, il pool768.

Posa: loader originale, scale di sessione, margini, stessi bersagli/maschere e
selezione. Train1907finestre/40utenti/200registrazioni; held18548/15/1950.
Conservato l'asse temporale; nessun preprocessing di classificazione.
E2 sceglie i partner storici sui bersagli completi, poi valuta sul supporto
comune crop6. Questa E2 è esplicitamente diversa dall'E2 storica a supporto
intero; si riportano conteggi/hash del supporto, E_dyn, livello, correlazione,
varianza predetta/target e gap E2. Nessuna scelta globale di un unico lettore
posa: i due pannelli primari sono sempre riportati affiancati. Un eventuale
disaccordo resta tale, senza scegliere il pannello favorevole. Disegno e
lettura approvati dalla review e2753c; con15utenti held gli IC restano grossolani.

Riutilizzo19k/pool: cache e OOF storiche T100-G, con hash e audit verificati;
le nuove estrazioni19k devono rispettare identità degli input/ordine/supporto e
parità del pool entro atol1e−6/rtol1e−5. Il piano sceglie esplicitamente
declared_historical_cache: tolleranza numerica non è identità delle nuove
predizioni di sonda. Il riferimento è il cache/lettore storico nominato.
Vengono anche registrati uguaglianza esatta e scarto di parità.
I13artefatti storici sono stati riletti e verificati il05/11 alle16:11CET.

**R-1b(B), contesto numerico:** estrazioni con1thread; benchmark, fit e confronti
di classificazione e posa con8thread, come i launcher storici della stessa
fase. La sintesi JSON usa1thread entro le2CPU assegnate. OMP/MKL/OpenBLAS
e i pool attivi sono verificati. Le sonde nuove richiedono le stesse versioni
di Python, numpy, scipy, sklearn e lo stesso features_dtype dei report storici
P1/P2 pinnati. Ciascun pool storico deve essere presente, con almeno la
molteplicità storica e gli stessi sei campi (internal_api, prefix, version,
num_threads, architecture, threading_layer). Sono ammessi in più soltanto
runtime OpenMP a8thread, anche per le occorrenze eccedenti di un pool storico,
elencati integralmente nella ricevuta e nel report.
Un solo `numeric_sha256` resta obbligatorio fra tutti i benchmark e fit nuovi.
`exact_match` con lo storico è registrato per ciò che misura: può essere false
pur passando la compatibilità esplicita. Nel processo nuovo è caricato anche
Torch; questa regola non dimostra che le predizioni siano numericamente identiche.
Versioni/pool storici mancanti o modificati, extraBLAS e extraOpenMP con thread
diversi bloccano benchmark/fit/confronto. La posa
verifica gli8thread e il medesimo ambiente fra il proprio benchmark e i fit;
non si dichiara identico lo schema di ambiente della posa a quello delle sonde.
Nessun refit19k/pool aggiuntivo: resta il riferimento storico dichiarato.

Verifica reale prima del checkout: due processi sul login node Leonardo,
moduli profile/deeplrn e cineca-ai/4.3.0, venvWORK e8thread, import numpy/scipy/
sklearn.linear_model con e senza Torch. Senza Torch: tre pool e hash storico
07b776ba…; con Torch: cinque pool, dueOpenMP aggiuntivi (un secondo libgomp e
libomp), entrambi8thread. La regola B li accetta e exact_match è false.
Uscita e comandi in numeric_preflight_20261105.json. Sono soli import di pochi
secondi CPU, non benchmark o prove di equivalenza delle predizioni; le fasi
compute verificano nuovamente il contesto. Nessun checkout/job aggiuntivo.

Lettura V-R1: tutte le differenze, bootstrap appaiato per soggetto2000/seed20261102,
1pp e non-inferiorità−1pp per classificazione,2% relativo di E_dyn per posa,
IC favorevoli secondo regola già concordata. La replica T25-L−B19618 è
riportata con i propri IC, senza trattarla come pavimento universale del rumore.
Un risultato non risolto non implica equivalenza. Dati di sviluppo, seme unico.

## Risorse concrete

| Fase | Job | GPU / CPU / RAM per job | Tempo massimo | Tetto GPUh / coreh |
|---|---:|---|---|---|
|Estrazioni|12|1 / 8 /100GiB|45min|9 /72|
|Benchmark classificazione|15|0 /8 /30GiB|30min|0 /60|
|Benchmark posa|1|0 /8 /30GiB|15min|0 /2|
|Sonde classificazione|48|0 /8 /30GiB|4h|0 /1536|
|Fit/analisi posa|3|0 /8 /30GiB|1h|0 /24|
|Confronti classificazione e posa|2|0 /8 /30GiB|1h|0 /16|
|Sintesi finale|1|0 /2 /2GiB|10min|0 /0,333333|
|Checkout separati per4fasi|4|0 /1richiesta,2prudenziali /2GiB|10min|0 /1,333333|
|Totale|86|82calcolo +4checkout|limiti per job|**9 /1711,666667**|

Questo è un tetto di allocazione, non il consumo atteso. Riferimenti osservati:
estrazioni storiche per dose1729s posa +895sP1 +1688sP2, quindi3,5933GPUh
per tre dosi prima dell'overhead delle nuove viste. Sonde storiche T100-G:
14,28coreh per cella pool a cinque dataset; otto nuove celle darebbero
114,24coreh per estrapolazione, senza garantire gli stessi tempi sulle nuove
viste. Il nuovo benchmark determina il preventivo CPU effettivo prima dei fit.

DB2 è fissato in due shard[0,1]/[2,3] per ciascuna delle8celle nuove;
gli altri32job completano tutti4fold. Quindi40celle logiche e48job fisici.
I tempi storici dimostrano che la policy prudenziale sui4fold supererebbe il
limite13800s; sulle coppie rientra. Entrambe le coppie hanno4h e gate13800s.
Nessuno split automatico scelto dopo i risultati.

Benchmark classificazione: tutti6C sul train completo del fold0, nessuno score
validation/held. Costo misurato più refit al costo del candidato più lento,
scalato per righe/classi dei fold, con pavimento dai costi storici completi,
×3 e ×1,5 per dati/predizione, +600s per job. Memoria RSS×3 entro85% di30GiB.
Tutti48shard devono passare prima di qualsiasi fit classificazione.
La misura e il pavimento storico ora usano entrambi8thread. Manteniamo i
48shard fissati: nessun passaggio facoltativo a72job o aumento del tetto.
Il margine del pavimento è limitato (DB2 prima coppia circa1,25×, EPN1,33×
rispetto ai tempi storici); un gate fallito ferma la fase, non autorizza a
ridurre i margini, cambiare shard o ritoccare il runtime dopo le estrazioni.
È una proiezione empirica, non una garanzia temporale su nuovi checkpoint.

Benchmark posa: solo train19k, statistiche/eigensistemi, fit a alpha fisso,
predizione, E2, sintesi/bootstrap diagnostici e scritture compresse misurati.
Proiezione sul held per numero di finestre, quadratica per pairing E2, margine2
e120s fissi, inclusi artefatti e hash; nessun payload held aperto.
Gate3300s per dose e RSS proiettata entro85% di30GiB. Coefficienti e predizioni
dei fit sono persistenti; un bordo alpha irrisolto non viene nascosto.

## Esecuzione dopo review e pubblicazione

Piano/82ID/output in results/infra/consolidation_long_20261105/launch_plan.json.
Owner operativo Claude, submitter scripts/submit_job.py e ledger comune.
Claude crea il ramo di esecuzione figlio della consegna, pinna la quarta
review e i commit di fase; cambia solo metadata/pin/stati, non i sorgenti.

1. Convergenza reale di Claude; pin del suo commit nel piano e degli SHA
   esecutivi, commit/push verificato. Nuovo checkout immutabile per estrazioni.
2. Controllare le12ricevute complete0:0, parità, dose, supporti, risorse e hash.
   Pubblicare cache_manifest con pin di cache e ricevuta per ciascun artefatto.
3. Nuovo checkout per16benchmark CPU. Pubblicare ricevute, proiezioni e
   ammissione di tutti i48shard classificazione e della posa prima dei fit.
4. Nuovo checkout con pin misurati per48sonde +3fit posa. A fine matrice
   pin dei report/OOF/audit, quarto checkout per i due confronti. Dopo averne
   raccolto i report, pubblicare il piano finale con i due pin. Il jobSUMMARY
   già previsto usa lo stesso quarto checkout e un pianoJSON esterno, copiato
   dal payload pubblicato e verificato per hash in un wrapper operativo.
   Nessun quinto checkout o modifica del quarto; procedura completa nella nota.
5. Registro in entrambe le sezioni, tabella, decisioni, consuntivo SLURM e review
   della lettura. Nessun giudizio finale su una matrice incompleta.

**R-2 / C-2, congelamento e invalidazione:** ogni fase usa esattamente lo
stesso `source_sha256`; i piani aggiornano soltanto pin reali e stati.
Da prima del primo checkout fino alla chiusura della catena, nessun commit
del ramo di esecuzione modifica `src/**/*.py`, `scripts/*.py`,
`scripts/slurm/consolidation_long.sbatch` o `scripts/slurm/lib_code_dir.sh`.
Anche codice destinato a FT15 va sviluppato su un ramo/worktree separato e
non integrato in questo ramo durante la catena. Documentazione, test e
risultati possono essere registrati: non appartengono alla firma dei sorgenti.

È scelta la regola R-2(i), non una nuova compatibilità delle cache. Un cambio
di uno qualsiasi dei sorgenti coperti **invalida tutte le ricevute/cache di
estrazione rispetto al nuovo codice**, anche se la modifica riguarda solo
tempi, thread, shard o lettori. La catena si ferma. Non si riscrivono hash o
ricevute e non si introducono elenchi di sorgenti alternativi accettati.
Ripartire con codice modificato richiederebbe nuova review, piano con nuovi
ID/output e ripetizione delle12estrazioni (fino a9GPUh aggiuntive, più checkout,
benchmark e CPU da preventivare). Questa spesa **non è inclusa** nel tetto
9GPUh/1711,666667coreh corrente e non è autorizzata automaticamente: serve una
nuova decisione di Simone sulla ripetizione e sul preventivo completo.
Nessuna modifica dei checkout già creati o dei run FT15 in corso.
CPU a ondate, al massimo8job outstanding dell'intero utente; verificare la
coda comune prima di ciascun invio per rispettare il limite seriale10.
Le ricevute di prerequisiti già completati valgono anche dopo purge dell'ID
dalla memoria SLURM; non rilanciare un checkout per questo motivo.

**Storage autorizzato da Simone,05/11 sera:** «si puo scrivere sullo scratch
 grande? no li terremo per piu di 40 giorni, cosi nonabbimao problemi di spazio».
Tutti i nuovi artefatti scientifici (cache, OOF, coefficienti, ricevute, report
 e manifest) vanno sotto:
`/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/consolidation_long_20261105`.
È una directory reale, senza symlink; controlli su percorsi freschi, assenza
 di overwrite e identità sono invariati. Le fonti e i checkpoint inWORK non
 si spostano. I log SLURM rimangono nella directoryWORK esistente per consentire
 l'apertura del log prima dell'avvio dello script.

**Retention:** uso previsto inferiore a40giorni. Cache e OOF sono rigenerabili
 dai checkpoint inWORK insieme a dati originali, calibrazione e codice/ricetta
 pinnati; rigenerarle è un nuovo costo, non un retry automatico. Nessuna promessa
 di disponibilità dopo la purge. A fine di ogni fase Claude copia sul Mac
 ricevute, report, manifest, audit e accounting; committa i metadata prima di
 rilasciare la fase successiva, verifica SHA remoto e copie locali. PayloadNPZ/PT
 non entrano inGit; eventuale archivio pesante Mac resta esterno al repository.
 Se la catena rischia di superare40giorni, fermare e dichiarare il problema di
 conservazione prima della scadenza, senza modificare in silenzio la retention.

**C-1 aggiornata:** sono due controlli distinti dell'owner, non del submitter.
Per SCRATCH la stima rimane24GiB: prima estrazione almeno32GiB liberi, poi
8GiB di margine più gli output già ammessi ancora da scrivere sulla stessa
filesystem. Verificare spazio e inode reali e l'eventuale quota personale/
progetto vigente; se esiste un limite usare quello più stretto, non il solo df.
SuWORK restano quattro checkout e log: i blob tracciati a51b4574 occupano
87.146.517byte, quattro copie348.586.068byte (0,325GiB). Riserva log128MiB;
**budget aggiuntivo prudenteWORK1GiB** complessivo per checkout, metadataGit
 e log, da ricontrollare ai commit di fase. Prima degli invii servono8GiB
 di margineWORK +1GiB preventivati +outputWORK già ammessi ancora da scrivere,
 inclusi i checkpoint FT15. Non contare due volte byte già presenti.
Le letture devono essere fresche≤1h e ripetute immediatamente prima del primo
invio di estrazione. Nessun gate dichiarato passato con la vecchia lettura.

Il precedente blocco33,143GiBcontro33,724GiB riguardava24GiBnuovi suWORK;
la decisione di Simone cambia esplicitamente la destinazione, non abbassa
la riserva. Con lo stesso inventario storico FT15 il nuovo requisitoWORK
sarebbe10,724GiB, ma Claude deve verificarlo e verificare SCRATCH al lancio.
Nessun file cancellato/spostato daGPT, nessun cambio ai jobEPN15 esistenti.

TIMEOUT/OOM/errore/pin o gate fallito: conservare evidenze, fermare la fase,
diagnosticare. No requeue, retry ciechi, overwrite, dose/seme/loss aggiunti,
scarto di fold o cambi delle soglie dopo gli score. Nessun monitor automatico.

## Verifiche e limiti della consegna

186test CPU passati nella suite integrata, più4nuovi test del contratto
del piano concreto (190complessivi), incluso il codice già revisionato;
bash−n del launcher passato. Test di selezione, ordine train/held, parità delle
viste, bootstrap/maschere, hash/receipt/audit, lettori generici in D, griglie,
shard e gate globali. Review interna senza blocchi residui: ha fatto includere
il costo di scrittura/hash delle predizioni nel benchmark posa.
Nessuna misura CUDA delle nuove viste e nessun benchmark nuovo eseguiti prima
della review esterna. Gli SHA di checkpoint e degli artefatti storici sono reali;
i pin futuri rimangono null e impediscono l'esecuzione delle fasi successive.
La vista pre della posa è conservata in FP16 e controllata per finitezza;
non è stata aggiunta una misura dell'errore di arrotondamento (nota N-1).
Le verifiche storiche della correzione R-1/R-2 sono riportate separatamente in
results/infra/consolidation_long_20261105/rereview_validation.json; i190test
sopra descrivono la consegna originale, non il conteggio della rilettura.
La correzione R-1b e la verifica reale sono documentate in
results/infra/consolidation_long_20261105/third_review_validation.json.

Le verifiche della consegna SCRATCH/owner/N-4 sono in
results/infra/consolidation_long_20261105/handoff_validation.json.
