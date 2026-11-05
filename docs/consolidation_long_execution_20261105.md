# Valutazione del lungo T100-G — pacchetto per Claude

Stato: **implementato e testato, in attesa della review esterna di Claude**.
Incarico di Simone tramite coordinamento: completare questa valutazione, farla
revisionare a Claude, poi lanciarla appena convergente. Il lancio della catena è
già autorizzato; questa consegna al revisore non chiede un secondo consenso.
La matrice FT15 è una campagna separata.

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
posa: le proposte di riferimento restano per pannello, da convergere con Claude.

Riutilizzo19k/pool: cache e OOF storiche T100-G, con hash e audit verificati;
le nuove estrazioni19k devono rispettare identità degli input/ordine/supporto e
parità del pool entro atol1e−6/rtol1e−5. Il piano sceglie esplicitamente
declared_historical_cache: tolleranza numerica non è identità delle nuove
predizioni di sonda. Il riferimento è il cache/lettore storico nominato.
Vengono anche registrati uguaglianza esatta e scarto di parità.
I13artefatti storici sono stati riletti e verificati il05/11 alle16:11CET.

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
È una proiezione empirica, non una garanzia temporale su nuovi checkpoint.

Benchmark posa: solo train19k, statistiche/eigensistemi, fit a alpha fisso,
predizione, E2, sintesi/bootstrap diagnostici e scritture compresse misurati.
Proiezione sul held per numero di finestre, quadratica per pairing E2, margine2
e120s fissi, inclusi artefatti e hash; nessun payload held aperto.
Gate3300s per dose e RSS proiettata entro85% di30GiB. Coefficienti e predizioni
dei fit sono persistenti; un bordo alpha irrisolto non viene nascosto.

## Esecuzione dopo review e pubblicazione

Piano/82ID/output in results/infra/consolidation_long_20261105/launch_plan.json.
Un solo owner, submitter scripts/submit_job.py e ledger comune.

1. Convergenza reale di Claude; pin del suo commit nel piano e degli SHA
   esecutivi, commit/push verificato. Nuovo checkout immutabile per estrazioni.
2. Controllare le12ricevute complete0:0, parità, dose, supporti, risorse e hash.
   Pubblicare cache_manifest con pin di cache e ricevuta per ciascun artefatto.
3. Nuovo checkout per16benchmark CPU. Pubblicare ricevute, proiezioni e
   ammissione di tutti i48shard classificazione e della posa prima dei fit.
4. Nuovo checkout con pin misurati per48sonde +3fit posa. A fine matrice
   pin dei report/OOF/audit, nuovo checkout per confronti e sintesi.
5. Registro in entrambe le sezioni, tabella, decisioni, consuntivo SLURM e review
   della lettura. Nessun giudizio finale su una matrice incompleta.

Ogni fase usa lo stesso source_sha256 scientifico; i piani aggiornano soltanto
pin reali e stati. Cambi del codice richiedono review pertinente e benchmark
nuovo se cambia la fonte della misura. Nessuna modifica di checkout in uso.
CPU a ondate, al massimo8job outstanding dell'intero utente; verificare la
coda comune prima di ciascun invio per rispettare il limite seriale10.
Le ricevute di prerequisiti già completati valgono anche dopo purge dell'ID
dalla memoria SLURM; non rilanciare un checkout per questo motivo.

Artefatti stimati24GiB; prima estrazione richiede32GiB liberi in WORK.
Prima di ogni invio: quota di progetto fresca≤1h, almeno8GiB di margine più
la stima conservativa degli output ammessi ancora da scrivere; tenere conto
dei job pending/running senza contare due volte byte già prodotti.
È quota condivisa, nessuna prenotazione implicita. Nessun dato su SCRATCH come
unica copia scientifica nuova; nessun raw/NPZ/checkpoint in Git.

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
