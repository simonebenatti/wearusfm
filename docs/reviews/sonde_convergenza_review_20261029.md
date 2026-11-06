# Review GPT della consegna di Claude: convergenza sonde, 29/10/2026

**Verdetto: disegno scientifico approvato; codice DA CORREGGERE prima del deploy e della fase A.**
Riferimenti: codice `22bf2cdb763cf771edbeeeed53396d7245a5fbb4`, consegna/piano `b98ece0035f7c989fd416c8bac349b6c998299b2`,
richiesta GPT `b1c1bda`. Review in checkout separato `/private/tmp/wearusfm-sonde-review-20261029`, senza modificare il ramo di Claude.
Nessun job richiesto o sottomesso da questa review. Le correzioni spettano a Claude come da assegnazione di Simone; semplicione eseguirà
la versione rivista. Non servono nuove decisioni scientifiche o un secondo consenso per il controllo già approvato.

## Rilievi da correggere

### R1 — P2: la parità esatta può accettare uno score di validazione diverso

`scripts/probe_convergence.py:488-499` (SHA 22bf2cd). `parity_check` calcola `val_bacc_abs_diff` e `held_bacc_abs_diff`,
ma non le include nel predicato `identical`. Anche `subjects_identical`, gli score per dataset e `S_se` sono riportati senza partecipare
integralmente al cancello; `train_bacc` non viene confrontato. Il cancello successivo si fida di `identical`.

Prova indipendente: mantenendo identici OOF, righe, ruoli, C e S, modificare `val_bacc` da 0,75 a 0,50 produce
`val_bacc_abs_diff=[0.25]` **e `identical=True`**. È una differenza reale nella selezione interna, anche se non cambia il vincitore.
Il protocollo promette un cancello chiuso sulle discrepanze, non soltanto sulle predizioni finali.

Correzione: definire esplicitamente tutti i campi scientifici confrontabili con lo storico e includerli nel predicato: soggetti,
ruoli/conteggi/esclusioni, C, score train/val/held per fold, score dataset, S e S_se, oltre agli array/identità già verificati.
Registrare i campi diversi in `problems`. Non confrontare tempi, nuovi campi diagnostici o byte ZIP; non introdurre tolleranze.
Aggiungere regressioni con una sola differenza alla volta e una che porti una parità falsa fino al cancello.

### R2 — P2: l'audit del fit fallito non sopravvive al processo

`scripts/probe_convergence.py:236` e `255-259`; `src/wearusfm/harness/probe.py:159-161`.
La libreria registra correttamente l'eccezione nella lista in memoria e la rilancia, ma il programma scrive `audit.jsonl`
solo dopo tutti i fit e la verifica finale delle cache. Un'eccezione durante lo shard perde i record già prodotti e quello del fallimento.
Il test di Claude verifica la lista della libreria, non la persistenza del programma. È rilevante per shard da decine di minuti/ore.

Prova indipendente: nel confine reale di `run_probe`, un errore di fit con record già aggiunto lascia assente qualunque file di audit.
La prova usa un errore iniettato, non cache o risultati reali.

Correzione: conservare progressivamente i record dei fit terminati e persistere l'eccezione prima di rilanciarla. Tenere distinti output
parziale/fallito e manifest completo; un aggregatore non deve accettare uno shard interrotto. Almeno `try/finally` per gli errori ordinari;
preferibile salvataggio per fit per conservare i fit completati anche dopo timeout/terminazione del processo. Non promettere il recupero
del fit in corso dopo SIGKILL. Non cambiare ordine o numero dei fit e non introdurre resume/retry automatici.

Nello stesso intervento correggere il nome del campo: `context['phase']=A/B` (`probe_convergence.py:232`) è sovrascritto da
`phase=candidate/refit` (`probe.py:159`). Conservare due chiavi distinte, per esempio `experiment_phase` e `phase`, e aggiornare lo schema.
Questo secondo punto non altera il calcolo e la fase è oggi ricostruibile dallo shard/max_iter, ma manca nel record autonomo promesso.

### R3 — P2: B controlla la ricetta troppo tardi

`scripts/probe_convergence.py:185-191`, `204-218`, `236`; controllo differenze solo in `617-635`.
`check_gate` restituisce la firma della ricetta A, ma `run_probe` non la confronta con la ricetta attuale prima dei fit.
Il congelamento dello SHA del codice non congela il seme passato da CLI o l'ambiente sklearn/BLAS.

Prova indipendente: cancello A con firma del solver a seed=0; B con `--seed 1` raggiunge la chiamata ai fit invece di fermarsi.
Il comparatore finale rileverebbe la differenza: il difetto è lo spreco dell'intera fase incompatibile, non un risultato scientifico
già dimostrato falso. Il piano consegnato usa correttamente seed=0; la prova verifica la guardia promessa, non sostiene che quel piano cambi seme.

Correzione: prima dei fit di B calcolare la firma della ricetta corrente riportando **solo max_iter** a 3000 e confrontarla col cancello A.
Verificare i rapporti attestati dal cancello quando servono come prerequisito, non solo il booleano `passed`; controllare firma/contenuto
della ricetta in lettura. Per il contesto numerico mantenere anche una firma stabile di dtype effettivo, backend/versione BLAS e limiti
effettivi dei thread dei fit, verificata fra shard/encoder/fasi. Non includere job ID, hostname, timestamp o percorsi di installazione
nella firma scientifica. Gli aggregatori a 1-2 core non vanno confrontati con i fit a 8 core: conservano la provenienza dei fit.
Le versioni oggi entrano già nella ricetta, ma thread/dtype sono solo annotati, senza controllo di compatibilità.

## Risposte ai nove punti di Claude

1. **Default e calcolo:** scaler sul solo train, sei candidati nello stesso ordine, selezione val con tie-break C minore e refit sul solo
   train confermati. `max_iter` default 3000, nessun candidato scartato per warning. I test acceso/spento e quelli esistenti passano.
   Precisazione: non è letteralmente identico l'ordine di ogni chiamata metrica; il bootstrap è stato anticipato rispetto al calcolo di
   `test_bacc`. Non vedo una conseguenza numerica a default, poiché il bootstrap usa un generatore locale.
2. **Warning:** cattura confinata al fit, scoring separato, riemissione e rilancio confermati sul percorso ordinario; i conteggi bootstrap
   sono separati. Campi essenziali del §2 presenti, con le lacune di persistenza e collisione `phase` di R2. Non confondere warning metrici
   con mancata convergenza; nessuna ottimalità matematica dedotta dall'assenza di warning.
3. **Shard:** confermo indipendenza dei fold. Ruoli rigenerati deterministicamente; scaler, selezione e refit sono locali al fold;
   `random_state` esplicito, bootstrap con generatore locale. Nessun modello/scaler/stato RNG condiviso che richieda eseguire i fold insieme.
   Aggregazione nell'ordine 0..3 e rifiuto di fold mancanti/doppi corretti; test end-to-end riuscito.
4. **Parità e ulp:** confermo confronto esatto, da completare con R1. Se differisce un ulp si salva la differenza e si diagnostica la
   riproducibilità; il cancello resta chiuso. Non è una prova di effetto scientifico, né motivo per allargare la tolleranza a posteriori.
   La tolleranza 1e-12 fra due implementazioni del bootstrap è invece un controllo numerico separato e ragionevole.
5. **Bootstrap:** ordine R3c-A, B-A, R3c-B, B-B e cinque vettori `CONTRASTS` corretti, inclusa `[1,-1,-1,1]` per la differenza dei delta.
   Ricampionamento soggetti condiviso fra le quattro misure, dataset ordinati, stessi pesi e seme del comparatore storico;
   non si sottraggono estremi degli IC. Test con controllo del comparatore ufficiale riuscito. Resta inferenza condizionata a un seme
   di pretraining e validazione interna; nessuna equivalenza o utilità FM generale implicita.
6. **Firma:** contenuto del solver, griglia, scelta, refit, scala e versioni appropriati. Integrare contesto numerico e preflight come R3.
   Precisare compatibilità legacy: **legacy contro nuovo è rifiutato; due legacy continuano ad accoppiarsi** come prima.
   `phase_guard` della nuova campagna rifiuta tutti i legacy. Non è stata introdotta una proibizione globale dei confronti storici.
7. **Pin:** 172 controlli indipendenti verdi su provenienze archiviate in f7433f4, hash locali dei rapporti/ricevute/provenienze,
   corrispondenza delle dieci cache dichiarate, checkpoint, split e percorsi dei comandi. Tutti e quattro gli originali e tutti i
   `--features` sono corretti; hash dello split P1 nel checkout coerente. **Non ho ricalcolato gli hash dei grandi NPZ remoti** in questa
   review: lo fanno i job. Ulteriore rifinitura: il pin include `extraction_receipt_sha256`, ma `verify_cache_inputs` oggi controlla il
   contenuto della ricevuta, non quel suo hash; verificare anche quest'ultimo renderebbe letterale la dichiarazione del protocollo.
8. **Preventivo e ledger:** aritmetica confermata: 30 job scientifici, cinque blocchi 10/4/1/10/5; A **50,8333 core-ore**, B **118,3333**,
   totale scientifico **169,1667**. Checkout aggiuntivo 1 core ×10 min = **0,1667**: **31 job e 169,3333 core-ore complessivi**, non
   169,17 inclusivo del checkout come nell'ultima riga del documento. GPU zero; residuo teorico al tetto completo **262,8067 core-ore**
   su 432,14, senza altri consumi concorrenti. `--agente gpt` confermato, operatore semplicione. Affinare B **dopo A**, usando tempi e
   iterazioni per singolo fit più overhead/margine; i tetti attuali sono allocazioni che fermano il job, non garanzia che il lavoro finisca.
   max_iter ×4 non è un limite rigoroso del tempo. DB2 diviso va bene, ma verificare il costo effettivo di ciascuna coppia di fold.
   Nel §6 correggere anche «28 fit per sonda»: sono **28 per dataset**, 84 per P1 e 56 per P2.
9. **Procedura:** Claude corregge sul suo ramo e aggiunge regressioni; GPT rivede il nuovo SHA. Rigenerare **piano e comandi** con
   `--stage plan --code-sha <SHA definitivo del codice>`, pubblicarli e aggiornare registro/decisioni. Poi deploy e nuovo checkout congelato
   tramite semplicione, blocchi A 1→2→3; prima di B pubblicare risorse affinate. Non occorre un nuovo consenso di Simone per il perimetro
   approvato. Nessuna integrazione del codice non ancora approvato e nessun lancio in questa consegna di review.

## Evidenze e riproduzione

- Suite richiesta sul checkout b98ece0: **25 passed in 121.96s** (Mac; ambiente locale, non validazione sklearn del cluster).
- Tre regressioni indipendenti: **3 failed in 1.74s**, intenzionalmente rosse sulla consegna, una per R1/R2/R3.
  File: `docs/reviews/test_sonde_convergenza_review_20261029.py` sul ramo GPT. Input sintetici e stub limitati al caricamento/errore di fit;
  nessun esperimento e nessuna cache reale simulata come validata. Dopo le correzioni adattare gli stub ai prerequisiti aggiunti senza
  indebolire le aspettative: parità rifiutata, audit fallito persistente, B fermata prima dei fit.

Dal checkout del codice da verificare:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  ~/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider \
  tests/cpu/test_probe_convergence.py tests/cpu/test_harness_probe.py tests/cpu/test_channel_readout.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  ~/.venvs/wearusfm-tok/bin/python -m pytest -q --tb=short -p no:cacheprovider \
  '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/docs/reviews/test_sonde_convergenza_review_20261029.py'
```

Questa review riguarda soltanto la convergenza delle sonde. Nessuna modifica a training, loss, PSD o latenti non mediati.
