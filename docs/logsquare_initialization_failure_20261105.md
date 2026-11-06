# T100-LQ: arresto prima del training e diagnosi proposta

Il job **63532335** è FAILED 1:0 dopo 21 s (05/11, 06:35:40–06:36:01,
orari sacct). **63532370** è CANCELLED, senza avvio. Nessun aggiornamento
di pesi, checkpoint o risultato scientifico: il controller si arresta in
`verify_initialization`, prima di creare output e chiamare torchrun.
Le directory del run e delle due ricevute sono assenti. Il checkout
`b2934b07f296778eefe97d4bdb8ca08f2b4e29be` è ancora pulito; non è stato modificato.

Evidenze raccolte in
`results/infra/logsquare_training_20261104/failure_63532335_20261105/`:
log integrale verificato contro SHA256 remoto
`bf8f591f52f89c530870fadf0d9127cc914b1901b5b62835ee88bdc5eff6ce2a`,
sacct e verifica dei percorsi. Costo allocato: **0,023333 GPUh e 0,186667
coreh**; la continuazione costa zero. Il checkout precedente resta a
0,012222 coreh. Nessun rilancio e nessun monitor attivo.

## Quanto sappiamo e quanto manca

L'errore è `fresh shared initialization differs from historical T100-L`.
Il controller confronta lo SHA esatto dello stato appena costruito, esclusa
la nuova matrice W_log, con `38cb4cce…` del collaudo. L'errore non riporta
l'hash osservato né il gain, quindi **la causa non è ancora dimostrata**.

Il critico ha recuperato un precedente misurato sul cluster:
`results/passo6/diag_initial_state_20261014.json`, job 60561493, documentato
in decisioni e `docs/step1_random_downstream.md`. L'ordine degli import,
anche con un thread, cambiava solo `tokenizer.fe_gain` di pochi ULP.
Una prima correzione limitata ai thread non aveva risolto. Ora il preflight
importa NumPy prima di Torch, il controller Torch prima di NumPy: è una
pista concreta, non una diagnosi già conclusa.

Sul Mac la calibrazione è sensibile ai thread: gain
548,6939187567017 con un thread, 548,6939187567021 con due/quattro;
lo stato senza gain rimane identico. Il valore storico è
548,6939187567023. Sostituirlo nella copia locale NON recupera l'hash
storico: piattaforme/runtime diversi. Questa osservazione non basta a
localizzare il guasto sul cluster. CPU→CUDA da solo non spiega il problema:
l'hash riporta i byte su CPU e il modello non ricalibra durante `.to`.
Nessuna differenza nei sorgenti scientifici fra collaudo 43cad1f e b2934b0.

## Consegna diagnostica da rivedere

Script nuovo `scripts/logsquare_initialization_diagnostic.py`; launcher
omonimo in `scripts/slurm/`. Piano e risorse in
`results/infra/logsquare_initialization_20261105/`, stato **REVIEW_REQUIRED**.
Non cambia alcun sorgente scientifico o controller esistente, né il piano
di training/rilascio originale. Non viene sottomesso in questa consegna.

Verifiche: 56 test CPU passati in 4,55 s, sintassi del launcher valida;
smoke locale con modelli completi in cinque processi nuovi, L/LQ appaiati
in ciascun contesto. CUDA non provata. Review interna favorevole dopo due
correzioni di contesto, in
`docs/reviews/logsquare_initialization_diagnostic_critic_20261105.md`.

Cinque processi Python nuovi, con import effettivi dei moduli originali:

| Contesto | Prima costruzione | Dettaglio |
| --- | --- | --- |
| controller | LQ | Contesto del controllo che è fallito |
| preflight | L | Import del collaudo; intra/inter-op a 1 |
| trainer | LQ | Costruzione equivalente alla parte non-main, senza DDP |
| trainer_rank0 | Conteggio parametri, poi LQ | `count_parameters` costruisce prima un modello non addestrato |
| controller_threads1 | LQ | Stesso controller, intra/inter-op espliciti a 1 |

Il parent importa solo standard library. Prima delle costruzioni non
interroga i dettagli del runtime Torch; le interrogazioni complete vengono
registrate dopo. Per ogni modello: SHA completo/condiviso/per tensore,
gain, RNG, W_log nullo, copia CUDA e verifica dei byte. I pin originali e
tutti i 102 sorgenti sono verificati prima e dopo; nessun dato EMG o
checkpoint caricato. La calibrazione sintetica interna al costruttore è
quella originale, non una calibrazione nuova su dati.

Solo su una **copia diagnostica** dello stato si sostituisce il gain con
quello del report storico. Se questa unica sostituzione recupera lo SHA
esatto, individua una differenza sufficiente a spiegare il controllo;
non corregge il modello e non autorizza ad allentare l'hash. Se non lo
recupera, la differenza non è spiegata dal solo gain. L'originale viene
ricontrollato immutato. Un mismatch è un'osservazione, non un successo
scientifico o una ricevuta di training.

Limite: non si avviano DDP, teacher, optimizer, forward, loader o training.
La diagnostica confronta import/costruzione/trasferimento, non riproduce
l'intero ciclo del preflight o del trainer. Le differenze del secondo
modello in ogni processo restano descrittive. Un risultato ambiguo ferma
la fase; nessun retry cieco o cambio del riferimento storico.

## Risorse e seguito

| Job proposto | Risorse | Tetto | Costo massimo |
| --- | --- | --- | --- |
| LOGSQ-initdiag-checkout-20261105 | 2 CPU, 4G, seriale | 10 min | 0,333333 coreh |
| LOGSQ-initdiag-run-20261105 | 4 GPU, 32 CPU, 240G, boost_qos_dbg | 10 min | 0,666667 GPUh / 5,333333 coreh |

Totale massimo **0,666667 GPUh / 5,666667 coreh**. Quattro GPU preservano
il contesto di allocazione/visibilità del controller fallito; la copia
usa solo cuda:0. Cinque figli limitati a 90 s (450 s), 150 s di margine
per avvio, pin e I/O. Non abbiamo una misura affidabile di durata e non
spacciamo il tetto per una stima. Nessuna risorsa di nuovo training inclusa.

Dopo review Claude: pubblicare stato APPROVED_FOR_DIAGNOSTIC e pin della
review, nuovo SHA/checkout `logsquare-initdiag-20261105`, inviare tramite
submitter/ledger comuni. Output nuovo:
`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/logsquare_initialization_20261105/diagnostic`.
L'autorizzazione operativa di Simone resta valida; manca la review tecnica
di questa nuova consegna, non un nuovo consenso operativo.

Dopo il rapporto, correggere solo la causa provata e far rivedere il diff.
**Correggere soltanto il modello temporaneo del controllo non basta**:
il trainer ne costruisce un altro. Qualunque correzione deve verificare
anche lo stato effettivamente usato prima degli aggiornamenti. Il training
rimane fermo fino a questa verifica e alla release aggiornata; checkout,
log e ID falliti si conservano. Epsilon, architettura, loss, dose e seme
restano quelli approvati.
