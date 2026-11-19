# Linea1: pacchetto operativo300M a49.046

19/11/2026. Operatore unico: chat **1-30/300 launcher**, ramo
`codex/scaling-49046-launcher-20261118`. Base scientifica218afe13; consegna
scientifica f511ce90. Il confronto19k64785568 è chiuso. Questa consegna prepara
il seguito già deciso; **nessun job49k e nessun deploy eseguito**. Rilascio
subordinato alla review tecnica unica della chat scientifica, poi esecuzione
senza nuovo consenso di Simone e senza review per singolo invio. La review interna
non sostituisce quella consegna.

## Ricetta, checkpoint e verifiche reali

Nessuna modifica ai byte di `src/`, `train_jepa.py`, `training/run.py`, target,
loss o schedule. T100-G storico: tokenizer100ms/GELU/stride25ms, targetRMS25ms
e spettro rapido50ms. Nessun merge dei nuovi target da main.

- 300M19.618 accettato: job63979977, SHA256
  `d7ebe9d87a08882ae0c1c836b62988ea21c7f1246998a5f82613878a0499387e`,
  4.673.783.740byte. Inventario12file e ricevuta accettata pinnati nel piano.
- 30M49.046: checkpoint persistenteWORK del job63525809, SHA256
  `bfa6141fc2ed6505d2f582a46f0c04f36fd35f867e8cff2b11555b241947a490`,
  454.381.728byte. Il19/11 è stato davvero eseguito `checkpoint_identity`
  su questi byte contro configurazione storica49k, firma dati, optimizer/RNG e
  statistiche di posa. Esito positivo, trained-state SHA
  `f69ac99800ba6ac071e508751a2411b62b29f78245642c8b9f0cabadf88e2cc4`.
  Comando e stdout in `launcher/control_checkpoint_actual_verification.json`.
- La prima acquisizione Mac ha verificato entrambi i byte stream dei checkpoint,
  poi è fallita nella fase distinta dei metadati (limite2MB su un JSON da3,2MB).
  Non presentiamo quel primo script come una raccolta completa: limite e
  compatibilitàPython3.6 sono corretti nella raccolta successiva. Il verbale
  `checkpoint_stream_verification.json` distingue il superamento delle due
  asserzioni dal digest non stampato di quella prima esecuzione.

## Training in due nuove directory

| Segmento | Update | GPU / CPU / RAM | Walltime | Timeout trainer | GPUh richieste |
|---|---:|---|---|---|---:|
| 19.618→39.237 | 19.619 | 4 /32 /480G |19h45 |19h15 |79 |
| 39.237→49.046 |9.809 |4 /32 /480G |10h |9h30 |40 |

Un nodo, boost_usr_prod, QoSnormal, batchglobale32, accum1. Entrambi i timeout
lasciano30min al wrapper. Atteso dai tempi precedenti circa96,1GPUh;
prenotazione119GPUh,119,4 col margine scheduler180s/job; CPU allegate952coreh,
955,2 col margine. Partizione riletta19/11:UP, max24h; QoSnormal senza limiti
più restrittivi nei campi interrogati. OverTimeLimit globale3min; è un margine
prudente, non una garanzia assoluta di kill al secondo.

`scaling_continue49046.py` importa torch prima di NumPy. Copia verificata di
checkpoint.pt e tre identità, sorgente in sola lettura, nuovo run, nessun restart
optimizer/EMA/RNG/stream/gain/allarmi. Piani canonici in
`evaluation/training/`; mirror nel launcher obbligatoriamente byte-identici.
Entrambe le copie sono attestate, e i costi sono ricalcolati dai job effettivi.

Conservati total_steps49046, warmup1962, decay_fraction0,2 e keep_checkpoints.
Il39.237 è il nativo `checkpoint_predecay.pt`; update39.238 usa ancora LRpicco
per l'indicizzazione originale. Finale `checkpoint_postdecay.pt`. Metriche
attese esattamente per gli update del segmento e pannelli alle sole cadenze
native; nessun pannello aggiunto artificialmente all'endpoint.

Il segmento2 **non è accodato in anticipo**: viene inviato dopo COMPLETED0:0 e
raccolta verificata del segmento1. A runtime verifica anche jobID del predecessore,
ricevuta, piano, checkpoint e inventario effettivi. La variabile stdout di sacct
è separata dal piano: regressione specifica della correzione inclusa.

Gate memoria: allocato≤90%dellaGPU per ciascun rank; riservato registrato
separatamente. Riferimento37,7518GiB, non si usa il riservato come allocato.
La bit-identità della ripresa attraverso il decay è provata su un piccolo
trainingCPU reale, non misurata suGPU.

## Archivio e retention

Minimo20GiB effettiviSCRATCH prima del segmento; gateWORK30GiB prima di deploy e
stadi. Quota progetto20147512 riletta19/11,45,9485GiB liberi, non `df` della mount.
Il gate legge di nuovo la quota reale prima dell'uso; una riduzione fa fermare
lo stadio. Evidenza grezza in `launcher/live_limits_20261119.json`.

I file del run sorgente hanno date12/11, scadenzaSCRATCH40giorni22/12. Orizzonte
congelato19/12/2026UTC: il wrapper controlla la retention dei pinSCRATCH letti;
il controller rifiuta l'uso oltre quell'orizzonte. Nessuna promessa di permanenza
indefinita di cache/log suSCRATCH. I materiali della catena19k restano nei loro
percorsi congelati, e le prove/risultati piccoli sono già pubblicati nel repo.

Dopo il training, **prima delle estrazioni**, jobCPU di archivio:2CPU,30G,1h,
2coreh richieste,2,1 col margine. Copia19k/predecay/finale in nuova directoryWORK,
~13,06GiB totali, con SHA/dimensioni prima/dopo, fsync e sorgenti immutabili.
I checkpoint futuri sono legati alle ricevute dei due piani congelati; nessun pin
fittizio oggi. Ricevuta, hash delle tre copie, sacct grezzo e log raccolti.
WORK non subisce il purge automaticoSCRATCH; nessuna cancellazione prevista.

## Perché sei estrazioni, compreso il controllo

Inventariate cache eOOF49k della linea2: i manifest, piani e ricevute originali
sono in `launcher/control_line2_evidence/` e `metadata_snapshot.json`. La posa
linea2 usa schema pre/post e ricevute consolidation-long; classificazione eOOF
hanno provenienze e contesto numerico diversi dalla catena scaling. Non sono
input direttamente certificati dai validatori scaling. Un adattatore potrebbe
consentirne il riuso, ma non è implementato né provato in questo pacchetto.
Non affermiamo che quelle feature siano scientificamente invalide.

**Scelta del pacchetto:** estrarre entrambe le taglie dai checkpoint esistenti con
la stessa pipeline scaling e rifare i lettori nella stessa procedura. Nessun
nuovo training30M. Costo aggiuntivo massimo del controllo:7GPUh richieste.
Questa scelta è esplicita nella review del pacchetto, non un riuso silenzioso
né una correzione a posteriori. Posa/P1/P2 per ogni taglia:150/120/150min,
1GPU,8CPU,100G. Totale14GPUh richieste,14,3 col margine; CPU allegate112coreh,
114,4 col margine. La proiezione896→384 non richiede un'altra estrazione.

P1 usa la finalizzazioneP1-r2 già approvata, collegata nel percorso reale
`execute`, con logJSONL completo e validazione del contenuto. Hash/check sugli
input restano stretti. Pose/P2 mantengono la guardia stretta precedente; nessun
allentamento generale di `file_identity` o `check_unchanged` dei lettori.
I sei job raccolgono ricevute, provenienze, log di estrazione e log della guardia.

## Catena CPU e budget separato

La vecchia catena19k chiude94,732778coreh sul proprio tetto351. Nessun residuo
riutilizzato. Nuovo tettoCPU-only400coreh, nessunGPU nei lettori.

| Stadio | Job massimi | Coreh con margine |
|---|---:|---:|
| Cinque checkout2CPU/5min |5 |1,333333 |
| Archivio2CPU/1h |1 |2,1 |
| Validazione posa2CPU/30min |1 |1,1 |
| Benchmark posa8CPU/10min |3 |5,2 |
| Benchmark classificazione train-only |15 |76,666667 |
| Ridge posa8CPU/1h |1 |8,4 |
| Fit classificazione8CPU |≤15 |≤246 |
| Sei aggregati2CPU/10min |6 |2,6 |
| Confronto8CPU/30min |1 |4,4 |
| **Massimo complessivo** | |**347,8 /400** |

Riserva52,2coreh non estende i tetti dei singoli stadi e non autorizza retry.
I fit usano la formula train-only e packing esatto già rivisti: max3h/shard,
max15shard,240coreh richieste,30G. Se benchmark/RSS/tempo non rientrano, stop
prima del fit. Nessuna nuova scelta di C/alpha sul held. Sono pareggiate le
procedure, non i valori selezionati. BudgetGPU totale richiesto133h;
133,7 con margine. CPU allegate aGPU1069,6coreh con margine, separate dalle
347,8CPU-only: non sono costi intercambiabili.

## Ordine operativo e cancelli

L'operatore esegue comandi una riga alla volta; non esiste runner automatico.
Generazione di righe non equivale a invio. Solo `scripts/submit_job.py`, agenteGPT,
campagna `scaling-300m-20261118`, ledger comune. ID prefissatiSCALING-49046.
Nessun invio di altre chat, nessun monitor/automazione nuovo.

1. Review favorevole del pacchetto pubblicato, pinSHA remoto verificato.
2. `gate checkout-code --sha CODE_SHA --reviewed REVIEWED_SHA`, quindi
   `lines checkout-code --sha CODE_SHA`: deployref nuovo e checkout congelato.
   Raccogli `collect checkout-code`, verifica COMPLETED0:0 e sacct.
3. `gate train-predecay`→`lines train-predecay`; raccogli lo stadio concluso.
4. `gate train-final`→`lines train-final`; raccogli lo stadio concluso.
5. `gate archive`→`lines archive`; raccogli e verifica le tre copie persistenti.
6. `extraction-plan`: costruito solo dalla ricevuta finale raccolta, piano originale
   del controllo e pin effettivi. Commit/push, checkout-extraction documentale,
   raccolta. `gate extractions`→`lines extractions`: sei estrazioni indipendenti.
7. Raccolta `extractions`; `reader-plans`, commit/push, checkout-plans e raccolta.
8. Validazione posa prima dei benchmark posa;15benchmark classificazione.
   Per ogni stadio: gate→lines, attendi/raccogli l'esito prima dei dipendenti.
9. `budgets` dai benchmark reali; commit/push, checkout-budgets e raccolta;
   ridge e tranche di fit ammesse. Max10job nella QoSseriale, max4di linea.
10. Raccogli tutti i fit; prima degli aggregati verifica420fit e convergenza.
    Sei aggregati, raccolta; `comparison-inputs` dai soli artefatti accettati;
    commit/push, checkout-inputs, raccolta; confronto finale e raccolta.

Per ogni checkout successivo il commit può solo aggiungere documenti/evidenze di
questa catena o aggiornare docs. I byte di codice non cambiano dopo la review.
Ogni gate controlla HEAD/albero remoto, byte documenti nel checkout che li contiene,
pin nei commit, sorgenti operative locali attestate al checkoutCODE, prerequisiti
raccolti e sacct fresco, output assenti, ledger, postiQoS e quota. La copia del
piano di estrazione vive nel checkout-extraction, e il launcher riceve quel
percorso assoluto e il suo hash, non un percorso inesistente nel checkoutCODE.

`lines` richiede l'ultimo verbale superato, età≤1800s, ledger/documenti/prove
immutati e stessa tranche; `--max` può solo accorciare. La coda illeggibile,
anche con stdout vuoto, non è una coda vuota. Un gate fallito non emette righe.
Primo errore reale, TIMEOUT o incongruenza: stop e diagnosi, nessun retry cieco.
Solo attesa di job già accettati significa attendere, non risottomettere.

## Lettura e consegna finale

Primari e criteri scientifici218afe13 invariati: EPN612 ed E_dyn naturali,
bootstrap soggetti2000/seme20261102, P1/P2 con margine1pp, proiezione fissa
secondaria. Single seed e utenti di sviluppo. Nessuna selezione del checkpoint
migliore, nuovo seme, loss, fine-tuning o prosecuzione oltre49.046.

`dose_interaction.json` è **sempre descrittivo in questa consegna**: riporta le
traiettorie e la differenza dei delta, `comparable=False`, nessunIC d'interazione.
Non è implementato un certificatore di identità inter-dose di righe/target/fold/
ricetta/runtime: conteggi uguali o somme di varianza uguali non bastano.
Questa è l'alternativa prudente già consentita dal piano scientifico; non blocca
i primari validi e non sottrae IC separati.

Raccogli report, ricevute, sacct grezzo affiancato all'accounting, consuntivi, log,
verifica proiezione/primari e aggiorna tabella, decisioni e registro in entrambe
le sezioni, con commit/push e SHA remoto verificato. La lettura finale torna alla
chat scientifica. Nessuna estensione implicita del perimetro.

## Riproduzione locale e file

- `scripts/scaling_continue49046.py` e `slurm/scaling_continue49046.sbatch`.
- `scripts/scaling_plans49046.py`, `scaling_operations49046.py`,
  `scaling_terminal_builders.py`: costruttori, raccolta, gate e righe.
- `scripts/scaling_derive49046.py`: derivazione deterministica con hash delle
  cinque sorgenti accettate19k, controllo originale30M e finalizzazioneP1.
- `scripts/scaling_preserve49046.py` e relativo launcher.
- Piani e preventivi in `results/infra/scaling_49046_20261118/launcher/`;
  documenti canonici iniziali in `evaluation/training/`.
- `launcher/verify_release.py`: suiteCPU mirata nuova e guardiaoutput, stdout completo,
  exitcode, tempi e hash dei sorgenti in `verification.json`. Non invia nulla.
- Test specifici `tests/cpu/test_scaling_*49046*.py`: ripresaCPU reale attraverso
  decay, predecessore/scheduler, piano generato, gate→lines, percorso checkout→
  raccolta→piano estrazioni, entrambi i modelli nel launcher reale simulato,
  mirror mutato, P1-r2 e secondario descrittivo.

Comando: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src
~/.venvs/wearusfm-tok/bin/python -B results/infra/scaling_49046_20261118/launcher/verify_release.py`.
La suite non misura l'esecuzioneGPU né garantisce tempi di coda o memoria futura.


Nota sulle verifiche: un tentativo esteso su tutti42fileCPU scaling è stato
interrotto (exit130) senza avanzamento nel vecchio test di ripresa con2worker
locali, prima del primo update; non viene presentato come suite superata né
come difetto dimostrato della produzione. Verbale in
`launcher/broad_suite_interrupted.json`. La verifica di rilascio è quella
mirata e limitata a600s in `verification.json`, con test nuovi e regressioni19k.


Il giro di regressione19k più nuovi test ha dato881passati e un solo fallimento
nel test storico `test_logical_ids_and_outputs_are_unique_new_and_outside_the_code`:
consulta il ledger comune live ma cerca ricevute19k nel vecchio percorso.
La stessa identica asserzione fallisce anche sul ramo19k concluso2df4ca68;
stdout e riferimento in `launcher/baseline_ledger_test_failure.json`. Non è
stato modificato il test storico né alterato il ledger. Il giro di regressione
resta registrato come non interamente verde. La verifica finale mirata ripete i
test nuovi e la guardiaoutput dopo l'ultima correzione dei pin; l'esito è in
`verification.json`. I due risultati non vengono sommati come test distinti.

I byte dei checkpoint39k/49k non sono predetti: derivano dal `run_inventory`
della ricevuta nativa, con stessoSHA della ricevuta, tipo intero positivo,
limite6GiB per checkpoint e20GiB totali d'archivio. Il4.673.783.740 è il dato
misurato del19k e la base della stima di storage, non un pin fittizio del49k.


Esito finale di rilascio: **250test passati in178,24s**, sorgenti invariati durante
la prova. `source_inventory.json` fissa i byte operativi e i test;
`scientific_integrity.json` conferma103sorgenti scientifici immutati.
`release_catalogue.json` riserva56ID massimi univoci e inutilizzati alla
preparazione, non è un piano di fit inventato. Nessun job o deploy49k eseguito.
