# Review GPT della correzione dell'inizializzazione T100-LQ

Data: 05/11/2026. Codice esaminato: `fd874d75dcc5252076399c3c8d2d5ae97584570c`,
ramo Claude `claude/logsquare-init-fix-20261105`, base `4d5a3ef`.
Integrato per fast-forward nel ramo GPT `codex/tokenizer-logcontrast-20261104`,
senza riscrivere i commit. Due review indipendenti: guard/diagnostica e ingresso dei rank.

**Verdetto: A e B APPROVATI; C approvato per lettura e risorse, DA CORREGGERE
per l'esecuzione dei timeout. Non sottomettere ancora la diagnostica.**
Nessun nuovo job, nessuna release di training, nessuna modifica GPT ai sorgenti di produzione.
Il piano pubblicato resta `REVIEW_REQUIRED`. L'autorizzazione operativa di Simone resta valida.

## A. Stato esatto accettato

Approvata la regola: tutti i tensori condivisi corrispondono al riferimento del preflight
quando su una copia si sostituisce il solo gain; il gain originale deve invece coincidere
bit per bit con quello dei checkpoint storici T100-L; `W_log` deve essere nullo.

- Hash del riferimento: `38cb4cce85f9ba01cfd9ceceaff10ff202136089e8f6757562f684efab5345ea`.
- Gain del preflight: `548.6939187567023`, `0x1.1258d2546f0f0p+9`.
- Gain storico del training: `548.6939187567021`, `0x1.1258d2546f0efp+9`.
- Guard: `9cc3c37efc6ee9443b366799d5e8da4d78036f2dc2f81229b63deaf15fafea9c`.

È un emendamento esplicito del riferimento accettato, motivato da evidenze storiche
indipendenti dal modello osservato oggi. Non è l'adozione di un nuovo hash misurato
dopo il fallimento, né una tolleranza o una scrittura nel modello. Lo stato esatto del
preflight è correttamente rifiutato. Il guard rilegge le fonti storiche e rifiuta anche
una modifica di un solo ULP su un parametro reale.

Limite da mantenere in tutte le sintesi: i checkpoint storici attestano il gain, ma non
conservano l'hash completo dello stato iniziale. Chiamiamo lo stato accettato
**ricostruzione registrata dell'inizializzazione storica**. Non abbiamo dimostrato
retrospettivamente che ogni altro tensore storico fosse identico. Il gain del vecchio
controller fallito non è stato registrato; l'ordine degli import resta una spiegazione
da verificare con la diagnostica dei contesti, non una misura di quel processo passato.

## B. Controllo nel trainer

Approvato l'ingresso per rank. Il wrapper intercetta lo studente costruito dal vero
trainer dopo il seed, lo verifica prima che venga restituito e restituisce lo stesso
oggetto senza modificarlo. L'eccezione per `count_parameters` è limitata al modello
temporaneo del rank 0; i ruoli esatti sono poi richiesti dal controller. Il percorso
fresco esclude i successivi meccanismi che potrebbero sostituire lo stato iniziale.

Quattro attestazioni concordanti sono richieste; il gain di student e teacher viene
ricontrollato nei checkpoint. Il percorso `--verify-only` si ferma dentro il costruttore,
prima di teacher, optimizer e wrapper DDP. Può leggere manifest, indici e scale: la
formulazione precisa è **nessun batch di training o forward sui segnali**, non assenza
assoluta di letture dei dati. La prova del wrapper con quattro rank CPU/Gloo verifica
due update bit-identici al trainer normale. CUDA/NCCL e lo stato a 30M su Leonardo
restano oggetto della diagnostica reale.

## C. Lettura e risorse

Approvati i tre esiti prefissati e il rifiuto di letture computabili quando un contesto
fallisce. `cause_demonstrated` va letto come riproduzione operativa della discrepanza
nei contesti attuali, con i limiti storici sopra; il rapporto non autorizza il training.

Approvato il tetto: 15 minuti, 4 GPU e 32 CPU = **1 GPU-ora e 8 core-ore**;
checkout 2 CPU per 10 minuti = **0,333333 core-ore** aggiuntive.
La correzione dei timeout deve includere il tempo di pulizia nei limiti dichiarati.
Un cambiamento delle durate richiede un aggiornamento esplicito del piano prima del lancio.

## Rilievo bloccante R-TIMEOUT

**Priorità P2. File `scripts/logsquare_initialization_diagnostic.py`:295–304 e 436–445.**

`run_bounded` crea una sessione nuova per il launcher e al timeout invia subito
`SIGKILL` al suo gruppo. Questo non raggiunge i worker creati in sessioni separate e
impedisce al launcher di effettuare la propria pulizia. Il watchdog esterno usa invece
`subprocess.run(timeout=...)`, che termina il solo figlio diretto. I limiti annidati
300/270 secondi non dimostrano che il timeout interno avvenga sempre per primo: prima
del lancio ci sono import, controlli e costruzioni del modello, senza un tetto di 30 secondi.

Conferma in sola lettura su Leonardo: Torch `2.2.0a0+git6c8c5ad`, ambiente
`cineca-ai/4.3.0`, file
`/leonardo/prod/opt/libraries/cineca-ai/4.3.0/none/cineca-ai-env/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/api.py`:
il `SubprocessHandler` imposta `start_new_session=True` alle righe 563–566 e la propria
chiusura segnala il gruppo di ciascun worker alla riga 584. Non è solo una differenza
del Torch installato sul Mac.

Due regressioni indipendenti in `docs/reviews/test_logsquare_detached_timeout_20261105.py`
falliscono sul commit esaminato:

1. Timeout interno: il worker separato continua a scrivere un heartbeat dopo il ritorno
   di `run_bounded`.
2. Timeout esterno di `execute`: helper, launcher e worker annidati; il worker continua
   a scrivere dopo il ritorno del watchdog esterno.

La prova controlla avanzamento effettivo, non soltanto esistenza di un PID zombie.
I processi generati dai test sono stati ripuliti e la pulizia verificata. Esito completo
in `results/infra/logsquare_initialization_20261105/gpt_review_timeout_reproduction.json`.

Serve una terminazione coordinata e limitata nel tempo che copra entrambi i watchdog
e anche i discendenti in nuove sessioni: consentire prima la chiusura ordinata di
torchrun, poi eventuale escalation mirata ai processi della sola invocazione, senza
perdere i discendenti quando termina il padre. Conservare log ed errore del contesto.
Non basta aumentare il timeout o sostituire SIGKILL con SIGTERM senza un controllo
dei discendenti residui. Il walltime SLURM resta una protezione esterna: non sosteniamo
che i worker sopravvivano necessariamente alla fine dell'allocazione. Il difetto è
nel limite e nella pulizia del singolo contesto, che il codice promette di garantire.

## Verifiche

- Quattro file originali LQ: prima esecuzione **153 passati, 1 fallito** per il solo
  divieto sandbox di bind su localhost. Il singolo test quattro-rank, rieseguito fuori
  sandbox, **passa**. Totale dei 154 test mirati verificato, senza CUDA.
- Le due regressioni aggiuntive del timeout: **2 fallimenti attesi**, in 2,49 secondi.
- `src/`, i preflight e `train_jepa.py` invariati rispetto a `4d5a3ef`.
- Pin della diagnostica validati su una copia temporanea con il solo stato impostato
  ad approvato. Nessun cambio di stato del piano pubblicato.
- La suite CPU completa 2446/2447 è un risultato comunicato da Claude, non rieseguito
  in questa review. Il fallimento preesistente citato da Claude non interessa questo diff.

## Seguito

Claude corregge R-TIMEOUT e le due formulazioni descrittive; GPT rivede il diff mirato.
Restano invariati guard, criterio sui rank, epsilon, ricetta, 102 sorgenti scientifici,
vecchia release e artefatti del fallimento. Solo dopo la chiusura del rilievo: approvazione
del piano diagnostico, commit/push, checkout congelato e diagnostica già autorizzata.
Nuova release di training soltanto dopo averne letto l'esito.
