# Review indipendente: finestra e tokenizer non lineare — 02/11/2026

**Esito: favorevole alla consegna preparatoria; nessun difetto materiale o blocco algoritmico individuato.**
La modifica implementa i tre bracci del [protocollo](../tokenizer_window_nonlinear_protocol_20261102.md)
senza cambiare i default storici. Questo esito riguarda codice e configurazioni locali: non certifica
apprendimento, miglioramento del FM o costo GPU dei nuovi tokenizer. Supporti sui batch reali,
timing/memoria e piano operativo restano le verifiche concrete già previste, non nuove richieste di consenso.

Review del subagent `critico`, in sola lettura sul worktree `wearusfm-gpt-science`, diff rispetto a
`7e1c26f2f4e882242fb46a247432f63f78b7b013`. L'unico file scritto per questa review è il presente rapporto.
Nessun comando remoto, job, fit o duplicazione delle suite. Gli esiti dei test sotto sono quelli
comunicati dagli esecutori; il reviewer ha letto implementazioni e test e verificato localmente
hash e differenze delle configurazioni, senza presentare le esecuzioni altrui come proprie.

## Perimetro e verifiche

File runtime: `src/wearusfm/model/fm.py`, `patch_tokens.py`, `src/wearusfm/training/jepa.py`,
`run.py`, `scripts/train_jepa.py`. Letti anche `tests/cpu/test_frontend_window.py`,
`test_tokenizer_activation.py`, `test_tokenizer_comparison.py`, oltre ai percorsi invariati
di mascheramento, frontend, Perceiver e caricamento dei checkpoint pertinenti al diff.

| Aspetto | Evidenza verificata | Esito |
|---|---|---|
| Default e API | `FMConfig.tokenizer_activation="identity"`; `JEPAConfig.masked_loss_radius=None`; flag CLI opzionali. Il nuovo campo del modello è aggiunto in fondo alla dataclass. | Il percorso storico resta selezionato in assenza dei nuovi campi. |
| Posizione della GELU | Applicata alle 64 risposte dopo `fe_gain`, prima della proiezione 64→D, senza cambiare padding o validità delle patch. | Non linearità effettiva rispetto al segnale, non una modifica del solo readout. |
| Parametri e inizializzazione | GELU funzionale, nessun parametro/buffer addizionale, nessuna estrazione casuale; gain e pesi coincidono fra Identity/GELU a pari seed. | T100-L/G hanno pari parametri e supporto. |
| Apprendibilità | I kernel Fourier/spline/MLP e la proiezione rimangono nel grafo; test con gradienti finiti e non nulli verso tutte le famiglie. | La GELU non deve avere parametri propri per rendere apprendibile il tokenizer composto. |
| Supporto comune delle loss | `masked_query_radius` è chiamato sia da `loss_counts` sia da `_jepa_compute`; il valore è validato prima di encode e all'ingresso del trainer. | Override 2 coerente tra query, numeratori e denominatori; nessuna dipendenza dal peso JEPA. |
| Protezione minima | Override intero non negativo; bool/non interi rifiutati; override inferiore al `core_radius` nominale rifiutato. | Non si può ridurre accidentalmente la protezione richiesta dal tokenizer lungo. |
| Posa | Continua a usare `pose_time_valid(enc.patch_valid, ...)` e le annotazioni originali; i conteggi della posa non usano le query erose. | Il confronto non cambia implicitamente supporto o bersaglio della posa. |
| Checkpoint storici | Il loader del modello usa il default Identity quando manca il campo; `_config_with_defaults` normalizza Identity/None senza mutare il dizionario caricato. | I vecchi checkpoint non vengono reinterpretati come GELU. |
| Resume e branch | Le guardie confrontano `model` e `jepa`, inclusi attivazione e raggio; cambiare questi campi non è una modifica operativa ammessa. | Il riavvio non cambia silenziosamente la semantica dei parametri o della loss. |
| Encoder congelato | `tokenizer_activation` è normalizzato ma resta fuori da `BRANCH_MODEL_FIELDS`, usato anche come insieme di esclusione nel controllo frozen. | Il caricamento frozen rifiuta Identity→GELU invece di ignorare il cambio. |

Il normale `init_from` rimane un warm-start distinto da resume, come prima del diff; non è il percorso
della campagna proposta. Le configurazioni dei tre bracci partono da zero e non caricano un checkpoint
da 25 ms. Non si deduce la semantica dell'attivazione dai soli tensori dello state_dict: serve il config,
come verificato dal test di roundtrip e dal controllo negativo Identity/GELU con gli stessi pesi.

## Mascheramento e portata della garanzia

`WearUsFM.encode` continua ad azzerare i campioni delle patch nascoste **prima** del tokenizer.
Local encoder e Perceiver escludono le chiavi nascoste. La GELU è applicata dopo tale azzeramento,
quindi non crea una via alternativa per recuperare i campioni occultati. I test perturbano solo
quei campioni a 200/1.000/2.000 Hz, verificano identità delle uscite student e includono un controllo
positivo senza maschera, in cui la perturbazione deve essere rilevabile.

La garanzia è sul segnale ricevuto dal modello, dopo il filtro del loader: non dimostra isolamento
dei campioni originali prima del prefiltraggio. `core_radius=2` protegge il supporto nominale da
100 ms; non rende invisibili le code anti-alias o il contesto globale del teacher. Le docstring
corrette descrivono questo limite. Non è necessario cambiare il raggio a 6 per introdurre GELU;
una diversa definizione del target completamente locale sarebbe un'altra modifica scientifica.

Il test comune dei tre bracci usa JEPA **0**, ancore e posa, entrambe le riduzioni delle loss,
un campione più corto e un canale QC escluso. L'oracolo indipendente attende **306 query**,
con offset `[0, 240, 306]`, contro 388 nel controllo storico a raggio 0; la posa conserva 850 elementi
validi, oppure due finestre nella riduzione per finestra. Conteggi runtime e preconteggi coincidono.
RVQ è spento nelle configurazioni della campagna; non è stato acceso per ottenere copertura artificiale.

## Coerenza tra frequenze e test osservati

La prima lettura aveva segnalato un limite di copertura: la prova D8 riguardava il banco lineare,
mentre il test GELU usava segnali diversi alle diverse frequenze. Il limite è stato affrontato prima
della chiusura con `test_gelu_preserves_cross_sampling_rate_features_with_the_declared_d8_tolerance`:
stesso segnale fisico a banda limitata, 2 kHz contro 1 kHz/200 Hz, confronto dopo gain+GELU e prima
della proiezione, per famiglia, lontano dai bordi, tolleranza relativa 2% già utilizzata nel frontend.
Non viene rivendicata una nuova misura di equivalenza su tutto il corpus o su pesi dopo training.

| Esecuzione comunicata | Esito osservato dal reviewer tramite gli esecutori |
|---|---|
| Prima suite attivazione + finestre | 19 nuovi test di attivazione e 18 test frontend, tutti passati, 9,30 s. |
| Suite estesa a quattro file, con confronto dei bracci e D8 dopo GELU | **62 passed, 18,16 s**, comunicato da `b19618_integrity`. |
| Delta finale: confronto tre bracci reso esplicitamente JEPA=0 | **16 passed, 4,71 s**, solo `test_tokenizer_comparison.py`, comunicato da `b19618_integrity`. |
| Regressioni root: jepa, pose_training, training_run, objective_arms | **67 passati: 65 + 2 DDP**. I due torchrun erano inizialmente impediti dalla restrizione socket del sandbox, prima del training; la riesecuzione identica fuori sandbox è passata in 11,91 s tramite autorizzazione già esistente. |

Le prime tre righe descrivono esecuzioni sovrapposte, non vanno sommate. Il rifiuto iniziale del socket
non è classificato come errore algoritmico; il risultato DDP positivo è quello della riesecuzione effettiva.
Nessun test GPU o nuovo training è stato eseguito dal reviewer.

## Configurazioni appaiate e limiti della consegna

Letto [comparison_configurations.json](../../results/infra/frontend_window100_20261102/comparison_configurations.json)
e ricalcolati gli SHA dei tre JSON:

| Braccio | SHA-256 configurazione |
|---|---|
| T25-L | `e7f056b9d9036cbd411891e132a19f487873d6ae9f37a1b2e5a894dc445e2af9` |
| T100-L | `14c742f53660fe10283ca86ab6e525a1e69d92a36d827fbee26538f0afe999c8` |
| T100-G | `f48b643dbc1aacdce0a264033ef8f47e923ee041325b1f416eb6822bf9b132c1` |

Confronto indipendente ricorsivo dei JSON: T25-L→T100-L cambia soltanto `model.window_ms`;
T100-L→T100-G cambia soltanto `model.tokenizer_activation`. Rispetto al config B storico, T25-L
aggiunge il raggio 2, Identity esplicita, `log_batch_origin=true` e checkpoint conservati a 5.000/19.618;
le due ultime variazioni sono comuni e dichiarate. Non emergono nuove loss, dati o iperparametri nascosti.

Resta corretto misurare sui dati reali quanti target delle ancore sopravvivono, verificare l'appaiamento
del flusso e stimare costo/memoria. I test sintetici non sostituiscono tali misure, né dimostrano che
il vantaggio atteso della finestra o della GELU esista. La preparazione è coerente con il confronto
autorizzato; l'esecuzione seguirà il piano operativo completo, senza nuove barriere di approvazione
inventate da questa review.
