# Valutazione lunga: proposta di emendamento delle risorse, 06/11/2026

**Stato: proposta per la decisione di Simone, poi implementazione e review di
Claude. Non e' un piano eseguibile.** Runtime, `plan.json`, `launch_plan.json`,
pin, ledger e checkout attivi restano invariati. Operatore unico: Claude.

Fonti: ricevute pubblicate da Claude in `2a20316a0bf77c93e354d4ce3f5acee94db459df`,
registro `a5d57019a76638840468f927517000b396d58bbb`. Verifica locale, senza cluster.
Ricalcolo indipendente in `docs/reviews/check_long_resource_options_20261106.py`;
matrici, pin, costi e alternative in
`results/infra/consolidation_long_20261105/resource_amendment_proposal/options.json`.

## 1. Gate verificato e motivo dell'emendamento

Le 16 ricevute sono complete; hash contro l'audit pubblicato corretti. Le 15
celle di classificazione contengono sei fit train-only convergenti ciascuna,
senza valutazione validation/held, stessa firma numerica `1ac8481e...` e regola B
verso lo storico. I 30 record SLURM archiviati (12 estrazioni, 16 benchmark,
due checkout) sono COMPLETED 0:0 senza riavvii. Consumo confermato:
3,373611 GPUh e 42,218889 coreh.

La formula del gate e' ricalcolata sia con il runtime congelato sia con codice
indipendente: **15 shard respinti su 48, 10 celle su 40**. Proiezione complessiva
1.087,621716 coreh. Memoria ammessa ovunque; il problema e' il tempo per job.
Il riferimento storico 114,24 coreh viene sostituito nelle future stime dalla
misura completa, senza confondere proiezione prudenziale e consumo effettivo.

| Caso | Proiezione per shard attuale | Soglia |
|---|---:|---:|
| DB2 pre, due coppie di fold | 21.515,6 / 20.454,2 s | 13.800 s |
| DB2 pool, due coppie di fold | 16.166,2 / 15.376,3 s | 13.800 s |
| DB3 pre, quattro fold insieme | 26.523,0 s | 13.800 s |
| DB6 pool, quattro fold insieme | 16.858,7 s | 13.800 s |

Il fermo di Claude e' corretto: non sono esiti scientifici negativi, e non si
aggira il gate per far partire soltanto le celle che passano.

## 2. Suddivisione proposta

| Cella | Prima | Proposta | Job prima → dopo, su tutte le dosi |
|---|---|---|---:|
| DB2 pre | [0,1], [2,3] | [0], [1], [2], [3] | 6 → 12 |
| DB2 pool | [0,1], [2,3] | [0], [1], [2], [3] | 4 → 8 |
| DB3 pre | [0,1,2,3] | [0], [1], [2], [3] | 3 → 12 |
| DB6 pool | [0,1,2,3] | [0,1], [2,3] | 2 → 4 |
| Tutti gli altri casi | Invariati | Invariati | 33 → 33 |
| **Classificazione** | | | **48 → 69** |

Restano **40 celle scientifiche**, gli stessi quattro fold per cella, tutte le
classi, la griglia di C, i sei candidati e il refit selezionato, stessi solver,
tolleranza, limiti di iterazioni, numerica a otto thread, dati, endpoint e letture.
Il pool a 19.618 resta il riferimento storico riusato, non un nuovo fit.

Si divide esclusivamente il lavoro indipendente per fold. Non si eliminano
candidati o fold, non si cambia il criterio di selezione e non si riducono i
fattori di sicurezza. Le scelte sono fissate sulle sole misure train-only dei
tempi, senza leggere punteggi o predizioni del confronto lungo.

DB3 pre potrebbe passare anche con due coppie, ma una avrebbe soltanto 65,9 s
di margine rispetto al gate a quattro ore. I singoli costano appena 24 coreh
allocate in piu' rispetto all'alternativa con tempi calibrati: li preferisco.
Le alternative sono conservate nel JSON, non vengono scelte dopo i risultati.

## 3. Tempo richiesto e nuovo preventivo

La proiezione P di ogni shard mantiene esattamente la formula attuale:
massimo fra tempo misurato e floor storico per fold, fattori 3 e 1,5, 600 s di
overhead per job e letture. Il tempo SLURM W viene fissato ora da:

```text
W = 900 × ceil((P + 600) / 900)
gate: P <= W − 600 e W <= 14.400 secondi
```

Si conserva quindi anche la riserva finale di 600 s oltre alla proiezione.
Non e' una regola di adattamento durante l'esecuzione: i 69 valori sono
pubblicati prima dei fit. Richiesta costante di 8 CPU e 30 GiB; walltime fra
30 e 225 minuti, tutto entro il limite di quattro ore di `lrd_all_serial`.
La ricostruzione del gate resta globale per tutte le 69 celle fisiche; nessun
fit di classificazione parte se ne fallisce una. Memoria: limite invariato.

| Voce | Core-ore |
|---|---:|
| Proiezione prudenziale dei 69 fit | 1.115,658874 |
| Allocazione massima dei 69 fit | 1.276 |
| Altre allocazioni future, inclusi due checkout | 41 |
| **Massimo futuro complessivo** | **1.317** |
| Gia' consumato | 42,218889 |
| **Consumato + massimo futuro** | **1.359,218889** |
| Totale conservativo, mantenendo anche i vecchi tetti delle fasi concluse | **1.451,666667** |
| Tetto approvato | **1.711,666667** |

**Nessun aumento del tetto richiesto.** La contabilita' piu' conservativa lascia
260 coreh di margine. Nessuna GPU nuova se viene approvato il riuso verificato
descritto sotto; il cumulato GPU resta 3,373611 sul tetto originario di 9.
La maggiore frammentazione aggiunge circa 28,04 coreh alla proiezione, per gli
overhead ripetuti; non moltiplica il lavoro scientifico.

69 fit da quattro ore ciascuno porterebbero invece il totale a 2.383,67 coreh:
non e' l'opzione raccomandata. La proposta risparmia allocazioni sui casi veloci,
senza abbassare i margini della formula di ammissione.

## 4. Ricevute, source_identity e riuso senza nuove misure

R-2(i) attuale vieta il riuso dopo ogni cambio ai sorgenti coperti. Non la
consideriamo gia' superata: la decisione e la review dell'emendamento devono
approvare un'eccezione esplicita, limitata a questi input.

Propongo un **manifest immutabile di compatibilita' per le sole 28 ricevute**:
12 estrazioni e 16 benchmark. Il JSON della proposta ne contiene gia' i pin
esatti. Conservare ricevute e `cache_manifest` originali, senza riscrivere source,
SHA del codice, piani o hash dei 21 payload.

- Identita' originale: source `1ca7aead5cd48a1fd9ff398338d80f2471a8d4041ccc0ac7fcdb3c3c03d0b1e1`;
  recipe `b2356a8e09fd7bc560a2a677c2c93f8446afcff96254c91454452c937f865422`.
- Il manifest verifica per ciascun input SHA/byte, tipo, dose/task/dataset/vista,
  codice e piano originali, ricetta, provenienza della cache e artefatti associati.
  Nessun altro source o input viene ammesso implicitamente.
- `source_identity()` rimane piena e obbligatoria per il nuovo codice: non si
  escludono genericamente file o directory dalla firma. Tutti i nuovi fit,
  confronti e sintesi avranno la stessa nuova source congelata. I report nuovi
  registreranno il pin del manifest e le provenienze effettive degli input.
- La review confronta le mappe complete dei sorgenti e il diff puntuale:
  estrattore, rappresentazioni, matematica dei fit, selezione e valutazione
  devono restare invariati. Le sole modifiche ammesse sono confezione dei job,
  ammissione delle risorse, provenienza e rilascio delle fasi. Se la prova di
  compatibilita' non riesce, si ferma: nessuna riclassificazione delle ricevute.
- I benchmark conservano la policy originale. Si verifica quella e l'identita'
  della misura, poi si ricalcola la nuova suddivisione sulle **stesse ricevute**.
  Non basta accettare il vecchio source confrontando il resto con la policy nuova.
- Per la posa vanno verificati anche il source del manifest e quello nella
  configurazione incorporata nelle NPZ. Si conserva l'identita' originaria degli
  input senza sostituire `ctx.source_sha256`, che identifica il nuovo eseguibile.

**Non serve ripetere i benchmark**: le misure sono complete, numerica e algoritmo
del fit non cambiano, e il lavoro per fold viene solo distribuito fra job.
Questo vale a condizione che l'implementazione e la review confermino gli
invarianti. Nessuna ripetizione automatica di estrazioni o benchmark e nessun
nuovo costo GPU e' implicitamente autorizzato dalla proposta.

## 5. Numero di job, identificatori e posa

La campagna diventerebbe **103 job di calcolo + 4 checkout = 107 allocazioni**.
30 sono gia' concluse; ne restano 77. I 69 ID proposti terminano in `-A1`, con
output nuovi sotto `fits_a1/classification/`; il JSON contiene l'intera lista e
la corrispondenza con i 48 ID precedenti, mai inviati secondo la consegna Claude.
Gli ID precedenti vanno marcati nel piano come sostituiti, non come job falliti.
Il ledger non viene riscritto e prima dei futuri invii Claude verifica eventuali
conflitti. Nessun ID della proposta e' stato riservato o sottomesso da GPT.

I tre fit posa, i due confronti, la sintesi e i due checkout futuri conservano
gli ID non ancora usati. Non si aggiunge un quinto checkout. Gli artefatti di
estrazione e benchmark gia' completati mantengono tutti i loro riferimenti.

La posa passa il suo gate: **583,269 s per dose** contro 3.300, memoria 12,530 GiB
contro 25,5. Propongo di rilasciare **prima i tre fit posa**, dopo la review
dell'emendamento comune e sullo stesso nuovo checkout dei fit. Il suo gate e'
indipendente da quello della classificazione; il nuovo prontuario deve renderlo
esplicito. I criteri e i pannelli scientifici restano tutti quelli concordati.

Non anticiperei la posa con il vecchio codice: richiederebbe compatibilita'
anche fra nuovi risultati di versioni differenti e complicherebbe la sintesi.
Tutti i nuovi output rimangono sotto un'unica source congelata.

Corsia: **al massimo due job fit della linea 2 pendenti o in esecuzione in
totale**, inclusa la posa, finche' il confronto dello scaling non e' in coda.
L'operatore Claude verifica quel prerequisito con il coordinamento; nessun
aumento automatico della concorrenza da parte di GPT. Restano i limiti globali
di coda e la priorita' concordata con la linea 1.

## 6. Consegna esecutiva dopo la decisione

L'implementazione dovra' aggiornare insieme policy per dataset/vista, matrice,
walltime per job, controlli del launcher e dei manifest, conteggi del confronto,
bridge limitato agli input e prontuario. La regola di lettura scientifica e la
ricetta non cambiano. I nuovi pin di codice saranno reali, dopo il commit e la
review; non sono inventati nella presente proposta.

Verifiche richieste prima del rilascio: gate esatto sulle 15 ricevute e sulla
posa; copertura identica delle 40 celle/fold senza duplicati; 69 ID/output
distinti; limiti e contabilita' riprodotti; rifiuto di source/recipe/input alterati,
input non elencati e vecchi report dei fit non previsti; confronto scientifico
identico a prescindere dalla partizione dei fold; tutti gli input originari
immutati. La review di Claude precede pin e deployment; poi opera lui nella
catena approvata, senza una seconda richiesta di via per quei lanci.

La chiusura EPN15 richiesta da Claude e' **gia' verificata** in `bb08cd0c1bd56f57efe3a85fc8dbc5713889b6e0`:
[review scientifica](reviews/finetune15_results_20261106.md), aggregatore identico
e ricalcolo indipendente. Non e' un prerequisito ancora pendente per questa scelta.

Rilettura del critico GPT: nessuna contraddizione bloccante nel disegno, nella
distinzione fra proposta e rilascio, nel bridge o nell'ordine delle fasi. Non
sostituisce la review di Claude della futura implementazione.
