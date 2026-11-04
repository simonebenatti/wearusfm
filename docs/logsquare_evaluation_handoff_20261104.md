# T100-LQ: consegna degli adattatori di valutazione — 04/11/2026

**Stato: analisi delle lacune, non implementazione né autorizzazione di lancio.**
Simone ha già autorizzato il seguito operativo; rimane la review di Claude prima
 del deploy. La review `b0fac8f` richiede orchestratori ed estrattori adattati e
verificati prima del training, e `rho_infer` misurato prima delle estrazioni.
Questa nota specifica il lavoro necessario; non soddisfa da sola quel requisito.
Nessuna modifica di dose, loss, seed, readout, griglia di ricerca o criterio.

## Disegno da conservare

| Endpoint | Nuova estrazione LQ | Nuovo fit | Controlli riusati |
| --- | --- | --- | --- |
| 5.000 | posa train + val, teacher | ridge istantanea | T100-L primario, T25-L e T100-G secondari alla stessa dose |
| 19.618 | posa train + val, teacher | ridge istantanea | stessi tre controlli a 19.618 |
| 19.618 | P1: DB2/DB3/DB6; P2: EPN612/UCI, teacher | logistica pool768 convergente | OOF dei tre controlli a 19.618 |

Quattro job di estrazione scientifica per il candidato: due posa, uno P1, uno
P2. Producono nove cache NPZ: quattro posa e cinque classificazione. I controlli
non si riaddestrano né si riestraggono se ricevute, pin, ricette e supporti sono
compatibili. I numeri attesi non sono un preventivo: quest'ultimo dipende dal
benchmark di inferenza e dai benchmark CPU previsti.

## Lacune osservate nel codice attuale

| Componente | Vincolo attuale | Adattamento necessario |
| --- | --- | --- |
| `tokenizer_evaluation.py` | `ARMS=P.ARMS`, protocollo della ricevuta training storico, `comparison.arm_configurations[arm]` a tre bracci | Controller LQ esplicito; config e ricevuta del training LQ, pin dei due endpoint e validazione reale del ramo |
| `validate_checkpoint` | costruisce il teacher con strict load ma verifica solo finestra/attivazione fra proprietà del tokenizer | Verificare anche flag LQ, epsilon, presenza/forma/valore della matrice log, carico del gain e config completa |
| config storiche | `FMConfig` attuale aggiunge due campi assenti nei JSON storici; confronti grezzi a `asdict(model.cfg)`/`config_to_dict` non coincidono | Canonicalizzare esclusivamente `tokenizer_logsquare=False`, `tokenizer_logsquare_eps=0.001` quando assenti; conservare hash e byte originali; nessun altro default implicito |
| `tokenizer_pose_readout.py` | `validate_plan`, `run`, `paired_comparison` e CLI richiedono esattamente tre bracci | Fit del solo candidato; confronto a quattro celle con tre contrasti LQ dichiarati, importazione verificata delle celle storiche |
| `tokenizer_probe.py` | piano a tre bracci; `compare` richiede sei aggregati con stesso piano/codice | Fit e aggregazione del candidato; confronto di otto aggregati, ognuno con provenienza originale, senza fingere identità di codice/piano fra campagne |
| launcher storici | whitelist tre bracci e walltime fissi; confronti a tre celle | Launcher nuovi LOGSQ, risorse da piano misurato, nessun bypass delle whitelist |

`fm_features.load_model` già ricostruisce tutti i campi riconosciuti di `FMConfig`
e usa `model.load_state_dict(...)` in modalità strict: il meccanismo può caricare
LQ. Occorre provarlo con pesi log non nulli, non solo con il ramo a zero, e con
checkpoint storici privi dei due campi. Non serve modificare il caricatore per
ignorare chiavi, errori o differenze di configurazione.

## Strategia minima proposta

Conservare gli orchestratori storici e introdurre moduli LOGSQ separati che
riusino le funzioni numeriche già verificate. Non cambiare `ARMS` tramite monkey
patch, non rinominare LQ in un braccio storico, non riscrivere ricevute o OOF.
Le nuove validazioni e aggregazioni devono essere esplicite; i fit rimangono
quelli esistenti. Il codice condiviso necessario deve essere pinnato nel piano.

File da implementare e sottoporre a review:

- `scripts/logsquare_evaluation.py` e `scripts/slurm/logsquare_evaluation.sbatch`;
- `scripts/logsquare_infer_benchmark.py` e relativo launcher;
- `scripts/logsquare_pose_readout.py` e relativo launcher CPU;
- `scripts/logsquare_probe.py` e relativo launcher CPU;
- test CPU omonimi, più test dei launcher;
- piani in `results/infra/logsquare_evaluation_20261104/`: estrazione, benchmark
  inferenza, lettore posa per dose, sonde endpoint e risorse.

### 1. Estrazione e checkpoint

Interfaccia proposta, parallela a quella attuale:

```
logsquare_evaluation.py --stage validate|extract-pose|extract-classification
  --plan PLAN --step 5000|19618 [--task p1|p2]
  --expected-code-sha SHA --expected-plan-sha256 SHA256 --out NEW_DIR --device cuda
```

Il braccio è fissato a `T100-LQ`, non un argomento libero. Riusare
`tokenizer_evaluation.extraction_commands`, `pose_sources`,
`classification_sources`, `verify_pose_cache`, `verify_classification_cache` e
le relative verifiche di immutabilità. Per `load_context` e
`validate_checkpoint` serve la variante LQ, perché protocollo e identità del
training differiscono; una chiamata al guard storico non è sufficiente.

Il piano registra config candidata e piano training, input originali
`csv/manifest/scales/windows/pose_stats/splits`, radici, riferimenti storici,
allocazioni e benchmark `rho_infer`. Per ciascun endpoint il pin conserva:
`path`, `sha256`, `config_path`, `config_sha256`, `receipt_path`,
`receipt_sha256`; aggiungere SHA del codice training atteso e della release.
Prima che esista il checkpoint il piano è preparatorio; nessun valore fittizio
può essere interpretato come pin valido. Dopo il training si pubblicano i pin
reali in un nuovo commit e checkout per la valutazione.

Guard obbligatori: endpoint nominato immutabile, assenza STOP, ricevuta completa
`tokenizer-logsquare-training-20261104-v1`, stessa config/dose/dati/statistiche,
caricamento strict del teacher, `window_ms=100`, attivazione identity,
`tokenizer_logsquare=True`, epsilon esatto 0.001, matrice log `(384,64)` uguale
al checkpoint, gain salvato uguale. Il teacher viene caricato in lettura e non
ricalibrato sui segnali. Lo stato dei file si verifica anche a fine estrazione.

### 2. Benchmark di inferenza sul percorso effettivo

**I percorsi attuali non hanno la stessa precisione:**

- Posa: `pose_eval.extract` chiama `model.encode(inp,None)`, `latent_mean` e
  `model.pose_head` con `no_grad`, **senza autocast**. Il batch massimo richiesto
  è 32, ma il ciclo è per registrazione con dieci finestre: i batch effettivi
  possono essere dieci o meno. Misurare le forme realmente consegnate.
- Classificazione: `channel_readout.extract_p1/extract_p2` chiama
  `CR.extract_channel_readouts`, con BF16 autocast CUDA. `CR.readouts` usa
  `forward_layers`, compreso il decoder su tutte le query valide, e produce
  pool/local/dec. La sonda usa poi solo pool768, ma il costo attuale include gli
  altri readout. Sostituirlo con solo `model.encode` falserebbe il preventivo.

Benchmark proposto: due teacher congelati T100-L e LQ allo stesso endpoint
(5.000 è sufficiente per prepararlo), eval e stessi ingressi, ordine dei due
bracci alternato in modo fissato, warmup e numero di ripetizioni pubblicati
prima del lancio. Usare selezioni deterministiche dei soli dati di train dei
percorsi reali: posa train; soggetti train di un fold congelato per P1/P2,
inclusi EPN612 a 200 Hz e i gruppi nativi degli altri dataset. Non selezionare
batch in base a loss o score. Ripetere le **stesse forme e lo stesso supporto**
per i due modelli; registrare identità dell'ingresso, fs, canali, n e patch.

Implementazione preferita: strumentare chiamate delle funzioni di estrazione
esistenti sul sottoinsieme dichiarato, senza duplicarne preprocessing,
scaling, crop o batching. Cronometrare con sincronizzazione CUDA prima e dopo
il percorso di forward e dei readout; registrare separatamente loader,
preprocessing, trasferimenti e scrittura, oltre al totale. I payload scientifici
prodotti dal campione non entrano nei fit e non si leggono score val.

Report per percorso e gruppo di forme: tempo dopo warmup, n di campioni/batch,
memoria e rapporto LQ/L. La regola di aggregazione delle forme e dei dataset
per ottenere i rapporti di posa, P1 e P2 va fissata nel piano **prima** del job:
non scegliere il rapporto più favorevole dopo averli misurati. Alternativa
conservativa da sottoporre a review: massimo dei rapporti dei gruppi
rappresentativi del rispettivo job, con copertura obbligatoria e nessuna media
pesata inventata dopo i dati. Non è ancora una regola approvata.

Poi R2: stima di ciascun job = durata storica omologa × rapporto; cap = walltime
storico × max(1,rapporto), arrotondato per eccesso a 300 s. L'effetto su loader/I/O
rimane dichiarato. Il benchmark ha il proprio preventivo GPU/CPU e ricevuta;
`rho_update` non lo sostituisce. Prima delle estrazioni si pubblicano misura,
cap e pin del benchmark. Oltre limiti o memoria: fermare e revisionare.

### 3. Ridge posa: un nuovo fit, celle storiche verificate

Riusare senza modifiche `Q.read_data`, `Q.fit_reader`, `Q.predict`,
`Q.summarize`, `Q.benchmark_train` (`Q=tokenizer_pose_readout`). Restano
LayerNorm stateless FP32, mean K64 salvata FP16, crop6, mask centrale, CV sui
40 utenti di train in due fold alternati, griglia/estensioni alpha e lettura
val dopo la selezione. Il benchmark apre solo train; alpha irrisolto ferma
la lettura, senza cambiare griglia o scegliere un controllo diverso.

Interfaccia proposta: stage `validate|benchmark|ridge|compare`, piano pinnato,
SHA codice/piano, output nuovo. `ridge` calcola solo LQ e salva coefficienti,
predizioni e per-user sufficienti. `compare` richiede la cella LQ più il report
storico completo alla stessa dose, pin delle cache, ricetta, source hashes,
coefficienti e predizioni storici verificati. I report storici di partenza sono
sotto `results/infra/tokenizer_evaluation_20261103/cpu_5000/final/` e quelli
omologhi dell'endpoint; i percorsi effettivi vanno presi dai loro receipt,
non dedotti dal nome della cartella.

Confronto esplicito a quattro celle, contrasti soltanto LQ−T100-L (primario),
LQ−T25-L e LQ−T100-G (secondari). Conservare tutte le misure e bootstrap utenti
2.000/seed20261102, stessi draw fra celle, denominatori/supporto identici,
soglia pratica 2% e gruppi dynamic/all/static. Nessuna intersezione silenziosa.
Non è consentito riscrivere le vecchie celle assegnando loro il codice nuovo.

### 4. Logistica: un candidato, quattro fold, OOF originali

Riusare `tokenizer_probe.fit_fold`, `AuditLog`, `audit_summary`,
`validate_oof`, `numeric_context` e le funzioni metriche. La parte nuova deve
validare piano/caches LQ e aggregare i suoi shard con le stesse verifiche del
controller storico. Restano pool768, niente PCA, scaler sul train del fold,
C = 1e-3...100, lbfgs/12.000/tol1e-4, balanced, selezione su val del fold,
pareggio al C minore, refit sul solo train, esclusione test ufficiali.

Interfaccia proposta: `fit-shard|aggregate|compare`, SHA piano/codice e output
nuovo; i primi due operano solo su LQ, con shard storici DB2-01/DB2-23/DB3/DB6/P2.
`compare` importa i due aggregati LQ e i sei storici, verificando per ciascuno
il proprio protocollo/piano/codice, OOF e audit originali. Stessa ricetta e
contesto numerico sono condizioni esplicite; una differenza non va nascosta
relassando la verifica esistente di `numeric_sha256`.

Controllare identità ordinata row/y/s/fold, target testuali EPN612 compresi,
conteggi train/val/held, selezione C, sette fit per fold, score ricalcolato
sulle OOF. Rigenerare bootstrap appaiati sulle quattro celle con la stessa
regola storica, pubblicando draw hash, tutti e tre i contrasti LQ, EPN612
primario di classificazione e NI1pp per P1/P2. Se un fit resta irrisolto,
conservare numeri descrittivi e sospendere la decisione pertinente; nessun
retry o aumento automatico di max_iter. Prima dei fit, benchmark e cap CPU.

## Test e review necessari

- Checkpoint LQ reale sintetico con W_log non nullo, gain deliberatamente
  diverso dal costruttore, strict load e uguaglianza dei pesi/output; rifiuto
  di flag/epsilon/chiavi/forme diversi, stato incompleto, step/receipt/STOP errati.
- Legacy T25-L/T100-L/T100-G: ammettere solo i due default LQ mancanti;
  rifiutare qualsiasi altro campo mancante o mutato, con pin aggiornati nel test.
- Parità delle chiamate di estrazione e dei target/righe/crop/supporti;
  leak/val vietata nel benchmark e nelle scelte dei fit, output parziale senza
  ricevuta completa, hash e file-sorgente ricontrollati.
- Timing sintetico deterministico: warmup escluso, sync prima/dopo, alternanza
  dei bracci, ratio positivo e finito; nessuna sostituzione FP32/BF16 o pool-only.
- Fit ridge/logistica: risultato numerico uguale alla funzione storica sugli
  stessi dati, candidato unico, nessuna lettura val per alpha/scaler.
- Confronti a quattro celle: zero delta per predizioni uguali, segno corretto,
  bootstrap condiviso; rifiuto di cella/dose/OOF/fold/denominatore mancante o
  diverso, sorgenti miste non dichiarate, convergenza irrisolta e classi testuali.
- Launcher: sola lettura del checkout, SHA e piano contenuti, output nuovi
  esterni, risorse esatte dal piano, nessuna sottomissione nascosta.

Prima del training: implementazione degli adattatori, regressioni storiche,
piani strutturali, review della lettura R1/del preventivo e review di questa
consegna effettivamente implementata. I pin dei checkpoint futuri arriveranno
alla loro produzione; non servono placeholder accettati dal runtime.
Prima delle estrazioni: checkpoint completi verificati, pin reali pubblicati,
`rho_infer` e cap misurati, checkout congelato. Prima dei fit: cache verificate,
benchmark CPU, ricette e contesto numerico compatibili.

Nessuna di queste attività introduce contrastiva, nuovi layer, energia di
banda o altri rami: sono preparazione e verifica della sola domanda LQ.
