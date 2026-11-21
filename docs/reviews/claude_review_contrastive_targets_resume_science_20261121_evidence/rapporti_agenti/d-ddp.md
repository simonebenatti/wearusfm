# Compito D (d-ddp): meccanica DDP e ordine di riduzione

Clone: `918245b6cee89fb71e159c0fe17f4a7c2e746fa7` (sola lettura). Cartella di lavoro e script:
`/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_resume/agents/d-ddp/`
(`ddp_rebuild_probe.py` + `.out`, `ddp_lines.py`, `gradnorm.py`, `fields.py`, `ab.py`, `trace_eq.py`, `ring_order_sim.py`).
Percorsi delle evidenze relativi a `results/infra/contrastive_targets_20261118/`. Nessun job, nessun ssh, nessuna modifica al clone.

## Verdetto in breve

- Nella ripresa **solo la riduzione dell'update 545** usa i bucket iniziali (`114202292, 68480` byte, cioe' un solo
  bucket fp32 piu' uno piccolo, senza ricostruzione). Dal 546 la ripresa usa gli stessi 7 bucket ricostruiti del continuo.
  Il campo `has_rebuilt_buckets` delle tracce e' in ritardo di un update: per questo GPT (prompt) scrive «545-546» e
  il documento GPT e la nota precedente di Claude scrivono «dal 547». Entrambe le letture sono sfasate. [verificato]
- Gli artefatti esistenti localizzano gia' la prima differenza nella riduzione DDP del 545. A 545 le norme fp64 dei
  gradienti globali per obiettivo coincidono esattamente fra continuo e ripresa (`contrast_global_grad_base/nce/rnc`).
  Quelle norme passano da una riduzione esplicita, con layout fisso e indipendente dai bucket. Differiscono invece le norme
  calcolate dopo la riduzione DDP: `grad_norm` e `contrast_projector_grad_norm`. A 546 in CN la norma del proiettore
  dopo la riduzione torna identica. [verificato sui valori; l'interpretazione causale e' dedotta]
- L'ordine di grandezza (2,2e-12 e 3,4e-13 sulla norma a 545) e' compatibile con un diverso ordine di somma fp32 a 4 rank. [dedotto, simulazione CPU]
- Controllo minimo consigliato: **una sola ripresa CN con un'iterazione di riscaldamento DDP a effetto nullo prima del 545**.
  L'esito decisivo e' l'identita' byte per byte con il checkpoint finale del continuo `4743abaf…`, oppure la sua assenza.
  Costo stimato 0,25-0,3 GPU-ore (una ripresa misurata 205-265 s su 4 GPU). Richiede codice nuovo nell'inventario,
  quindi una review e una release. [proposta]

## 1. In quale punto viene campionato `has_rebuilt_buckets`, e quale riduzione differisce

### Codice che scrive la traccia
`scripts/contrastive_target_resume.py:189-201` (`observe_backward`): chiama `accumulate(...)`, cioe' il forward DDP e il
backward dell'update, e solo **dopo** legge `model._get_ddp_logging_data()` e scrive `iteration, has_rebuilt_buckets,
bucket_sizes, rebuilt_bucket_sizes, rebuilt_per_bucket_param_indices` (righe 194-200). Il campione cade quindi dopo
il forward e il backward dell'update.

### Che cosa contiene quel campo (sorgente Python di torch, 2.14 sul Mac)
`torch/nn/parallel/distributed.py:1755-1778` (`_pre_forward`) esegue, nell'ordine:
1. `self.logger.set_runtime_stats_and_log()` (riga 1758): copia nel logger lo stato del reducer **in quel momento**, compreso `has_rebuilt_bucket_`;
2. `self.reducer.prepare_for_forward()`;
3. `self.reducer._rebuild_buckets()` (riga 1776): ricostruisce i bucket, una sola volta, con l'ordine di prontezza dei gradienti registrato nel backward precedente.

Il valore registrato descrive quindi lo stato **prima** dell'eventuale ricostruzione fatta nello stesso forward. La
riduzione del backward che segue usa gia' i bucket nuovi. Il logger campiona solo nelle prime iterazioni (fino a 10) e
poi ogni 100. Lo mostrano le tracce: nelle riprese `iteration` sale 0..10 e poi resta a 10; nel continuo vale sempre 500.

Il layout iniziale: `distributed.py:150-159`. Con `find_unused_parameters=False` e senza static graph,
`bucket_size_limits = [sys.maxsize]`, quindi tutti i parametri dello stesso tipo stanno in un solo bucket fino alla
ricostruzione. Il trainer crea il DDP con `find_unused_parameters` falso, se non ci sono teste RVQ o keep
(`src/wearusfm/training/run.py:430-438`). Il wrapper si crea una volta per processo (`run.py:1469`), quindi da capo in
ogni ripresa.

### Prova empirica (torch 2.14, CPU, gloo, 2 processi)
`agents/d-ddp/ddp_rebuild_probe.py`. Il probe campiona dopo il backward come la traccia del progetto e usa un comm hook
che registra i bucket effettivamente ridotti. Output in `ddp_rebuild_probe.out`:

```
loop_iter 0: log_iteration 0, log_has_rebuilt null, buckets ridotti [[0, 8840]]          <- layout iniziale (1 bucket)
loop_iter 1: py_has_rebuilt_after_fwd true, log_iteration 1, log_has_rebuilt 0, buckets ridotti [[0,4680],[1,4160]]  <- gia' ricostruiti
loop_iter 2: log_has_rebuilt 1, rebuilt_sizes "18720, 16640", buckets ridotti [[0,4680],[1,4160]]
```

Il log riporta `has_rebuilt_buckets=0` all'iterazione 1, mentre il bucket ridotto in quella iterazione e' gia' quello
ricostruito. Lo riporta a 1 solo all'iterazione 2.

### Tracce di Leonardo (torch `2.2.0a0+git6c8c5ad`, da `reloads_execution_20261121/CN_A/numeric_profile.rank0.json`, `flags.torch_version`)
Tutti i rank delle riprese CN_A, CN_B, CR_A e CR_B hanno lo stesso schema (`agents/d-ddp/ddp_lines.py`):

| update | iteration | has_rebuilt_buckets | rebuilt (hash di sizes+indici) |
|---|---|---|---|
| 545 | 0 | assente | assente |
| 546 | 1 | 0 | assente |
| 547 | 2 | 1 | `c946e781` |
| 548-576 | 3..10 | 1 | `c946e781` |

Nel continuo, su ogni rank e a ogni update 545-576: `iteration 500`, `has_rebuilt_buckets 1`, rebuilt `c946e781`
(stessi 7 bucket `3241700, 26631168, 27800064, 28399104, 27912144, 68480, 218112` e stessi indici).
`bucket_sizes` vale sempre `114202292, 68480` anche nel continuo: e' il layout di costruzione, non quello in uso.

Lo schema 2.2 e' identico a quello del probe 2.14: assente, 0, 1. Quindi:
- **update 545** (iterazione 0 del nuovo DDP): riduzione con il layout iniziale, diverso dal continuo;
- **update 546** (iterazione 1): `_rebuild_buckets` all'inizio del forward; riduzione con i 7 bucket, **uguale al continuo**;
- dal 547 il log mostra soltanto il nuovo stato, in ritardo.

Il trainer fa un solo forward sincronizzato per update. Con la contrastiva `accum_steps` deve valere 1
(`run.py:742-744`); il `no_sync` serve solo ai microbatch non finali (`run.py:759-761`). Lo conferma la traccia:
`iteration` = update − 545 nelle riprese. [verificato]

**Conclusione 1.** Differisce solo la riduzione del 545. «545-546» (prompt di GPT) e «dal 547»
(`docs/reviews/contrastive_targets_resume_science_20261121.md:28` e la nota precedente di Claude) leggono il campo in
ritardo. Anche la tabella `readout_execution_20261121/ddp_trace_readout.json` (`rows[*].differences`,
`differing_update_ids: [545, 546]`) descrive il log e non la riduzione effettiva. [verificato per torch 2.14 con il probe;
per 2.2 dedotto dall'identita' dello schema di log e dal sorgente]

### Riscontro numerico indipendente negli artefatti esistenti (`agents/d-ddp/fields.py`, `gradnorm.py`, `ab.py`)
Continuo e ripresa A, campi numerici di `metrics.jsonl` diversi (A e B sono identici per tutti i campi di gradiente):

- **CN 545**: differiscono `grad_norm` (rel. 2,20e-12), `grad_clip_factor` (2,20e-12) e `contrast_projector_grad_norm`
  (1,02e-13; norma fp64 di `p.grad` del proiettore dopo la riduzione DDP, `run.py:1691-1692`). Sono **identici**
  `contrast_global_grad_base`, `contrast_global_grad_nce`, `contrast_global_grad_rnc` e i coseni. Questi provengono da
  `autograd.grad` e da un `all_reduce` esplicito di un vettore concatenato in ordine fisso dei parametri
  (`src/wearusfm/training/contrastive_objectives.py:415-439`), quindi indipendente dai bucket DDP.
- **CR 545**: differiscono `grad_norm` e `grad_clip_factor` (3,40e-13); `contrast_global_grad_*` identici.
- **CN 546**: `grad_norm`, `grad_clip_factor`, `contrast_projector_grad_norm` e `contrast_global_grad_*` identici;
  restano solo i coseni a 5e-16 e le loss di log (vedi sotto).
- **CR 546**: `grad_norm` a 3,43e-06 e `contrast_global_grad_nce` a 1,58e-05. La differenza e' gia' a monte della
  riduzione e viene dallo stato dopo il 545, non dal layout.
- Le loss di log (`jepa`, `log_env`, `log_rms`, `ms_fast`, `ms_slow`, `band_shape`, `total`) differiscono di 1e-8...4e-7
  **anche fra A e B**, a ogni update, con gradienti e checkpoint identici: circa 7 campi per update, 219 (CN) e 213 (CR)
  differenze in 32 update (`ab.py`). E' rumore del solo calcolo delle loss registrate e non entra nei gradienti, quindi
  non serve come prova di divergenza.
- Le tracce per update (batch, input, RNG prima e dopo, coda, copertura e confine a 544) coincidono fra continuo e
  ripresa in 4 rank × 2 braccia × 2 riprese (`agents/d-ddp/trace_eq.py`: `diffs none`, `boundary_eq True`).

Lettura (dedotta): con stato, input e RNG identici, i gradienti locali del 545 coincidono. La riduzione esplicita a layout
fisso lo conferma con norme fp64 identiche. La prima differenza compare **dopo la riduzione DDP**, l'unica operazione il
cui layout cambia. Il 546 di CN, a layout di nuovo uguale, ridà un gradiente ridotto con norme identiche.
Questo osservabile esclude un difetto di ripristino che alteri forward o backward del 545. Non esclude da solo un difetto
che agisca solo nell'update di AdamW. Il boundary verificato a 544 copre momenti, gruppi e `step`
(`contrastive_target_resume.py:181-187`), quindi un tale difetto dovrebbe stare fuori dallo stato salvato.

## 2. Effetto numerico di una diversa composizione dei bucket

- DDP moltiplica ogni gradiente per 1/world_size (0,25, esatto) e poi fa un all-reduce SUM del buffer del bucket. I
  gradienti sono fp32: i bucket sono in byte e `114202292/4` e' intero.
- NCCL ring a 4 rank: il buffer si divide in canali e chunk. Ogni elemento si somma in sequenza lungo l'anello, partendo
  da un rank che dipende dal chunk che lo contiene. Dimensione del chunk e numero di canali dipendono dalla dimensione
  del messaggio e dal protocollo; NCCL puo' anche scegliere tree o ring in base alla dimensione. Spostare un parametro
  da un buffer da 114 MB all'inizio di un buffer da 3,2 MB (i parametri 266-269 aprono il primo bucket ricostruito:
  `rebuilt_per_bucket_param_indices` comincia con `268 269 266 267`) cambia, per buona parte degli elementi, l'ordine
  delle 4 somme fp32. Ne segue una differenza di circa 1 ulp sugli elementi interessati, maggiore dove c'e' cancellazione.
  [dedotto dal funzionamento noto di NCCL; non verificabile sul Mac]
- Simulazione CPU (`agents/d-ddp/ring_order_sim.py`, 4e6 elementi a code pesanti, due assegnazioni dell'inizio
  dell'anello): differisce il 27,6% degli elementi; ||Δ||/||g|| = 4,2e-8; variazione relativa della norma globale in
  fp64 = 5,0e-11, che scala circa come 1/√N. Il trainer usa circa 2,86e7 elementi e norme per tensore in fp32 combinate in
  fp64 (vedi sotto): l'ordine atteso e' 1e-12...1e-11. Osservato 2,2e-12 (CN) e 3,4e-13 (CR): **compatibile**. [dedotto]
- `grad_norm` non e' un numero fp32 (`4.336711902204087` non e' rappresentabile in fp32; verificato con numpy). La
  differenza assoluta a 545 in CN e' 9,54e-12, sotto l'ulp fp32 della norma (4,8e-7). La norma totale combina in fp64
  norme per tensore: c'e' anche un bucket di un altro tipo, `68480` byte, e `torch.stack` promuove. Una variazione di 1
  ulp nella norma fp32 di qualche tensore produce scarti relativi di questa misura. [verificato il formato; dedotta la causa della promozione]
- CN 546, zero esatto: **compatibile** con un layout uguale al 546 e con parametri dopo il 545 identici, o diversi solo
  sotto la risoluzione bf16 dei pesi convertiti dall'autocast. A 545 il gradiente differisce di circa 1 ulp; l'update
  AdamW, con lr per gradiente normalizzato molto piccolo rispetto al peso, sposta il peso fp32 di quantita' che quasi
  sempre si arrotondano allo stesso valore. I momenti `exp_avg`/`exp_avg_sq`, che hanno la scala del gradiente,
  conservano invece la differenza. CR 546 a 3,4e-6: compatibile con qualche peso che, dopo il 545, attraversa un confine
  di arrotondamento fp32 o bf16. Da li' la divergenza cresce come in un sistema caotico (CN 547: 2,2e-5; poi 1e-4...1e-3).
  [dedotto]

## 3. Controllo d'intervento piu' economico

### Proposta I1 (consigliata): ripresa con riscaldamento DDP a effetto nullo
Nella ripresa, prima del calcolo del confine e degli hash RNG «before» del 545, cioe' all'inizio di `observe_backward`
quando `runtime.step == fork_step` (`contrastive_target_resume.py:168-187`):
1. si salvano RNG (cpu, cuda, python, numpy) con `rng_state`, gia' in `scripts/contrastive_target_parity.py:41-53`;
2. si esegue un forward e backward **con il DDP vivo** e il batch del 545, ma con un runtime contrastivo clonato,
   costruito come in `_measure_case` (`contrastive_target_parity.py:98-120`: `load_state_dict` del deepcopy, poi `prepare`).
   Coda, copertura e contatori del runtime vivo restano cosi' intatti;
3. `opt.zero_grad(set_to_none=True)`, senza `opt.step()`; si ripristinano gli RNG;
4. il percorso registrato prosegue invariato. Il confine e gli hash RNG gia' scritti nella traccia fanno da controllo del
   ripristino: devono coincidere con quelli del continuo. Lo fa gia' l'osservatore della parita' con
   `live_state_restored` (`contrastive_target_parity.py:290-296`).

Effetto atteso: il reducer registra l'ordine di prontezza al riscaldamento. Il 545 ricostruisce i bucket all'inizio del
forward e riduce con i 7 bucket del continuo. Si verifica che `rebuilt_per_bucket_param_indices` abbia l'hash
`c946e781` e si registra anche il flag Python `model._has_rebuilt_buckets`, che non e' in ritardo.

Esito che dimostrerebbe H2 come causa sufficiente:
- a 545 `grad_norm` = `4.336711902204087` e `contrast_projector_grad_norm` uguale al continuo, bit per bit;
- **stato finale a 576 identico byte per byte al continuo `4743abaff631d43c0f9267b886087fdcbefdfc156332bf34d43b668297668dec`**,
  con lo stesso hash del readout, e diverso dalle riprese `bb39f339…`.

E' un test a un lato: un ripristino incompleto puo' solo rompere l'identita', non crearla per caso. Esiti alternativi:
- 545 identico e 576 diverso: altra dipendenza dall'eta' del processo dopo il 545 (allocatore, comunicatore). Va indagata
  e H1 resta aperta;
- 545 ancora diverso con trace e hash identici: il layout non e' la causa o non e' stato pareggiato. Lo si distingue con
  l'hash dei bucket ricostruiti.

Rischi in questo codice:
- RNG o runtime consumati dal riscaldamento: coperti dal ripristino e dal confronto degli hash della traccia;
- memoria: un forward/backward in piu', come uno step normale;
- l'ordine di prontezza del riscaldamento deve coincidere con quello registrato al passo 1 del continuo. Lo e' gia' nella
  ripresa reale (`c946e781` identico), quindi e' atteso, e la traccia lo verifica;
- broadcast dei buffer nel forward DDP: valori uguali fra rank, nessun effetto atteso;
- `deterministic_algorithms` e' falso (`numeric_profile.flags`), ma A e B identici byte per byte mostrano che la parte
  di gradiente e' ripetibile.

Costo: una ripresa CN, 205-265 s su 4 GPU misurati (`reloads_execution_20261121/allocation.sacct.tsv`), circa
0,25-0,3 GPU-ore; tetto prudente 0,5 GPU-ore. Il confronto si fa con un job CPU come quello del readout. Inventario: la
modifica tocca `scripts/contrastive_target_resume.py` o un nuovo wrapper che lo importa, quindi servono nuova review e
nuova release del digest. Non e' un parametro di configurazione: in torch 2.2 non esiste un'API per impostare il layout
dei bucket. `bucket_cap_mb` non agisce sul layout iniziale (`sys.maxsize`). `static_graph=True` ricostruisce comunque
dopo la prima iterazione.

### Proposta I2 (alternativa simmetrica, piu' cara): ricreare il DDP nel continuo subito dopo il salvataggio del fork
Si fa `del` del vecchio wrapper e si chiama `_ddp_wrap` di nuovo: il distruttore del Reducer rimuove i suoi hook e il
costruttore risincronizza i parametri, uguali, dal rank 0. Il 545 del continuo usa cosi' il layout iniziale come le riprese.
Esito decisivo: continuo finale = `bb39f339…`. Non consuma RNG e non richiede ripristini, ma costa un continuo intero
(784 s × 4 GPU, circa 0,87 GPU-ore) e tocca `run.py:1469`, nel cuore del trainer. Utile come seconda prova solo se I1 fosse
ambiguo: le due insieme separano il layout dall'eta' del processo.

### Sulla proposta di GPT (gradienti locali, ridotti, clipping e momenti ai passi 545-547)
E' coerente, ma il risultato del 545 e' gia' previsto dagli artefatti: gradienti locali uguali e prima differenza dopo
la riduzione. Al 546-547 non c'e' alcuna differenza di riduzione da misurare. Per osservare senza perturbare, un comm
hook che clona il buffer del bucket prima e dopo e poi esegue la stessa divisione esatta e lo stesso `all_reduce` dello
stesso buffer non cambia l'aritmetica. Va pero' registrato prima della prima iterazione di entrambi i percorsi, quindi
il continuo va rifatto da 0. La prova di non perturbazione e' che il continuo strumentato ridia `4743abaf…` e la ripresa
strumentata `bb39f339…`. Costa circa 4 volte I1 e non da' un esito binario sulla sufficienza di H2.

## 4. Correzione strutturale per il futuro (non implementata)
Obiettivo: una riduzione che non dipenda da quante iterazioni ha gia' fatto il processo.
- **(a) Riduzione canonica esplicita.** Un comm hook che non riduce (restituisce il bucket) oppure `no_sync` su tutto, poi
  dopo il backward un solo `all_reduce` di un buffer piatto in ordine fisso dei parametri e di dimensione fissa, come gia'
  fa `global_gradients`. Costo: si perde la sovrapposizione fra comunicazione e calcolo per circa 114 MB fp32 a step,
  pochi ms su NVLink contro circa 1,36 s/step (784 s/576), sotto l'1%; piu' 114 MB di buffer. Per renderla indipendente
  anche dalle scelte di NCCL (algoritmo e canali, che dipendono dalla topologia) si fissano `NCCL_ALGO`/`NCCL_PROTO`
  oppure si usa `all_gather` piu' una somma locale in ordine fisso dei rank: traffico 4 volte, alcune decine di ms, ancora
  piccolo.
- **(b) Riscaldamento a effetto nullo a ogni avvio** (training nuovo e ripresa), come in I1: un forward/backward per
  processo, ma piu' codice di ripristino da mantenere.
- Non bastano `static_graph=True` (ricostruisce comunque) e `find_unused_parameters=True` (cambia il layout iniziale e
  blocca la ricostruzione, con costo per passo).

Tutte queste soluzioni cambiano i numeri del trainer a livello di ulp rispetto alle campagne storiche. Richiedono nuova
release e test d'uguaglianza a tolleranza, non bit per bit, con lo storico.

## Divergenze dalla lettura di GPT
1. «Ai passi 545-546 ... raggruppano diversamente» (prompt) e «dal 547» (`docs/reviews/...:28`): differisce solo la riduzione del 545; il campo di log e' in ritardo di un update.
2. «Scarto sulla norma totale del gradiente InfoNCE circa 9,5e-12» (`docs/reviews/...:28`): 9,54e-12 e' la differenza
   assoluta di `grad_norm`, la norma del gradiente totale dopo DDP e clipping, non della sola InfoNCE.
   `contrast_global_grad_nce` a 545 e' identico.
3. GPT considera aperta la localizzazione e chiede di misurare i gradienti locali. Gli artefatti esistenti
   (`contrast_global_grad_*` identici, `contrast_projector_grad_norm` diverso a 545 e identico a 546 in CN) la danno gia'
   in forma indiretta. Resta da dimostrare la **sufficienza** del layout, e per questo I1 e' piu' economico e decisivo.

## Punti aperti
- La semantica 2.2 del Reducer C++ non e' letta nel sorgente: e' dedotta dal Python 2.14 e dall'identita' dello schema di
  log con le tracce 2.2.
- La scelta dell'algoritmo NCCL su Leonardo (ring/tree, canali) non e' nota dagli artefatti.
- Non e' verificato che i parametri dopo il 545 in CN siano bit-identici: e' solo compatibile con le norme del 546.
- Che cosa calcola esattamente l'hash `bb39f339…`/`4743abaf…` (file o stato canonico) va riletto in
  `reloads_execution_20261121/collection_readout.json` prima di fissare il criterio di I1.
