# Compito E: valutazione della prova GPU proposta da GPT (chiave e-prova-gpt)

Clone `codex/contrastive-target-windows-20261118` a `918245b6cee89fb71e159c0fe17f4a7c2e746fa7` (verificato con `git log -1`).
Rapporto originale: `data/preflight_readout.json` (sha256 d4300a2e...). Nessun job, nessuna rete, nessuna modifica al clone.
Abbreviazioni: E = `results/infra/contrastive_targets_20261118`; C = continuo, A/B = riprese.

## 0. Verdetto in breve

- Gli artefatti esistenti bastano per escludere in pratica un difetto grossolano di ripristino (H1) e mostrano che il
  fallimento dipende dal criterio (H3) applicato a una divergenza caotica partita da uno scarto di arrotondamento al
  passo 545 (H2). La prova GPU di GPT localizzerebbe solo il seme di 545 (locale o riduzione): utile per la causa, **non
  decisiva** per la scelta di Simone, e non dice nulla su H3.
- Il criterio di GPT "una differenza gia' locale sposta l'indagine a monte" vale **solo al passo 545**. Al 546 vale solo
  se pesi e momenti in ingresso risultano identici bit per bit; al 547 non vale piu'.
- Divergenza da GPT (dedotta, forte): nella ripresa il raggruppamento DDP diverso riguarda **solo il passo 545**, non
  545-546. Il campo `has_rebuilt_buckets: 0` letto al 546 e' un'istantanea presa prima della ricostruzione nella stessa
  forward. Le metriche CN al 546 sono identiche bit per bit al continuo.
- Primo controllo consigliato: lettura CPU (job seriale, 0 GPU-ore) dei checkpoint 544/576 per i momenti, piu' le
  tabelle gia' presenti nel JSON. Se Simone vuole la causa, conviene la variante d'intervento (riduzione a ordine fisso)
  oppure la prova di GPT con hook di comunicazione e identita' dei byte come validazione. Il costo dichiarato da GPT
  (1,11-1,17 GPU-ore) e' corretto. Il tetto di 2 GPU-ore vale solo se le due fasi stanno in un solo job da 30 minuti.

## 1. Come il trainer fa backward, riduzione, clipping e step (verificato sul codice)

- DDP: `src/wearusfm/training/run.py:430-438`, `DistributedDataParallel(_Objective(student), device_ids=..., find_unused_parameters=unused)`,
  con `unused = rvq_codes or keep_readout`. Nessun `bucket_cap_mb`, nessun comm hook, nessun `static_graph`.
- Backward: `run.py:761` (`no_sync` su tutti i micro tranne l'ultimo; qui `accum_steps 1`, da config.json) e `run.py:798` `loss.backward()`.
  Allo stesso punto il ramo congiunto chiama `losses["total"].backward()` (`run.py:1687`).
- Norma del proiettore prima del clipping, in float64, sui gradienti gia' ridotti: `run.py:1689-1693` (`contrast_projector_grad_norm`).
- Clipping: `run.py:1700`, `clip_grad_norm_(student.parameters(), cfg.grad_clip)` con `grad_clip 1.0`. Step: `opt.step()` a `run.py:1709`.
  Poi EMA (`run.py:1720`) e `contrastive.commit()`.
- `grad_clip_factor = min(1, clip/max(gnorm,1e-12))` (`run.py:1735`): e' una funzione di `grad_norm`, quindi non porta
  informazione propria (scarti relativi uguali e opposti, vedi sotto).
- Front-end in float64 (docstring `run.py:12`). I gruppi DDP sono due nel layout iniziale e sette in quello ricostruito.
  Il gruppo da 68.480 byte (8.560 double) e' identico nei due layout (vedi §2).
- Contesto numerico (`E/benchmark_execution_20261121/CN/numeric_profile.rank0.json`): torch `2.2.0a0+git6c8c5ad`, CUDA 12.1,
  `deterministic_algorithms false`, `cudnn_benchmark false`, `matmul_tf32 false`, `bf16_reduced_precision_reduction true`;
  config: `amp_bf16 true`, `model.grad_checkpoint true`.

### Osservatore esistente

`scripts/contrastive_target_resume.py:148-208` applica patch a `AdamW.__init__`, `ContrastiveRuntime.prepare` e
`trainer._accumulate_terms`. Scrive hash di batch, ingressi e RNG prima e dopo, e al passo del fork gli hash del confine.
**Dopo** `_accumulate_terms`, quindi dopo il backward e prima del clipping, legge `model._get_ddp_logging_data()` (righe 194-202).
Non osserva gradienti, clipping o step. La strumentazione richiesta va aggiunta nello stesso punto (dopo l'accumulo:
gradienti ridotti), dopo `clip_grad_norm_` e dopo `opt.step()`. Il trainer non ha un punto di patch fra clip e step:
`clip_grad_norm_` e `AdamW.step` si possono avvolgere con `patch.object`, come gia' si fa per `AdamW.__init__`.

## 2. Il raggruppamento DDP: solo il 545 e' diverso (dedotto, divergenza da GPT)

Fatti (verificati, `trace-0.ddp.jsonl`):
- C (`E/benchmark_execution_20261121/CN/trace-0.ddp.jsonl`), passi 545-548: `has_rebuilt_buckets 1`, sette gruppi
  `3241700, 26631168, 27800064, 28399104, 27912144, 68480, 218112`. Il primo gruppo inizia con gli indici `268 269 266 267`.
- CN_A, passo 545: `iteration 0`, senza `has_rebuilt_buckets`, `bucket_sizes "114202292, 68480"`. Passo 546: `has_rebuilt_buckets 0`.
  Passo 547: ricostruito, stessa composizione del continuo. CR_A si comporta allo stesso modo.
- Somma dei gruppi ricostruiti = 114.270.772 = 114.202.292 + 68.480 (calcolo mio). Lo stesso vettore viene quindi
  ridotto prima come un'unica allreduce float32 da circa 114 MB, poi in cinque gruppi float32 da 3-28 MB.

Lettura dei tempi (dedotta): in torch, `DistributedDataParallel._pre_forward` chiama
`logger.set_runtime_stats_and_log()` **prima** di `reducer._rebuild_buckets()` nella stessa forward. L'ho verificato
nel torch locale 2.14 (`distributed.py:1755-1776`); per 2.2 lo presumo, perche' questa parte e' stabile, ma non l'ho
verificato. `_get_ddp_logging_data()` letto dopo il backward dell'update N riporta dunque lo stato registrato
all'inizio della forward di N, prima della ricostruzione. Ne segue che la forward del 546 ricostruisce i gruppi e il
backward del 546 usa gia' il layout del continuo.

Riscontro indipendente nelle metriche (verificato, `metrics.jsonl`; script in linea, output qui sotto):

| campo (CN, C vs A) | 545 | 546 | 547 |
|---|---|---|---|
| grad_norm | 4.336711902204087 vs 4.336711902194546 (rel −2,2e-12) | identici (9.023137809933427) | rel +2,2e-5 |
| contrast_projector_grad_norm (float64) | 0.13962099177834264 vs 0.1396209917783284 (rel −1,0e-13) | identici (0.21809202574133565) | rel +6,5e-5 |
| contrast_nce | identici | identici | rel +2,4e-6 |
| contrast_global_grad_nce/base/rnc | identici | identici | diversi |

Al 546 la norma float64 dei gradienti ridotti del proiettore coincide bit per bit, mentre al 545 differisce. Se al 546 la
riduzione avesse usato ancora il gruppo unico, ci si aspetterebbero scarti di ulp come al 545. Questo sostiene: un solo
passo con raggruppamento diverso (545). Nel braccio CR il 546 differisce gia' (`grad_norm` rel −3,4e-6, `contrast_nce`
−2,2e-6): la forward al 546 vede pesi diversi, quindi CR non permette questa verifica.

Campi non affidabili per il confronto: `band_shape`, `log_rms`, `ms_fast`, `ms_slow`, `log_env`, `jepa`, `total` e, a tratti,
`contrast_rnc` differiscono **fra A e B** in 29-32 passi su 32, benche' i checkpoint di A e B siano identici. Per CN
l'output e' `{'band_shape': 32, 'contrast_rnc': 2, 'jepa': 29, 'log_env': 32, 'log_rms': 32, 'ms_fast': 29, 'ms_slow': 31, 'total': 32}`.
Sono valori di perdita calcolati con riduzioni non deterministiche (`deterministic_algorithms false`) che non entrano nel
gradiente. Restano affidabili, cioe' uguali fra A e B in tutti i 32 passi: `grad_norm`, `grad_clip_factor`,
`contrast_nce`, `contrast_global_grad_*` e `contrast_projector_grad_norm`. Ne segue anche che l'identita' A/B non prova
il determinismo di tutti i kernel, ma solo del percorso del gradiente nei processi nuovi (dedotto).

## 3. Cosa mostrerebbe verosimilmente la prova a 545, 546 e 547

Previsione sotto H2-riduzione (dedotta dalle metriche sopra):
- **545**: gradienti locali di ogni rank identici bit per bit fra C e ripresa, per tutti i 270 parametri. Ridotti
  diversi di 1-2 ulp in una parte degli elementi float32; il gruppo float64 identico. Fattore di clipping diverso di circa
  2e-12 in relativo. Gradienti dopo il clipping e `exp_avg`/`exp_avg_sq` dopo AdamW diversi di ulp. Pesi dopo il 545
  identici o diversi in pochissimi elementi di 1 ulp float32: lo scarto dell'aggiornamento, circa lr·1e-7 = 3e-12, sta
  sotto l'ulp dei pesi.
- **546**: in CN, locali e ridotti identici, come richiedono le metriche uguali bit per bit; norma e clipping identici.
  `exp_avg` resta diverso perche' porta lo scarto del 545 (β1=0,9). I pesi dopo il 546 differiscono in alcuni elementi.
- **547**: tutto diverso, a partire dalla forward (`contrast_nce` rel 2,4e-6). Gli scarti locali sono qui conseguenza
  degli aggiornamenti precedenti, non indicazione di una causa a monte.

Esiti e ipotesi:
| esito a 545 | lettura |
|---|---|
| locali identici, ridotti diversi di poche ulp, layout diverso | H2-riduzione confermata osservativamente (non causalmente) |
| locali gia' diversi con stato al confine identico (hash gia' verificati) | seme numerico a monte, legato alla storia del processo: primo uso dei kernel in un processo nuovo, scelta degli algoritmi, inizializzazione pigra di cuBLAS/NCCL, ricalcolo del checkpointing. Resta H2 in senso largo. H1 solo se lo scarto ha una grandezza non da arrotondamento, per esempio ≥1e-6 relativo su un tensore |
| tutto identico a 545-547 ma checkpoint 576 diverso | incoerente con le metriche esistenti: la strumentazione ha cambiato il percorso. Inconclusiva |
| checkpoint strumentati diversi da 4743abaf... / bb39f339... | la prova non riproduce il fenomeno originale. Inconclusiva sull'originale |

**H3 non e' toccata dalla prova**: riguarda denominatore e soglia, si giudica dagli artefatti (§4).
Il criterio di GPT ("una differenza gia' locale sposta a monte") vale quindi per il **solo 545**. Per il 546 vale solo
verificando che pesi, momenti e buffer in ingresso siano identici bit per bit, da registrare come hash; per il 547 no.
Al 545 contano soprattutto il confronto fra i ridotti e il layout (offset di ogni parametro nel gruppo). Il fattore di
clipping e' ridondante con la norma.

## 4. Le prove gia' disponibili su H3 (verificate sul JSON)

Da `arms.CN.resume.pairs.continuous__reload_a.floating_tensor_relative_to_left` (6.520 tensori):
- CN `exp_avg` (270): mediana 0,461%, p90 0,588%, massimo 1,316%. I primi quattro sono 268 (1,316), 266 (1,222),
  269 (1,108) e 267 (1,034); seguono 194 (0,714), 158 (0,704), 212 (0,702) e 176 (0,696).
- CR `exp_avg`: mediana 0,288%, p90 0,403%, massimo 0,591% (tensore 47).
- `exp_avg_sq`: CN massimo 0,168% (proiettore), CR massimo 0,034%.
- Chiavi della coda: CN fino a 0,919%, CR fino a 0,604%.
- Blocchi dello studente rispetto allo spostamento dal 544 (`by_block_relative_to_left_displacement_from_fork.student`), CN:
  `contrastive_head` 0,932%, `local` 1,027%, `pool` 1,047%, `backbone` 0,612%. **I pesi del proiettore non sono anomali**;
  lo sono solo i loro primi momenti. In CR `contrastive_head` vale 0, perche' il proiettore non viene usato.

Lettura (dedotta): la divergenza e' diffusa su tutti i 270 tensori in entrambi i bracci; in CN e' circa 1,6 volte quella
di CR (mediane 0,46% contro 0,29%). La soglia dell'1% coglie solo la coda: i quattro primi momenti del proiettore stanno
a circa 2-3 volte la mediana. Un `exp_avg` e' una media dei gradienti con segno su circa 10 passi; la sua norma si
riduce per le cancellazioni, quindi il rapporto scarto/norma cresce rispetto a quello dei gradienti. Questo e' coerente
con H3 e con un seme di arrotondamento amplificato in modo caotico (H2). Non indica uno stato mancante.

## 5. Strumentazione senza alterare il fenomeno

Opzioni per i gradienti locali prima della riduzione:
1. **Comm hook di DDP** (`model.register_comm_hook`) che clona `bucket.buffer()` e chiama poi la stessa allreduce del
   default (`default_hooks.allreduce_hook`: divisione per world size e `all_reduce`). Vantaggi: e' esattamente il vettore
   ridotto, si ottengono anche `bucket.parameters()` e gli offset per mappare il layout, e la futura del risultato da' il
   ridotto. Rischi: il percorso Python sostituisce quello C++ del reducer senza hook. Il default C++ moltiplica per 1/4
   alla copia nel gruppo, l'hook divide per 4: con 4 rank sono operazioni esatte salvo subnormali, quindi il risultato
   dovrebbe restare bit per bit uguale, ma va verificato. L'hook va registrato prima della prima forward e in entrambe
   le fasi. Per evitare sincronizzazioni dentro il backward, clonare sul dispositivo e copiare su CPU dopo lo step.
2. **Hook sui tensori dei parametri** (`p.register_hook`), che scattano prima di AccumulateGrad; gli hook del reducer
   DDP sono post-hook di AccumulateGrad, quindi l'ordine non cambia. Forniscono il gradiente locale non diviso. Rischi:
   allocazioni aggiuntive (114 MB per rank per passo) e kernel di copia sullo stream. Le scelte degli algoritmi
   cuBLAS/cuDNN, con `cudnn_benchmark false`, non dipendono dalla memoria libera salvo una mancanza di workspace,
   improbabile su 64 GB. Nessun consumo di RNG.
3. **Replica separata senza DDP**: sconsigliata. Una seconda forward/backward consuma RNG, se non con `fork_rng` anche
   CUDA, usa allocazioni e kernel diversi e non garantisce gradienti uguali bit per bit a quelli del DDP.

Attivare la strumentazione solo per gli update 545-547, condizionata al passo; registrare hash per parametro e copie
complete solo dei 4 tensori del proiettore e del vettore float32 di un rank. Non cambiare variabili d'ambiente NCCL
(niente `NCCL_ALGO` o `NCCL_PROTO`; `NCCL_DEBUG` solo se gia' presente).

Verifica di non alterazione (criteri da fissare prima):
- sha256 del checkpoint finale del continuo strumentato = `4743abaff631d43c0f9267b886087fdcbefdfc156332bf34d43b668297668dec`
  e della ripresa = `bb39f33998df27c2a0b7b6ca7554c77afccf08bb7405831a17a28c4ef6594301` (dal JSON,
  `arms.CN.resume.pairs.continuous__reload_a.final_checkpoint_sha256`);
- `trace-*.jsonl` e `trace-*.ddp.jsonl` identici a quelli archiviati;
- campi affidabili di `metrics.jsonl` (§2) identici in tutti i passi. Non usare i campi di perdita non deterministici.
Condizione: stesso checkout, stessa immagine software e stesso tipo di nodo. Non ho verificato se A e B siano girati
sullo stesso nodo (il nodo non compare nei file sacct): e' un punto aperto.

## 6. Costi (verificati sui file sacct)

`E/benchmark_execution_20261121/allocation.sacct.tsv`: CN = 65017310, ElapsedRaw 784, 4 GPU, 32 CPU, Timelimit 30 min.
`E/reloads_execution_20261121/allocation.sacct.tsv`: CN_A = 65033016 (265 s), CN_B = 65033756 (213 s); CR_A 205 s e CR_B 210 s.
L'associazione degli ID ai bracci viene dai file `*_submission.txt`.
- Continuo + ripresa: 997-1.049 s × 4 GPU = **1,108-1,166 GPU-ore**; × 32 CPU = 8,86-9,32 core-ore. I conti di GPT sono corretti.
- Tetto di 2 GPU-ore = 30 min × 4 GPU: vale per **un solo job** che esegua in serie continuo e ripresa (circa 17,5 min piu'
  la strumentazione, quindi ci sta). Con due job separati da 30 minuti, come nel disegno eseguito, il preventivo secondo
  la regola GPU × walltime massimo e' di **4 GPU-ore**, oppure 2,67 con la ripresa limitata a 10 minuti. Il tetto di
  consumo di 2 resta realistico.
- Durata dei passi: `t_step_s` 2,2-3,6 s nei passi 545-551 del continuo; la somma dei passi 545-576 e' 79,7 s. Primo passo della
  ripresa 23,5 s: inizializzazione pigra. Overhead stimato della strumentazione su 3 passi: copie GPU di 114 MB per stadio,
  hash sha256 su CPU (circa 0,1-0,2 s per 114 MB) e scrittura di circa 1-1,5 GB per rank per fase. In tutto **< 30 s per fase,
  sotto il 3%** (stima, da misurare).

## 7. Alternative e raccomandazione

- (a) **Lettura CPU dei checkpoint** C/A a 576 e del fork a 544, con un job seriale su Leonardo, 0 GPU-ore e circa 0,3
  core-ore come la raccolta precedente. Per i 4 tensori e per tutti i 270: ‖m_C‖, ‖m_A‖, ‖Δm‖, coseno, ‖m_576 − m_544‖,
  piu' gli stessi numeri per CR. Risponde alla rilevanza pratica (coseno atteso ≥0,9999 se ‖Δm‖/‖m‖≈1,3%) e a H3. Una regola
  basata sullo spostamento dal 544 sarebbe **post hoc** e va dichiarata tale.
- (b) **Metriche e JSON gia' archiviati**: fatto qui (§2, §4). Seme di grandezza da arrotondamento al 545, CN identico al 546,
  divergenza diffusa in entrambi i bracci, proiettore anomalo solo nei momenti.
- (c) **Intervento**: un comm hook a riduzione indipendente dal layout, per esempio somma in float64 dei 4 contributi poi
  arrotondata a float32, oppure all_gather e somma in ordine fisso dei rank, in un continuo e in una ripresa nuovi. Se le
  due fasi risultano identiche bit per bit a 576, H2-riduzione e' provata causalmente e si ottiene anche un rimedio per riprese
  esatte. Se differiscono, il seme e' locale. Costo uguale alla prova di GPT, circa 1,1-1,2 GPU-ore; non riproduce i byte
  originali per costruzione, quindi la validazione e' l'identita' C/ripresa.
- Raccomandazione: prima (a); se Simone vuole la causa, una sola coppia GPU. Va bene la proposta di GPT con comm hook
  osservativo validato sui byte, oppure (c), piu' decisiva sulla causa. La GPU non serve per decidere se tornare ai
  training lunghi, ma solo per capire o eliminare il seme.
- Risultato che convincerebbe a tornare ai training lunghi: dalla lettura (a), coseni ≥0,9999 e ‖Δm‖ dei 4 tensori
  dello stesso ordine della coda della distribuzione dei 270 (gia' 2-3 volte la mediana). Se si fa la GPU: al 545 locali
  identici e primo scarto nei ridotti di poche ulp, con checkpoint strumentati identici ai byte esistenti; oppure in (c)
  identita' bit per bit continuo/ripresa.
