# Compito B (b-stato-ripristino): che cosa si salva, si ripristina e si verifica al fork 544 (ipotesi H1)

Clone `918245b6cee89fb71e159c0fe17f4a7c2e746fa7` (verificato con `git log -1`), sola lettura. Rapporto originale:
`data/preflight_readout.json` (sha256 d4300a2e…, fornito). Script miei, tutti in sola lettura sugli artefatti:
`agents/b-stato-ripristino/{cmp.py,metrics_cmp.py,ab_cmp.py}` (percorsi sotto `.../scratchpad/rev_resume/`).
Legenda: **[V]** verificato (codice/artefatto/comando), **[D]** dedotto.

## Verdetto breve
Gli artefatti escludono un difetto di ripristino dello stato salvato. Tutto cio' che entra nel gradiente di InfoNCE al passo 545
(studente compreso il proiettore, teacher, momenti e passo di AdamW, lr, coda con chiavi e ordine, copertura, RNG CPU/CUDA per rank,
batch e input) coincide con l'impronta del fork su tutti e 4 i rank, nel continuo e nelle due riprese. Inoltre le loss deterministiche
del forward 545 (`contrast_nce`, `contrast_rnc`, `pose`) sono **identiche bit per bit** fra continuo e riprese. La prima grandezza che
diverge e' la **norma del gradiente dopo la riduzione DDP al 545**: 2,2e-12 relativo, cioe' una differenza di arrotondamento. La sola
differenza di stato di processo documentata e' la disposizione dei bucket DDP, che non sta nel checkpoint. H1, nel senso di uno
stato salvato e ricaricato male, non e' sostenuta.

## 1. Inventario salvato / ripristinato / verificato

| Elemento | Salvato | Ricaricato | Verificato al confine 545 |
|---|---|---|---|
| Studente (incl. `contrastive_head`, 4 parametri) | `payload` run.py:1480 | run.py:1433 | impronta `boundary.student`, 4 rank x 3 fasi [V] |
| Teacher EMA | run.py:1480 | run.py:1434, poi broadcast dal rank 0 run.py:1467-1469 | `boundary.teacher` [V] |
| AdamW: `state` (exp_avg, exp_avg_sq, step) | run.py:1480 `opt.state_dict()` | run.py:1435 | `boundary.optimizer_moments` [V]; l'impronta include dtype e forma ma **non il dispositivo** (pairing.py:17-20) |
| AdamW: `param_groups` | idem | idem; lr riscritto ogni passo da `_lr_at(step)` (run.py, ciclo) | `boundary.optimizer_groups_at_next_lr` (target_resume.py:314-319) [V]; `lr` uguale in metrics per 32 passi [V] |
| Stato contrastivo: coda (chiavi, identita', step, global_index, last_step, dtype, feature_dim), copertura, rapporti di copertura, dataset attesi | `ContrastiveRuntime.state_dict` contrastive_objectives.py:176-181, salvato solo dal rank 0 | `load_state_dict` contrastive_objectives.py:183-194, **su ogni rank dallo stesso file** | `boundary.contrastive_state` su ogni rank [V]; ogni passo: `queue_order_before_sha256`, `queue_last_step_before`, `queue_length_before`, `coverage_before_sha256` [V] |
| Coda: per rank o replicata? | Replicata: le chiavi sono raccolte da tutti i rank (`gather_variable`, contrastive_objectives.py:255-257, 270) e ogni rank fa lo stesso `commit`; la copia del rank 0 basta | | l'impronta di confine e' uguale sui 4 rank del continuo, quindi le 4 copie erano identiche gia' prima della ripresa [V] |
| Precisione delle chiavi | float32 (normalize_projection, contrastive.py:66-73), clone CPU (contrastive_memory.py:211), salvate come tensori | dtype controllato al caricamento (contrastive_memory.py:290), riportate a dispositivo/dtype della chiave corrente (riga 220) | entries 0..5069 (CN) e 0..5039 (CR) con errore **esattamente 0** a 576 [V] |
| RNG per rank (CPU e CUDA) | `all_gather_object` in `save` run.py:1593-1597 | run.py:1447-1450 | `rng_{cpu,cuda}_before` al 545 confrontati col fork (target_resume.py:367-371) e `before/after` uguali per 32 passi [V] |
| Posizione dello stream | non serve: `BatchStream(start=step)` run.py (DataLoader), generatore proprio con seme fisso | controllo `batch.index != step` | `batch_sha256` e `input_sha256` uguali per 32 passi e 4 rank [V] |
| Schedule lr / EMA / pesi contrastivi | lr funzione del passo; EMA costante `cfg.jepa.ema_momentum` (jepa.py:606); pesi `ContrastiveConfig.weights(step)` (contrastive_objectives.py:102-105) | ricalcolati | `contrast_nce_weight`/`contrast_rnc_weight` uguali e uguali al valore atteso (target_resume.py:381-385) [V] |
| GradScaler | assente (autocast bf16, nessun `GradScaler` in run.py, grep vuoto) | | n/a [V] |
| Contatori allarmi, config, stream, firma dati, pose_stats, provenienza | payload | run.py:1445 e guardie | confrontati esattamente a 576 (`_nested_equal`, target_resume.py:259-265) [V] |
| **Bucket DDP del reducer** | **non salvati** (stato di processo) | DDP nuovo con bucket iniziali; ricostruiti dopo la prima iterazione | tracce `.ddp.jsonl`: continuo al 545 gia' ricostruito (`has_rebuilt_buckets: 1`), ripresa `iteration 0` con `bucket_sizes "114202292, 68480"` [V] |

Esito del rapporto: `arms.CN.resume.boundary_mismatches = []`, `arms.CR.resume.boundary_mismatches = []`,
`missing_active_support = []` [V].

## 2. Confronto delle tracce (cmp.py)
- `trace-<rank>.jsonl`, tutti i campi esclusa `phase`, per 32 aggiornamenti e 4 rank: **nessuna differenza** fra continuo e A, continuo
  e B, A e B, in CN e in CR. Il campo `boundary` e' uguale nelle tre fasi su ogni rank [V].
- `trace-<rank>.ddp.jsonl`: l'ordine finale ricostruito (`rebuilt_per_bucket_param_indices`) delle riprese e' uguale a quello del
  continuo su tutti i rank. La ripresa registra `has_rebuilt_buckets: 0` al 546 e `1` al 547 [V].

## 3. Metriche per aggiornamento (metrics_cmp.py, ab_cmp.py)
- **A contro B**, con checkpoint finali identici byte per byte: `grad_norm`, `contrast_projector_grad_norm`, `contrast_nce` e `pose` sono identici nei 32 passi;
  `jepa`, `log_rms`, `ms_fast`, `ms_slow`, `log_env`, `band_shape`, `total` differiscono in 29-32 passi fino a 5e-7 relativo; `contrast_rnc` differisce in 2-3 passi
  fino a 8e-8 [V]. Spiegazione [D]: le perdite delle ancore e del JEPA sommano per finestra con `index_add` (anchors.py:47-48, 65-66). Su GPU
  l'ordine delle somme atomiche e' libero. Cambia il valore registrato, non il gradiente: la derivata di una somma non dipende dall'ordine.
  E' rumore di sola lettura, presente anche fra due esecuzioni identiche.
- **Continuo contro ripresa al 545 (CN)**: `contrast_nce` 3.67859810590744, `contrast_rnc` e `pose` identici; `lr` identico;
  `jepa`/`log_rms`/`ms_fast` differiscono di 9e-8-3,8e-7, entro il rumore A/B; `grad_norm` 4.336711902204087 contro 4.336711902194546
  (2,2e-12 rel, 9,5e-12 assoluto); `contrast_projector_grad_norm` 1,0e-13 rel [V].
- **Al 546 (CN)**: `grad_norm`, `grad_clip_factor`, `contrast_projector_grad_norm` e `contrast_nce` sono **identici**. `contrast_nce` diverge
  per la prima volta al 547 (2,4e-6), `pose` e `contrast_rnc` al 548. Al 576 `contrast_projector_grad_norm` differisce di 9,2e-4 rel [V].
- **CR**: al 545 `contrast_nce` (solo metrica), `contrast_rnc`, `pose` e `jepa` sono identici; `grad_norm` differisce di 3,4e-13 rel; dal 546 divergono `grad_norm` e `contrast_nce` [V].
- **Proiettore in CR**: `contrast_projector_grad_norm = 0.0` in tutti i passi, e i momenti 266-269 hanno errore relativo `0.0` a 576
  (`floating_tensor_relative_to_left`) [V]. In CR il proiettore non riceve gradiente (peso NCE nullo), quindi **CR non mette alla prova i
  quattro tensori che falliscono in CN**. Che CR superi le tre coppie non dice nulla sulla loro sensibilita'.

## 4. Candidati di stato non ripristinato o non verificato
| Candidato | Stato |
|---|---|
| Chiavi della coda con dtype o precisione diversi | **Escluso**: fp32 salvato come tensore, dtype controllato; entries pre-fork con errore 0 a 576; `contrast_nce` del 545 bit-identico [V] |
| Coda per rank ricaricata dal solo rank 0 | **Escluso**: la coda e' replicata per costruzione; l'impronta e' uguale sui 4 rank del continuo e delle riprese [V] |
| Ordine dei negativi, eta', deduplicazione, metadati | **Esclusi**: `queue_order_before_sha256` (identita', step, global_index), lunghezza e last_step uguali per 32 passi; i contatori delle duplicazioni (`contrast_nce_duplicate_current_discarded`, `previous_keys_replaced`) non sono fra i campi diversi al 545 [V] |
| Buffer del proiettore | **Escluso**: `ProjectionHead` e' Linear-GELU-Linear senza buffer (contrastive.py:15-27); i parametri stanno nello studente e nell'impronta [V] |
| Stato dell'autocast o flag numerici | **Escluso**: autocast e' un contesto per passo (run.py, `amp`); flag `numeric_profile` uguali fra continuo e ripresa (diff vuoto sul rank 0) [V] |
| RNG (dropout del muscolo, worker) | **Escluso**: RNG prima e dopo ogni passo uguali [V] |
| Dispositivo o percorso di AdamW (step su CPU, foreach o fused) | **Aperto, improbabile**: l'impronta non include il dispositivo. Dal codice di torch, `load_state_dict` lascia `step` su CPU se non e' capturable e porta i momenti sul dispositivo del parametro, come nel continuo [D]. Si esclude registrando dispositivo e dtype di `step`/`exp_avg` e il percorso foreach dopo il caricamento |
| Bucket DDP | **Differenza reale, ma non e' stato del checkpoint**: la ripresa riduce il 545 con i bucket iniziali, il continuo con quelli ricostruiti [V]. In torch 2.14 locale `set_runtime_stats_and_log` (riga 1758) precede `_rebuild_buckets()` (riga 1776) in `_pre_forward`. Il `has_rebuilt_buckets: 0` registrato al 546 e' quindi gia' compatibile con bucket ricostruiti al 546 [D, versione di Leonardo 2.2.0a0 non verificata]. Lo conferma `grad_norm` identico al 546 in CN [V]. La disposizione diversa riguarderebbe **il solo passo 545** |
| Algoritmi NCCL, cuBLAS o cuDNN | **Aperto, improbabile**: cudnn.benchmark=False; A e B identici. Nei sacct non ci sono i nodi, quindi non so se continuo e riprese abbiano girato sullo stesso nodo |

## 5. Conclusione su H1
- **Esclusi dagli artefatti**: studente e teacher, compreso il proiettore; momenti, step e gruppi di AdamW; coda (chiavi, precisione,
  ordine, eta', deduplicazione, replica per rank); copertura; RNG per rank; posizione dei dati; schedule e pesi delle loss; contatori.
  Prova decisiva: impronte di confine uguali, piu' forward bit-identico al 545 per le loss deterministiche.
- **Resta possibile**, ma non e' un difetto dei dati salvati: lo stato di processo dopo il riavvio. La disposizione dei bucket DDP e'
  diversa al 545 (osservata). Il percorso di AdamW non e' verificato quanto al dispositivo.
- **Osservazione che chiuderebbe il punto**: nella prova proposta da GPT, gradienti locali per rank uguali al 545 e diversi solo dopo la
  riduzione. Basterebbe anche un rerun con `bucket_cap_mb` o un ordine di bucket fissato uguale nei due percorsi, oppure il
  continuo riavviato con DDP ricostruito prima del 545 (per esempio un passo a vuoto con `no_sync` escluso dal confronto). Lo scarto a 576
  dovrebbe sparire o restare confinato all'arrotondamento.

## Divergenze dalla lettura di GPT
1. GPT parla di raggruppamento diverso fino al 547 (e il compito di 545-546). Le metriche indicano che la disposizione diversa riguarda
   probabilmente il solo 545: al 546 `grad_norm` e `contrast_projector_grad_norm` di CN sono identici. Il campo registrato al 546
   precede la ricostruzione nello stesso `_pre_forward` [D].
2. GPT non osserva che in CR il proiettore ha gradiente nullo e momenti con errore 0. CR quindi non controlla i tensori 266-269, e il
   confronto con CR non distingue una sensibilita' propria del proiettore da un difetto.
3. GPT scrive che lo scarto di 9,5e-12 e' sulla «norma totale del gradiente InfoNCE». E' la differenza assoluta della norma totale del
   gradiente del braccio CN (2,2e-12 relativo), non della sola componente InfoNCE.
4. Le chiavi della coda scritte dopo il fork differiscono fino a 0,92% in CN (`contrastive_state.queue.entries.5538.key`) e 0,60% in CR.
   Sono sotto soglia ma vicine, e GPT non lo segnala. Le chiavi precedenti al fork hanno errore esattamente 0.
