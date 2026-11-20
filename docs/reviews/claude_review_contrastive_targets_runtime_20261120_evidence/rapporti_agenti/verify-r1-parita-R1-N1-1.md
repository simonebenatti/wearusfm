# Scettico n.1 — R1-N1 (r1-parita): probe con reducer nuovi contro DDP vivo dopo la ricostruzione dei bucket

Inizio 20/11. Copia di sola lettura a 0efa4d45 (codice fbfde797). Lavoro in rev_runtime/tmp/sk1_r1n1.

## Lettura documenti e codice (fatta)
- Prontuario docs/contrastive_targets_execution_20261119.md:34-35 a fbfde797: «I sei casi usano reducer nuovi uguali tra loro;
  non copiano lo stato interno del reducer vivo.» -> la scelta e' dichiarata esplicitamente.
- scripts/contrastive_target_parity.py a fbfde797: docstring del modulo (r.1-7) «Six isolated DDP clones ... The live reducer ...
  never advanced by a probe»; docstring di _measure_case (r.99) «Fresh reducer per case»; campo del rapporto (r.316)
  "measurement": "live C0 pre-update states; six isolated reducers; production global batch32". Nessuna frase afferma
  identita' al bit fra probe e backward vivo.
- Decisione scientifica (docs/reviews/contrastive_common_diagnostics_science_20261119.md, punto 1): «identita' dei gradienti
  ridotti per byte entro e tra i bracci» -> confronto probe/probe, non probe/vivo.
- Consumatori del rapporto: solo contrastive_benchmark_readout.py:503-521 (validate_parity_report, probe/probe + legami);
  nessun confronto fra gradient_norm/gradient_sha256 dei probe e grad_norm vivo in metrics.jsonl.
- Produzione: review_configs/{C0,CN,CR}.config.json rvq_codes=None, keep_readout=False -> run.py:436-438
  find_unused_parameters=False -> il DDP vivo ricostruisce i bucket dopo la prima iterazione anche in produzione
  (la premessa del rilievo vale anche su Leonardo).
- Dato del revisore da precisare: nei suoi exp2.rank*.json l'update 2 riporta has_rebuilt_buckets=0 e bucket_sizes
  invariati; il logging DDP e' aggiornato in _pre_forward PRIMA di _rebuild_buckets (torch distributed.py:1758 e 1776),
  quindi ritarda di un'iterazione. Verifico con _has_rebuilt_buckets e con il layout reale del reducer.

## Riproduzione indipendente (fatta, 05:15-05:17)
Script mio: rev_runtime/tmp/sk1_r1n1/scripts/probe_live.py (trainer reale R.train, observe_control reale, gloo 4 rank CPU,
torch 2.14.0, config di test_contrastive_runtime._config con batch 32, amp_bf16, prefetch, muscle_dropout .4, fill/ramp 256,
grad_steps (4,); P.UPDATES=(1,2,3,4); P.assert_numeric_profile stub su CPU). Gradiente vivo letto dopo il backward vivo;
gradienti dei probe catturati avvolgendo la funzione backward passata ai probe (meccanismo diverso da exp2.py); layout reale
dei bucket letto con reducer._get_zeros_like_grad_buckets(); _has_rebuilt_buckets dal DDP vivo.
- live (run_L1234), identico sui 4 rank: update 1 layout vivo = iniziale [98,8 parametri] = probe, vivo == 6/6 probe al byte;
  update 2/3/4 _has_rebuilt_buckets=True gia' all'update 2 (il logging DDP dice ancora 0 all'update 2: ritardo), layout vivo
  [8,98] != layout dei probe [98,8]; vivo != probe: 94/106, 95/106, 94/106 tensori; max|diff| 1,91e-6; rel L2 5,06e-8,
  5,42e-8, 5,57e-8. I 6 probe: 1 solo hash per punto e fra rank; gate: 1 hash, errori nessuno (INCONCLUSIVE solo memoria CUDA).
- controfattuale norebuild (run_N1234): proxy del reducer vivo con _rebuild_buckets -> False (layout vivo resta [98,8]):
  vivo == 6/6 probe al byte a tutti e 4 gli update (0/106 tensori diversi). -> l'unica causa dello scarto e' il layout dei
  bucket (ordine di riduzione); forward/backward dei probe riproducono esattamente il calcolo vivo.
Esito: il fatto citato nel rilievo e' riprodotto (numeri identici a exp2.py, CPU deterministica); la sua lettura dei campi
di logging all'update 2 e' imprecisa (has_rebuilt_buckets=1 compare solo dal 3), la conclusione sulla ricostruzione e' giusta.
- variante grad_checkpoint=True (run_G124, update 1/2/4): stesso quadro (update 1 vivo == probe; 2 e 4 layout [8,98] contro
  [98,8], 94/106 tensori, rel L2 5,06e-8 e 5,57e-8; 1 hash fra i probe).
- Clone di sola lettura invariato (git status vuoto). Checkpoint e store cancellati; restano script e JSON in sk1_r1n1.

## Valutazione
- Non e' un difetto: la scelta e' dichiarata dove il rilievo chiede di scriverla. Prontuario r.34-35 («I sei casi usano
  reducer nuovi uguali tra loro; non copiano lo stato interno del reducer vivo»); parity.py r.100 «Fresh reducer per case»;
  r.3-6 «Six isolated DDP clones ... The live reducer ... never advanced by a probe»; r.317 "six isolated reducers";
  plan.json numerical_amendment.neutrality.criterion «reduced gradients byte-identical within and between arms and ranks».
  Nessun testo afferma che il probe riproduca al bit il backward vivo; nessun codice lo usa (readout:503-521 solo
  validate_parity_report, probe/probe; live_state_restored confronta i .grad vivi prima/dopo, None/None).
- Il controfattuale mostra che i probe riproducono esattamente il calcolo vivo a parita' di layout: il disegno e' fedele
  e il gate (probe/probe) non e' toccato. A peso zero il braccio cambia solo i pesi (contrastive_objectives.py:104-105),
  quindi i tre run vivi ricostruiscono gli stessi bucket: nessuna differenza fra bracci nascosta dal layout dei probe.
- Il «fix» proposto e' gia' presente; resta solo, facoltativa, una frase sulla conseguenza (gradiente probe != vivo dopo
  il rebuild) o l'hash vivo come descrittivo. Gravita' se reale: NOTA. Esito: CONFUTATO come difetto (fatto riprodotto,
  per disegno e dichiarato). Imprecisione nell'evidenza del rilievo: all'update 2 il logging DDP riporta ancora
  has_rebuilt_buckets=0 (ritardo di un'iterazione), non 1.
