# R1 - Fedelta' della ripresa e neutralita' del riscaldamento (commit 9d34dc69)

Cartella di lavoro: /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_probe/agents/r1-fedelta-ripresa

## Verdetto della parte R1
La ripresa CN_A e' ricostruita fedelmente e il riscaldamento e' a effetto nullo (verificato col trainer vero
sotto DDP Gloo). C'e' pero' un blocco nuovo, oltre a B-1, nel confronto finale della gamba warm.

## B-2 (bloccante) - la gamba warm non puo' mai eguagliare i metadati del continuo: `cuda_rng[1..3]`
- File: scripts/diagnostics/contrastive_cn_restart_compare.py:241 e :248 (anche :246 per `exact_payload`).
- Meccanismo (src/wearusfm/training/run.py): riga 1383 `torch.manual_seed(cfg.seed)` semina tutte le GPU visibili;
  il continuo (partenza da zero) alle righe 1462-1463 risemina tutte le GPU con `seed*1_000_003+rank+1`; una ripresa
  (righe 1446-1450) ripristina solo la GPU del proprio rank. Il payload (riga 1482) salva `torch.cuda.get_rng_state_all()`
  del rank 0, cioe' quattro stati: quelli dei dispositivi 1-3 dipendono dal tipo di avvio, non dalla traiettoria.
- Prova sui file veri (lettura di Claude su Leonardo, confronto_checkpoint_veri_20261121.out.txt):
  `cuda_rng ... uguali per dispositivo: [True, False, False, False]`;
  `continuo vs A ... metadata_or_schema_different ['cuda_rng.1', 'cuda_rng.2', 'cuda_rng.3'] n 3`; A contro B identici.
- La gamba warm e' una ripresa (release resume_a): anche se la traiettoria recupera il continuo bit per bit,
  `checkpoint_differences(continuo, warm)` da' quei tre percorsi non float, e la riga 248 solleva ValueError,
  il report diventa INCONCLUSIVE. Tolta la riga 248, `exact_payload` resterebbe False e SUPPORTED sarebbe
  irraggiungibile: al massimo PARTIAL_SUPPORT, attribuito male.
- Simulazione eseguita (exp_cuda_rng.py, CPU): output
  `{'floating_tensors_different': [], 'metadata_or_schema_different': ['cuda_rng.1','cuda_rng.2','cuda_rng.3'], 'exact_payload': False}`
  -> `ValueError warm endpoint metadata/schema or nonfloat state differs -> INCONCLUSIVE`; con classify ->
  `PARTIAL_SUPPORT ... warm trajectory does not recover continuous`.
- I test non lo vedono: test_cn_restart_compare.py usa payload sintetici senza questo schema.
- Correzione minima: nel solo confronto warm contro continuo escludere `cuda_rng.1..3` (i dispositivi non posseduti
  dal rank 0), richiedendo invece che siano uguali a quelli della gamba ordinary/CN_A (prova che sono l'artefatto
  dell'avvio); `cuda_rng.0` e `rank_rng` restano confrontati. Dichiarare l'esclusione nel report e nel protocollo e
  aggiungere un test con questo schema esatto.

## Rilievi minori
- I-1 (importante) device_identity (leg.py:30-43): ramo ctypes mai eseguito su CUDA (i test lo sostituiscono), chiamato
  solo all'update 545 dentro il backward, dopo l'avvio completo. Firme corrette (cuDeviceGet(CUdevice*, int),
  cuDeviceGetUuid(CUuuid*, CUdevice), CUresult 0); cuInit e' gia' fatto dal runtime (contesto creato da set_device/NCCL) e
  `CDLL("libcuda.so.1")` riprende la libreria gia' caricata, quindi dovrebbe funzionare. Ma un fallimento qui fa
  RuntimeError -> la gamba cade -> INCONCLUSIVE dopo il consumo delle GPU. Correzione: calcolare l'identita' una volta
  all'avvio della gamba (dopo `torch.cuda.set_device(LOCAL_RANK)`), con `cuInit(0)` esplicito e `argtypes`; in caso di
  errore registrare `gpu_uuid=None` piu' `CUDA_VISIBLE_DEVICES`/`LOCAL_RANK`/indice e confrontare quelli
  (allocation.json ha gia' l'inventario nvidia-smi), invece di abortire.
- N-1 (nota) radici dei dati passate come stringhe non risolte (contrastive_cn_restart.py:126-129) mentre lo storico
  passava `resolve(strict=True)`; il controllo in validate_argv confronta i risolti e le radici non entrano in payload,
  tracce o firme (load_index tiene le radici a parte), quindi nessun effetto atteso.

## Verifiche positive
1. Riga di comando: stessi 14 argomenti dello storico (contrastive_job.build_command, mode resume): config, release,
   stage benchmark, out-dir, time-limit 1770, manifest, scales, window-seconds, due root, audit-report, cohort,
   calibration-report, calibration-run-dir. Nessun passthrough. Il runner riconosce resume_a dalla release CN_A
   (target_resume, fork a 544, prepare_reload con fase reload_a). Il controllo `out_dir.exists()` e' fatto da tutti
   i rank prima della barriera di init (TORCH_DIST_INIT_BARRIER=1), come nello storico. Il payload non dipende dalla
   cartella: config, stream, data_signature (sha di manifest/scale/finestre), provenance (source_digest invariato:
   i file nuovi non stanno nei glob `scripts/contrastive_*.py` e `scripts/slurm/contrastive*.sbatch`). CN_A e CN_B
   hanno lo stesso sha con cartelle diverse.
2. Ordine delle chiamate al 545: trace.observe_backward (pending, input, RNG before, boundary) -> dispatch:
   ReductionObservation registra l'hook (prima di ogni forward/backward DDP del processo) -> snapshot -> [warm:
   null_backward col vero _accumulate_terms su runtime deepcopy, hook non in cattura, zero_grad(None), buffer e RNG
   ripristinati] -> snapshot -> boundary -> start -> backward vero (warm: ricostruzione dei bucket nel forward) ->
   finish (verifica ridotto == param.grad) -> hook dell'ottimizzatore -> trace RNG after -> trainer:
   projector norm, before_update, PRIMO clip reale (run.py:1700), opt.step (pre-hook: gradienti dopo il clip;
   post-hook: stati Adam). Gli hook dell'ottimizzatore si tolgono alla chiamata del 546.
3. Il modello non ha stato Python mutato nel forward (scansione AST di model/, jepa.py, contrastive_objectives.py):
   le sole mutazioni sono del runtime (pending, pending_stats, failure_reason, ...), isolate dalla deepcopy.
   Coda su CPU, snapshot della coda puro. Dropout del muscolo sul generatore globale CUDA, coperto dal ripristino.
4. Eseguito: tests/cpu/test_cn_restart_leg.py 4 passati in 6,5 s.
5. Eseguito: exp_ddp_warm.py, WearUsFM piccolo con testa contrastiva, ContrastiveRuntime vero con coda, trainer
   _accumulate_terms vero, DDP Gloo a 2 rank, dropout 0,4, gradienti globali attivi al 545, due aggiornamenti.
   Varianti nohook / ordinary / warm: gradienti locali di 544 e 545, valori, norme, studente, teacher, ottimizzatore,
   runtime e RNG finali identici bit per bit in tutte e tre; boundary before==after in entrambe; local e reduced uguali
   per tutti i parametri; la sola differenza e' lo stato DDP (warm: iterazione 2, bucket ricostruiti). Simulazione CPU,
   non CUDA/NCCL; per il modello piccolo ho adattato il conteggio di 270 parametri con uno shim.
6. Tracce storiche: continuo, CN_A e CN_B hanno TRACE_EQUAL e SUPPORT_FIELDS identici su 545-576 per i 4 rank e lo
   stesso boundary; quindi i confronti tracce warm-continuo non falliscono per costruzione.
