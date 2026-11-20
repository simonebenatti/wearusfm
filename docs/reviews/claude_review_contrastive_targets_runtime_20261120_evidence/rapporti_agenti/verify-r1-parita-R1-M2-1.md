# Scettico n. 1 su R1-M2 (r1-parita) — CUBLAS_WORKSPACE_CONFIG

Avvio: copia a 0efa4d45, grep CUBLAS_WORKSPACE_CONFIG riprodotto:
- scripts/contrastive_numeric_profile.py:66 (solo registrazione)
- scripts/diagnostics/contrastive_common_state_cuda.py:94 (registrazione), :392 (impostazione :4096:8 nell'env dei figli)
- docs/contrastive_common_diagnostics_20261118.md:22 (dichiarazione)
Nessuna occorrenza in src/, tests/, results/ (al primo grep).

## Fatti riprodotti (fbfde797 / 0efa4d45)
- numeric_profile.py:66 registra `cublas_workspace_config`; assert_numeric_flags (77-93) non lo richiede. VERO.
- diagnostics/contrastive_common_state_cuda.py:392 `env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", ...}` per i 4 profili. VERO.
- contrastive_preflight_gpu.sbatch:29 esporta solo OMP/OPENBLAS/MKL/PYTHONDONTWRITEBYTECODE; contrastive_job.py:188 os.execv con env ereditato; nessun altro punto del repo (git grep -i cublas|workspace) lo imposta. VERO (modulo cineca-ai non verificabile senza ssh).
- La diagnostica storica usava gia' NCCL + DDP (_ddp_wrap in snapshot(), init_process_group("nccl") riga 274): lo stream NCCL c'era anche li'.
- Stream del prefetch (run.py:627-682) esegue solo _stage -> _prepare -> _to_device (run.py:591-604, 890-895): copie .to(device), nessuna GEMM/cuBLAS.
- Ricevute: numeric_flags (incluso cublas_workspace_config) scritto per rank in numeric_profile.rank{r}.json (contrastive_run.py:359-364); numeric_context_across_runs (readout:345-389) confronta TUTTI i campi fra le 28 ricevute -> il valore (None) e' documentato e uniforme.
- Prontuario 21-27: profilo definito dai soli flag, "l'identita' viene misurata"; decisione faa4a8d7 r.3 "Restano aperte la riproducibilita' al batch di produzione", r.46 "Una difficolta' di ... ripetibilita' riporta il piano a Simone".

## Verifiche tecniche (fonti esterne pubbliche, sola lettura)
- PyTorch v2.2.1 aten/src/ATen/cuda/CublasHandlePool.cpp, getCurrentCUDABlasHandle(): sotto `#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION < 12200` crea un workspace per coppia (handle, stream) (`cublas_handle_stream_to_workspace`, chiave make_tuple(handle, stream)) e chiama `cublasSetWorkspace(handle, ..., getChosenWorkspaceSize())`. parseChosenWorkspaceSize: la variabile decide SOLO la dimensione (default A100 4096*1024*2+16*1024*8 = ~8,1 MiB; con :4096:8 = 32 MiB).
- PyTorch v2.2.1 CUDABlas.cpp: il percorso cuBLASLt (gemm_and_bias, Linear con bias) legge solo CUBLASLT_WORKSPACE_SIZE (default 1 MiB) e alloca il workspace a ogni chiamata dal caching allocator; non legge CUBLAS_WORKSPACE_CONFIG.
- cuBLAS 12.1, sez. 2.1.4 Results reproducibility: la non determinazione multi-stream e' dovuta alla scelta del workspace interno per routine in stream paralleli; rimedi equivalenti: (1) workspace separato per stream con cublasSetWorkspace, (2) un handle per stream, (3) cublasLtMatmul con workspace dell'utente, (4) CUBLAS_WORKSPACE_CONFIG. "Any of those settings" basta. Lo stack di produzione (torch 2.2.1, torch.version.cuda "12.1") applica gia' (1) e (3).
- Diagnostica faa4a8d7 (cuda/*.json): torch 2.2.0a0+git6c8c5ad, cuda 12.1, NCCL 2.19.1, cuDNN 8907; workspace ':4096:8' su tutti i rank di TUTTI e quattro i profili; bf16_auto 6 hash distinti su 6 casi, bf16_math/fp32_math/deterministic 1 hash. La variabile era tenuta costante: la non ripetibilita' osservata e' attribuita da quei dati a SDPA/TF32 cuDNN, non al workspace cuBLAS.
- La variabile nella diagnostica serviva al profilo bf16_math_deterministic (use_deterministic_algorithms(True) la esige con CUDA>=10.2); la produzione esclude per disegno gli algoritmi deterministici (numeric_profile.py:106-107).

## Prove locali (venv wearusfm-tok)
- check_flags.py: assert_numeric_flags accetta cublas_workspace_config None, ':4096:8', ':16:8', 'garbage' e anche il campo assente -> conferma "non richiesto".
- check_context.py: numeric_context_across_runs con 28 ricevute tutte None -> PASSED; un solo rank (CR_B/rank2) con ':4096:8' -> FAILED, fields ['cublas_workspace_config'] -> il valore e' documentato e uniforme nella coorte.

## Esito
Premessa fattuale VERA; difetto CONFUTATO. Il meccanismo invocato (avvertenza multi-stream di cuBLAS) e' neutralizzato in torch 2.2.1/CUDA 12.1 da workspace per (handle, stream) e workspace Lt per chiamata; gli stream in piu' (NCCL, presente anche nella diagnostica; prefetch, solo copie H2D) non eseguono routine cuBLAS. La variabile cambia solo la dimensione del workspace (eventuale scelta di kernel diversa, ma deterministica e uguale fra i probe dello stesso processo); il cancello confronta probe con probe, mai con i gradienti diagnostici. Nessuna confusione di causa: ogni ricevuta registra il valore e il readout ne impone l'uniformita'. Prontuario e decisione non dichiarano identita' con l'ambiente diagnostico: "l'identita' viene misurata", riproducibilita' al batch di produzione "aperta", difficolta' -> Simone.
Residuo: NOTA documentale (il prontuario potrebbe dire che la diagnostica aveva :4096:8 e la coorte lo lascia non impostato, registrato come null).
Non verificato: esecuzione CUDA; se il modulo cineca-ai/4.3.0 esporti la variabile (le ricevute lo diranno); scelta degli algoritmi cuDNN a batch 32 (rischio diverso, fuori da questo rilievo).
