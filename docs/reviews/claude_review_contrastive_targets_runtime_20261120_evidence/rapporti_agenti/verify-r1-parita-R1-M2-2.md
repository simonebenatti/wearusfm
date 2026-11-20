# Verifica scettica n. 2 di R1-M2 (contesto cuBLAS) — avanzamento

Copia a 0efa4d45 (sola lettura). Comandi miei.

## Prove riprodotte
- grep -rn CUBLAS_WORKSPACE_CONFIG scripts src docs results tests: 4 righe, come citato
  (numeric_profile.py:66 registra; diagnostics/contrastive_common_state_cuda.py:94 registra e :392 imposta ':4096:8';
  docs/contrastive_common_diagnostics_20261118.md:22). Nessuna occorrenza nel launcher GPU ne' nel prontuario.
- contrastive_preflight_gpu.sbatch:29 esporta solo OMP/OPENBLAS/MKL/PYTHONDONTWRITEBYTECODE (confermato).
- assert_numeric_flags (numeric_profile.py:77-93) non controlla cublas_workspace_config (confermato).

## Elementi contro il rilievo (in corso)
- Il prefetcher (src/wearusfm/training/run.py:647-668, _prepare 591-599, _to_device 890-895, _pose_tensors 568-588)
  esegue sul suo stream solo copie H2D (.to(device)); i target delle ancore restano su CPU. Nessuna routine cuBLAS.
- Gli stream NCCL del DDP eseguono collettive NCCL, non routine cuBLAS. Il caveat cuBLAS riguarda routine cuBLAS
  su piu' stream concorrenti che condividono un handle.
- La parita' (contrastive_target_parity.py:246-301) confronta sei backward nello stesso processo/ambiente.
- numeric_context_across_runs (contrastive_benchmark_readout.py:345-390) confronta TUTTI i campi dei flag,
  compreso cublas_workspace_config, sulle 28 ricevute: un ambiente misto blocca.

## Esito: CONFUTATO (refuted=true). Gravita' reale: al piu' NOTA

Le premesse fattuali sono corrette (variabile impostata nella diagnostica, non imposta ne' richiesta in produzione).
Il meccanismo di rischio invocato non si applica a torch 2.2.1 / CUDA 12.1:

1. Doc cuBLAS 12.1, par. 2.1.4 "Results Reproducibility": la non determinazione multi-stream e' "due to library
   optimizations in selecting internal workspace for the routines running in parallel streams"; rimedi equivalenti:
   workspace separato per stream con cublasSetWorkspace(), cublasLtMatmul con workspace dell'utente, oppure la variabile;
   "Any of those settings will allow for deterministic behavior even with multiple concurrent streams sharing a single
   cuBLAS handle". Default atomics CUBLAS_ATOMICS_NOT_ALLOWED ("each cuBLAS routine should produce the same results from
   one run to the other").
2. PyTorch v2.2.1 (= git6c8c5ad; ricevute diagnostiche: torch 2.2.0a0+git6c8c5ad, cuda 12.1):
   - CublasHandlePool.cpp getCurrentCUDABlasHandle: handle per thread, cublasSetStream e, sotto
     `#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION < 12200`, workspace dedicato per coppia
     (handle, stream) con cublasSetWorkspace = primo rimedio, applicato CON O SENZA la variabile.
   - CUDABlas.cpp gemm_and_bias (cuBLASLt): workspace proprio da CUBLASLT_WORKSPACE_SIZE (default 1024 KiB) = terzo rimedio.
   - La variabile in 2.2.1 serve solo a dimensionare il workspace cuBLAS (default 4096*1024*2+16*1024*8 = 8.519.680 byte
     su A100; ':4096:8' = 33.554.432) e al controllo alertCuBLASConfigNotDeterministic (Context.cpp), attivo solo con
     use_deterministic_algorithms(True). Per questo la diagnostica, che comprendeva bf16_math_deterministic, doveva
     impostarla per tutti i profili. Il profilo di produzione forza use_deterministic_algorithms(False)
     (contrastive_numeric_profile.py:111) e il prontuario (docs/contrastive_targets_execution_20261119.md:23-24) dichiara
     "Non si impone l'algoritmo deterministico come sostituto del test: l'identità viene misurata".
3. Gli stream extra non eseguono routine cuBLAS: prefetch (attivo, review_configs/C0.config.json:173 "prefetch": true)
   fa solo copie H2D (run.py:647-668, 591-599, 890-895, 568-588; target delle ancore su CPU, run.py:608); NCCL esegue
   collettive. La diagnostica aveva gia' DDP/NCCL a 4 rank (snapshot -> trainer._ddp_wrap; init_process_group nccl).
4. La parita' confronta sei backward nello stesso processo e ambiente (contrastive_target_parity.py:246-301); i flag,
   compreso cublas_workspace_config, sono registrati per caso (righe 121, 149) e numeric_context_across_runs
   (contrastive_benchmark_readout.py:345-390) li confronta tutti sulle 28 ricevute. Prova sintetica mia:
   uniforme -> PASSED; una ricevuta con ':4096:8' -> FAILED, fields ['cublas_workspace_config'].
5. La decisione faa4a8d7 non trasferisce la prova: r.3 "Restano aperte la riproducibilità al batch di produzione"; r.46
   ripetibilita' insufficiente riporta il piano a Simone. La fixture diagnostica (8 finestre emg2pose, stato di
   calibrazione, coda traslata) differisce comunque dalla produzione (32 misto, stato vivo): impostare la variabile non
   eliminerebbe il confondimento di un eventuale fallimento.

Residuo: il prontuario non dice che la variabile della diagnostica non fa parte del profilo (solo chiarezza: NOTA).

## Verifiche positive
- pytest tests/cpu/test_contrastive_numeric_profile.py: 16 passed in 0,96 s; copia git pulita (0 righe di status).
- Ricevute diagnostiche in faa4a8d7 (cuda/*.json): workspace ':4096:8' in tutti i 4 profili x 4 rank; bf16_auto scarto
  ALL 2,83e-4..2,96e-4 pur con la variabile; bf16_math/fp32_math/deterministico 0,0.

## Non verificato
- Comportamento reale su Leonardo (nessun accesso/job); versione della libcublas caricata da cineca-ai/4.3.0 (il ramo
  PyTorch dipende da CUDA_VERSION di compilazione, 12.1).
- Se 8 MiB e 32 MiB portino a kernel diversi: scelta deterministica, irrilevante per la ripetizione entro processo.
