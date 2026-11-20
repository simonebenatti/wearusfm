# R1 — Parita' a peso zero (rapporto incrementale)

Inizio: lettura di scripts/contrastive_target_parity.py, contrastive_numeric_profile.py, contrastive_run.py (diff 438c8d65..fbfde797).

## Lettura (fatta)
- Nessuna modifica in src/ fra 438c8d65 e fbfde797: l'aggancio e' solo per monkeypatch di `trainer._accumulate_terms` e `ContrastiveRuntime.prepare` (observe_control, contrastive_target_parity.py:336-362), attivato in contrastive_run.py:355-357 solo per benchmark/full/C0.
- Indici: il trainer chiama prepare(batch, step) e _accumulate_terms con step zero-based = update completati; l'osservatore misura quando runtime.step+1 in (128,256), cioe' step 127/255: 127 e 255 update completati. OK.
- Probe: nuovo WearUsFM + testa, load_state_dict dalle copie CPU, make_teacher + load teacher, nuovo DDP (_ddp_wrap) sul clone: nessun parametro condiviso con il DDP vivo.
- state_sha256 porta i tensori su CPU (device non entra nell'impronta), dtype/shape/bytes si'.
- weights(step): fill 256, ramp 256 -> fattore 0 a step 127 e 255; wants_gradients vero a update 256 (grad_steps) -> global_gradients/allineamento anche nei probe.

## Esperimento E1 (fatto): trainer reale, DDP gloo 4 rank, con/senza osservatore
Script: rev_runtime/tmp/r1-exp/scripts/exp_ddp_observer.py e compare_runs.py. Config _config(pose) di test_contrastive_runtime con batch 32, amp_bf16 True, prefetch True, muscle_dropout .4, fill/ramp 256 (pesi zero), grad_steps (4,); P.UPDATES patchato a (2,4) (schedule di test a 4 update), P.assert_numeric_profile patchato (CPU).
- Baseline A vs B: checkpoint finale (student, teacher, optimizer, contrastive_state, rank_rng) e metrics.jsonl (senza tempi) identici.
- Baseline A vs osservatore O: tutto identico (student/teacher/optimizer/contrastive_state/rank_rng/metriche) -> i 12 backward dei probe non perturbano il run vivo.
- Report O: a entrambi i punti un solo hash di gradiente su 6 casi x 4 rank, live_state_restored True su 4 rank, supporto NCE 32, RnC 32 finestre/~4e5 coppie, pesi 0, finestre 32; stato INCONCLUSIVE solo per memoria CUDA assente (atteso su CPU).

## E2 (fatto): strumentazione degli hook e dei puntatori
- Run I (osservatore + hook su tutti i parametri vivi durante measure, puntatori data_ptr dei probe): live_hook_calls 0 su 4 rank (nessun gradiente arriva ai parametri vivi), sovrapposizione puntatori probe/vivo 0 su 12 probe x 4 rank, .grad vivi None prima e dopo; checkpoint e metriche ancora identici al baseline A.
- Controllo negativo N1 (P.restore_rng no-op): i probe consumano RNG (muscle dropout) e l'osservatore ferma il run con "parity probe modified the live control state" -> il controllo e' efficace e il ripristino reale e' cio' che rende E1 identico.

## E3 (fatto): costo
- Storico bf16 auto (benchmarks_completed_bf16_20261118, metrics.jsonl.gz): elapsed C0 638 s, CN 657 s, CR 742 s; t_step normale 0,83 s, diagnostico 1,73 s; picco 13,72 GiB.
- Setup di un probe col modello 30M di produzione su CPU Mac (proxy): ~0,8 s (costruzione 0,2, load+teacher 0,2, runtime con coda 6.293 voci 0,2, impronte 0,4). Stima probe: 6x(setup ~1-1,5 s + backward ~0,7 s) a 128 e 6x(~1-1,5 + ~1,6 s) a 256 -> ~30-50 s totali; entra nei 1.770 s finche' il full C0 in math resta sotto ~1.650 s (cioe' < ~2,5x lo storico). Non misurabile qui.

## Test (fatti)
- test_contrastive_target_parity.py + test_contrastive_numeric_profile.py: 45 passed in 5,11 s.
- test_contrastive_targets_readout.py + test_contrastive_run.py: 37 passed in 5,84 s.
- Copertura: nessun test esegue observe_control attorno a un train() DDP reale (il test "real CPU replays" usa DistContext() senza DDP; quello dell'osservatore usa _measure_case finto). Lacuna coperta da E1/E2.

## Verifiche di codice (fatte)
- (c) assess_point: profilo/rank/casi/supporto/memoria mancanti -> INCONCLUSIVE; stato/RNG/pesi/finestre/misure/hash diversi con evidenza completa -> FAILED; readout lega source, config canonico, summary C0, initial_state, provenance (contrastive_benchmark_readout.py:503-525); INCONCLUSIVE -> INCONCLUSIVE_REQUIRES_REVIEW -> authenticate_preflight rifiuta -> nessun training. Traiettorie 1..256 solo descrittive (trajectory_drift, rtol 1e-4/atol 1e-6).
- (d) profilo: numeric_profile avvolge R.train per calibrazione/benchmark/training (contrastive_run.py:338-369) e il main del CUDA check; probe sotto gli stessi flag di processo e riasserzione per caso dentro autocast. Compatibilita' 2.2: is_autocast_enabled METH_NOARGS -> TypeError -> ramo enabled(); get_autocast_dtype assente -> get_autocast_gpu_dtype(); cudnn_sdp via hasattr (None ammesso).
- Note emerse: ricevute numeriche della calibrazione scritte ma non lette da nessuno (solo le 28 dei 7 run); CUBLAS_WORKSPACE_CONFIG non impostato dal launcher GPU (diagnostica storica con :4096:8); memoria per caso = picco cumulativo del processo.

## Ripresa (20/11 03:02): nuova sessione dell agente; riverifico i punti critici a partire dal codice a fbfde797
- 03:10 test rieseguiti: test_contrastive_target_parity.py + test_contrastive_numeric_profile.py -> 45 passed in 5,39 s (basetemp r1b-pt1).
- Lettura codice a fbfde797 confermata: aggancio per patch di ContrastiveRuntime.prepare e trainer._accumulate_terms (contrastive_target_parity.py:336-362), attivo solo per benchmark/full/C0 (contrastive_run.py:355-357); opt.zero_grad(set_to_none) precede prepare (run.py:1625-1629) -> .grad vivi None all'ingresso dell'osservatore.
- In corso: riesecuzione indipendente degli esperimenti DDP gloo (r1b-exp) con controllo negativo nuovo N2 (probe che condivide i parametri vivi).
- 03:20 riesecuzione indipendente (r1b-exp, script exp.py/compare.py, gloo 4 rank, trainer reale, UPDATES=(2,4), batch 32, muscle_dropout .4, prefetch):
  * A (senza osservatore) vs O (osservatore): student/teacher/optimizer/contrastive_state/rank_rng/step e metriche (senza tempi) identici; grad_norm identici ai 4 update.
  * Rapporto O: 1 hash di gradiente su 6 casi x 4 rank a ciascun punto; live_state_restored True x4; supporto NCE 32, RnC 32 finestre, ~4,0e5 coppie, pesi 0; stato INCONCLUSIVE solo per memoria CUDA (atteso su CPU); config_sha256 == sha canonico di config.json, summary_sha256 == sha(summary.json), initial_state == summary: True.
  * I (strumentato): 0 chiamate di hook sui parametri vivi, sovrapposizione data_ptr 0 su 12 probe x 4 rank, .grad vivi None prima/dopo, stato di logging del DDP vivo invariato (chiavi non temporali), reducer vivo finalizzato; checkpoint e metriche = A.
  * N1 (restore_rng inerte) e N2 (probe che riusa i parametri vivi in un nuovo DDP): l'osservatore ferma il run su 4 rank ("parity probe modified the live control state").
  * Osservazione: il gradiente ridotto del backward VIVO allo stesso stato NON e' byte-identico al gradiente del probe C0a (update 2 e 4). Verifico se e' solo l'ordine di riduzione (bucket ricostruiti dal DDP vivo) o un difetto di fedelta'.
- 03:35 exp2.py (gradiente vivo dopo il backward vivo contro probe C0a, stesso stato):
  * update 1 (DDP vivo con i bucket iniziali): vivo == probe byte per byte (max |diff| 0, 106/106 tensori uguali).
  * update 2 e 4 (DDP vivo con bucket ricostruiti dopo la prima iterazione; logging: bucket_sizes 82140,27008 -> 27008,82140, has_rebuilt_buckets 1): vivo != probe, scarto L2 relativo 5,1e-8 / 5,6e-8, max |diff| 1,9e-6, 94/106 tensori diversi; i 6 probe restano uguali fra loro e fra rank.
  * Lettura: i probe riproducono esattamente il calcolo vivo; l'unica differenza e' il layout dei bucket del reducer nuovo (ordine di riduzione). Coerente con il prontuario ("non copiano lo stato interno del reducer vivo"): il gate confronta probe con probe. NOTA, non difetto.
- Test: test_contrastive_targets_readout.py + test_contrastive_run.py + test_contrastive_cuda_check.py -> 41 passed in 14,17 s.
- Costo (e): storico bf16 auto C0 elapsed_s 638,2 (sum t_step 546,5; normale 0,82 s, diagnostico 1,71 s; update 128: 0,80 s, 256: 1,91 s), CN 657,0, CR 741,8; picco 13,72 GiB; calibrazione storica 749,2 s (512 passi, 0,79/1,74 s), picco 16,40 GiB.
  Setup per probe (proxy CPU Mac, modello 28.559.133 parametri, coda 6.293 voci): 0,71-0,72 s con impronte.
  Stima osservatore: ~14 s + f*(6*0,8 + 6*1,9) s = 30/38/46/63 s per f = 1/1,5/2/3 (f = rallentamento math). Il C0 full entra nei 1.770 s se f <= ~2,9; vincoli piu' stretti: calibrazione f <= ~2,5, CR f <= ~2,8. Tempo/memoria math a batch 32 mai misurati (diagnostica storica a 8 finestre, docs/reviews/contrastive_common_diagnostics_science_20261119.md).
  L'oracolo CUDA usa batch globale 8 (contrastive_cuda_check.py:42): la prima misura math a batch 32 e' la calibrazione.
- 03:50 variante con grad_checkpoint e 2 strati (GA/GI, exp3.py): stesse conclusioni (checkpoint e metriche identici, 0 hook, overlap 0, 1 hash, live_restored x4).
- Simulazione API torch 2.2 (cudnn_sdp/enable_cudnn_sdp assenti, is_autocast_enabled senza argomenti, get_autocast_dtype assente): profilo applicato (cudnn_tf32 True->False, solo math, cudnn_sdp None), autocast bf16 riconosciuto, ripristino identico.
- Contaminazione dei tempi (prova CPU): t_step_s A [1.87,1.64,1.76,2.29] contro O [1.92,9.89,1.54,15.37]; elapsed 11,73 contro 32,65 s. In produzione entra in measured_allocation del C0 (readout:65-92): stima ~+5-9% (~+2,7 GPU-ore) sulla proiezione del training C0.
- CUBLAS_WORKSPACE_CONFIG: diagnostica storica con ':4096:8' (diagnostics/contrastive_common_state_cuda.py:392; docs/contrastive_common_diagnostics_20261118.md:22), produzione non lo imposta ne' lo richiede (numeric_profile.py:66 solo registra; contrastive_preflight_gpu.sbatch:29).

## Esito sintetico
Nessun BLOCCANTE nel punto 1. MINORE: (M1) tempo dell'osservatore nella proiezione delle risorse del C0; (M2) contesto cuBLAS diverso dalla diagnostica che ha motivato il profilo. NOTE: probe != vivo dopo la ricostruzione dei bucket (per disegno); PAIRS non usate e nessuna attribuzione entro/fra bracci; alcuni campi mancanti -> FAILED anziche' INCONCLUSIVE; ricevute numeriche di calibrazione/oracolo non confrontate con le 28 e contesto senza cuDNN/NCCL/driver; memoria per caso = picco cumulativo; a peso zero i bracci differiscono solo per l'etichetta.
