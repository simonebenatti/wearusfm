# verify-r4-risorse-R4-N6-1 — scettico n. 1 sul rilievo R4-N6

Inizio: 2026-11-20. Copia rev_targets/clone, HEAD 0efa4d45; codice a fbfde797.
Rilievo: "Tempi del profilo math non misurati: margine storico circa 2,3-2,4 volte" (NOTA).

## Progressi

## 1. Prove riprodotte (comandi miei, copia a 0efa4d45 pulita)
- summary bf16 18/11 (`benchmarks_completed_bf16_20261118/benchmark/{C0,CN,CR}/summary.json`): steps 576, elapsed_s 638,24 / 656,98 / 741,77; gpu_max_memory_gib 13,72. Confermato.
- `benchmarks_completed_bf16_20261118/accounting.txt`: 64771457/478/595 (full) 661/683/769 s; 64771859/64772690 (split 544+ripresa, `contrastive_split_benchmark.py`) 775/759 s. Confermato.
- `calibration_completed_20261117/accounting_and_hashes.txt:1`: 64701482 774 s; `run/summary.json`: amp_bf16 true, 512 passi, elapsed 749,16 s, **picco 16,40 GiB** (non 13,7: il "picco storico" del rilievo e' sottostimato; irrilevante rispetto a 64).
- Margini su 1770: CR 2,386; calibrazione 2,363 (train) / 2,287 (job); C0 2,773. "circa 2,3-2,4" corretto per il caso peggiore.
- `contrastive_job.py:132,159` passa `--time-limit-s 1770` a calibrazione e benchmark; `run.py:1243` t_start all'ingresso di train, `:1609` controllo per passo, `:1821` elapsed_s; `contrastive_run.py:338-371` il profilo avvolge tutto `R.train` e il mancato `max_steps` solleva "phase incomplete ... do not silently retry". Confermato.
- `preflight_plan.json` resources.measured_new_recipe_seconds = null. Confermato.
- Profilo (`contrastive_numeric_profile.py:111-121`); prova locale: rispetto ai default di torch cambiano SOLO cudnn_tf32 True->False, flash/mem_efficient/cudnn SDPA True->False (matmul TF32, deterministic, cudnn.benchmark, precisione fp32 gia' ai default). 16 test di test_contrastive_numeric_profile.py passati.
- Storico: nessun flag di backend nei checkout 8905e346/4fc51525 (git grep vuoto) -> default di torch. Nessuna misura di tempo in profilo math: i diagnostici 365cb1e2 (resume CN/CR in amp_bf16 automatico, 8-12 passi; CUDA common-state senza tempi).

## 2. Motivi di confutazione trovati
Il rilievo non individua difetti ne' lacune documentali: tutto il suo contenuto e' gia' dichiarato e implementato.
- Decisione scientifica faa4a8d7 (docs/reviews/contrastive_common_diagnostics_science_20261119.md:21) descrive gia' il passaggio "da SDPA con piu' backend ammessi e TF32 di cuDNN attivo a solo math e TF32 di cuDNN spento"; :46 "Tempo e memoria al batch misto globale di 32 e costo dei benchmark restano da misurare. Una difficolta' di memoria, tempo o ripetibilita' riporta il piano a Simone prima di qualunque cambio di backend"; :50 "Misurare ... memoria e throughput nella configurazione candidata".
- Prontuario docs/contrastive_targets_execution_20261119.md:25-27 ("Il benchmark di produzione deve misurare tempo e memoria a batch32 prima di proporre il profilo nei training lunghi. Memoria, tempo, esiti o supporto insufficienti mantengono il blocco"), :91-92 (1800/1770), :114-115 ("Nessuna misura nuova di throughput e' ancora disponibile"), :195.
- preflight_plan.json: measured_new_recipe_seconds null; training_release "Absent; full-batch math time/memory, successful preflight, measured training budget required"; failure_policy "Failed or inconclusive gate stops descendants; no retry/time/backend/...".
- La correzione proposta e' gia' nel codice del lettore: contrastive_benchmark_readout.py:205-225 (576 passi, stopped max_steps, flag math su ogni rank, gpu_max_memory_gib obbligatorio), :65-92 (measured_allocation da elapsed_s e t_step_s misurati nel profilo math, riserva 1,5x + 300 s per 5000/14618), :470-474 (full_batch_gpu_peak_gib, profilo, allocazioni per i training). Il confronto con 1770 s e 64 GB e' automatico: limite interno -> "phase incomplete" (FAILED), OOM -> crash; afterok e failure_policy fermano i discendenti.
- Il fattore "TF32 di cuDNN" non incide sui tempi: nessuna convoluzione attiva (unico Conv1d in spectral_branches.py:215, costruito solo se cfg.spectral_branches, che e' [] nel config base f9d03184). Con grad_checkpoint=true su local/backbone/decoder (fm.py:198-199; config base) la matrice d'attenzione del math non resta salvata oltre il blocco: aumento di memoria transitorio.
- Mia stima (non misura): traffico HBM aggiuntivo del math ~0,1 s/passo contro 1,1-1,3 s/passo storici; margine 2,3x ampio.

## Esito
CONFUTATO come rilievo: fatti corretti (salvo il picco storico 16,40 GiB della calibrazione), ma nessun difetto e nessuna lacuna; non misurazione, obbligo di misura e arresto sono dichiarati nella decisione faa4a8d7, nel prontuario e nel preflight, e la correzione suggerita e' gia' implementata nel lettore. Gravita' reale: NESSUNO.
Non verificato: tempi e memoria reali del profilo math su A100/torch 2.2 (solo sul cluster, vietato); kernel SDPA effettivamente scelto nello storico (ignoto anche per GPT, faa4a8d7:21); costo del loader con i target 100/125 nel nodo GPU.
Nessun file della copia modificato (git status vuoto); file temporanei n6_* cancellati alla fine.
