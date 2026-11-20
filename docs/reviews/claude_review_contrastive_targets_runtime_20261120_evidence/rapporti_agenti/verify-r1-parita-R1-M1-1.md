# Verifica scettica n. 1 del rilievo R1-M1 (r1-parita)

Inizio 20/11. Compito: confutare se possibile. Lavoro in rev_runtime/tmp/sv1_r1m1.

## Lettura del codice a fbfde797 (fatta)
- run.py:1613 `t0 = time.time()` a inizio passo; run.py:1629 prepare, 1631 `_accumulate_terms(...)` (nome globale del modulo, quindi vede il patch); run.py:1731 `t_step_s = time.time() - t0`; run.py:1243 `t_start`, 1821 `elapsed_s = time.time() - t_start`. use_terms vero in produzione (run.py:1314).
- parity.py:348-356: `observed_backward` chiama `observer.measure(...)` PRIMA del backward vivo, dentro la stessa chiamata; measure costruisce 6 probe per punto (parity.py:285-288) solo per C0, update 128/256 (parity.py:248).
- parity.py:295 registra `observer_seconds_including_setup`; nessun lettore lo usa (git grep: solo parity.py:295).
- readout:65-92 `measured_allocation`: per_update = max(elapsed/576, mean(normal)+0.02*premio); chiamata per OGNI braccio (readout:470), incluso C0, sul summary/metrics del full con osservatore (readout:465).
- contrastive_run.py:140-159: la somma delle tre allocazioni <= approved_training_gpu_hours della release e il time_limit del braccio <= allocazione per fase del braccio.

## Riproduzione indipendente (fatta, script miei in sv1_r1m1/scripts: obs_timing.py, analyse.py, hist_alloc.py)
- Trainer reale a fbfde797, 4 rank gloo CPU, C0, batch 32, accum 1, prefetch; UPDATES=(2,4) e assert_numeric_profile patchati (CPU). Ogni chiamata a measure cronometrata.
- Coppia A1/O1: t_step_s A [1,295; 1,496; 2,883; 4,611] contro O [1,696; 10,021; 1,445; 15,079]; measure su rank0 = 8,593 s (update 2) e 13,052 s (update 4); O - measure = 1,428 e 2,028 s (in linea con gli altri passi). elapsed_s A 15,289, O 32,526 (diff 17,2; somma measure 21,6; rumore CPU sul passo 3-4 di A). observer_seconds_including_setup = 8,56-8,57 e 13,01-13,03 s sui 4 rank (coincide con measure).
- measured_allocation reale (hist_alloc.py) sui benchmark storici bf16: C0 elapsed 638,2 -> 1,1080 s/update, fasi [8700, 24600], 37,0 GPUh; con +X s dell'osservatore (40% a 128, 60% a 256 e in elapsed): +10 s 38,0; +20 s 38,33; +30 s 39,0; +40 s 39,67 (+2,67 GPUh, +7,2%); +60 s 40,67. CN 38,33, CR 43,33 invariati. Record 128 normale, 256 diagnostico (grad_steps contiene 256); normali 497, diagnostici 71 in records[8:]. elapsed/576 domina sempre.
- Coppia A2/O2 (back-to-back, meno rumore): t_step_s A [1,015; 0,842; 0,873; 1,388], O [1,019; 6,348; 0,876; 10,189]; measure 5,491 e 8,783 s; O - measure = 0,858 e 1,406 (= A entro 20 ms); elapsed A 6,943, O 21,236: diff 14,294 = somma measure 14,274 (entro 20 ms). Identita' 1:1 confermata: il tempo dell'osservatore si somma esattamente a t_step_s degli update osservati e a elapsed_s.
- Nota operativa: con base relativa '.' lo store file:// relativo non parte (nessun output); rieseguito con percorso assoluto.

## Ricerca di motivi per confutare (fatta)
- Nessuna sottrazione altrove: `observer_seconds_including_setup` compare solo in parity.py:295; il readout incorpora il rapporto di parita' (readout:516) ma measured_allocation (readout:470) gira prima e non lo legge.
- measured_allocation e' invariata fra 438c8d65 e fbfde797 (il diff del readout non tocca 65-92): precede l'osservatore; il suo "diagnostic overhead" e' il premio dei passi con contrast_global_grad_base riscalato al 2%, non l'osservatore.
- Prontuario, risposta R-1..N-7, piano e preflight_plan: nessuna dichiarazione che la proiezione del C0 includa l'osservatore (unica menzione dell'osservatore: prontuario riga 36). preflight_plan: long_training_budget null; plan: measured_new_runtime/allocated_caps null.
- L'osservatore e' attivo solo in benchmark/full/C0 (contrastive_run.py:355-357), mai nel training: il costo non si ripete.
- Simmetria parziale: CN/CR full hanno trace_resume_updates (solo impronte, update 545-576, confine a 544, nessun forward/backward in piu'): costo di pochi secondi contro ~20-40 s dell'osservatore. Il checkpoint 544 e' in tutti e tre i bracci (contrastive_prepare.py:404-405).
- Proxy host del probe col modello di produzione (28.559.133 parametri, Mac CPU): costruzione 0,15 s, load+teacher 0,03, impronte studente/teacher 0,09, gradiente copia+hash+norma 0,09-0,11 -> ~0,36 s per probe senza DDP/coda/forward-backward. Limite inferiore realistico su GPU: 6 backward normali + 6 diagnostici (storico 0,8 e 1,7 s) + setup -> almeno ~20 s; stima 20-40 s.

## Effetto e gravita'
- Effetto certo nel segno e nel meccanismo; ampiezza GPU non misurabile qui: +20 s -> +1,33 GPUh (+3,6%), +40 s -> +2,67 GPUh (+7,2%) sul C0 storico. Inquina anche diagnostic_update_mean_s del C0 (~+20% con ~24 s a 256) e normal_update_mean_s (~+4%), letture descrittive del throughput del profilo.
- Non bloccante: sovrastima (prudente). validate_measured_budget (contrastive_run.py:140-159) usa le allocazioni come tetto superiore del time_limit e come somma <= cap approvato; il cap si fissa dopo la lettura (budget_decision_pin), quindi nessun cancello reso impossibile. Unico scenario negativo: cap fissato in modo indipendente nella finestra di ~1-3 GPUh fra somma vera e somma gonfiata.
- Esito: NON confutato. Gravita' MINORE confermata.
- Test: test_contrastive_benchmark_readout.py -k measured_allocation: 1 passed (basetemp sv1_r1m1_pt). Copia di sola lettura pulita (git status vuoto).
- Pulizia: cancellati run_*, store_*, data e basetemp; restano solo gli script in sv1_r1m1/scripts e i log O2.log/A2.log.
