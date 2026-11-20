# Verifica indipendente n. 2 del rilievo R1-M1 (r1-parita)

Inizio: 20/11. Rilievo: il tempo dell'osservatore entra nel throughput del C0 e nella proiezione delle risorse del training.
Piano: (1) leggere run.py (t_step_s, elapsed_s), contrastive_target_parity.py (observe_control), contrastive_benchmark_readout.py (measured_allocation), contrastive_run.py (attivazione); (2) cercare compensazioni/dichiarazioni in prontuario e piani; (3) riprodurre con un esperimento mio; (4) valutare la gravita'.

## Lettura del codice a fbfde797 (fatta)
- run.py:1243 `t_start = time.time()` all'inizio di train(); run.py:1613 `t0` a inizio passo; run.py:1631 chiamata a `_accumulate_terms` (sempre usata con contrastive: use_terms, run.py:1314); run.py:1731 `t_step_s = time.time() - t0`; run.py:547 `_merge_records`: t_step_s = massimo fra i rank; run.py:1821 `elapsed_s = time.time() - t_start`.
- contrastive_target_parity.py:348-356: `observed_backward` chiama `observer.measure(...)` PRIMA del backward vivo, dentro la stessa chiamata a `_accumulate_terms` -> il tempo dell'osservatore cade dentro t0..t_step_s dell'update 128 e 256 e dentro elapsed_s.
- contrastive_run.py:355-357: observe_control solo per stage benchmark, phase full, arm C0. readout:464-470 usa per il C0 la STESSA cartella full_dirs['C0'] (summary + metrics) per measured_allocation e (readout:503) per zero_weight_parity.json.
- readout:65-92 measured_allocation: per_update = max(elapsed/576, mean(normal)+0.02*premium); nessuna sottrazione di observer_seconds_including_setup (parity.py:295; git grep: il campo non e' letto da nessun modulo).
- Anche CN/CR full hanno sovraccarichi solo-benchmark (trace_resume_updates 545-576 con hash, impronte al confine 544, checkpoint nominato 544: contrastive_run.py:350-354, contrastive_target_resume.py:138-208), ma senza forward/backward aggiuntivi: ordine di grandezza molto minore.
- Valle: contrastive_run.py:140-159 (validate_measured_budget): somma delle tre allocazioni <= release.approved_training_gpu_hours; time_limit_s del braccio <= allocazione per fase del braccio. plan.json: resources.allocated_caps null, requires_measured_benchmark_before_training_release true -> il tetto del training non e' ancora fissato; sara' deciso sul preventivo prodotto dal readout.

## Quantificazione sullo storico (fatta, script mio v2r1m1_scripts/alloc_hist.py, measured_allocation di fbfde797)
- Storico bf16 auto senza osservatore: C0 elapsed 638,2 s, somma t_step 546,5 s, normal 0,8305 (497), diag 1,7272 (71); per_update 1,1080 = elapsed/576 (domina il ramo elapsed); fasi [8700, 24600]; 37,000 GPUh. CN 38,333; CR 43,333.
- Osservatore iniettato (elapsed + X, X ripartito 40/60 sui record 128/256): X=20 s -> 38,333 (+1,333, +3,6%); 30 s -> 39,000 (+2,0, +5,4%); 40 s -> 39,667 (+2,667, +7,2%); 50 s -> 40,0 (+3,0); 63 s -> 40,667 (+3,667).
- Il ramo elapsed domina con e senza osservatore: conta solo l'incremento di elapsed_s; i record 128/256 spostano il ramo t_step di <0,05 s, che resta sotto elapsed/576. L'aritmetica del rilievo (1,108 -> ~1,18; 8.700+24.600 -> 9.300+26.400; +2,7 GPUh, +7%) e' confermata.

## Esperimento indipendente (fatto): trainer reale, 4 rank gloo su CPU, C0, batch globale 32, pesi zero
Script miei: rev_runtime/tmp/v2r1m1_scripts/obs_timing.py (patch solo in memoria: P.UPDATES=(2,4), P.assert_numeric_profile -> flag finti perche' i sei probe girino su CPU, cronometro attorno a ControlParityObserver.measure) e compare_timing.py. Nessun file della copia modificato.
- A1 baseline: t_step_s [1,58; 1,27; 1,09; 2,00], elapsed_s 9,45. A2 baseline: [1,60; 1,18; 1,01; 1,66], elapsed_s 8,97.
- O1 osservatore: t_step_s [3,36; 11,46; 1,65; 15,89], elapsed_s 37,56; measure() 10,00 s (update 2) e 13,19 s (update 4) su ogni rank; observer_seconds_including_setup 9,98 / 13,18; 6 casi x 4 rank, live_state_restored True x4.
- O2 osservatore: t_step_s [1,81; 9,56; 1,38; 15,29], elapsed_s 31,80; measure() 8,23 / 13,14; campo 8,22 / 13,13.
- Incremento degli update osservati = tempo di measure() (O2: +8,34 s contro 8,23; +13,46 contro 13,14); incremento di elapsed ~ somma dei due (O2: +22,6 s contro 21,4; O1 +28,4 contro 23,2 con rumore sui passi 1 e 3). Meccanismo del rilievo riprodotto.
- Il campo observer_seconds_including_setup esclude la preparazione prima di `started` (rng_state, deepcopy dello stato contrastivo, _cpu_state, impronte: parity.py:266-277) e gather/assess finali (parity.py:296-298): qui 0,01-0,02 s, su GPU col 30M probabilmente frazioni di secondo. Una sottrazione basata sul campo lascia un piccolo residuo prudente.

## Ricerca di motivi per confutare (fatta)
1. Percorso di produzione: si'. Osservatore solo nel full C0 (contrastive_run.py:355-357); stessa cartella usata da measured_allocation (readout:464-474); le allocazioni entrano in validate_measured_budget (contrastive_run.py:140-159) e nel preventivo dei training da presentare a Simone (decisione faa4a8d7, punto 3).
2. Dichiarazioni: nessuna. git grep su prontuario, risposta, decisione, piani, decisioni.md e registro: nessuna frase dice che la misura/proiezione del C0 include l'osservatore; plan.json:363 chiede ricevute di throughput nuove; prontuario:25-26 «Il benchmark di produzione deve misurare tempo e memoria a batch32».
3. Compensazioni: nessuna. observer_seconds_including_setup e' scritto (parity.py:295) e mai letto.
4. «Proiezione conservativa» (docstring readout:66-71): copre loader e sovraccarico diagnostico alla frequenza futura, non uno strumento solo-benchmark. Inoltre i valori riportati come misure del C0 risultano distorti: con 40 s di osservatore diagnostic_update_mean_s 1,727 -> 2,065 (+20%), normal 0,8305 -> 0,8627.
5. Asimmetria: anche CN/CR full portano sovraccarichi solo-benchmark (trace 545-576, impronte al confine 544, checkpoint nominato 544), ma senza forward/backward aggiuntivi: ordine di grandezza inferiore. L'asimmetria resta.
6. Dominanza del ramo elapsed: strutturale, non dipende dai dati. sum_t_step/576 > mean(normal) + 0,02*premio perche' i 71 aggiornamenti diagnostici su 576 (12%) pesano piu' del 2%; quindi per_update = elapsed/576 e ogni secondo dell'osservatore entra 1:1 in elapsed/576 (circa 0,057 GPUh per secondo, prima degli arrotondamenti a 300 s).

## Gravita'
Non BLOCCANTE: errore prudente. Il tetto del training non e' ancora fissato (plan.json resources.allocated_caps null; la release del training pinna il tetto dopo il readout), quindi non puo' essere superato. Un time_limit piu' lungo non ferma il training, nessun cancello e nessun esito scientifico viene alterato o attribuito male. Resta un difetto reale introdotto dal diff: osservatore nuovo e measured_allocation invariato. Sovrastima il preventivo del solo C0 di +1,3...+3,7 GPUh (+3,6...+9,9%) per 20-63 s di osservatore e distorce le medie per update riportate per il C0. MINORE, al limite basso (correggibile anche con la sola dichiarazione nel prontuario).

## Esito
NON confutato. Rilievo riprodotto con prove mie; gravita' MINORE confermata.
