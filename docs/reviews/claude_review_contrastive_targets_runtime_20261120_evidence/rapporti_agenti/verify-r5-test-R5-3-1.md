# Scettico n. 1 su R5-3 (r5-test): ripresa, invarianti senza test

Avvio: 2026-11-20. Copia 0efa4d45 (contenuti fbfde797). Rilievo: MINORE, guardie di
compare_three_paths / validate_fork / prepare_reload senza test che le isolino.

## Lettura del codice (fbfde797)
- scripts/contrastive_target_resume.py: righe citate coincidono (start.json 324-329, endpoint 333-340,
  confine stato 364-366, confine RNG 367-371, invarianti di riga 373-380, pesi 381-385, traccia=metriche 389-391,
  rank_rng 77-82, copia byte-identica 104-105).
- Usi in produzione: contrastive_run.py:265 (validate_fork via validate_target_fork), :344 (prepare_reload),
  :354 (trace_resume_updates); contrastive_benchmark_readout.py:478-487 (compare_three_paths).
- Unico file di test che chiama le funzioni: tests/cpu/test_contrastive_target_resume.py;
  test_contrastive_targets_readout.py:131 sostituisce compare_three_paths con monkeypatch.
- Documenti: risposta R-1 ripresa dice solo "Test con trainer CPU reale e due processi nuovi";
  prontuario 49-71 dice "Config, hash dei byte iniziali, mapping delle origini e stato al confine sono controllati"
  (nessuna dichiarazione di test negativi per queste guardie, nessuna dichiarazione di limite).

## Riproduzione (script mio tmp/sk1_r53_scripts/mut.py, clone tmp/sk1_r53_clone a 0efa4d45)
Base non mutata: test_contrastive_target_resume.py -> 18 passed in 48.19s.
Mutazioni singole, solo file di test mirato (18 test):
- boundary_state (resume.py:365 -> `if False:`): 18 passed (SOPRAVVIVE)
- boundary_rng (:370 -> `if False:`): 18 passed (SOPRAVVIVE)
- start_json (:325 `if False and (...`): 18 passed (SOPRAVVIVE)
- endpoint (:333 `if False and (...`): 18 passed (SOPRAVVIVE)
- summary_counters (:342, extra non citato): 18 passed (SOPRAVVIVE)
- row_invariants (:373 `if False and (...`): 18 passed (SOPRAVVIVE)
- weights (:384 -> `if False:`): 18 passed (SOPRAVVIVE)
- trace_metrics (:390 -> `if False:`): 18 passed (SOPRAVVIVE)
- rank_rng (:77 `if ranks > 1:` -> `if False:`): 18 passed (SOPRAVVIVE)
- copy_identity (:104 -> `if False:`): 18 passed (SOPRAVVIVE)
- boundary_state_lastrow (:365 confronta trace[-1], variante plausibile di regressione): 18 passed (SOPRAVVIVE)
- origin_mapping (:353, extra non citato): 18 passed (SOPRAVVIVE)

## Insieme della ricevuta (29 file, 465 test) con le nove guardie del rilievo disattivate INSIEME
boundary_state+boundary_rng+start_json+endpoint+row_invariants+weights+trace_metrics+rank_rng+copy_identity
-> rc 0, "465 passed in 173.72s"; albero del clone pulito dopo il ripristino. Il fatto del rilievo e' confermato.

## Sonde negative mie (tmp/sk1_r53_scripts/test_r53_tamper.py, fuori dalla copia, fixture vera common_paths)
Codice NON mutato: 9 passed in 18.74s -> ogni guardia rileva davvero l'alterazione che solo lei puo' vedere
(confine stato su trace[0] di reload_a; RNG CPU di trace[0] uguale nei tre percorsi; start.json protocol; STOP;
phase di riga; peso rnc cambiato coerentemente in tracce e metriche; supporto metriche != traccia;
checkpoint sintetico world_size 4 con cuda_rng None; copia corrotta in prepare_reload).
(Prima versione della sonda di copia sbagliata da me: azzerare l'ultimo byte di uno zip non cambia nulla;
corretta con un byte invertito a meta'.)
Ogni mutazione del rilievo applicata da sola contro le sonde: uccisa esattamente dalla sua sonda (1 failed, 8 passed):
boundary_state->test_boundary_state_only; boundary_rng->test_boundary_rng_same_in_all_three_paths;
start_json->test_start_json_protocol; endpoint->test_endpoint_stop_file; row_invariants->test_row_phase_label;
weights->test_weights_changed_consistently_everywhere; trace_metrics->test_trace_disagrees_with_metrics;
rank_rng->test_validate_fork_four_ranks_without_cuda_rng; copy_identity->test_prepare_reload_detects_corrupted_copy.

## Righe eseguite (plugin mio tmp/sk1_r53_scripts/r53_lines.py, sys.settrace solo su scripts/contrastive_target_resume.py)
test_contrastive_target_resume.py (18 passed): MAI eseguite le righe 78-82 (ramo ranks>1 di validate_fork), 105, 329,
340, 344, 366, 371, 379-380, 385, 388, 391; a 362 solo il ramo else (ranks=1). Nessun altro test della ricevuta chiama
le funzioni: test_contrastive_targets_readout.py:131 sostituisce compare_three_paths con uno stub; la produzione
(readout.py:487-488) usa ranks=4, cioe' i rami 362/378-379 mai eseguiti dai test.

## Motivi cercati per confutare (nessuno basta)
- Ridondanze in produzione (riducono l'effetto, non il fatto): endpoint 333-340 e contatori 341-344 sono ripetuti da
  readout._run_evidence (readout.py:205-209, 226-229, 255-262) su continuo e reload prima di compare_three_paths (465, 483);
  la copia 104-105 da contrastive_run.py:348 e da start.json copied_checkpoint_sha256 (resume.py:327, anch'essa senza test);
  rank_rng in parte da readout.py:258 (solo checkpoint finali) e da resume.py:362/368-371. Restano di primo livello e senza
  doppione: confine dello stato 364-366, traccia=metriche 389-391, invarianti di riga 373-380 (RNG 367-371 solo in parte coperto
  dal confronto a coppie, che non vede un'alterazione uguale nei tre percorsi).
- Prontuario 49-71 e risposta R-1 non dichiarano un limite di test per queste guardie (dicono "controllati", "Test con trainer
  CPU reale e due processi nuovi"); il precedente diagnostico storico (365cb1e2, test_contrastive_common_checkpoint.py:52-58)
  aveva invece un test negativo del confine RNG.
- Imprecisione minima del rilievo: test_missing_support_is_inconclusive_not_success:172-174 altera anche trace[0], ma solo
  contrast_rnc_pairs, mai i campi di confine; "mai trace[0]" vale per i campi di confine/RNG.
- Precisazione sulla correzione: un'alterazione di rng_cpu_before_sha256 o dei pesi su un solo reload e' gia' presa dal
  confronto a coppie (TRACE_EQUAL), quindi il test deve asserire boundary_mismatches o alterare in modo uguale i tre percorsi;
  per traccia=metriche l'alterazione va fatta solo nelle metriche.

## Esito
NON confutato. Gravita' MINORE (come M-1 della review 94f5094): le guardie esistono e funzionano sul codice non mutato
(9/9 sonde), quindi la catena non accetta oggi evidenze non valide; manca solo il test che le isoli e il ramo ranks>1
non e' mai eseguito. Non BLOCCANTE; non NOTA (difetto reale della verifica, non di documentazione).
Non verificato: il ramo ranks=4 reale (all_gather_object degli RNG, CUDA/NCCL) non e' eseguibile qui.
Pulizia: clone tmp/sk1_r53_clone e basetemp cancellati; restano solo script e risultati (52 KB) in tmp/sk1_r53_scripts;
copia di sola lettura pulita a 0efa4d45. Fine.
