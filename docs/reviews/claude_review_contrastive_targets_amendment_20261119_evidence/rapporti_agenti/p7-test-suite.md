# p7 - Test e ricevuta della suite (revisore Claude, 19/11/2026)

Copia: clone a 438c8d65. Diff della consegna e388a1de..0efdc29c.

## Progressi
- avvio: struttura del repository e diff della consegna
- 65bc6022..438c8d65: `git diff --name-only 65bc6022..438c8d65 -- src scripts tests configs` vuoto (solo docs/ e results/infra/{contrastive_targets,target_windows_integration}_20261118/).
- 0efdc29c..65bc6022 aggiunge solo scripts/diagnostics/contrastive_common_{checkpoint,state_cuda}.py, scripts/slurm/diagnose_contrastive_common.sbatch e i loro due test.
- Insieme (a) diretto: 9 file, 173 test raccolti; insieme indiretto (script che importano contrastive_prepare): audit, cuda_check, rank_exit, common_checkpoint, common_state_cuda = 32 test.

## (a) Esecuzione dei test a 438c8d65 (Mac, venv wearusfm-tok, thread=1)
- Diretto, 9 file: 173 passed in 14.72s (wall 16 s), 0 failed. Per file: prepare 43, job 29, historical_calibration 9, historical_callsites 5, benchmark_readout 26, mmap_compatibility 7, run 14, source_compat 32, target_windows_fix 8.
- Indiretto, 5 file (audit, cuda_check, rank_exit, common_checkpoint, common_state_cuda): 32 passed in 21.44s (wall 23 s), 0 failed.

## (c) Ricevuta della suite (ricostruzione da parse_suite.py, sola lettura)
- sha256 delle 5 evidenze dichiarate in verification.json: tutti corrispondenti; local_validation.full_cpu_suite identico a verification.json.
- collection.log: 3723 nodi unici in 233 file = tutti i 233 tests/cpu/test_*.py di 65bc6022; 4 gruppi + ddp coprono 233 file una volta sola.
- Per parte: passed + failed + errori di setup = test raccolti dei file della parte (854, 876, 951, 1036, 6). Somma passed originali 3592.
- 131 nodi non passati (5 FAILED, 126 ERROR at setup, 114 ERROR at teardown sugli stessi nodi): tutti nei 13 file del recupero; recovery_remaining.xml 263 testcase = tutti i raccolti dei 13 file, 263 passed, 0 errori/fallimenti/skip. Distinti passati ricostruiti: 3723.
- Code tails di cpu_suite.json = ultime righe dei log; recovery_remaining.log = tail del json.
- NOTA: i log originali cpu_part*.log, cpu_ddp.log e recovery_remaining.log non hanno sha256 in verification.json (legati solo dal commit git).
- contrastive_suite_initial.log (0efdc29c): 7 failed / 300 passed, 1 storico poi spostato al reader congelato + 6 multiprocesso nella sandbox; README lo dichiara non verdetto. targeted_tests 121 di local_validation senza log/comando.

## (b) Mutazioni (clone tmp/p7-mut a 438c8d65; M0 senza mutazione: 173 + 32 passed)
- M1a guardia baseline spenta: 3 failed (test_review_builder_rejects_a_nonhistorical_baseline[rms,fast], ..._target_version).
- M1b/M1c/M1d guardia senza versione/RMS/fast: ciascuna 1 failed sul test del componente tolto. Guardia protetta.
- M2a/M2b/M2c (RMS 25, fast 50, versione v1 su base), M2d (solo calibrazione RMS 25), M2e (solo CN fast 50): 1 failed ciascuna, sempre e solo test_review_configs_share_recipe_and_training_weights_are_not_executable; indiretto 32 passed. M2f (tutto 25/50/v1): 3 failed (anche test_old_calibration_cannot_authorize_new_target_recipe e test_unmodified_historical_calibration_does_not_qualify_new_targets). Assegnazione protetta, da un solo test per componente.
- M3a config_from_dict senza with_legacy_target_windows: 41 failed (30 prepare, 1 historical_calibration, 10 run). Protetta.
- M3b with_legacy_target_windows -> R._config_with_defaults: 173 + 32 passed, NON rilevata (sostituzione con l'helper largo vietato da CLAUDE.md nei confronti esatti).
- M7 PLAN riportato al piano storico contrastive_comparison_20261116/plan.json: 173 + 32 passed, NON rilevata. Solo contrastive_prepare legge PLAN; sezioni coverage/input_sha256 e pin calibration.historical_reference identici fra i due piani, quindi l'effetto e' la provenienza (_reader_provenance.plan) della nuova ricevuta.
- I file test_contrastive_{compare,core,memory,runtime,diagnostics,callsite_oracle} non importano i moduli modificati: non possono osservare queste mutazioni.
- M4a audit 3600, M4b regola vecchia (1800/3600), M4c accetta piu' corto, M4d accetta piu' lungo, M5a lettura 7200, M5b controllo kind tolto, M5c GPU 3600: tutte rilevate (3, 3, 4, 6, 3, 3, 10 failed in test_contrastive_job.py). Allocazioni protette.
- M6 testo interpretation del readout riportato al vecchio «Only update 1 may match exactly»: 173 passed, NON rilevata (il test controlla solo descriptive_only e assenza di 'passed').
- M3b e' una mutazione equivalente dentro config_from_dict: i riempimenti di _config_with_defaults coincidono con i default delle dataclass salvo spectral_branches ([] -> tuple ()) e le due finestre (25/50, identiche nei due helper). Non e' una lacuna di test.

## Verifiche aggiuntive
- `contrastive_prepare.py prepare --out tmp`: C0/CN/CR/calibration byte-identici a review_configs/*.json; preparation.json differisce solo nei path (sha256 e bytes uguali). Tutti 100/125/v3, lambda null (calibrazione 1/1 inerte, 512 passi).
- source_digest a 438c8d65 = 643a8937... = local_validation.source_sha256 = preflight_plan.source_sha256; nessun file del perimetro del digest cambia fra 0efdc29c e 438c8d65.
- preflight_plan.json: secondi delle 9 celle = ALLOCATED_SECONDS per modo (audit 7200, GPU 1800, lettura 3600), override `--time=02:00:00` dichiarato. Nessun test lega preflight_plan o plan.json al codice.
- `pytest --collect-only tests/cpu` a 438c8d65: 3723 node id identici a collection.log della ricevuta (7 s).
- Storico: e388a1de..438c8d65 aggiunge solo file in contrastive_comparison_20261116/common_state_diagnostics_20261118/ (merge diagnostici), nessun file storico modificato.
- I 245 blocchi di errore dei log: parti 0-2 tutte con ENOSPC, parte 3 (12 errori di setup) "could not create numbered dir ... after 10 tries" (creazione tmp_path fallita).
- Il rifiuto delle evidenze storiche a sorgenti correnti avviene per "current execution source inventory/pins differ" (audit, calibration_run, calibration_receipt).

## Rilievi (bozza)
- MINORE: M7, il passaggio di PLAN al nuovo piano non e' protetto da test (provenienza della nuova ricevuta; piano non legato al generatore).
- MINORE: M6, testo interpretation del readout non protetto.
- NOTA: log originali della suite non pinnati in verification.json; targeted_tests 121 senza comando.
- NOTA: test_historical_audit_does_not_qualify_new_sources senza match.
- pulizia: rimossi tmp/p7-mut, tmp/p7-logs, tmp/p7-scripts; clone pulito a 438c8d65.
