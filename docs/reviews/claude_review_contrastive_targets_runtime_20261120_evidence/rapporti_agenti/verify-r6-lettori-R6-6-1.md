# Scettico n. 1 su R6-6 (r6-lettori): profilo numerico della calibrazione mai riletto

Avvio: 04:16:43

## Progressi
- Letti a fbfde797: contrastive_run.py:269-373 (main), contrastive_prepare.py:266-408 (calibration_report, authenticate_calibration, calibrated_configs), contrastive_benchmark_readout.py:95-111, 201-273, 345-389, 392-526, contrastive_release.py:195-243, contrastive_job.py:62-159, contrastive_numeric_profile.py, prontuario, risposta, decisione scientifica, piani.
- Scrittura: run.py:338-340 entra in numeric_profile(require_cuda=True) per ogni stadio; 358-364 scrive numeric_profile.rank{r}.json anche con --stage calibration (job.py:130,158 usa contrastive_run.py per la calibrazione).
- Lettori (git grep 'numeric_profile.rank' a 0efa4d45, esclusi test): solo readout:213/272 (_run_evidence) e readout:362 (numeric_context_across_runs), entrambi sui sette run (readout:355, 465, 483). calibration_report pinna solo config/summary/metrics (prepare:364); _initial_from_calibration (checkpoint+config), initial_calibration_description (metrics), validate_cuda_prerequisite (checkpoint): nessuno legge le ricevute della calibrazione.
- Probe rev_runtime/tmp/v_r6_6_1/probe_calib_profile.py (fixture_run di test_contrastive_prepare, nessuna scrittura nella copia; git status pulito prima e dopo):
  - senza ricevute: calibration_report PASSED_FOR_REVIEW, inputs pinnati solo config.json/metrics.jsonl/summary.json;
  - con quattro ricevute volutamente errate (TF32 acceso, flash/mem-efficient SDPA, profilo diverso, rank 9, source/config/phase falsi): report identico (True), authenticate_calibration e calibrated_configs accettano (λ invariate su C0/CN/CR);
  - numeric_context_across_runs con un ottavo run CALIB -> ValueError 'numeric context requires all seven full/reload runs'.
- Pytest test_contrastive_targets_readout.py + test_contrastive_prepare.py: 66 passati (8,4 s). Basetemp e output del probe cancellati.

## Tentativi di confutazione
- Controllo altrove? No: nessun lettore (release, job, cuda_check, parity, resume, local_validation) apre le ricevute della calibrazione; job.py esegue con os.execv, nessun controllo dopo il run; la dipendenza 'calibration_accepted' del piano passa da calibration-report (prepare main), che non le legge.
- Percorso non di produzione? No: la calibrazione di produzione passa da contrastive_run.py --stage calibration (job.py:130,158) e scrive le ricevute (run.py:358-364).
- Dichiarazione nei documenti? Parziale: prontuario r. 21-23 «registrate su tutti i rank e confrontate fra i sette run (28 ricevute)» delimita il confronto ai sette run; la decisione scientifica (r. 46) chiede il profilo «fissato e registrato» dalla calibrazione, non riletto. Nessun testo dichiara che le λ non hanno evidenza di profilo letta.
- Garanzia del codice: vera solo per il contratto minimo. calibration_report esige provenance.source_sha256 == source_digest() (prepare:288), che copre contrastive_run.py e contrastive_numeric_profile.py; numeric_profile asserisce i 10 campi di assert_numeric_flags + cuDNN SDPA e solleva se non ottenibili; il trainer (src invariato) non tocca i flag. NON garantiti dal codice ne' confrontati per la calibrazione: bf16/fp16_reduced_precision_reduction, torch_version, cuda_version, CUBLAS_WORKSPACE_CONFIG (confrontati invece fra i sette run, readout:380-384); torch/CUDA compaiono solo in summary.environment (pinnato, mai confrontato). Il rilievo, semmai, e' leggermente sottostimato nella formulazione.
- Effetto: nessuno sui cancelli. Le λ sono un numero unico per i tre bracci, il cancello [1e-4,1] e' lontano da scarti dell'ordine 0,03-0,07%, e i rapporti a pesi attivi sono rimisurati nei benchmark con profilo verificato. Nessun esito attribuito male, nessun cancello falsificato.

## Esito
- NON confutato. Fatto riprodotto (ricevute della calibrazione scritte e mai rilette, non pinnate nella ricevuta). Gravita' NOTA confermata.
