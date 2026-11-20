# Scettico n. 1 su R1-N4 (rilievo di r1-parita)

Rilievo: ricevute numeriche di calibrazione e oracolo fuori dal confronto delle 28; contesto senza cuDNN/NCCL/driver/GPU.
Inizio: lettura di scripts/contrastive_numeric_profile.py, contrastive_run.py, contrastive_benchmark_readout.py a fbfde797.

## Riproduzione (fatta, comandi miei; a fbfde797, copia a 0efa4d45 identica per il codice)
- `git grep -n numeric_profile fbfde797 -- scripts/`: unici lettori dei file `numeric_profile.rank*.json` in
  contrastive_benchmark_readout.py:210-223 (_run_evidence, 7 run), 272 (pin), 345-389 (numeric_context_across_runs);
  oracolo solo per rank con assert_numeric_flags (readout:299-301, anche via release:195-202). Scrittura per ogni stadio,
  calibrazione compresa: contrastive_run.py:359-364 (dispatcher: contrastive_job.py:130,158 `--stage calibration`).
  calibration_report (contrastive_prepare.py:266-301) legge solo config/summary/metrics.
- Prova dinamica (script $T/sk1_r1n4/scripts/test_sk1_r1n4.py, pytest con readout_fixture dei test, basetemp sk1_r1n4/pt1): 3 passed.
  * ricevute della calibrazione con torch_version, cublas_workspace_config e bf16_reduced_precision_reduction diversi:
    read_preflight PASSED_FOR_REVIEW, contesto PASSED (7 run); cancellate: ancora PASSED. Controllo: stessa divergenza
    in CR_A -> FAILED_REQUIRES_REVIEW (differenze ['torch_version'] x4).
  * _cuda_evidence accetta un rank 2 con torch_version/bf16_reduced_precision_reduction diversi; rifiuta TF32 acceso
    (NumericProfileUnavailable: matmul_tf32).
  * chiavi di numeric_flags: 19, nessuna cudnn/nccl/driver/nome GPU.
  => I fatti del rilievo sono esatti.

## Motivi per cui non e' un difetto (verificati)
1. Dichiarazione esplicita del perimetro: prontuario docs/contrastive_targets_execution_20261119.md:21-23 «Le opzioni effettive
   sono registrate su tutti i rank e confrontate fra i sette run (28 ricevute)»; risposta N-3 (contrastive_targets_response:17)
   «Il readout verifica le 28 ricevute numeriche dei sette run». Il comportamento coincide alla lettera: registrate ovunque,
   confrontate fra i sette. La decisione faa4a8d7 (science_20261119.md:46) chiede il profilo «fissato e registrato su tutti i
   rank ... dalla calibrazione ai training»: non chiede il confronto della calibrazione con i benchmark. plan.json:104-111
   non chiede versioni o hardware.
2. Garanzia non «solo indiretta»: calibrazione e oracolo girano dentro numeric_profile(), che imposta e ASSERISCE il contratto
   prima di qualunque calcolo (numeric_profile.py:109-122; run.py:340; cuda_check.py:300-303 e asserzione per rank a 281):
   fallisce chiuso. La sorgente eseguita e' legata dal source digest (summary.contrastive.provenance calcolato a runtime,
   run.py:334-337; prepare.py:288), che copre anche i launcher contrastive*.sbatch con lo stack dei moduli (prepare.py:192-201).
   Restano non confrontati solo i campi fuori contratto (versioni, riduzioni, CUBLAS_WORKSPACE_CONFIG).
3. cuDNN e NCCL sono gia' registrati per ogni run, calibrazione compresa: summary.json -> environment.modules_loaded
   (src/wearusfm/training/run.py:180-188, 1821-1827; src/ e scripts/slurm invariati fra 438c8d65 e fbfde797). Storico line 3:
   calibration_completed_20261117/run/summary.json e benchmark C0 registrano cuda/12.1, nccl/2.19.1-1, cudnn/8.9.7.29-12,
   cineca-ai/4.3.0, torch 2.2.0a0+git6c8c5ad. Nome della GPU nell'oracolo (cuda_check.py:286-288, rank 0); memoria totale del
   device nel rapporto di parita' (target_parity.py:94). Manca davvero solo il driver (e il nome GPU nei 7 run).
4. Nessun cancello dipende dal confronto mancante: parita' a peso zero tutta nello stesso processo; oracolo autoreferenziato
   (riferimenti e DDP nello stesso processo); λ e stato iniziale della calibrazione sono comuni ai tre bracci e autenticati per
   ricalcolo. Un ambiente diverso nelle riprese darebbe al piu' un FAILED/INCONCLUSIVE (direzione conservativa), non un rilascio.

## Esito
Fatti riprodotti, ma il rilievo descrive un perimetro dichiarato e in parte e' impreciso (cuDNN/NCCL registrati in summary.json;
contratto asserito a runtime). Residuo: driver e nome GPU non nei 7 run, non richiesti da piani o decisione. refuted=true;
se lo si volesse tenere, gravita' massima NOTA.
Non verificato: comportamento reale su Leonardo (LOADEDMODULES nei nuovi run, omogeneita' dei driver), nessun accesso al cluster.
