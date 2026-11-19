# P2 - Nessuna eredita di evidenze storiche (rapporto incrementale)

Commit: HEAD 438c8d65, consegna e388a1de..0efdc29c

## a) Riferimenti alla coorte storica nei tre file nuovi
- plan.json:166-169 calibration.historical_reference -> review_response/gradient_reference.json (ed4e5b5c...): moltiplicatore 0,1179564 (politica D3 del 16/11, misurato sul controllo 30M storico 25/50). Entra nella formula di lambda (prepare.py:343-348, historical_gradient_multiplier) e quindi nel cancello [1e-4,1]; plan.json:186 lo dichiara "historical prior, not measured attenuation for100/125". Nessun cancello lo usa come evidenza della nuova coorte.
- plan.json:201-204 coverage.manifest_reference -> review_response/coverage_manifest_reference.json (b0d96443...): quote/esenzioni da manifest e durate; indipendente dai target. Usato da build_review_configs (quote) e validate_input_bindings (sha manifest/window_seconds).
- preflight_plan.json:197: solo testo "old_evidence ... invalid". preparation.json: nessun riferimento alla coorte storica (solo baseline T100-G tokenizer_execution_20261103 e review_configs nuovi).
- input_sha256 identici al piano storico (dati, non target).

## Test nominati (HEAD 438c8d65)
- 4 passati in 3,76 s.

## b) Esecuzioni concrete (tmp/p2/probe_b.py, output copiato in reports/p2-probe_b.out.json)
- source_digest() a HEAD = 643a8937..., uguale a preflight_plan.json.source_sha256.
- Audit storico (sha 5c720fad = pin compat): compat guard e validate_audit -> "current execution source inventory/pins differ". Audit storico rietichettato con sorgente corrente: "audit stream/config differs from the calibration recipe" (config_sha256 = config storico 25/50; con il config storico come controllo sarebbe accettato -> la barriera indipendente e' il config 100/125).
- Calibrazione storica (lambda 0,11942686/0,33044627): calibration_report, authenticate_calibration, calibrated_configs -> "calibration run config differs from the frozen 512-batch recipe"; compat guard su run e ricevuta canonica (4f302517 = pin) -> inventario differente. Config rietichettato 100/125 con provenienza vecchia -> compat guard. SOLO falsificazione deliberata di config.json + summary.source_sha256 passa calibration_report (PASSED_FOR_REVIEW, lambda vecchie): non e' accettazione automatica (vedi NOTA).
- 19 job storici (audit/calibration/cuda/benchmarks_20261117/mmap/bf16/readout_bf16): build_command -> "wrong mode, release, source or allocation". 16 release storiche: validate_release -> "release source/config pins differ".
- Release NUOVA (sorgente/config nuovi) che pinna l'audit vecchio -> compat guard. Release benchmark NUOVA con audit sintetico nuovo valido ma ricevuta+coorte bf16 vecchie -> "calibration run config differs".
- Benchmark bf16 C0 (25/50/v1) con ricetta nuova e lambda vecchie: _run_evidence -> "incomplete/unmatched benchmark C0". cuda.json bf16 -> "CUDA/NCCL check ... different source/cohort/inputs". read_preflight con tutte le evidenze bf16 -> "calibration run config differs". authenticate_preflight(preflight_readout bf16) -> "did not pass every frozen gate" (la ricevuta era FAILED_REQUIRES_REVIEW: ripresa CN 0,0325>0,01).
- Eccezione TIMEOUT 64652559: esiste solo come documento releases/audit_recovery_20261117.json (APPROVED_EXACT_AUDIT_RECOVERY); nessun codice lo legge (grep audit_recovery/APPROVED_EXACT/slurm_state vuoto in scripts/src/tests contrastivi). Nessun lettore controlla lo stato SLURM: la non-eredita' e' solo dichiarativa (preflight_plan.json:196).

## c) Test nominati e mutazioni (clone temporaneo tmp/p2/mut a 438c8d65; 8 file contrastivi, 164 casi; log in reports/p2-mutations/)
Controllo M0: 164 passati; sonde: vecchie evidenze rifiutate.
- MA (via 100/125 da build_review_configs): falliscono unmodified_historical_calibration, old_calibration_cannot_authorize, review_configs_share_recipe.
- MC (confronto config che ignora finestre/versione): falliscono unmodified_historical_calibration (match) e old_calibration_cannot_authorize.
- MB (whitelist compat allargata ai sorgenti correnti + manifest rigenerato): fallisce historical_audit_does_not_qualify_new_sources (+11 test sintetici compat). Produzione: audit ancora rifiutato da config_sha256, calibrazione da config.
- MC+MB: calibrazione storica ACCETTATA, CN 100/125/v3 con lambda 0,11942686/0,33044627 -> 3 test nominati falliscono. Rilevato.
- MD+MB: audit storico ACCETTATO per la ricetta nuova -> fallisce solo historical_audit_does_not_qualify_new_sources (per MB).
- MD da sola (validate_audit senza config_sha256), MG (read_preflight senza config_sha256), MF (authenticate_calibration senza guardia ricevuta): 164/164 passati; con MD+MG+MF tutti i 337 test contrastivi passano. Barriere di secondo livello non coperte.
- MH (niente confronto ricevuta/ricalcolo): fallisce solo il test sintetico receipt_is_recomputed; test_modified_historical_receipt_is_rejected[4] resta verde: ora si ferma alla barriera config (vacuo per il suo nome).
- test_pinned_historical_receipt_still_requires_authenticated_recalculation: gira solo nel tree congelato 8905e346 (git archive); verde in TUTTE le mutazioni; non verifica il commento "Current 100/125 sources must not inherit".
- Tutti i 337 test contrastivi (tests/cpu/test_contrastive_*.py + test_rank_contrast.py) con MD+MG+MF applicate: 337 passati in 88,6 s (reports/p2-mutations/all_contrastive_MD_MG_MF.txt).
- Ricevute rank-exit storiche (3) con validate_receipt(source corrente): tutte rifiutate.

## d) Nessuna autorizzazione/ricevuta vecchia nella cartella nuova
- grep in results/infra/contrastive_targets_20261118 (compresi full_suite e log): nessun APPROVED_FOR_PREFLIGHT/PREPARATION/TRAINING, APPROVED_EXACT, PASSED_FOR_REVIEW, CALIBRATED_COHORT; nessuna lambda 0,1194/0,3304; nessun ID 6465*-6478* (unico "6478" e' dentro l'hash scales); nessun hash di ricevute/coorti/audit/sorgenti vecchi (f57d8731, 4f302517, 5c720fad, 46938f31, ab2d6460, 6b421b75, ba8a8657).
- Hash a 64 cifre condivisi con la cartella storica: solo i 4 input dati, la baseline T100-G (f9d03184) e i due riferimenti di politica (ed4e5b5c, b0d96443).
- Celle preflight: status REVIEW_REQUIRED, code_sha null, release null; fresh_root nuova.
- NOTA: plan.json:303-308 copia testualmente scientific_decisions del 16/11, inclusa "authorization": "Simone authorizes launches after convergence; ..."; nessun codice la legge, il piano aggiunge target_windows.execution "Prepare and present new costs before any launch"; portata da esplicitare.

## Conclusione P2
Nessun BLOCCANTE: ogni evidenza storica provata (audit, run e ricevuta di calibrazione, coorte bf16, 19 job, 16 release, benchmark, cuda, rank-exit, ricevuta preflight) e' rifiutata dal codice corrente da almeno due barriere (guardia sorgenti + ricetta/config; per i job/release anche source_sha256).
- MINORE: barriera di ricetta dell'audit (config_sha256 in contrastive_run.py:99-101 e contrastive_benchmark_readout.py:331-334) senza test (MD/MG: 164/164; MD+MG+MF: 337/337); test_historical_audit_does_not_qualify_new_sources copre solo la guardia sorgenti, senza match.
- NOTA: test_pinned_historical_receipt... gira solo su 8905e346 (non tocca il codice corrente; MF non rilevata); test_modified_historical_receipt_is_rejected[4] ora si ferma alla barriera config.
- NOTA: non-eredita' del TIMEOUT solo dichiarativa; nessun controllo di stato SLURM/durata; OverTimeLimit 3 min.
- NOTA: provenienza basata sul contenuto (prova 2g; cuda_check accetta checkpoint strutturalmente validi; la lettura rifiuta a valle).
- NOTA: plan.json:303-308 copia la riga di autorizzazione del 16/11; portata da esplicitare.
Pulizia: clone mutato e work rimossi; basetemp rimossi; copia git pulita a 438c8d65.
