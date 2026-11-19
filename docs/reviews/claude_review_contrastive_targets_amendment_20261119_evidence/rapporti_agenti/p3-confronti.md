# P3 confronti - rapporto incrementale

Inizio: 2026-11-19 10:07:11
Copia: clone @ 438c8d65 (detached)

## (a) plan.json nuovo vs storico (stesso protocollo v2)

Diff campo per campo (script ../tmp/p3_plandiff.py):
- CHANGED scope (testo descrittivo) -> dichiarato (doc: "Il nuovo piano e' ...").
- ADDED arms.{C0,CN,CR}.{rms_window_ms=100, ms_fast_window_ms=125, target_cli_args} -> dichiarato (doc righe "Tutti i piani dei tre bracci...").
- ADDED common.{rms_window_ms, ms_fast_window_ms, target_version=fast125_v3, target_cli_args} -> dichiarato (tabella ricetta).
- ADDED calibration.historical_multiplier_interpretation -> dichiarato (doc "Il coefficiente storico ... resta il prior").
- limitations 5->7: +"Historical T100-G differs in two target supports; no matched historical attribution", +"Historical gradient attenuation is a prior..." -> dichiarati.
- ADDED scientific_decisions.target_windows (citazioni di Simone) -> dichiarato; citazioni coerenti con CLAUDE.md (la seconda e' un estratto).
- ADDED amendment="contrastive-target-windows-20261118-v1", target_recipe_amendment{...} -> dichiarati.
- INVARIATI: evaluation intero (pairs, primary/descriptive endpoints, weights, bootstrap 100000/20261116/family 9/alpha .05/0.99444, margins, reading, monte_carlo_boundary_rule, comparator), activation, coverage (soglie e manifest_reference storico), nce, rnc, calibration (formula, range, selection, benchmark_ratio_rule), resources, release=null, input_sha256, reference_recipe.
Esito (a): nessuna differenza non dichiarata; protocollo statistico invariato.

Review configs nuove vs storiche (configs/ storico): uniche differenze loader.rms_window_ms (MISSING->100), loader.ms_fast_window_ms (MISSING->125), target_version (v1->fast125_v3), per C0/CN/CR/calibration.
Fra C0/CN/CR nuovi: unica differenza contrastive.arm. Calibrazione: calibration_only, pesi 1/1 inerti, max_steps 512.

## (a) comparatore vs piano
- scripts/contrastive_compare.py e tests/cpu/test_contrastive_compare.py invariati da c680e95d (stesso commit dell'ultimo plan storico): git diff c680e95d..438c8d65 vuoto.
- Costanti del comparatore == plan nuovo .evaluation (== storico): protocol, pairs, endpoint, 100000/20261116/9/.05/0.99444, pesi P1/P2, margini, 3SE, 420 fit (script ../tmp/p3_compare_consts.py: tutti True).
- Il comparatore NON legge il piano: costanti codificate; nessun test lega plan.evaluation alle costanti (grep vuoto).

## (b) lettori del piano e parita' dei bracci
- contrastive_prepare.PLAN -> plan nuovo (riga 30). Usato da policy_reference (coverage.manifest_reference, calibration.historical_reference), validate_input_bindings (input_sha256), _reader_provenance (pin del piano). Importato da job/audit/run/readout/cuda/split.
- contrastive_source_compat.PLAN_RELATIVE_PATH -> plan storico (solo validazione di evidenze storiche).
- build_review_configs(baseline) riproduce byte-identici i 4 config committati (../tmp/p3_repro.py): 100/125/v3 per C0/CN/CR/calibrazione; baseline restituita 25/50/v1.
- Fra C0/CN/CR unica differenza: contrastive.arm.
- Guardie di stesso-ricetta: calibrazione (config == build_review_configs calibration), benchmark readout (_run_evidence: cfg raw == expected[arm] da calibrated_configs), training (validate_recipe + validate_cohort: bundle di tre bracci dallo stesso receipt). Nessuna guardia nel comparatore finale (contrastive_compare.py non legge config, target_version, amendment o cohort).

- Prova: _run_evidence (benchmark readout) rifiuta un C0 con config storico (campi assenti + v1) contro expected 100/125: "incomplete/unmatched benchmark C0" (../tmp/p3_bench_mixed.py su fixture dei test).
- Prova: contrastive_compare.compare accetta celle posa annotate con ricette diverse per braccio e l'output non porta amendment/ricetta/cohort (../tmp/p3_mixed_recipe.py). grep di rms_window|ms_fast|target_version|amendment|cohort|config nel comparatore: nessuna occorrenza.
- pytest test_contrastive_compare.py + test_contrastive_prepare.py + test_contrastive_benchmark_readout.py: 101 passed (6,3 s).
- Pin: preflight_plan.science_plan.sha256 == sha256(plan.json) = 6081832...; source_digest HEAD = 643a893... == preflight_plan/local_validation.

## (c) T100-G storico
- Dove compare: base dei config (build_review_configs: verifica 25/50/v1 poi 100/125/v3), prior del moltiplicatore lambda (calibration.historical_reference -> review_response/gradient_reference.json, run control30m 25/50), riferimento descrittivo dei primi 8 update (contrastive_benchmark_readout.initial_calibration_description, descriptive_only, nessun 'passed', test dedicato), plan.limitations[5], plan.target_recipe_amendment.historical_comparison, doc emendamento righe 27-30, protocollo par. 6 riga 132 ("C0-T100-G storico a 19.618 come controllo descrittivo di compatibilita'").
- Nessun codice calcola C0-T100-G; contrastive_compare.ARMS = C0/CN/CR; nessun cancello usa T100-G. La "parita' iniziale" fallita nello storico e' C0 vs CN/CR (zero-weight 256 update), non T100-G (preflight_readout bf16: 'CN/CR: zero-weight256-update parity failed').
- Diff config T100-G vs C0 nuovo: blocco contrastive (impalcatura), loader.rms/ms_fast (assenti->100/125), target_version v1->fast125_v3, campi LQ serializzati inerti (false/0.001). amp_bf16 uguale (True).
- Nessuna release APPROVED_FOR_TRAINING nello storico: nessun checkpoint C0/CN/CR a 19.618 da mescolare.

## Rilievi
- MINORE M-1: contrastive_compare.py non lega i tre bracci alla coorte calibrata ne' alla ricetta target; con protocollo v2 invariato l'uscita non registra l'emendamento. Non blocca il preflight (guardie a calibrazione/benchmark/training presenti; manifest del confronto e' consegna futura).
- NOTA N-1: campi target del plan (arms/common/target_cli_args/target_recipe_amendment) non letti dal codice; 100/125 codificati in build_review_configs; nessun test di coerenza piano<->config (oggi coerenti).
- NOTA N-2: protocollo par. 6 conserva l'etichetta "controllo descrittivo di compatibilita'" per C0-T100-G; per la nuova coorte non e' piu' un controllo di compatibilita' (l'emendamento lo dice in sostanza, non rinomina il ruolo).
Fine: nessun BLOCCANTE per il punto 3.
