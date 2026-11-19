# Punto 4 — lambda come prior, ricalcolo, cancelli invariati

Rapporto incrementale (agente p4-lambda). Copia: clone @ 438c8d65; consegna e388a1de..0efdc29c.

## Progressi
- avvio: lettura del diff e dei file di calibrazione

## (a) Formula di lambda — verifiche
- scripts/contrastive_prepare.py:85-95 historical_gradient_multiplier(): legge policy_reference("calibration","historical_reference") dal PLAN nuovo (riga 30 = contrastive_targets_20261118/plan.json), richiede reference_updates [513,19618] e initial_updates [1,8], ricalcola reference_mean/initial_mean dal file pinnato e lo confronta col campo multiplier (rel_tol 1e-14). Nessun calcolo da gradienti nuovi.
- plan nuovo calibration.historical_reference = review_response/gradient_reference.json sha ed4e5b5c... (shasum locale identico); multiplier 0.1179564123141922 = 3.5140269780276796/29.790894018271878; status APPROVED_BY_SIMONE; D3 "2_historical_active_period_mean".
- Sezione calibration del piano nuovo identica alla storica (diff JSON: solo aggiunta historical_multiplier_interpretation).
- contrastive_prepare.py:343-353: ratios = base/term sui 32 selezionati (records[256:], primi 32 congiuntamente informativi, nessuna estensione oltre 512), weight = .1*multiplier*median; fuori [1e-4,1] -> failure_reasons, nessun clipping (valore riportato), non finito -> None.
- authenticate_calibration (377-394) richiede PASSED_FOR_REVIEW e failure_reasons == [] e ricalcola tutto: un solo λ fuori intervallo blocca calibrated_configs per tutti e tre i bracci.
- Difesa ulteriore: ContrastiveConfig.__post_init__ (contrastive_objectives.py:58-63) rifiuta pesi non finiti, None o fuori [1e-4,1].
- source_digest a HEAD = 643a8937... = preflight_plan.source_sha256.

## (b) Inserimento delle λ misurate — verifiche
- review_configs nuovi: C0/CN/CR nce_weight=rnc_weight=null, 100/125/v3, max_steps 19618; calibration: arm C0, calibration_only, pesi 1/1 inerti, 100/125/v3, 512 passi. preparation.json: "training_weights: null until calibration review and dedicated release".
- Unico punto d'inserimento: contrastive_prepare.calibrated_configs (397-409) -> authenticate_calibration -> calibration_report(run_dir) ricalcola tutto; usa measured["weights"]. Chiamato da: CLI calibrated-configs (cohort.json con calibration_sha256/source_sha256/initial_state_sha256), contrastive_run.validate_recipe (benchmark/training: config deve essere == calibrated_configs(...)[arm]), validate_cohort, readout.read_preflight.
- calibration_report (268-273): config.json del run deve essere == build_review_configs(baseline)["calibration"] (100/125/v3) dopo with_legacy_target_windows; poi digest sorgenti (se diverso -> contrastive_source_compat, che a HEAD fallisce: inventario corrente != manifest).
- Prova eseguita (tmp/p4/check_hist.py) con le evidenze storiche 64701482 (calibration_completed_20261117): calibration_report / authenticate_calibration / calibrated_configs -> "calibration run config differs from the frozen 512-batch recipe"; config riscritto a 100/125/v3 con digest vecchio -> "historical evidence compatibility: current execution source inventory/pins differ". Solo un run interamente falsificato (config + provenance riscritti) passa: non e' riuso ma falsificazione, fuori modello.
- Radice d'uscita: il lettore e' volutamente indipendente dalla posizione (test_calibration_receipt_is_identical_after_evidence_directory_relocation); fresh_root di preflight_plan non e' controllata dal codice (dispatcher: solo fresh output sotto $WORK/wearusfm_runs). Protezione dalle evidenze storiche = contenuto (config v3 + digest). -> NOTA.
- Controllo positivo (tmp/p4/check_new.py, fixture del test): ricevuta PASSED pinna il piano nuovo (6081832627ce...); calibrated_configs -> C0/CN/CR 100/125/v3, stessa coppia (λN,λR) nei tre config, pesi effettivi per braccio a 600: C0 (0,0), CN (λN,0), CR (0,λR). λ=1 esatto accettato (estremo incluso); λ=1,0001 -> FAILED_REQUIRES_REVIEW con valore non clippato e calibrated_configs rifiutato.
- Valori storici λN 0,1194.../λR 0,3304... compaiono solo in documenti/ricevute storiche, non in codice, piano nuovo o preflight.

## (c) Regola dei rapporti — verifiche
- plan nuovo calibration.benchmark_ratio_rule identico allo storico (diff JSON: nessuna modifica in /calibration salvo la chiave interpretativa aggiunta).
- readout RATIO_RULE (riga 22) {.005,.5,1.,.25,4.} introdotto in c680e95d e mai cambiato (git log -L); component_ratios 513..576 completo (64), ρ=λ·||g_term||/||g_base||, mediana e p95 interpolati, cancello solo sul termine attivo; rapporto CN/CR righe 386-390. Hunk del diff c680e95d..HEAD nel readout: nessuno tocca queste righe.
- Protocollo storico §5 (righe 104-110) e decisioni.md:7835 (D-3) coerenti.

## (d) initial_calibration_description — verifiche
- Usi (git grep): readout 24-25, 95-111, 360 (solo campo del risultato read_preflight); test_contrastive_benchmark_readout.py:80-85 (valori 2x storico -> mean_ratios 2, nessun errore); riferimento storico e ricevuta storica bf16. Nessun altro lettore in scripts/, src/ o tests/.
- Solleva solo per integrita' (sha del riferimento, update 1..8 presenti, valori finiti), mai per differenze; nessun confronto con historical_first8/historical_means in readout, calibrazione, preflight, cuda_check, audit o run.
- Ricevuta storica bf16: update 1 diverso dallo storico in tutti e sei i campi gia' nella vecchia coorte (dunque nessun cancello implicito).

## Test eseguiti
- test_contrastive_prepare.py + test_contrastive_benchmark_readout.py: 69 passed.
- test_contrastive_historical_calibration.py + test_contrastive_historical_callsites.py: 14 passed.

## Esito provvisorio
Nessun blocco per il punto 4. NOTA: radice d'uscita nuova non verificata dal codice (solo dichiarata).

## Chiusura
- Nessun BLOCCANTE, nessun MINORE. Una NOTA: fresh_root (preflight_plan.json) non controllato dal codice (git grep: solo PLAN in contrastive_prepare.py:30; contrastive_job.py:71-73 richiede solo output nuovo sotto $WORK/wearusfm_runs). Da pinnare nella release nuova.
- File temporanei rimossi (basetemp pytest e cartelle di lavoro); restano solo gli script tmp/p4/check_hist.py e tmp/p4/check_new.py per la riproduzione. Copia git pulita.
