# Verifica scettica n. 1 del rilievo P4-N1 (radice d'uscita nuova solo dichiarata)

Copia: clone @ 438c8d65 (sola lettura). Rapporto incrementale.

## Progressi
- avvio: lettura delle prove citate e ricerca di controlli altrove
- git grep 'contrastive_targets_20261118|fresh_root' in scripts/src @HEAD: solo contrastive_prepare.py:30 (PLAN). Riprodotto.
- contrastive_job.py:70-73: unico controllo = out sotto $WORK/wearusfm_runs e inesistente; introdotto in c680e95d (16/11), NON toccato dalla consegna (diff e388a1de..0efdc29c cambia solo ALLOCATED_SECONDS).
- test di rilocazione (test_contrastive_prepare.py:284-293) introdotto in c680e95d (16/11): indipendenza dalla posizione voluta e preesistente.
- Precedente storico: preflight_plan storico aveva 'fresh_outputs' = "WORK/wearusfm_runs/results/contrastive_preflight_20261116_v2/<cell>", anch'esso solo dichiarativo; la review di rilascio (docs/reviews/contrastive_calibration_release_20261117.md:43) lo verificava leggendo i job spec.
- Job spec storici: le release bf16_20261118 e mmap_20261118 scrivevano out sotto radici nuove (contrastive_preflight_20261118_bf16/_mmap) ma pinnavano calibration_summary sotto la radice 20261116_v2: il riuso fra radici della calibrazione era voluto dentro la stessa coorte; la radice e' convenzione delle USCITE, non condizione della ricevuta.

## Riproduzioni mie (script in tmp/skep1_p4n1/scripts/)
- dispatcher_root.py (job readout fittizio, WORK temporaneo): out sotto fresh_root -> accettato; out NUOVO sotto la radice VECCHIA contrastive_preflight_20261116_v2 -> accettato; fuori da wearusfm_runs -> rifiutato; out esistente -> rifiutato. Il fatto citato e' vero: nessun codice confronta out con fresh_root.
- content_gate.py: run di calibrazione storico (calibration_completed_20261117/run) -> calibration_report, authenticate_calibration, calibrated_configs RIFIUTATI ("calibration run config differs from the frozen 512-batch recipe"); stessi file COPIATI sotto <WORK>/wearusfm_runs/results/contrastive_targets_20261118/calibration -> stesso rifiuto; config riscritto a 100/125/v3 -> "historical evidence compatibility: current execution source inventory/pins differ". Un controllo di radice sarebbe strettamente piu' debole del cancello di contenuto (le prove vecchie copiate nella radice nuova lo passerebbero).
- Tutti i lettori della calibrazione (contrastive_run.py:46,58,66; contrastive_benchmark_readout.py:313-314; CLI calibrated-configs) passano da calibration_report.
- pytest: test_calibration_receipt_is_identical_after_evidence_directory_relocation + test_jobs_reject_output_escape_existing_output_and_wrong_roots: 2 passed.

## Motivi per cui non e' un difetto
1. Nessun documento rende la radice una condizione della ricevuta di calibrazione: emendamento:80 dice solo dove vanno le uscite; plan.json non contiene 'fresh'/'root'; plan.json/target_recipe_amendment usa old_receipts_reusable=false + required_new_evidence, garantiti dal contenuto. La premessa "condizione chiesta per la ricevuta" non e' documentata.
2. preflight_plan.json non e' un percorso eseguibile: executable=false, celle con release/code_sha null, nessun codice lo legge (git grep preflight_plan in scripts/src/tests contrastivi: nulla). La radice si materializza nei job spec della release futura, che devono stare nel checkout congelato (contrastive_job.py:159) e pinnare la review (riga 69); required_before_first_submission include "single package review and source pins".
3. Stessa architettura gia' usata e rivista nella coorte storica: fresh_outputs dichiarativo, REMOTE costante nei build_release.py (bf16/mmap), review di rilascio che legge l'output (contrastive_calibration_release_20261117.md:43). Le release bf16/mmap riusavano di proposito calibration_summary di un'altra radice: la radice non e' mai stata un criterio di validita'.
4. Parte invariata: controllo del dispatcher e indipendenza dalla posizione introdotti in c680e95d (16/11), non toccati da e388a1de..0efdc29c ne' dai commit successivi (solo diagnostica).
5. Peggior caso (out di una release futura sotto un'altra radice di wearusfm_runs): l'uscita resta nuova (out.exists rifiutato), prodotta dalla ricetta/sorgente nuove e autenticata per contenuto: nessun effetto su ricetta, riuso, attribuzione, tetti o fallimenti. Solo una cartella con nome fuorviante.

## Esito
Fatto riprodotto, difetto confutato: refuted=true. Se il revisore volesse conservarlo, al massimo come promemoria di release (NOTA), non come lacuna di questa consegna; il fix suggerito per calibration_summary sotto fresh_root sarebbe un vincolo nuovo, contrario al riuso fra radici previsto dal disegno storico.
Pulizia: fakework e basetemp rimossi; copia git pulita (status vuoto).
