# Review Claude: rilettura del diff runtime del pacchetto TCN frozen (20/11/2026)

Oggetto: la revisione di GPT seguita alla mia review operativa `527f8da`
(`docs/reviews/claude_review_pose_temponet_runtime_20261119.md`). Ramo `codex/pose-temponet-implementation-20261119`; diff
letto: `git diff 2f982114 01252b6c`.

**SHA rivisti:**
- codice `01252b6c3023c49046b312e003b7915d8f3f6119`;
- consegna `aa3c8047ab8078d5dcc4e061537c3e5b35189bc8`, che rispetto al codice cambia solo il pin;
- `source_sha256` del piano `027bb7876b372ceb537610322c11f1903c8c2224abfca27484c403bbc297d732`, 246 sorgenti (245 più il nuovo
  dispatcher).

Richiesta: messaggio di GPT incollato da Simone il 20/11. Rilettura del solo diff, senza agenti, come concordato nella sezione 7
della `527f8da`.

## Metodo

- Letti la risposta punto per punto (`docs/reviews/pose_temponet_runtime_response_20261119.md`), il prompt, il diff dei
  sorgenti, dei launcher, dei test, del piano e del documento d'esecuzione.
- Checkout isolato e staccato di `aa3c8047`, nel mio worktree; il worktree di GPT non è stato toccato.
- Test sul Mac:
  - i 20 file TCN: 263 passati in 140,75 s, compresa la catena dei 24 stadi;
  - i 5 file storici: 125 passati e 1 fallito, `test_concrete_campaign_remains_review_gated_and_all_endpoints_pinned`, lo stesso
    della base;
  - suite completa con solo `src` nel path: 2.622 passati e lo stesso fallimento in quattro parti senza il test DDP, poi il test DDP da
    solo, 6 passati: in tutto 2.628 passati su 2.629;
  - `bash -n` pulito sui tre launcher.
- Nessun agente, nessun job, nessun accesso al cluster.

## Esito

**APPROVATO: review operativa favorevole del pacchetto `01252b6c` (consegna `aa3c8047`).** I quattro blocchi e i sei importanti
sono chiusi nel codice, ciascuno con il suo test. Le minori sono corrette, oppure dichiarate nel documento d'esecuzione come
limiti operativi. Ricetta v3, proposta scientifica, soglie, grafo dei 24 job e tetti non cambiano.

Che cosa vuol dire. L'operatore «2-TCN head launcher» può eseguire `release-plan` con questa review: la review nomina lo SHA
completo del codice e il `source_sha256`, come ora richiede il codice. Poi procede con la sequenza e i gate del documento
d'esecuzione:
1. preflight e parità R-1;
2. benchmark;
3. `resource-plan` con la misura nuova;
4. WLS, calibrazione e gate in forma chiusa;
5. controlli, selezioni, rifit, inferenza bloccata, metriche e raccolta.

Che cosa non vuol dire: non attesta tempi, memoria, determinismo o ambiente su A100 e su Leonardo, né l'esito dei gate sui dati
veri. Li decidono i gate, che possono ancora fermare la catena.

## 1. Chiusure

| Rilievo | Correzione verificata nel codice | Test |
| --- | --- | --- |
| B-1 | `pose_temponet.sbatch:14` antepone `$CODE_DIR/src:$CODE_DIR/scripts` al `PYTHONPATH` del modulo | `test_launcher_preserves_module_pythonpath_and_dispatches_frozen_code` |
| B-2 | `resources._job` arrotonda per eccesso al minuto, e il costo usa gli stessi minuti. `validate_graph`, `apply_training_pack` e `submit_command` rifiutano secondi non multipli di 60 | `test_measured_walltime_rounds_before_packing_commands_and_allocation` (fino a `_allocation`, con il TimeLimit di Slurm), `test_graph_and_pack_reject_non_minute_training_allocations` |
| B-3 | `--agente gpt` | `test_submitter_agent_is_accepted_by_the_actual_parser` (contro il parser vero del submitter) |
| B-4 | `_progress` legge `pair.points` e `pair.lr`. Al passo zero stampa gli hash iniziali dei due fold; i rifit stampano `refit_initialization` prima degli update | `test_progress_records_unfinished_curve_and_initial_hashes`, `test_refits_print_initial_hash_before_any_update`, catena dei 24 stadi |
| I-1 | `GateNotPassed(status, gate)` in `controls.py`. Dai controlli in poi ogni stadio scrive `combined_gate.json`. Un gate fallito scrive `failed_gate.json` e `gate_status` con stato `gate_failed`. WLS conserva lo stato vero; il teacher degenere dà `control_non_calibrated`. `commands --stage controls` richiede `--closed-reference-gate-output`, salva il gate anche negativo prima di stampare qualsiasi riga, e con gate negativo non ne stampa nessuna | `test_nondiagnostic_outer_gate_is_saved_before_neural_fit`, `test_calibration_gate_status_survives_failure_receipt`, `test_no_neural_command_before_combined_closed_form_gate`, `test_commands_persist_closed_form_gate_before_emitting_any_command`, `test_controls_commands_require_an_explicit_new_gate_artifact_path` |
| I-2 | Il benchmark valuta con `workflow._score_model`, compreso il punteggio pulito dei controlli, e misura l'inferenza con `data_api.view_inputs`, come la produzione | `test_benchmark_score_runs_production_cast_mask_finite_and_metric_path` |
| I-3 | `identity` e stato riscaldati prima dei 10 campioni. Le istantanee in memoria sono misurate a parte rispetto a scrittura e lettura degli artefatti, conteggiate 2 per fold e LR e 1 per seme di rifit | `test_benchmark_measures_real_artifacts_separately_from_memory_snapshots`, `test_resource_formula_counts_memory_snapshots_separately_from_artifact_io` |
| I-4 | Margine temporale di L0 e L13 con fit su `y*`, valutato su `y`; statistiche e residui registrati | `test_temporal_margin_fits_clean_target_and_scores_noisy_target` (L0 entro 1e-12 da 0) |
| I-5 | `commands` pretende 40 esadecimali e un percorso relativo prima di leggere piano o git. Il wrapper verifica commit, hash del piano (64 esadecimali, letto dal commit pubblicato) e percorso prima di creare il checkout | `test_commands_reject_non_full_plan_commit_before_any_git_or_plan_read`, `test_checkout_checks_plan_before_any_campaign_mutation` |
| I-6 | `resource-plan --operator-checks`: `observed_at` con fuso orario, successivo ai `finished_at` dei due benchmark autenticati; la misura entra nel piano prima del gate | `test_resource_plan_requires_new_checks_after_both_authenticated_benchmarks`, `test_resource_plan_replaces_stale_quota_and_binds_new_file` |

Minori:
- **L1-2:** chiusa con gli stessi eventi di B-4.
- **L3-6:** `release-plan` pretende nella review lo SHA completo del codice o il `source_sha256`
  (`test_review_must_name_actual_code_or_source_before_materializing_it`).
- **L3-7:** gli stadi dopo il preflight passano da `pose_temponet_dispatch.sbatch`, che verifica `EXPECTED_SHA` ed esegue il
  launcher del checkout congelato.
- **L4-4:** il preventivo delle metriche comprende l'I/O, le due valutazioni R-1 e il batch 32.
- **L4-5:** misurata la prima lettura del runner; la page cache del sistema non è controllata, ed è dichiarato.
- **L5-2:** `absolute_E_dyn` nella candidatura, con test sul double subito sopra la soglia.
- **L5-3:** la seconda parità R-1 resta in `historical_R1.json`, nella ricevuta e nel rapporto.
- **L5-5:** la catena dei 24 stadi.

Dichiarati come limiti operativi nel documento d'esecuzione: L3-4, L3-8 e L3-9.

## 2. La catena dei 24 stadi

`tests/cpu/test_pose_temponet_e2e_chain.py` fa girare tutti i 24 stadi attraverso `run_stage`, con le funzioni vere di
selezione, rifit, inferenza bloccata, metriche e raccolta; `run_selection` e `run_refits` non sono sostituite.

Che cosa verifica:
- 24 ricevute complete;
- la cartella di sviluppo aperta solo dall'inferenza bloccata (post e pre) e dalle metriche (post);
- 11 selezioni bloccate e 5 rifit con 3 semi ciascuno e update > 0;
- 15 predizioni e un rapporto completo con 5 contrasti;
- la seconda R-1 salvata e uguale nel rapporto;
- almeno 44 eventi al passo zero con lo stesso hash iniziale nei due fold;
- 15 eventi di inizializzazione dei rifit, a zero update.

Che cosa simula, dichiarato nel test e nella risposta:
- CUDA su CPU e una ricetta di 1→2 epoche;
- preflight, benchmark, cache e predizioni storiche, con i valori R-1 della fixture;
- `qualify_controls`, che è eseguito davvero, ma la cui ammissione è forzata (`fixture_forced_admission`, con
  `fixture_unforced_status` conservato).

È il test che avrebbe visto B-4. Non qualifica i controlli sui dati veri, e non lo pretende.

## 3. Note residue (non bloccano)

1. Prima del rilascio e fra le fasi, l'operatore deve fare quattro cose, tutte già scritte nel documento d'esecuzione:
   - confrontare il modello di CPU dei due nodi seriali (L3-4);
   - prendere una misura nuova dopo i benchmark (I-6);
   - preparare le copie locali delle due `calibration.json` per `commands --stage controls` (I-1);
   - non rinfrescare SCRATCH mentre gira uno stadio (L3-8).
2. Un tentativo aggiuntivo richiede una decisione e ID e checkout nuovi (L3-9).
3. Il fallimento del test storico resta su questo ramo. Su `main` (`f546c9e4`) lo stesso test passa: dovrebbe sparire con
   l'integrazione.
4. Per `release-plan`: questa review nomina lo SHA completo del codice `01252b6c3023c49046b312e003b7915d8f3f6119` e il
   `source_sha256` `027bb7876b372ceb537610322c11f1903c8c2224abfca27484c403bbc297d732`. `aa3c8047` ha gli stessi sorgenti.

## 4. Prompt per GPT

```text
Rilettura Claude del diff runtime TCN (codice 01252b6c3023c49046b312e003b7915d8f3f6119, consegna aa3c8047, base 2f982114):
**APPROVATO, review operativa favorevole del pacchetto.**
Review: docs/reviews/claude_review_pose_temponet_runtime_rereview_20261120.md, ramo claude/temponet-db8-20261109 (commit nel
messaggio di Simone). Nomina lo SHA completo del codice e il source_sha256 027bb7876b372ceb537610322c11f1903c8c2224abfca27484c403bbc297d732,
come richiede release-plan.

Rilettura del solo diff, senza agenti. Test riprodotti: 263 TCN passati, compresa la catena dei 24 stadi; storici 125 passati e lo
stesso fallimento della base; suite completa 2.628 passati su 2.629, compresi i 6 del test DDP da solo, con lo
stesso fallimento; bash -n pulito sui tre launcher.

Chiusi nel codice, ciascuno con il suo test, B-1...B-4 e I-1...I-6; minori corrette o dichiarate come limiti operativi
(L3-4, L3-8, L3-9). La catena dei 24 stadi passa da run_stage con selezioni e rifit veri; le simulazioni e l'ammissione forzata
dei controlli sono dichiarate.

Nessuna nuova richiesta. Prima del rilascio e fra le fasi restano i passi dell'operatore già scritti nel documento d'esecuzione:
modello di CPU dei due nodi seriali, misura nuova dopo i benchmark per resource-plan, copie delle due calibration.json per
commands --stage controls, nessun rinfresco di SCRATCH durante uno stadio.
La review non attesta tempi, memoria o determinismo su A100, né l'esito dei gate sui dati veri: li decidono i gate.
Operatore unico: 2-TCN head launcher. Nessun lancio da parte mia.
```
