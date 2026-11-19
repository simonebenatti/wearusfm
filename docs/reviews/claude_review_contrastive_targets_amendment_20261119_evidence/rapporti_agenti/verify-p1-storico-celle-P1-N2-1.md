# Verifica scettica n. 1 di P1-N2 («with_legacy_target_windows non completa target_version»)

Esito: CONFUTATO come difetto (refuted=true), gravita' reale NESSUNO. La meccanica e' riprodotta esattamente, ma e' il
comportamento voluto, documentato, coperto da test, corretto nel merito, fail-closed e irraggiungibile nella linea 3.

## Riproduzione indipendente (copia a 438c8d65, pulita)
Script mio: tmp/skep1_p1n2/repro.py (venv, PYTHONPATH src:tests/cpu:scripts). Output essenziale:
- T100-G storico (sha f9d03184... = BASELINE_SHA256): config_from_dict -> (25.0, 50.0, v1), coerente.
- T100-G senza target_version: with_legacy_target_windows e _config_with_defaults NON aggiungono target_version;
  config_from_dict -> (25.0, 50.0, fast125_v3), incoerente (tvf(50) = v1).
- R.train(...) -> «target_version incompatible with loader.ms_fast_window_ms» (run.py:1248), cartella del run non creata.
- build_review_configs -> «baseline target windows/version are not the historical T100-G 25/50 recipe» (prepare.py:115-117).
- Unico config reale senza target_version nel repository: results/passo6/w1_collaudo_59624004.json (aggiunto il 07/10 alle
  15:58, prima di 0e8ba050 delle 19:36): config_from_dict lo accetta -> (25, 50, v3); build_review_configs lo rifiuta.
- Calibrazione nuova senza finestre e senza tv -> (25, 50, v3); validate_recipe(..., "calibration") la rifiuta
  («configuration differs from the frozen comparison recipe»); controllo positivo sul file vero -> (100, 125, v3).
- Ripresa: T100-G v1 coerente da un checkpoint il cui config non ha target_version -> «PSD target definition changed: use a
  NEW run directory, not resume» (run.py:1334-1337).
- config_to_dict scrive sempre target_version (asdict): ogni config prodotto dal 07/10 lo contiene.
- pytest (4 casi): test_legacy_target_windows_fill_only_the_two_new_fields, test_missing_target_windows_are_historical...,
  test_explicit_target_windows_survive_config_restore, test_config_roundtrip... -> 4 passed in 1,35 s (basetemp cancellato).

## Perche' non e' un difetto
1. Dichiarazione esplicita: run.py:221-223 («non completa nessun altro campo: i confronti esatti ... restano stretti»);
   docs/target_windows_fix_20261118.md:53-54 e 106-107 («un altro campo mancante resta un errore»);
   docs/contrastive_targets_amendment_20261118.md:23-25 (baseline verificata 25/50/v1, poi 100/125/v3 espliciti); CLAUDE.md;
   test tests/cpu/test_target_windows_fix.py:104-121 fissa proprio questa proprieta'.
2. Nel merito non esiste un completamento corretto: 0e8ba050 introduce target_version INSIEME al cambio di normalizzazione
   della PSD multiscala (anchor_targets._band_logpower: hann_periodogram x fs/n, usata per ms_fast 50 e ms_slow 500), e la
   decisione registrata quel giorno (docs/decisioni.md in 0e8ba050: «Versione target ...v1; resume di vecchi run multiscala
   vietato») tratta l'assenza del campo come definizione diversa da v1. Completare con v1 attribuirebbe male la ricetta
   pre-07/10; l'oggetto incoerente e' il segnale fail-closed giusto. Prima del 18/11 il default (v1) avrebbe etichettato in
   silenzio un config pre-07/10 come v1: ora fallisce in modo esplicito (piu' sicuro).
3. Controlli altrove: guardia del trainer, build_review_configs, validate_recipe (confronto esatto), _run_evidence del readout
   (cfg != expected_cfg, riga 205), pin sha256 del config in contrastive_cuda_check (226-228).
4. Irraggiungibile in linea 3: 54/54 config della linea 3 nel repository hanno target_version esplicito (50 v1 storici in
   contrastive_comparison_20261116/{configs,calibrated_cohort_*,calibration_completed_*,benchmarks_completed_bf16_*} e T100-G;
   4 v3 nei review_configs nuovi); l'unico config senza campo (w1_collaudo) e' citato solo in docs, da nessun .py.
5. Perimetro: with_legacy_target_windows e' del commit 0d91297 di Claude, «rivisto e integrato separatamente».

Residuo: la docstring di config_from_dict («without changing the recipe») non e' letterale per un input senza target_version,
ma per quell'input nessuna ricetta corrente lo rappresenterebbe e l'oggetto restituito e' inutilizzabile; la frase in piu'
proposta duplicherebbe documentazione gia' presente. Nessuna azione.

Pulizia: tmp/skep1_p1n2/work e basetemp cancellati; clone pulito (git status vuoto).
