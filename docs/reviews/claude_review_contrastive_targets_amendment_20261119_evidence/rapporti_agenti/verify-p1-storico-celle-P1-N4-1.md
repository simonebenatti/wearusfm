# Verifica scettica n. 1 del rilievo P1-N4 (run_frozen legge le evidenze di 8905e346)

Inizio 19/11. Copia a 438c8d65, solo lettura; file temporanei in tmp/verify-p1n4.

## Lettura
- tests/cpu/frozen_campaign_support.py:16-21 `archive`: `git archive sha *paths` dal commit, mai dal working tree; run_frozen (24-31) esegue il programma nel tree estratto.
- tests/cpu/test_contrastive_historical_calibration.py:17-19 FROZEN_READER=8905e346, FROZEN_PATHS con src, scripts,
  results/infra/contrastive_comparison_20261116 e la config T100-G; usati in 25 e 70. Stesso schema in
  test_contrastive_historical_callsites.py:120-123. Introdotti dalla consegna 0efdc29c (prima, e388a1de, leggevano HEAD con il lettore HEAD).
- Emendamento docs/contrastive_targets_amendment_20261118.md:54-58: «I vecchi piani, config, log e ricevute restano immutati ...
  I lettori congelati restano il modo di riprodurre le analisi vecchie». Dichiarazione esplicita del disegno; non dice che i test verificano la copia di HEAD.
- Consumatori a HEAD di artefatti storici nel percorso nuovo, tutti con pin sha256 a runtime:
  contrastive_prepare.py:28-29/269/425 (BASELINE T100-G, BASELINE_SHA256 f9d03184...), 75-84 policy_reference (pin del plan nuovo:
  review_response/gradient_reference.json e coverage_manifest_reference.json), contrastive_benchmark_readout.py:24-25/97
  (initial_historical_reference.json, HISTORICAL_INITIAL_SHA256 367afd99...). Lo script tmp/verify-p1n4/pins.py: i 3 pin del plan e
  il pin del preflight coincidono con i byte di HEAD.
- contrastive_source_compat.validate_historical_evidence (HEAD) fallisce gia' a :162 (inventario sorgenti corrente) prima dei pin degli
  artefatti: a HEAD ogni ricevuta storica e' rifiutata, quindi la copia HEAD di audit/calibrazione non alimenta la coorte nuova.

## Riproduzione (comandi miei)
- Guardia a HEAD (tmp/verify-p1n4/head_guard.py, copia sola lettura): audit, calibration_receipt e calibration_run tutti RIFIUTATI con
  «current execution source inventory/pins differ» (source_compat.py:162, prima dei pin degli artefatti); authenticate_calibration
  rifiuta con «calibration run config differs from the frozen 512-batch recipe». source_digest HEAD 643a8937...
- Integrita' attuale: `git diff [-M] --name-status --diff-filter=MDRT 8905e346 438c8d65 -- <cartella storica> <T100-G>` -> 0 righe;
  `git ls-tree -r`: 183/183 blob della cartella storica e 1/1 della config T100-G identici a HEAD; dopo 8905e346 solo 77 aggiunte
  (benchmarks/bf16/cuda bf16/common_state/gradient_parity/releases).
- Mutazioni nel working tree di un clone temporaneo --shared a 438c8d65 (tmp/verify-p1n4/mutate.py, run_mut.sh), suite
  tests/cpu/test_contrastive_*.py intera (base: 314 passed in 88 s):
  | mutazione della copia HEAD | esito |
  |---|---|
  | M1 ricevuta calibrazione, byte del peso nce (stesso double) | 314 passed |
  | M1b ricevuta, weights.nce 0,1194 -> 0,2194 | 314 passed |
  | M2 audit.json.gz (+\n) | 314 passed |
  | M3/M3b calibrated_cohort_20261117 CN/CR nce_weight 0,1194 -> 0,2194 | 314 passed |
  | M7 compatibility_20261118/manifest.json (+\n) | 314 passed |
  | M8 calibration run/summary.json (+\n) | 314 passed |
  | M9 plan.json storico (+\n) | 314 passed |
  | M4 config T100-G (+\n) | 16 failed (BASELINE_SHA256: prepare, run, historical_calibration) |
  | M5 review_response/gradient_reference.json (+\n) | 12 failed (pin del plan nuovo, policy_reference) |
  | M6 review_response/initial_historical_reference.json (+\n) | 1 failed (HISTORICAL_INITIAL_SHA256, readout) |
  Con M1b+M2+M8 insieme i 14 test dei due file storici, compresi i 3 run_frozen, passano (-v: 14 passed).
- Prima della consegna (e388a1de, merge di 0d91297) i 3 test ora congelati FALLIVANO gia' («current execution source inventory/pins
  differ»): la copertura della copia HEAD era sparita col cambio sorgenti, non e' stata tolta dalla consegna.
- Convenzione preesistente: frozen_campaign_support da 3263ee98 (06/11); test_consolidation.py:42-44, test_logsquare_probe.py:94-95,
  test_integration_frozen_contracts.py:17 leggono anch'essi le evidenze dal proprio SHA. Precedente del rimedio: test_consolidation.py:41
  confronta i byte HEAD del controller con `git show sha:...` (solo il controller, non le evidenze).

## Esito
Non confutato. Il fatto e' riprodotto e anzi piu' ampio: non solo i run_frozen, l'intera suite contrastiva (314) passa con le ricevute
storiche riscritte a HEAD, anche nel valore della λ storica. Gravita' reale NOTA: nessun consumatore a HEAD di quelle copie (guardia che
rifiuta tutto prima dei pin), quindi nessun effetto su ricetta, accettazione o attribuzione della coorte nuova; i soli artefatti storici
letti dalla coorte nuova (T100-G, gradient_reference, coverage_manifest_reference, initial_historical_reference) sono pinnati a runtime e
una loro riscrittura a HEAD fa fallire i test (M4/M5/M6). Precisazione di perimetro: la config T100-G, citata fra i FROZEN_PATHS, e'
protetta a HEAD da BASELINE_SHA256; il buco riguarda solo ricevute, audit, coorti calibrate, run, manifest di compatibilita' e piano storico.
Pulizia: basetemp cancellati dopo ogni corsa; clone temporaneo rimosso alla fine; copia sola lettura pulita a 438c8d65.
