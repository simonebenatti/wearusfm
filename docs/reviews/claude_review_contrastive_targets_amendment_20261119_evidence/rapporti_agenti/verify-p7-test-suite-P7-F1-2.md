# verify P7-F1 (scettico n. 2) - PLAN al nuovo piano non protetto da test

## Progressi
- avvio 19/11: clone a 438c8d65, lettura del rilievo; riproduzione autonoma in clone temporaneo.
- riproduzione (clone tmp/v2-p7f1/clone a 438c8d65, comandi miei):
  - M0 senza mutazione: 19 file tests/cpu/test_contrastive_*.py + test_target_windows_fix.py = 322 passed (104 s).
  - M7 riga 30 -> `results/infra/contrastive_comparison_20261116/plan.json` (non committata): 322 passed (113 s). RIPRODOTTO.
  - Nessun test che usi git/HEAD sui moduli correnti (run_frozen archivia FROZEN_READER 8905e346 fisso): committare la mutazione non cambierebbe l'esito.
  - Esiste solo tests/cpu; nessun file di test (fuori dai 19 contrastivi) importa contrastive_prepare o i suoi importatori (grep).
- effetto della mutazione (script tmp/v2-p7f1/scripts/plan_effect.py, PLAN commutato in memoria sullo stesso modulo):
  - configs, policy coverage, policy historical_reference, moltiplicatore, rifiuto degli input alterati, status, pesi: identici.
  - unica differenza: report/reader_provenance/plan/{path,bytes,sha256} (6081832627... nuovo vs 7733aa8d... storico).
  - una ricevuta prodotta con PLAN storico e' RIFIUTATA da authenticate_calibration del lettore congelato con PLAN nuovo
    ("calibration receipt differs from authenticated recalculation"): incoerenza fra lettori -> fail-closed.
- controlli altrove:
  - source_digest copre scripts/contrastive_*.py (riga 30 inclusa); preflight_plan.json pinna source_sha256 643a8937... e science_plan
    = nuovo plan.json (sha 6081832627...); contrastive_job.build_command rifiuta job.source_sha256 != source_digest();
    contrastive_run.validate_release rifiuta release.source_sha256 != source_digest(). Una riga 30 cambiata dopo la review non arriva all'esecuzione.
  - Lato codice dei "piani": generatore 100/125/v3 e ALLOCATED_SECONDS sono fissati da valori letterali nei test.
    Mutazione mia (audit 3600 + RMS base 25): 4 failed (3 in test_contrastive_job, 1 in test_contrastive_prepare).
  - Launcher sbatch: TimeLimit da scontrol -> --allocated-seconds; il dispatcher rifiuta ogni valore diverso da ALLOCATED_SECONDS (fail-closed).
  - plan.json common/arms e preflight_plan.json non sono letti da alcun codice (grep): documenti di review, oggi coerenti col codice (verificato).

## Esito (scettico n. 2)
- refuted = false: tutte le affermazioni fattuali del rilievo sono riprodotte (mutazione M7 invisibile ai test; PLAN letto solo
  in contrastive_prepare righe 76/107/219, anche tramite validate_input_bindings da audit/job/run; coverage, input_sha256 e
  calibration.historical_reference identici fra i due piani; nessun test legge plan.json, preflight_plan.json o local_validation.json).
- gravita' reale: NOTA, non MINORE.
  1. L'unico effetto della regressione e' un campo di provenienza (reader_provenance.plan) della ricevuta di calibrazione:
     ricetta, gate, lambda e pin degli input non cambiano. Nessun consumatore legge quel campo.
  2. Il valore revisionato della riga 30 e' congelato da source_digest (pinnato in preflight_plan.json e verificato da
     dispatcher e release): una modifica successiva fallisce in modo chiuso. Una ricevuta prodotta da un lettore con PLAN
     diverso viene rifiutata dal lettore congelato.
  3. "Un loro scostamento passerebbe inosservato" e' sovrastimato: i valori del codice (100/125/v3, 7200/3600/1800) sono
     fissati da test con letterali (mutazione: 4 failed); plan.json common/arms e preflight_plan.json non sono letti dal
     codice e il TimeLimit reale e' controllato in modo chiuso dal dispatcher.
- Il test proposto resta un irrigidimento economico e utile, non correzione di un difetto della consegna.
- pulizia: rimossi tmp/v2-p7f1/clone e tmp/v2-p7f1/work1 (basetemp gia' rimossi); resta solo tmp/v2-p7f1/scripts/plan_effect.py (8 KB).
  Clone principale pulito a 438c8d65.
- non verificato: suite completa (vietata sul Mac); l'eventuale uso futuro di reader_provenance.plan da parte di strumenti di
  release non ancora consegnati.
