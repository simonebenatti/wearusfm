# Verifica scettica n. 1 di P7-F1 (PLAN non protetto da test) — 19/11/2026

Copia di lavoro: tmp/v-p7f1-1 (clone --shared, detached 438c8d65).

## Progressi
- Letti prepare.py a 438c8d65 (PLAN riga 30; usi righe 76, 107, 219) e diff e388a1de..0efdc29c.
- grep: PLAN/contrastive_targets_20261118 compare solo in scripts/contrastive_prepare.py:30; nessun test cita il nuovo piano. test_contrastive_prepare.py:25 legge P._read(P.PLAN): segue la mutazione, non la rileva.
- Confronto piani (python3): coverage, input_sha256, calibration.historical_reference identici fra contrastive_comparison_20261116/plan.json (7733aa8d...) e contrastive_targets_20261118/plan.json (60818326...). Diversi: amendment, target_recipe_amendment, arms, common, calibration (altro), limitations, scientific_decisions, scope.
- reader_provenance e' usato solo in _reader_provenance e in authenticate_calibration (ricalcolo dallo stesso lettore): nessun controllo a valle del pin del piano.

## Riproduzione (comandi miei)
- M0 (senza mutazione, tmp/v-p7f1-1 a 438c8d65): 9 file diretti 173 passed; 5 indiretti + test_contrastive_compare 64 passed.
- M7 (riga 30 -> results/infra/contrastive_comparison_20261116/plan.json, sed in tmp/v-p7f1-1): 9 diretti 173 passed; 5 indiretti 32 passed; con compare 64 passed. RIPRODOTTO: nessun test rileva il ritorno al piano storico.
- `git grep` a 438c8d65: nessun file in tests/ cita contrastive_targets_20261118, lo sha 60818326... o preflight_plan/local_validation della linea 3; test_contrastive_prepare.py:25 legge P._read(P.PLAN) e quindi segue la mutazione.
- A e388a1de nessun test legava P.PLAN al piano storico: la lacuna c'era gia', la consegna non ha tolto una protezione.
- Effetto misurato (script tmp/v-p7f1-1-scripts/report_under_root.py, fixture del test): calibration_report e calibrated_configs identici in status, pesi (0,02359128246283844 / 0,04718256492567688), moltiplicatore, input_sha256 e config 100/125/v3 dei tre bracci; differiscono solo reader_provenance.plan (60818326... -> 7733aa8d..., 11809 -> 9110 byte) e le identita' sorgente (source_sha256 643a8937... -> 86cdcb6c..., sources[contrastive_prepare.py], source_manifest_sha256), come per qualunque modifica di prepare.py.
- Oggi concordano: plan.common/arms 100/125/v3 = build_review_configs (4 config); 9 celle di preflight_plan = ALLOCATED_SECONDS; preflight_plan.science_plan = _pin(PLAN) (60818326...); preflight_plan.source_sha256 = source_digest() = 643a8937....

## Attenuanti trovate (non confutano il fatto, pesano sulla gravita')
- reader_provenance ha scope dichiarato «reader checkout only; frozen run SHA/source equality remains a release check» (prepare.py:218); authenticate_calibration lo confronta solo con il ricalcolo dello stesso lettore (righe 381-389): nessuna decisione dipende dal pin del piano.
- PLAN fa parte del source_digest (scripts/contrastive_*.py, prepare.py:195). Con la mutazione il digest diventa 86cdcb6c...: contrastive_job.build_command rifiuta un job pinnato al digest rivisto 643a8937... («wrong mode, release, source or allocation», provato) e validate_release (contrastive_run.py) confronta release.source_sha256 con source_digest(). Un ritorno di PLAN dopo questa review non puo' girare con i pin rivisti: serve un nuovo rilascio e una nuova review del diff (una riga).
- plan.json e preflight_plan.json non sono letti dal dispatcher (nessun riferimento a science_plan/cells nel codice); l'autorita' esecutiva e' build_review_configs + ALLOCATED_SECONDS, entrambi protetti da test (mutazioni M2*, M4*, M5* del revisore; test_stage_allocations_cannot_be_swapped_or_extended). Uno scostamento futuro dei documenti sarebbe documentale, o fail-closed (ValueError prima della misura) per le celle.

## Esito
- NON confutato nei fatti: la mutazione M7 e' riprodotta (173 + 32 passed), PLAN e' letto solo da prepare.py (righe 76, 107, 219), le tre sezioni consumate sono identiche nei due piani, e nessun test lega plan.json/preflight_plan.json a generatore o dispatcher. Non e' una mutazione equivalente come M3b: il campo reader_provenance.plan della ricevuta cambia, e un test potrebbe vederlo.
- Gravita' rivista: NOTA, non MINORE. Il ritorno ipotetico di PLAN non cambierebbe pesi, config, gate o accettazione. Cambierebbe solo un'etichetta di provenienza con scope dichiarato non attestante. Inoltre non potrebbe girare con i pin rivisti: il source_digest cambia e dispatcher/release rifiutano (prova sopra). Lo scostamento fra documenti e codice e' documentale, o si ferma prima di misurare.
- Correzione proposta dal revisore: ragionevole e poco costosa (assert su P.PLAN e su _reader_provenance()['plan']['sha256'] == preflight_plan.science_plan.sha256; coerenza plan.common/arms con build_review_configs e celle con ALLOCATED_SECONDS). Non blocca il prossimo job.
- Pulizia: rimossi tmp/v-p7f1-1, i basetemp tmp/v-p7f1-1-bt-* e tmp/v-p7f1-1-work; resta solo tmp/v-p7f1-1-scripts/report_under_root.py (4 KB). Clone condiviso pulito a 438c8d65.
- Non verificato: suite completa (vincolo di disco). Al suo posto, git grep su tests/: nessun altro test legge contrastive_prepare.py, il nuovo piano o il digest 643a8937.
