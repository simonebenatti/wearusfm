# Verifica scettica n. 2 del rilievo R5-5 (r5-test)

Avvio: 2026-11-20. Copia di sola lettura a 0efa4d45; codice letto a fbfde797.
Rilievo: accounting e contesto numerico, controlli senza test che li isolino (MINORE); controllo 'flags differ across ranks' irraggiungibile (NOTA).

## Lettura del codice (fbfde797)
- scripts/contrastive_release.py:155-192 validate_audit_accounting; righe 164-166 (source accounting e source audit), 171-173 (job_id), 176-178 (colonne obbligatorie).
- scripts/contrastive_benchmark_readout.py:345-389 numeric_context_across_runs; 373-376 assert sui flag con NumericProfileUnavailable -> missing -> INCONCLUSIVE.
- scripts/contrastive_benchmark_readout.py:201-273 _run_evidence; 220 assert_numeric_flags (eccezione non catturata); 222-223 coerenza fra rank.
- scripts/contrastive_benchmark_readout.py:285-323 _cuda_evidence; 301 assert_numeric_flags sul profilo di ogni rank CUDA.

## Sonde mie (clone usa e getta rev_runtime/tmp/sk2_r55_clone a 0efa4d45; file tests/cpu/test_zz_sk2_r55.py copiato solo nel clone, sorgente in rev_runtime/tmp/sk2_r55_scripts/)
- 11 sonde sul codice NON mutato: 11 passed in 3.91 s. Ogni guardia citata funziona: C4d audit con source interno 0*64 e accounting coerente -> "not bound"; C4g TSV senza TimelimitRaw -> "unambiguous allocation fields"; C4h job_id "123.batch"/123/"" -> "job ID missing or changed"; C5c 28 ricevute coerenti con matmul_tf32=True -> INCONCLUSIVE_REQUIRES_REVIEW, 28 missing, 0 differenze, calls == []; backstop _run_evidence con TF32 acceso -> NumericProfileUnavailable (eccezione, non INCONCLUSIVE); C5f rank1 con bf16_reduced_precision_reduction diverso -> "differ across ranks"; C5m rank 2 CUDA con TF32 -> NumericProfileUnavailable.
- 222-223 NON e' strettamente irraggiungibile: se una ricevuta omette una chiave che altrove vale null (sul Mac 'cublas_workspace_config', 'cuda_version'), numeric_context_across_runs (confronto con .get) da' PASSED, differenze [] e missing [], mentre _run_evidence (confronto dict) solleva "production numeric flags differ across ranks". Con il produttore reale (contrastive_run scrive sempre tutte le chiavi di numeric_flags()) l'asimmetria non nasce: raggiungibile solo con ricevute non prodotte dal codice rivisto.

## Ricerca di ragioni per confutare (lettura)
- Nessun test fuori dalle 29 ricevute tocca queste guardie: git grep su tests/ trova validate_audit_accounting solo in test_contrastive_release.py (diretto) e stub in test_contrastive_job.py:27, test_contrastive_run.py:74, test_contrastive_targets_readout.py:112, test_contrastive_historical_callsites.py:112; numeric_context_across_runs solo in test_contrastive_targets_readout.py; _cuda_evidence diretto solo in test_contrastive_benchmark_readout.py:306-339 (mutazioni canonical/criterion/truncated, nessuna sul numeric_profile).
- Prontuario e risposta non dichiarano test per queste guardie: risposta R-4 elenca "test negativi di TIMEOUT, exit code, durata, pin, ID e duplicati"; N-3 dichiara "profilo richiesto mancante lascia inconcluso" senza citare un test. Nessuna esclusione dichiarata che renda il buco voluto.
- Contrappesi in produzione (riducono l'effetto, non smentiscono il buco):
  - C4d: contrastive_run.py:191 validate_audit (prima dell'accounting a 195) e contrastive_benchmark_readout.py:413-419 instradano un audit con source diverso in validate_historical_evidence, che per i sorgenti nuovi rifiuta (barriera contrastive_source_compat.py:162, testata), poi config_sha256 contro la calibrazione nuova (barriere M-1, testate).
  - C4g: senza la riga 177 una colonna mancante produce KeyError a 179/184/187 (fuori dall'except TypeError/ValueError): fallisce chiuso, cambia solo il tipo di errore.
  - C4h: nel sacct reale di Leonardo le righe di step hanno TimelimitRaw vuoto (results/infra/tokenizer_evaluation_20261103/probe_cpu_19618/initial_gate/sacct.txt: "63495005.batch|COMPLETED|0:0|596|8||825572K|"), quindi int('') -> "not integral"; la fixture del test ha invece 120 sulla riga .batch.
  - C5c: _run_evidence:220 solleva NumericProfileUnavailable (eccezione, nessun rapporto) invece dell'INCONCLUSIVE documentato; fail-closed.
  - C5m: il produttore contrastive_cuda_check.py:281 scrive numeric_profile = assert_numeric_profile(require_cuda=True) dentro numeric_profile() (301-303); un report legittimo non puo' avere un profilo non conforme.
  - Ricevute per rank: contrastive_run scrive flags dentro numeric_profile(require_cuda=True), che asserisce il profilo all'ingresso.
- Il contrappeso di C4d e' a sua volta testato: tests/cpu/test_contrastive_historical_calibration.py:90 (audit storico -> "current execution source inventory") e tests/cpu/test_contrastive_targets_guards.py:31/46 (barriere config_sha256 in validate_audit e read_preflight). Restano senza ricontrollo dell'audit solo make_job (release.py:240-242) e build_command (contrastive_job.py:80-81), che costruiscono/inviano il job; il runtime ricontrolla a contrastive_run.py:191 prima di misurare.

## Mutazioni mie (driver proprio rev_runtime/tmp/sk2_r55_scripts/mutants.py, una mutazione per volta in 3 cloni, poi sonde e insieme della ricevuta)
- In corso (carico macchina alto, ~25 su 11 CPU).
- NONE (nessuna mutazione): sonde 11 passed; insieme della ricevuta (29 file, PYTHONPATH src:scripts come la ricevuta) 465 passed in 169.40 s, rc 0.
- C4h (release.py:172 senza isinstance/isdigit): sonde 3 failed (job_id "123.batch", 123, ""); ricevuta 465 passed, rc 0 -> SOPRAVVIVE.
- C5f (readout.py:222 `if False:`): sonde 2 failed (unitaria e raggiungibilita' null/assente); ricevuta 465 passed, rc 0 -> SOPRAVVIVE.
- C5c nel clone 4 (solo la sonda, -s): con la riga 374 sostituita da `pass` e 28 ricevute coerenti con matmul_tf32=True la fixture del readout stampa "SK2_C5C_STATUS PASSED_FOR_REVIEW PASSED 0 0" (AssertionError: 'PASSED_FOR_REVIEW' == 'INCONCLUSIVE_REQUIRES_REVIEW'); la sonda del contrappeso (_run_evidence vero -> NumericProfileUnavailable) passa anche sotto la mutazione. In read_preflight non c'e' try/except attorno a _run_evidence (righe 463-465): nel flusso reale l'esito sarebbe un'eccezione, non PASSED.
- C4d (release.py:165 senza `or audit.get("source_sha256") != source`): sonde 1 failed (solo la sonda C4d); ricevuta 465 passed in 157.96 s, rc 0 -> SOPRAVVIVE.
- C5c (readout.py:374 `pass`): sonde 1 failed (solo la sonda dei 28 coerenti non conformi); ricevuta 465 passed, rc 0 -> SOPRAVVIVE.
- C5m (readout.py:301 `pass`): sonde 1 failed (solo la sonda CUDA); ricevuta 465 passed, rc 0 -> SOPRAVVIVE.
- Albero dei cloni pulito dopo ogni mutazione (solo il mio file di sonde non tracciato).
- C4g (release.py:177 `if not records:`): sonde 1 failed (solo la sonda C4g); ricevuta 465 passed in 147.17 s, rc 0 -> SOPRAVVIVE. Launcher: [0, 0, 0].
- Esiti effettivi dei mutanti (clone 4, tests/cpu/test_zz_sk2_r55_outcomes.py, -s): originale -> "unambiguous allocation fields" e "job ID missing or changed" (x2). C4g -> KeyError: 'TimelimitRaw' (fallisce chiuso, cambia solo il tipo). C4h -> job_id "123.batch" con la riga .batch della fixture (TimelimitRaw 120) ACCETTATO {"job_id": "123.batch", "elapsed_seconds": 7199}; con il formato sacct di Leonardo (TimelimitRaw vuoto sugli step) -> "elapsed/allocation is not integral".

## Esito
- NON confutato nella parte MINORE: le sei guardie esistono, funzionano (11/11 sonde) e nessun test delle 29 ricevute le isola: ognuna, rimossa da sola, lascia 465 passed rc 0. Le righe citate (release.py:165/172/177, readout.py:220/222-223/301/374) sono esatte. La descrizione del flusso e' confermata: con C5c la fixture da' PASSED_FOR_REVIEW, il flusso reale un'eccezione a 220.
- Correzione della NOTA: 222-223 non e' irraggiungibile in senso stretto. Una ricevuta che omette una chiave null negli altri rank (es. cublas_workspace_config: nessun launcher contrastivo imposta CUBLAS_WORKSPACE_CONFIG) da' contesto PASSED (confronto .get) e poi ValueError "differ across ranks" in _run_evidence (confronto dict). Con le ricevute del produttore (tutte le chiavi sempre scritte) non accade: va motivato, non tolto.
- Gravita' reale: MINORE, al limite basso. Nessuna evidenza non valida verrebbe accettata nel codice attuale; ogni guardia ha un contrappeso (C4d: validate_audit e validate_historical_evidence, testati; C4g: KeyError; C4h: TimelimitRaw vuoto sugli step nel sacct reale; C5c: eccezione a 220; C5m: assert nel produttore CUDA). Resta un buco reale della verifica consegnata: un comportamento documentato (N-3: profilo richiesto non disponibile -> INCONCLUSIVE) non ha test.
- Pulizia: cloni sk2_r55_clone{,2,3,4} e basetemp cancellati; restano solo script e risultati in rev_runtime/tmp/sk2_r55_scripts (48 KB). Copia di sola lettura pulita a 0efa4d45.
