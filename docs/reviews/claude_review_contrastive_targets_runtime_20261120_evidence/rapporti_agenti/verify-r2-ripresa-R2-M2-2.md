# verify-r2-ripresa-R2-M2-2 — scettico n. 2 sul rilievo R2-M2 (fbfde797)

Inizio: $(date)
Rilievo: readout:489 sovrascrive check['evidence_sha256'] di compare_three_paths con i soli pin dei reload.
Fri Nov 20 03:42:49 CET 2026

## Lettura del codice (fbfde797 = 0efa4d45 per questi file; il delta e' solo delivery_pin.json)
- readout:487-490: `check = compare_three_paths(...)`; `check['evidence_sha256'] = reload_pins`. Confermato.
- target_resume:347-361: evidence[phase] = checkpoint/summary/config/metrics, + start.json (reload), + origin, + trace-{rank}.jsonl; :423 restituito come "evidence_sha256". Confermato.
- A 438c8d65 la riga era `check['evidence_sha256'] = resumed_pins` su compare_resume, che NON restituiva evidence_sha256 (aveva full_sha256/resumed_sha256): la riga aggiungeva. A fbfde797 la stessa riga, adattata, sovrascrive. Probabile svista di trasposizione, non scelta documentata.
- Nessun altro legame delle tracce: git grep "trace-|ddp.jsonl|start.json" in scripts/src/prontuario/piani: solo scrittura (contrastive_run.py:354, target_resume:118/204) e lettura in compare_three_paths. Il job readout (contrastive_job.py) pinna solo summary.json dei sette run; contrastive_release.py non tocca le tracce; nessun consumatore di arms[*].resume.evidence_sha256.
- Prontuario: "La lettura richiede ... le tracce" (riga 183) e descrive le impronte per rank/update (riga 58), ma non dichiara che la ricevuta le lascia fuori. Risposta alla review e piani: nessuna dichiarazione in merito.
- Cosa resta nella ricevuta: arms[arm].evidence_sha256 del full (= continuo: checkpoint, summary, metrics, config, origin, numeric_profile); pairs[*].final_checkpoint_sha256 dei tre percorsi; fork_sha256; numeric_context_across_runs.profile_receipts_sha256. Mancano solo trace-0..3 x 3 fasi e start.json x 2 (per braccio).

## Riproduzione indipendente (03:47-03:52)
Script mio: rev_runtime/tmp/v2m2_scripts/test_v2m2_evidence.py, copiato in un clone temporaneo v2m2_clone (0efa4d45).
Usa compare_three_paths REALE sulla fixture viva common_paths (trainer CPU reale, CR, ranks=1, fork 2, end 4, due reload in processi nuovi), inserita in read_preflight reale tramite readout_fixture; ricevuta scritta su disco; poi authenticate_preflight.
Comando: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<clone>/src:<clone>/tests/cpu:<clone>/scripts python -B -m pytest -q -s -p no:cacheprovider --basetemp=... tests/cpu/test_v2m2_evidence.py
- Codice invariato: 1 passed in 14,03 s.
  - prodotto: continuous {checkpoint, config, metrics, origin, summary, trace-0}; reload_a/b {..., start.json, trace-0}.
  - pubblicato resume.evidence_sha256: solo reload_a/reload_b con i pin di _run_evidence (stub della fixture: origin, summary).
  - chiavi trace-*/start.json in tutta la ricevuta: [] (nessun altro legame).
  - authenticate_preflight prima delle modifiche: OK.
  - M1 (solo formato di reload_b/trace-0.jsonl): esito invariato, impronta prodotta cambiata, NON rilevata.
  - M2 (rng_cpu_after_sha256 e batch_sha256 dell'ultimo update riscritti in modo coerente nei tre percorsi): esito PASSED invariato, NON rilevata. Cioe' le identita' di RNG/batch registrate possono essere sostituite dopo la ricevuta senza che l'autenticazione se ne accorga.
- Con la correzione proposta nel clone (riga 489 -> check['reload_run_pins'] = reload_pins), V2M2_FIXED=1: 1 passed in 15,22 s; la ricevuta contiene le 10 chiavi trace/start (2 bracci), authenticate OK prima, M1 e M2 RILEVATE ("differs from authenticated recomputation"). La riga 489 e' l'unica causa.
- Con la correzione: test_contrastive_targets_readout.py + test_contrastive_benchmark_readout.py + test_contrastive_historical_callsites.py: 54 passed in 5,18 s (nessuna regressione; nessun test vincola il formato sovrascritto).
- Test del revisore (r2_scripts/test_r2_evidence_overwrite.py) rieseguito nel clone a codice invariato: 1 passed in 1,64 s, stesso output citato.

## Tentativi di confutazione
- Controllo altrove: nessuno. Il job readout pinna solo summary.json; release/run/job non leggono le tracce; nessun manifest finale nel pacchetto.
- Percorso non di produzione: no. read_preflight e' il readout di produzione (contrastive_job mode readout -> main) e authenticate_preflight e' chiamato da validate_measured_budget (stage training).
- Dichiarazione esplicita: nessuna nel prontuario, nella risposta alla review o nei piani; il prontuario dice anzi che la lettura "richiede ... le tracce".
- Scelta voluta: improbabile; a 438c8d65 la riga aggiungeva le impronte (compare_resume non ne restituiva), il nuovo modulo le calcola apposta per restituirle.
- Parziale legame implicito: la prima riga di ogni traccia (confine, RNG prima) e i campi di supporto (= metrics.jsonl, pinnato) sono vincolati dall'esito; il resto (batch/input/RNG/coda degli update successivi) solo a coppie, quindi modificabile in modo coerente (M2).

## Gravita'
- Non BLOCCANTE: al momento del calcolo il cancello usa le tracce vere; una modifica successiva che cambia l'esito e' rilevata (ricevuta != ricalcolo, fail-closed); una che non lo cambia non altera nessuna decisione; nessun tetto o cancello falsificato.
- Difetto reale di codice (impronte calcolate e scartate, in contrasto con il resto della ricevuta che pinna ogni evidenza) senza effetti sui cancelli: MINORE confermato.

## Esito: NON confutato, MINORE.
Pulizia: rimossi v2m2_clone e basetemp v2m2_pt1..4; resta solo v2m2_scripts/test_v2m2_evidence.py. Copia di riferimento pulita a 0efa4d45. Fine: Fri Nov 20 03:49:51 CET 2026
