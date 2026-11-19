# P6 - diagnostica bf16 (parita/ripresa) prima del nuovo rilascio

Avvio: 2026-11-19 10:07. Copia: clone @ 438c8d65.

## Progressi (1)

- b) File della diagnostica (scripts/diagnostics/*.py, scripts/slurm/diagnose_contrastive_common.sbatch, due test,
  common_state_diagnostics_20261118/*) : blob identici a 365cb1e2/35ea94ba in 65bc6022 e 438c8d65
  (submission_plan.json presente da 35ea94ba). sha256 pinnati nel submission_plan (driver, sbatch, checkout, submitter,
  resources, prepare_handoff, 5 file storici) coincidono al blob di 365cb1e2 e della punta.
- b) source_digest ricostruito dai blob git (script tmp/p6/digest_from_git.py, 117/119 file):
  8905e346 = 09eeaf93 = 365cb1e2 = 35ea94ba = d9020c20 = 46938f31... (= SOURCE del driver);
  e388a1de = c3aaae78...; 0efdc29c = 65bc6022 = afb89b45 = 438c8d65 = 643a8937... (= preflight_plan.source_sha256).
  Il nuovo albero NON ha il digest atteso dalla diagnostica -> va lanciata dal checkout 365cb1e2 (submission_plan.frozen_commit).
- c) contrastive_cuda_check.py byte-identico 8905e346/09eeaf93/438c8d65 (CRITERIA invariati). Readout: solo stringa
  'interpretation'. Job: solo mappa ALLOCATED_SECONDS (audit 7200). Vecchio plan.json invariato (blob d2711fed);
  nuovo plan vs vecchio: solo aggiunte (finestre, amendment, interpretazione prior) e testi scope/limitations.
- a) Nessun codice legge preflight_plan.json di contrastive_targets o 'diagnostic_resolution': cancello solo procedurale.

## Progressi (2)

- b) Prova eseguita dal nuovo albero (tmp/p6/probe_diag.py, copia @438c8d65): source_digest=643a8937 != SOURCE 46938f31;
  cuda.read_inputs -> ValueError "Frozen source digest differs" (prima riga di read_inputs, prima di input/checkpoint);
  checkpoint.run -> ValueError "frozen production source digest mismatch" (prima di pin input/checkpoint). Nessun file scritto.
  Nel driver reale: CUDA -> profilo bf16_auto "failed", break, status FAILED_REQUIRES_REVIEW, exit 2; ripresa -> failure.json.
- b) Test dei due diagnostici alla punta: 7 passed in 13,72 s (copia pulita dopo).
- c) readout: RESUME_RELATIVE_DISPLACEMENT=.01, RATIO_RULE, parita' a peso zero (rel 1e-4, abs 1e-6, 256 update), _nested_equal .01:
  invariati (unica modifica: stringa interpretation; initial_calibration_description era gia' descriptive_only).
- d) db66b507 (non antenato della punta): solo 5 documenti; submission_plan/resources/driver/sbatch invariati; digest 46938f31.
  N-1..N-4 recepite in modo coerente col codice (COMPARE ha le tre coppie entro braccio; _tensor_distance normalizza per
  ||a - boundary[key]||; build_batches toglie la posa alle prime due finestre = rank 0).

## Progressi (3)

- b) Controllo positivo in clone temporaneo @365cb1e2: source_digest() = 46938f31 = SOURCE; i due driver superano il controllo
  del digest e si fermano solo su FileNotFoundError dei percorsi Leonardo (atteso sul Mac). Dalla punta invece si fermano al digest.
- b) Rilancio/requeue: sbatch rifiuta output esistente (riga 42), entrambi i driver creano l'uscita con exist_ok=False: nessuna
  misura mista o sovrascritta; doc diagnostica: "nessun rilancio automatico".
- a) Dispatcher (contrastive_job.build_command) richiede status APPROVED_FOR_PREFLIGHT, digest del checkout, pin review, uscita
  nuova; nessun file job/release approvato nella nuova cartella; nessun campo o controllo per la diagnostica -> cancello solo
  procedurale ma esplicito in preflight_plan (depends + required_before_first_submission + failure_policy), plan.json
  (known_unresolved_gate), emendamento r.49-52, README, registro GPT 18/11.
- c) CUDA evidence nel readout: nessun percorso di compatibilita', richiede source == cohort e CRITERIA identici; costanti fisse
  .01/1e-5 ricontrollate. CPU launcher: SECS dal TimeLimit reale (override 2h -> 7200 coerente).
- d) Nota: la punta conserva le versioni pre-approvazione dei documenti diagnostici (REVIEW_REQUIRED, senza scheda di lettura):
  db66b507 (23:29) e' successivo al merge 65bc6022 (22:12) e non e' antenato di 438c8d65.
- d) db66b507 sostituisce "Le coppie che richiedono il profilo mancante restano inconclusive" con la sola regola sul profilo
  deterministico mancante: caso generale (fallimento di un profilo precedente -> break) non piu' detto esplicitamente.

## Progressi (4) e sintesi

- Test alla punta: readout + cuda_check + job = 59 passed (7,61 s). Configs nuovi amp_bf16=true come la coorte bf16 storica:
  la condizione dei fallimenti resta, quindi il cancello e' necessario.
- c) Elenco completo delle costanti cambiate (8905e346/09eeaf93 -> 438c8d65) nei 4 script: job audit 3600->7200 s
  (readout 3600 e GPU 1800 invariati); prepare: PLAN -> nuovo piano, baseline verificata 25/50/v1, bracci 100/125/v3;
  readout: solo stringa; cuda_check: nessuna. Nessuna soglia scientifica cambiata.

Esito P6: nessun BLOCCANTE, nessun MINORE. NOTE: (1) cancello solo procedurale e non identificato (pinnare report,
decisione e scheda nella release; ricalcolo tetti se la risoluzione cambia precisione/determinismo); (2) documenti diagnostici
della punta fermi alla versione pre-approvazione (db66b507 non integrato); (3) scheda db66b507: tolta la regola generale
"coppie che richiedono il profilo mancante = inconclusive".
Fine: pulizia di clone temporaneo, basetemp e file di lavoro fatta.
