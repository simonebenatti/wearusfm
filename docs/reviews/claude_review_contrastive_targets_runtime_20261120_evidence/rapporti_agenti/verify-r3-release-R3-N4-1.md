# Verifica scettica n. 1 del rilievo R3-N4 (r3-release)

Rilievo: «validate_release accetta un audit_sacct non pinnato e make_job non controlla --result-out» (NOTA).
Copia: rev_targets/clone, HEAD 0efa4d45 (= fbfde797 per tutti i file di codice; 0efa4d45 aggiunge solo delivery_pin.json).

## Progressi
- Letti contrastive_run.py:165-232 (validate_release), contrastive_release.py:155-251 (validate_audit_accounting, make_job), contrastive_job.py:62-159 (build_command).
- Righe citate confermate a fbfde797: contrastive_run.py:193-195; contrastive_release.py:159-160 (stringa -> Path.resolve senza pin), 236-237 (out=str(out) senza controlli), 241-242; contrastive_job.py:81, 82-86, 141.
- Unici launcher della coorte: scripts/slurm/contrastive_preflight_{cpu,gpu}.sbatch -> solo contrastive_job.py; contrastive_run.py e' raggiunto solo da contrastive_job.py:158 (split_benchmark storico invariato, non usato dai launcher nuovi).
- submit_job.py:12 «Un ID gia' usato non si risottomette» (sottomesso e' finale); prontuario:185 «un tentativo per ID, nessun retry»; prontuario:160 prescrive «output nuovo sotto la radice della coorte» (solo a parole).
- Clone temporaneo: rev_runtime/tmp/n4v1_clone (0efa4d45). Script: rev_runtime/tmp/n4v1_scripts/repro.py.

## Riproduzione indipendente (script mio, rev_runtime/tmp/n4v1_scripts/repro.py, output in repro.out)
Fixture sintetiche con l'ID logico reale dell'audit (CONTRAST-AUDIT-targets-v3-20261118) e source_digest reale 40d39f56...;
stub solo per i cancelli estranei (binding git della review, contenuto del report d'audit, pin di copertura degli input).
- A1 validate_audit_accounting(audit, "<percorso stringa>") ACCEPTED; A3 pin con sha errato REJECTED «pin mismatch».
- B1 validate_release, release.audit_sacct = stringa -> ACCEPTED (= V7 del revisore). B2 pin errato -> REJECTED.
- B3 release con stringa, poi i byte dell'accounting sostituiti con un'altra allocazione valida (job 999) -> ACCEPTED;
  B4 stessa sostituzione con release pinnata -> REJECTED. Quindi la stringa non lega i byte dell'accounting.
- C1 dispatcher, job.audit_sacct stringa -> REJECTED «complete path/SHA256 pin required» (contrastive_job.py:81).
- C2 dispatcher, job pinnato ma release con stringa -> REJECTED «job review/audit accounting differs from execution release» (:141).
- C0 controllo (job e release con lo stesso pin) -> comando costruito: le fixture arrivano fino in fondo.
- make_job con --result-out in un'altra coorte, uguale a fresh_root, fuori da WORK, relativo -> tutti ACCEPTED, out copiato invariato.
  Il dispatcher li rifiuta tutti «fresh output below WORK/wearusfm_runs required» (:82-86), prima di qualsiasi comando;
  controllo con out sotto fresh_root -> comando costruito.
- Estensione (non nel rilievo): make_job per una cella non-audit accetta anche audit_sacct come STRINGA (D5 ACCEPTED,
  contrastive_release.py:241-242 passa job["audit_sacct"] senza pin); il dispatcher poi lo rifiuta (C1). Stessa classe del punto out.
- pytest tests/cpu/test_contrastive_release.py tests/cpu/test_contrastive_job.py nel clone temporaneo: 52 passed.
  test_contrastive_run.py:74 sostituisce validate_audit_accounting con uno stub: il pin a runtime non e' testato.
  test_contrastive_release.py:145 chiama make_job con "/remote/new-output" e un preflight senza fresh_root: nessun controllo di out testato.

## Tentativi di confutazione
- Percorso non di produzione (punto 1): vero, ma il rilievo lo dichiara gia' («Nel flusso previsto non ha effetto»). I soli launcher
  (contrastive_preflight_{cpu,gpu}.sbatch) chiamano contrastive_job.py; contrastive_run.py e' raggiunto solo da contrastive_job.py:158 dopo :81 e :141.
  Lo stadio training non ha launcher ne' release (preflight: training_authorized false, training_release «Absent»). Non confuta: ne fissa la gravita'.
- Controllo altrove (punto 2): il dispatcher controlla out, ma solo nel job gia' inviato; nessun altro strumento (plan, local_validation,
  release) legge fresh_root (grep: solo contrastive_job.py:84 e il generatore del piano). Il prontuario:160 dice all'operatore
  «output nuovo sotto la radice della coorte», ma non lo fa verificare prima dell'invio.
- «Consuma l'ID»: confermato da submit_job.py:12 («Un ID gia' usato non si risottomette»; sottomesso e' finale) e prontuario:185.
- Compatibilita': audit_sacct compare per la prima volta in fbfde797 (git log -S), nessuna release storica usa la stringa; la doppia forma
  di validate_audit_accounting serve a check-audit e al dispatcher (dopo checked_file), non a validate_release.

## Esito
Non confutato. Fatti riprodotti tutti. Gravita' NOTA confermata: nessun effetto nel flusso previsto (punto 1); il punto 2 richiede un
errore dell'operatore contro l'istruzione del prontuario:160, fallisce in modo chiuso prima di qualsiasi misura con risorse trascurabili,
ma brucia l'ID a tentativo unico e richiede una nuova decisione. Nessun effetto BLOCCANTE; MINORE non giustificato.
Correzione suggerita: oltre a quanto proposto, in make_job usare checked_pin anche per audit_sacct (dict obbligatorio).

## Pulizia
Cancellati n4v1_clone, n4v1_data, n4v1_pytest. Restano solo rev_runtime/tmp/n4v1_scripts/{repro.py,repro.out} (per rieseguire:
clone condiviso a 0efa4d45, poi `python -B repro.py <clone> <dir_dati>`). Copia rev_targets/clone pulita, HEAD 0efa4d45.
