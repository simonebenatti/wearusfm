# verify-r6-lettori-R6-1-1 (scettico n. 1, rilievo R6-1: braccio della cella non legato a config/release eseguiti)

Avvio: lettura del codice a fbfde797 (job.py, release.py, run.py, readout, prontuario, piani).

## Riproduzione indipendente (04:1x)
- Clone temporaneo rev_runtime/tmp/sk1_r61_clone a 0efa4d45 (diff fbfde797..0efa4d45 = solo delivery_pin.json); review sintetica committata SOLO nel clone temporaneo (8283e870) per avere un binding reale.
- Probe mio: rev_runtime/tmp/sk1_r61_probe/test_sk1_r61_arm.py. Reali: celle del preflight_plan.json, bind_review/validate_review_binding (verify_git=True), validate_audit_accounting (sacct sintetico COMPLETED 0:0), make_job, build_command. Stub ortogonali: J.validate_input_bindings, J.DATA_ROOTS, input_sha256 del piano visto da make_job (file fisici di Leonardo assenti).
- Esito: 3 passed in 50,07 s.
  - "[SK1] cell CONTRAST-BENCH-C0-targets-v3-20261118 job.arm C0 -> executed config arm CN release arm CN script contrastive_run.py"
  - "[SK1] cell CONTRAST-RELOAD-CN-A-targets-v3-20261118 job.arm CN -> executed config arm CR release arm CR"
  - controlli: release resume_b per cella A respinta (repeat legato via phase); arm nell'evidenza respinto da make_job; etichetta job.arm cambiata respinta da build_command; config C0 su cella C0 accettata.
- RIPRODOTTO: il braccio della cella e' vincolato solo come etichetta; config/release eseguiti possono essere di un altro braccio.
- Riesecuzione da zero con rev_runtime/tmp/sk1_r61_probe/setup_and_run.sh (clone nuovo, review sintetica, pytest, pulizia): "3 passed in 59.96s"; clone e basetemp cancellati; copia di sola lettura pulita a 0efa4d45.

## Ricerca di motivi per confutare (nessuno trovato)
- Runtime: contrastive_run.validate_release (run.py:165-221) non riceve la cella; confronta solo release.arm con il config (205-206); fase non full richiede CN/CR (213-214); validate_target_fork lega il summary continuo a release.arm (256). Argparse (270-288) senza id/arm; il comando del dispatcher (job.py:158-159) non li passa. Nessun riferimento a PREFLIGHT/cells in run.py ne' in src/.
- make_job (release.py:215-243): vieta 'arm' nell'evidenza (233-235), copia arm dalla cella (236-237), non apre config ne' release.
- Launcher sbatch e submit_job.py non leggono il job JSON: nessun legame in codice fra ID del ledger e job JSON (gia' procedurale, indipendente da R6-1).
- Prontuario, risposta 20261119, prompt per Claude, piani, registro e decisioni nel diff: nessuna dichiarazione che affidi all'operatore il legame braccio-cella o che lo dichiari limite noto.
- Lettore: _run_evidence:205 rifiuta cfg != expected[arm]; i config a 576 differiscono solo per contrastive.arm (prepare.py:394-408): nessuna attribuzione scientifica sbagliata possibile.

## Valutazione
- NON CONFUTATO. Gravita' MINORE confermata.
- Non BLOCCANTE: serve un errore dell'operatore nella composizione dell'evidenza; il lettore rifiuta; nessun esito attribuito male; spreco <= 2 GPU-h (allocazione massima 4 GPU x 1800 s) dentro il tetto di 18; catena ferma per politica, ripresa solo con nuova decisione.
- Non NOTA: lacuna reale in una guardia introdotta in fbfde797 il cui scopo dichiarato e' legare il job alla cella fissa ("job differs from the reviewed fixed preflight cell"; make_job "Materialize a stage from its fixed cell"), asimmetrica rispetto al repeat.
- Precisazioni: arm assegnato a release.py:237 (istruzione 236-237); nel caso di un solo braccio sbagliato la discordanza emerge solo nella lettura finale, dopo le altre celle (le cui evidenze restano valide); nello scambio C0<->CN con mappatura per contenuto la lettura passa e attribuisce giusto, restano scambiati solo gli ID.
- Correzione proposta adeguata (solo full/resume); il fixture job_env attuale non ha arm in release/config e andrebbe aggiornato.
- Non verificato: validate_recipe/validate_cohort/validate_release reali con calibrazione autenticata (assente sul Mac): solo lettura del codice; nessuna prova CUDA o su Leonardo.
