# Verifica scettica n. 2 del rilievo R5-2 (r5-test)

Avvio: 2026-11-20. Copia di sola lettura a 0efa4d45; mutazioni solo in cloni temporanei sotto rev_runtime/tmp/v52b_*.

## Letture iniziali
- test_contrastive_job.py:26-27 e test_contrastive_run.py:73-74: `validate_review_binding` e `validate_audit_accounting` sostituiti con lambda che restituiscono None (letto a fbfde797/0efa4d45).
- contrastive_run.main e' chiamato da due test (test_contrastive_run.py:123 e :130), ma entrambi si fermano prima di validate_release (release troncata; --recipe-only). La frase «nessun test esegue contrastive_run.main» e' imprecisa alla lettera, corretta nella sostanza.
- Risposta R-2/R-4 e prontuario (righe 145-146) dichiarano i controlli nel codice (release, dispatcher, runtime); non dichiarano test del cablaggio di dispatcher/runtime.

## Riproduzione mia (clone v52b_wt a 0efa4d45, script mio v52b_scripts/mut.py: ogni modifica deve combaciare una sola volta)
- Base dell'intero insieme della ricevuta (29 file, ambiente della ricevuta + --basetemp): 465 passed in 332,94 s (CPU contesa, load ~80).
- Controlli positivi (devono morire): CTRL_J_cuda (job.py:145-146 tolto) -> 1 failed (KeyError job.py:145); CTRL_R_time (run.py:186-187 tolto) -> 1 failed (test_contrastive_run.py:98 DID NOT RAISE). L'harness rileva le uccisioni e il codice mutato e' quello importato.
- Mutazioni singole, test mirati:
  - job+release (52 test): J_bind_call (job.py:70->pass), J_job_review (71-72), J_cell (77-79), J_accounting (81->pass), J_fresh_root (85), J_release_eq (141-142), J_resume_pins (154), L_makejob_accounting (release.py:240-242): tutte 52 passed.
  - run+guards+targets_readout+historical_callsites+mmap+source_compat (87 test): R_bind_call (run.py:182->pass), R_review_pin (183-184), R_sacct_presence (193-194), R_accounting_call (195->pass), R_cuda_prereq (209->pass), R_phase_arm (211-214), R_target_fork (317->pass), R_barrier_sha (348-349), R_numeric_profile (340 -> flags={}): tutte 87 passed.
  - cuda_check+numeric_profile+mmap+source_compat+release+run+job+targets_readout (149 test): L_cuda_prereq_body (release.py:203-204 e 209-211), C_profile_ctx (cuda_check main senza numeric_profile), C_rank_flags (flag dei rank tolti dall'all_gather): tutte 149 passed.

## Ricerca di motivi per confutare
- Nessun altro test chiama build_command, validate_release, make_job (fuori dalla cella audit), validate_target_fork o validate_cuda_prerequisite: grep in tests/cpu (solo test_contrastive_job.py, test_contrastive_run.py, test_contrastive_release.py; make_job solo con la cella audit, che salta l'accounting). Nessun test esegue contrastive_job.py o contrastive_run.py come sottoprocesso; nessun test AST sulle chiamate di binding/accounting (l'unico AST e' quello dei siti mmap).
- Il readout usa invece una spia che registra le chiamate (test_contrastive_historical_callsites.py:104-129, accounting_calls): il contrasto descritto dal rilievo e' reale.
- Documenti: risposta R-2 («Dispatcher e runtime rifiutano review storiche...») e prontuario righe 145-146 («Il gate è eseguito nella release, nel dispatcher e nel runtime della calibrazione») descrivono il codice, non dichiarano test del cablaggio; R-4 dichiara test negativi di validate_audit_accounting in se', che esistono (test_contrastive_release.py:109-157). Nessuna dichiarazione che esenti il cablaggio dai test. Il prompt per Claude (punto 5) chiede proprio «integrazione fra produttori e lettori».
- Attenuante (non confutazione): il codice a fbfde797 contiene davvero i controlli (letti: job.py:69-85, 141-157; run.py:181-214, 317, 348-349; release.py:240-242) e il binding della review lega il source digest: un'eventuale rimozione dopo la review invaliderebbe il binding. La lacuna non cambia l'esecuzione della catena; riguarda il valore probatorio della ricevuta di 465 test.
- Imprecisione minore del rilievo: contrastive_run.main e' chiamato da due test (test_contrastive_run.py:123 e :130), che pero' si fermano prima di validate_release; i lambda sono `lambda value: None` e `lambda *args: None`. Non cambia la sostanza.

## Intero insieme della ricevuta (29 file, 465 test), mutazioni mie
- COMBO di tutte le 20 mutazioni sopra (job 7, run 9, release 2, cuda_check 2) applicate insieme: 465 passed in 149,44 s, rc 0.
- Singole: R_bind_call (run.py:182 -> pass, la riproduzione citata) 465 passed; J_bind_call (job.py:70 -> pass) 465 passed; J_accounting (job.py:81 -> pass) 465 passed; R_accounting_call+R_sacct_presence (run.py:193-195) 465 passed.
- Controllo positivo sul readout: CTRL_B_readout_accounting (benchmark_readout.py:421 -> accounting = {}) UCCISA da test_contrastive_historical_callsites.py:128 (assert [] == [...] della spia). Conferma «il readout e' coperto» e che la tecnica a spia, usata li', manca in dispatcher e runtime.
- Pulizia: cloni v52b_wt, v52b_wt2, v52b_wt3 cancellati; basetemp cancellate da run_mut.sh; copia di sola lettura pulita a 0efa4d45. Restano solo v52b_scripts (148 KB: mut.py, run_mut.sh, results.tsv, log).

## Esito
- Rilievo RIPRODOTTO, non confutato. La gravita' resta MINORE: il codice a fbfde797 contiene i controlli e il binding lega il digest dei sorgenti, quindi la catena non cambia. Manca pero' ogni prova automatica del cablaggio di dispatcher e runtime: binding, accounting, cella, prerequisito CUDA, fase/braccio, fork e profilo numerico. E' proprio la parte che il blocco contrastive-release-review della review attivera'.
- Non BLOCCANTE: nessun effetto dimostrato su esecuzione, evidenze accettate, cancelli o tetti.
- Imprecisioni del rilievo, senza conseguenze: main e' chiamato da due test che si fermano prima della release; i test negativi di R-4 esistono per validate_audit_accounting in se'.
- Non verificato: una prova d'integrazione con un binding vero attraverso build_command per i modi diversi da audit, e attraverso run.validate_release. Il cablaggio e' controllato solo per lettura, con tipi degli argomenti coerenti. Non verificato nemmeno il comportamento su Leonardo (git show nel checkout di campagna, torch 2.2).
