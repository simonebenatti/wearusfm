# verify-r3-release-R3-M2-2 — scettico indipendente n. 2 su R3-M2

Avvio: 2026-11-20. Compito: confutare R3-M2 (provenienza della review verificata solo in bind-review).

## Letture iniziali
- contrastive_release.py a fbfde797: validate_review_binding righe 106-134; review_commit usato solo a riga 109 (_full_sha); original_review_path mai letto; riga 132 confronta original_review_sha256 con file_sha della copia; commento righe 129-131 dichiara il limite.
- bind_review righe 137-152: unico punto in cui git_bytes(review_commit, review_path) viene confrontato con la copia.
- Chiamanti di validate_review_binding: contrastive_job.py:70 (dispatcher), contrastive_release.py:151 (bind), :221 (make_job), :279 (check-binding), contrastive_run.py:182 (runtime); contrastive_run.py:183 confronta review_pin con review_commit (entrambi dichiarati).
- Prontuario fbfde797 righe 126-131: bind-review controlla i byte originali; "L'autore/verdetto rimane responsabilita' del revisore e dell'operatore che ne legge la provenienza: non si pretende una firma crittografica d'autore". check-binding non compare nel prontuario.

## Riproduzione indipendente (clone tmp/vR3M2s2 a 0efa4d45; script in tmp/vR3M2s2_scripts, uscite in tmp/vR3M2s2_work/*.out)
- setup_commits.sh: commit di review simulati senza toccare il working tree: A dcd0f215 (APPROVED, ramo sim-claude-review, padre 0efa4d45), B 16bc393e (figlio di A, ritira l'approvazione, punta del ramo), O 687ab070 (orfano, nessun ramo).
- repro.py: copia della review = byte di A; bind-review CLI rc 0 (3,9 s); make-job audit CLI rc 0; commit di release b9906352 figlio di 0efa4d45, status pulito.
  - check-binding binding reale: rc 0.
  - check-binding binding con review_commit "2"*40 e original_review_path docs/reviews/non_esiste.md (copia fuori dal repo): rc 0, payload stampato. `git cat-file -e 2*40^{commit}` rc 128; il file non esiste.
  - Controlli attivi: original_review_sha256 manomesso -> rc 1 "review copy differs from authenticated original"; status del payload REJECTED -> rc 1 stesso messaggio; review_commit di 39 hex -> rc 1 "review provenance requires its full commit".
  - make-job con il binding falso: rc 0.
- repro_dispatch.py: build_command(job falso, cpu, 7200) -> KeyError 'config' come il job reale (evidenza vuota: passati binding, review e cella); controllo con original_review_sha256 manomesso -> ValueError del binding; cella diversa -> ValueError della cella.
  validate_release (calibrazione, solo gli sha dei 4 input simulati) con binding falso e review_pin "2"*40 -> si ferma dopo, a "CPU audit receipt missing or changed"; controlli: review_pin diverso -> "release review commit differs..."; original_review_sha256 manomesso -> errore del binding.
- repro_bind.out: bind-review accetta il commit orfano O (rc 0) e la versione superata A mentre la punta B ritira (rc 0). La riverifica proposta (git show review_commit:path == copia) passerebbe sul binding superato (True): non copre quel caso; respingerebbe solo il commit inesistente.
- pytest tests/cpu/test_contrastive_release.py nella copia di riferimento: 22 passed in 5,16 s; copia ancora pulita a 0efa4d45.
=> Fatti del rilievo RIPRODOTTI.

## Motivi per cui non e' un difetto (confutazione)
1. Dichiarazione esplicita nel codice, contrastive_release.py:129-131 a fbfde797: "Its original bytes are checked by bind-review; this copy remains pinned on the deploy branch. This is a provenance declaration, not a cryptographic author signature." Dice proprio che dopo bind-review si controlla solo la copia e che review_commit e' una dichiarazione, non solo che manca una firma.
2. Prontuario fbfde797:126-131: bind-review controlla i byte originali; i job portano binding e impronte; "L'autore/verdetto rimane responsabilita' del revisore e dell'operatore che ne legge la provenienza". Il verdetto corrente (versione superata) e il ramo (commit orfano) sono affidati per iscritto alla procedura. check-binding non compare nel prontuario.
3. Piano plan.json:439-445 (generatore contrastive_targets_plan.py:54-55): review_binds = reviewed_source_commit, source_sha256, science_plan_file_sha256, preflight_plan_file_sha256; historical_review_sufficient=false. Tutti e quattro sono riverificati in ogni stadio (validate_review_binding con verify_git=True: git show del commit rivisto per 125 sorgenti e i due piani). review_commit non e' un legame dichiarato.
4. Nessun effetto sui cancelli: con review_commit falso non passano review storiche, di un altro pacchetto, sorgenti o piani diversi, payload manomesso (controlli sopra). review_commit non entra nella provenienza dei run (contrastive_run.py:334-335 e 359-364) ne' nel readout (nessun uso di review/binding in contrastive_benchmark_readout.py): nessun esito attribuito male.
5. Percorso non di produzione: il job deve stare nel checkout congelato (contrastive_job.py:180); il launcher esige CODE_DIR in wearusfm_campaigns, HEAD = EXPECTED_SHA, nessuna modifica (contrastive_preflight_cpu.sbatch:15-18 e 28, lib_code_dir.sh:37 e 41); il checkout e' un worktree bloccato in sola scrittura negata (campaign_checkout.sbatch:45-53). Il binding di produzione e' quello scritto da bind-review (write_new esclusivo) e copiato da make-job; il dispatcher esige release.package_review == job.package_review (contrastive_job.py:141). "Modificato dopo bind-review" vuol dire solo una modifica a mano dell'operatore prima del commit.
6. Riverifica a runtime non realizzabile per costruzione: la release e' figlia del commit rivisto (prontuario:133-134), non del commit della review (ramo di Claude, pubblicato su origin; leonardo riceve main e deploy/*): sul cluster quel commit puo' mancare.
7. La correzione proposta per il Mac duplica bind-review (stessa macchina, stesso operatore) e non cambia cio' che si accetta: un operatore puo' creare un commit orfano con qualsiasi review e bind-review lo accetta; la versione superata passa anche la riverifica proposta. Cattura solo un errore di battitura in due campi di metadati.
8. Lo stesso revisore r3 lo aveva prima classificato NOTA (r3-release.md:23) e poi MINORE (riga 63).

## Esito
refuted = true. Comportamento riprodotto ma dichiarato nel codice, nel prontuario e nel piano, senza percorso di produzione e senza effetti su cancelli o attribuzioni. Gravita' reale: NOTA (chiarire nell'help o nel prontuario che check-binding non riverifica il commit originale; eventuale opzione Mac di riverifica come rinforzo, e procedura per verificare che la review sia la versione corrente).

## Chiusura
- Clone tmp/vR3M2s2 cancellato; conservati tmp/vR3M2s2_scripts (setup_commits.sh, repro.py, repro_dispatch.py) e le uscite tmp/vR3M2s2_work/{repro,repro_dispatch,repro_bind}.out. Copia di riferimento pulita a 0efa4d45.
