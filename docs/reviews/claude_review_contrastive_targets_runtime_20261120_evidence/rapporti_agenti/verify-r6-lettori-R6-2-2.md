# Verifica scettica n.2 del rilievo R6-2 (r6-lettori)

Inizio: lettura di scripts/contrastive_benchmark_readout.py e contrastive_target_resume.py a fbfde797.

## Lettura statica (fbfde797)
- scripts/contrastive_benchmark_readout.py:487-489: `check = compare_three_paths(...)` poi `check['evidence_sha256'] = reload_pins` (assegnazione, non merge).
- scripts/contrastive_target_resume.py:347-361: compare_three_paths costruisce `evidence[phase]` per continuous/reload_a/reload_b con checkpoint.pt, summary.json, config.json, metrics.jsonl, start.json (solo reload), origin_sessions.json (se log_batch_origin), trace-{rank}.jsonl; restituito a :423.
- reload_pins = pin di _run_evidence (readout.py:271-273): config, summary, metrics, checkpoint, origin_sessions, numeric_profile.rank0..3. Niente tracce ne' start.json; nessuna chiave `continuous`.
- Riga storica (438c8d65): `check = compare_resume(...)`, `check['evidence_sha256'] = resumed_pins`; compare_resume non restituiva evidence_sha256 -> residuo confermato.

## Sonda propria (sk2r62_probe, copia sola lettura, 3 passed in 26 s)
- P1 (readout_fixture + comparatore che calcola le impronte come target_resume:347-361): `resume.evidence_sha256` = {reload_a, reload_b} con le sole chiavi di _run_evidence; 'trace-' e 'start.json' assenti dall'intera ricevuta.
- P2: ricevuta scritta, authenticate_preflight OK; poi byte modificati in CN_A/trace-2.jsonl, CR/trace-0.jsonl (continuo), CR_B/start.json -> AUTHENTICATED.
- P3 (trainer CPU reale, fixture common_paths, ranks=1): evidence reale = per fase checkpoint/config/metrics/origin/summary/trace-0 (+start.json nei reload). Mutazioni neutre (start.json riformattato; rng_cpu_after_sha256 riscritto uguale nei tre percorsi): stato PASSED, risultato senza evidence identico, cambiano solo i pin trace-0 (3 fasi) e start.json (reload_a).

## Controfattuale (clone temporaneo sk2r62_clone a 0efa4d45, riga 489 -> chiave separata `reload_run_evidence_sha256`)
- P1: la ricevuta contiene continuous/reload_a/reload_b con trace-0..3 e start.json.
- P2: stesse modifiche ai byte -> REJECTED 'preflight receipt differs from authenticated recomputation'.
- I 67 test di test_contrastive_targets_readout.py, test_contrastive_benchmark_readout.py e test_contrastive_target_resume.py passano anche con la correzione (32 s): nessun consumatore dipende dalla struttura sovrascritta (git grep: nessun lettore di resume.evidence_sha256 in scripts/ o tests/).

## Ricerca di motivi per confutare (nessuno trovato)
- Nessun altro pin delle tracce: contrastive_job.py:89-102 (readout) pinna solo summary.json di ogni run; contrastive_release.py non tocca le evidenze del readout; validate_measured_budget (contrastive_run.py:115-162) usa solo status, coorte, benchmark_summaries e allocazioni.
- Prontuario (docs/contrastive_targets_execution_20261119.md, riga ~183) dice che la lettura "richiede ... le tracce", senza dichiarare che la ricevuta non le pinni; risposta 20261119 (R-1 ripresa) e piani muti sul punto; nessun test sul contenuto di resume.evidence_sha256.
- I file principali del continuo restano pinnati in arms[arm].evidence_sha256 e i checkpoint finali in pairs[*].final_checkpoint_sha256: la perdita riguarda solo le 24 tracce e i 4 start.json, come dice il rilievo.

## Gravita'
- Non BLOCCANTE: authenticate_preflight ricalcola tutto dai file correnti e confronta con una ricevuta PASSED; ogni modifica che cambia un campo del risultato e' respinta, una ricevuta FAILED non si autentica, e i pin non proteggono comunque da manomissioni anteriori al readout. Tracce scritte con open("x") in directory nuove: nessun percorso della catena le riscrive.
- Difetto reale di integrita'/provenienza (residuo della riga storica che scarta impronte calcolate apposta): MINORE confermato.
- Precisazione: "rileva solo le modifiche che cambiano l'esito" va letto come "che cambiano il risultato ricalcolato" (anche campi diversi dallo stato).
- Nota a margine fuori dal rilievo: trace-{rank}.ddp.jsonl non e' pinnato da nessuno (nemmeno da compare_three_paths); e' descrittivo e non letto dal comparatore.

## Esito: rilievo NON confutato, MINORE.

## Pulizia
Clone sk2r62_clone e basetemp sk2r62_bt_* cancellati; resta solo la sonda /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp/sk2r62_probe/test_sk2r62_probe.py. Copia di sola lettura pulita, HEAD 0efa4d45.
