# verify-r2-ripresa-R2-M2-1 — scettico n. 1 sul rilievo R2-M2 (fbfde797)

Inizio: 2026-11-20. Copia di riferimento a 0efa4d45, pulita; 0efa4d45 aggiunge solo delivery_pin.json (readout invariato fra fbfde797 e 0efa4d45).

## Lettura del codice
- readout:487-489 (fbfde797): `check = compare_three_paths(...)`, poi `check['evidence_sha256'] = reload_pins`: sovrascrittura confermata alla lettera.
- compare_three_paths (contrastive_target_resume.py:347-361, 423) restituisce evidence per fase continuous/reload_a/reload_b: checkpoint.pt, summary.json, config.json, metrics.jsonl, origin_sessions.json, trace-{0..3}.jsonl e start.json (solo reload).
- reload_pins = _run_evidence dei due reload (readout:271-272): config, summary, metrics, checkpoint, origin, numeric_profile.rank0..3. Nessuna traccia, nessun start.json.
- Restano nella ricevuta: fork_sha256 (checkpoint_step0000544.pt), pairs[*].final_checkpoint_sha256, arms[arm].evidence_sha256 del full (config/summary/metrics/checkpoint/origin/numeric). Perse: trace-0..3 x 3 fasi (12 per braccio) e start.json x 2.
- Origine: fino a 438c8d65 l'assegnazione `check['evidence_sha256'] = resumed_pins` stava su compare_resume, che NON produce quella chiave (restituisce full_sha256/resumed_sha256): nessuna perdita storica. Il passaggio a compare_three_paths in fbfde797 (diff del readout, righe 194-208) la conserva e ora collide con la chiave del modulo nuovo. Difetto nuovo, dentro il diff.

## Riproduzione indipendente (compare_three_paths REALE, non uno stub)
Script: rev_runtime/tmp/v_r2m2_1/test_verify_r2m2.py. Fork reale della suite (trainer CPU, 1 rank, fork 2/fine 4, due processi nuovi con spawn) dalla fixture common_paths; il contorno del readout dalla readout_fixture della suite; compare_three_paths reale chiamato dentro read_preflight.
Esito pytest: 2 passed in 14,02 s. Output essenziale:
- REAL compare_three_paths evidence: continuous/reload_a/reload_b, ciascuna con trace-0.jsonl; start.json nei due reload.
- PUBLISHED CN e CR: solo reload_a/reload_b con le chiavi di _run_evidence (nella fixture stub: origin, summary). Le impronte trace-0 delle tre fasi e i due start.json non compaiono IN NESSUN PUNTO del JSON della ricevuta.
- Dopo aver scritto la ricevuta: (1) chiave extra + formato compatto in tutte e tre le tracce; (2) batch_sha256/input_sha256 sostituiti con lo stesso valore finto nelle tre fasi: authenticate_preflight ACCETTA in entrambi i casi; le impronte scartate sarebbero cambiate in tutte e tre le fasi.
- Controllo: modifica che cambia il cancello (batch_sha256 solo in reload_b) -> "preflight receipt differs from authenticated recomputation". Il ricalcolo protegge l'esito, non i byte.

## Ricerca di motivi per confutare (nessuno regge)
- Nessun altro punto lega le tracce: _run_evidence (readout:271-272) pinna config/summary/metrics/checkpoint/origin/numeric_profile; il job readout (contrastive_job.py:96-101) pinna solo summary.json dei sette run; il runner (contrastive_run.py:350-354) scrive trace-{rank}.jsonl senza pin; contrastive_target_resume non ha main ne' rapporto proprio; unico chiamante di compare_three_paths = readout:487. Il prontuario (riga 183) chiede le tracce per la lettura ma non dichiara di non legarle; nessuna dichiarazione nei piani.
- Restano legati: fork_sha256 (checkpoint 544), final_checkpoint_sha256 delle tre fasi nelle coppie, evidence del full. La perdita riguarda le 12 tracce per braccio (3 fasi x 4 rank) e i due start.json.
- start.json: il suo contenuto e' gia' verificato contro valori legati (resume:324-329, 355-356: protocollo, fase, fork_sha, payload del fork, origin); fuori controllo solo source.path/step/end_step/world_size. La perdita del suo pin e' quasi irrilevante; quella delle tracce e' sostanziale (le tracce sono l'evidenza di RNG, batch, coda; la lettura storica faa4a8d7 si e' basata proprio sulle righe di traccia e sul loro manifest SHA).

## Gravita'
- Non BLOCCANTE: il cancello e' calcolato correttamente; authenticate_preflight (readout:529-540, chiamato da contrastive_run.py:119-120 per il training) ricalcola dalle tracce e rifiuta ogni modifica che cambi l'esito (controllo riprodotto). Nessuna evidenza non valida accettata, nessun esito attribuito male.
- MINORE confermato: difetto reale del codice (il readout butta via i pin che il modulo di ripresa calcola per la pubblicazione), lacuna di legame byte-a-byte della ricevuta committata.
- Fix proposto adeguato: conservare entrambe le impronte (chiave separata o fusione per fase) e un test che lo verifichi.

## Esito: refuted=false, MINORE.
Pulizia: rimossa la basetemp v_r2m2_1_pt; resta solo lo script (piccolo). Copia di riferimento pulita a 0efa4d45 (git status --ignored vuoto).
