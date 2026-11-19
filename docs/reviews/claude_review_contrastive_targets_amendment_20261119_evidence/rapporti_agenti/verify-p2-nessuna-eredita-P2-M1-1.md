# Verifica scettica P2-M1 (scettico n. 1)

Rilievo: Nessun test copre la barriera di ricetta dell'audit (MINORE).
HEAD copia: 438c8d65. Inizio lavoro.

## Letture (438c8d65)
- validate_audit (scripts/contrastive_run.py:91-111): ordine = sorgente (guardia compat se source_sha256 != source_digest) -> inputs -> steps/world/workers/config_sha256 (r.99-101, "audit stream/config differs") -> pannello. contrastive_run.py NON e' nel diff e388a1de..0efdc29c: barriera preesistente.
- read_preflight (contrastive_benchmark_readout.py:327-334): guardia compat se source != cohort.source, poi un solo if con config_sha256 != canonical_calibration ("CPU audit missing, failed or from different inputs/config/source"). Nel diff della consegna cambia solo la stringa interpretation (r.111).
- test_historical_audit_does_not_qualify_new_sources (test_contrastive_historical_calibration.py:90-94): solo validate_historical_evidence, pytest.raises(ValueError) senza match. Confermato.
- Test che toccano validate_audit/read_preflight: test_contrastive_run.py (release_fixture: audit con config_sha256 coerente, solo casi positivi/altre chiavi della release), test_contrastive_historical_callsites.py (config sintetico coerente, verifica solo l'invocazione della guardia). Nessun caso negativo su config_sha256 dell'audit (grep di "stream/config" e "inputs/config/source" in tests: 0 righe).
- Asimmetria: la consegna aggiunge test_old_calibration_cannot_authorize_new_target_recipe (calibrazione 25/50 sintetica, match "calibration run config differs"), nessun analogo per l'audit.
- Produttore: contrastive_audit.py main chiama validate_recipe(config,'calibration') e il dispatcher (contrastive_job.py:90-92) idem: un audit nuovo prodotto onestamente ha gia' config 100/125. La barriera config_sha256 in lettura e' difesa in profondita' (audit vecchi fermati prima dalla guardia sorgenti; audit forgiati/prodotti fuori dal dispatcher).

## Riproduzione indipendente (script miei in tmp/p2m1v1/: my_mutate.py, my_probe.py)
- Mutazioni mie: X1 = validate_audit senza config_sha256 (contrastive_run.py:100), X2 = read_preflight senza config_sha256 (contrastive_benchmark_readout.py:333). Cloni --shared a 438c8d65.
- pytest tests/cpu/test_contrastive_*.py tests/cpu/test_rank_contrast.py: X1+X2 "337 passed in 99.57s"; X1 sola "337 passed in 113.86s"; X2 sola "337 passed in 113.88s". Nessun test (nemmeno fuori dai contrastivi: grep validate_audit/read_preflight/messaggi solo in 2 file di test) rileva la rimozione.
- Pin: audit storico config_sha256 ab50bc11... = hash canonico di contrastive_comparison_20261116/configs/calibration.config.json (25/50); nuova calibrazione 7471e608... (= review_configs/calibration.config.json, 100/125/v3).
- my_probe.py su codice originale: (A) audit storico esatto -> "current execution source inventory/pins differ" (guardia sorgenti per prima); (B) stesso audit con source rietichettato al digest corrente e input coerenti -> "audit stream/config differs from the calibration recipe"; (B2) controllo con config_sha256 nuovo -> accettato; (C) read_preflight con audit (B) -> "CPU audit missing, failed or from different inputs/config/source"; (C2) controllo -> supera il cancello audit.
- my_probe.py su X1+X2: (B) accettato; (C) supera il cancello audit (arriva a _cuda_evidence). Il pannello 25/50 passa anche la copertura della ricetta 100/125 (sezione contrastive invariata): config_sha256 e' l'unico legame audit-ricetta.
- Nessun altro lettore lega l'audit alla ricetta: _cuda_evidence non pinna l'audit; release e job pinnano solo i byte scelti dall'autore (audit_sha256). Il produttore (contrastive_audit.py main, dispatcher modo audit) applica validate_recipe: difesa a monte, ma non in lettura.

## Scenario della whitelist allargata (mutazione mia W, tmp/p2m1v1/my_widen.py: manifest e ALLOWED_CHANGED_SOURCES rigenerati sulle sorgenti correnti, applicata per ultima)
- my_probe_w.py, byte ESATTI dell'audit storico (i suoi inputs = plan.json input_sha256 nuovi: True; sha patchato solo per i 4 sostituti degli input di Leonardo):
  - originale: D0/D1/D2 -> "current execution source inventory/pins differ".
  - W: guardia compat superata; validate_audit -> "audit stream/config differs from the calibration recipe"; read_preflight -> "CPU audit missing, failed or from different inputs/config/source". Con W la barriera di ricetta e' l'unica rimasta e funziona.
  - W+X1+X2: validate_audit ACCETTA l'audit 25/50 per la ricetta 100/125; read_preflight supera il cancello audit.
- pytest 337 casi: W "12 failed, 325 passed"; W+X1+X2 "12 failed, 325 passed"; insiemi dei fallimenti IDENTICI (diff vuoto): test_historical_audit_does_not_qualify_new_sources (DID NOT RAISE) + 11 test sintetici di test_contrastive_source_compat.py ("changed execution sources differ from the five reviewed paths"). Nessun fallimento dovuto alla ricetta. Affermazione del rilievo riprodotta.

## Motivi cercati per confutare (non sufficienti)
- Nessun test altrove: solo tests/cpu; run_frozen usa sempre commit storici fissi (8905e346, 97125bd, 63eba9a, ...), mai l'albero corrente; nessun test pinna 643a8937 o il nuovo piano.
- Attenuanti reali: (i) produttore: contrastive_audit.py main e dispatcher (modo audit) applicano validate_recipe(..,'calibration'), quindi un audit nuovo prodotto regolarmente porta il config 100/125; (ii) source_digest pinnato in job/release ferma regressioni successive alla review; (iii) emendamento r.57-58 vieta l'ampliamento della whitelist per il cambio dei target.
- Precisazione sul rilievo: W non e' "una futura correzione non scientifica" qualsiasi. Per far passare l'audit storico con le sorgenti correnti la whitelist deve includere 18 percorsi, fra cui anchor_targets.py, spectral_targets.py, pretraining_loader.py, training/run.py e train_jepa.py: e' proprio il cambio scientifico dei target escluso dall'emendamento. Lo scenario e' meno plausibile di come e' scritto, ma resta l'unico caso in cui la barriera e' l'ultima difesa meccanica.
- Strada alternativa verificata e scartata: una modifica della copertura in plan.json (fuori dal digest) cambia il config di calibrazione senza cambiare la sorgente, ma _coverage ricontrolla policy/quote/esenzioni del pannello: li' config_sha256 non e' l'unica barriera. Per il passaggio 25/50 -> 100/125 (solo loader e target_version) il pannello non vede la differenza: resta solo config_sha256.

## Esito
- refuted = false. Fatti riprodotti con comandi miei: barriera config_sha256 in validate_audit (contrastive_run.py:99-101) e read_preflight (contrastive_benchmark_readout.py:331-334) non coperta da alcun test (X1, X2, X1+X2: 337/337 verdi); test_historical_audit_does_not_qualify_new_sources senza match e legato solo alla guardia sorgenti; con W+X l'audit 25/50 qualifica la ricetta nuova e falliscono solo i 12 test della whitelist.
- Gravita': MINORE confermata (non BLOCCANTE: oggi nessuna accettazione, la guardia sorgenti ferma l'audit storico). Non la porto a NOTA come P7-F1/F2: qui la regressione non tocca un'etichetta o un testo, ma l'unico legame meccanico fra audit e ricetta in lettura. Nel caso W l'effetto sarebbe accettare evidenze vecchie.
- Correzione proposta: valida. I messaggi attesi sono confermati sul codice originale (sonde B/C). Suggerisco di aggiungere il controllo positivo (stesso audit con config_sha256 nuovo: validate_audit accetta, read_preflight supera il cancello, sonde B2/C2). Il messaggio di read_preflight e' condiviso da sette condizioni, e il controllo isola config_sha256. match='current execution source inventory' sul test storico e' corretto: e' il messaggio attuale.

## Pulizia e non verificato
- Rimossi i miei cloni (mut, mut_X1, mut_X2, mut_W, mut_WX), i basetemp e le cartelle di lavoro. In tmp/p2m1v1 restano script e output piccoli (116 KB). Copia di sola lettura pulita a 438c8d65.
- Non verificato: suite CPU completa (vincolo di disco; grep: nessun altro test importa i lettori dell'audit); audit reale su Leonardo (input fisici sostituiti con sha patchato).
