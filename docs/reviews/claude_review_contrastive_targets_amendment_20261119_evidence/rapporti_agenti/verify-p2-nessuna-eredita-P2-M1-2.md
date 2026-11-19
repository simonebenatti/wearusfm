# Verifica scettica P2-M1 (scettico n. 2)

Rilievo: nessun test copre la barriera di ricetta dell audit (config_sha256 in validate_audit e read_preflight), MINORE.
Copia: 438c8d65. Inizio lavoro (prova riprodotta con comandi propri).

## Riproduzione (comandi miei, clone tmp/skep2_p2m1/mut a 438c8d65, script in tmp/skep2_p2m1/scripts)
- Configurazioni: audit storico 5c720fad, config_sha256 ab50bc11 = sha canonico di configs/calibration.config.json storico (25/50 impliciti, v1). Calibrazione corrente 100/125/v3: 7471e608.
- Codice invariato (probe.py, out/probe_base.json):
  - A validate_audit(audit storico) -> "current execution source inventory/pins differ" (guardia sorgenti prima).
  - B sorgente rietichettata al digest corrente + input coerenti -> "audit stream/config differs from the calibration recipe"; B2 (guardia forzata ad accettare, byte originali) -> stesso errore.
  - C controllo: stesso audit rietichettato contro il config storico -> accettato. La barriera config_sha256 e' cio' che separa 25/50 da 100/125 dopo la guardia.
  - D/E/E2 read_preflight: D guardia sorgenti; E/E2 -> "CPU audit missing, failed or from different inputs/config/source".
- Mutazione MD+MG (mutate.py: tolto il confronto config_sha256 da contrastive_run.py:99-100 e contrastive_benchmark_readout.py:333):
  - A/D ancora rifiutati dalla guardia sorgenti (oggi nessuna accettazione).
  - B/B2 validate_audit ACCETTA l'audit 25/50 per la calibrazione 100/125; E/E2 read_preflight supera il cancello dell'audit (arriva a _cuda_evidence).
  - pytest tests/cpu/test_contrastive_*.py tests/cpu/test_rank_contrast.py (20 file): "337 passed in 106.93s" (out/pytest_mdmg.txt).
- grep: nessun test cita "audit stream/config" o "inputs/config/source"; nessun test fuori dai file contrastivi importa contrastive_run/contrastive_benchmark_readout (unico altro match: test_logsquare_train.py, validate_release di logsquare).
## Scenario della whitelist allargata, con la guardia VERA (clone tmp/skep2_p2m1/mutb, mb_mutate.py)
- MB mio: ALLOWED_CHANGED_SOURCES = i 18 percorsi davvero cambiati da ab2d6460 a 438c8d65, aggiunte ammesse = helper + harness/backbone_depth.py + harness/finetuning.py, manifest compat rigenerato sull'inventario corrente.
- MB da sola, audit storico byte-esatto (5c720fad) con guardia reale: validate_audit -> "audit stream/config differs from the calibration recipe"; read_preflight -> "CPU audit missing, failed or from different inputs/config/source". La barriera config_sha256 e' l'unica rimasta.
- MB+MD+MG (manifest rigenerato): validate_audit ACCETTA l'audit 25/50 per la calibrazione 100/125; read_preflight supera il cancello dell'audit.
- pytest 20 file contrastivi: MB "20 failed, 317 passed"; MB+MD+MG "20 failed, 317 passed"; insiemi dei falliti IDENTICI (diff vuoto). Fallisce test_historical_audit_does_not_qualify_new_sources con "DID NOT RAISE" (whitelist) + 19 test sintetici di compat. Il rilievo ne contava 11: dipende da come si allarga la whitelist, non cambia la conclusione.
## Ricerca di motivi per confutare
- Barriera preesistente: contrastive_run.py non e' nel diff; read_preflight cambia solo la stringa interpretation. Ma con la ricetta nuova e' proprio questa barriera a separare 25/50 da 100/125 (controllo C: lo stesso audit contro il config storico e' accettato).
- Guardia sorgenti: lega i byte esatti dei sorgenti 18/11 (che costruiscono 25/50), quindi oggi rifiuta qualsiasi albero nuovo; testata solo senza match. L'emendamento (r.57-58) esclude l'allargamento per il cambio target: lo scenario richiede due modifiche deliberate, la prima rompe 20 test.
- Produttore: contrastive_audit.py main applica validate_recipe('calibration'): un audit nuovo onesto ha gia' il config giusto. La barriera in lettura resta l'unico controllo per audit vecchi/forgiati se la guardia passa.
- Nessun test negativo su config_sha256 dell'audit: release_fixture (test_contrastive_run.py:72-83) e callsite (test_contrastive_historical_callsites.py:26-47) usano config coerenti; nessun test cita i due messaggi. Asimmetria con la calibrazione (test con match 'calibration run config differs' aggiunti nella consegna).
- Coerenza con la scala comune: p7 classifica MINORE comportamenti non protetti da test (M6, M7) e NOTA il solo match mancante.
## Esito
Non confutato. Gravita' MINORE (bassa: oggi nessuna accettazione, doppia barriera attiva). La correzione proposta e' discriminante: i casi B/E falliscono sul codice corrente con i messaggi indicati e passano sotto MD/MG.
## Non verificato
- Suite CPU completa (vietata): eseguiti i 20 file test_contrastive_*.py + test_rank_contrast.py; grep: tutti gli importatori diretti/indiretti di contrastive_run/contrastive_benchmark_readout sono in quei 20 file.
- Nessuna esecuzione su Leonardo; ingressi fisici dell'audit sostituiti da file fittizi con sha mappato ai pin dell'audit.
## Pulizia
Cloni mutati, basetemp e copie decompresse rimossi; restano solo tmp/skep2_p2m1/scripts. Evidenze piccole (script, probe JSON, log pytest, elenchi dei falliti) in reports/verify-p2-M1-2-evidence/. Copia git pulita a 438c8d65.
