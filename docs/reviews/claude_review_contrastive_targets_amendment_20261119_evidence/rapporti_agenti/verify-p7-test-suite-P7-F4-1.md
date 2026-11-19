# Scettico n. 1 — P7-F4 (test sul vecchio audit senza `match`)

## Passo 1 — riproduzione (438c8d65)
- `tests/cpu/test_contrastive_historical_calibration.py:90-94`: `with pytest.raises(ValueError):` senza `match` — confermato (`cat -n`).
- Script mio `tmp/p7f4v1_scripts/repro.py` (audit decompresso in tmp, helper del clone):
  `ValueError: historical evidence compatibility: current execution source inventory/pins differ`
  (riga 162 di `scripts/contrastive_source_compat.py`). Il motivo di oggi e' quello voluto.

## Passo 2 — ricerca di motivi per confutare
- Controllo altrove del meccanismo: `tests/cpu/test_contrastive_source_compat.py:148-158` usa `match="current execution source inventory"`, pero' su scene sintetiche (ROOT monkeypatchato), non sul manifest reale della HEAD.
- I test congelati (`run_frozen`, `git archive 8905e346`) leggono il manifest dell'archivio, non quello della HEAD (`frozen_campaign_support.py:16-21`). Manifest e audit identici fra 8905e346 e 438c8d65 (`git diff --stat` vuoto), ma nessun test lo asserisce.
- Nessuna dichiarazione nell'emendamento o nel registro che il test sia volutamente indifferente al motivo (righe 54-58 dell'emendamento: whitelist non ampliata).
- Produzione: `contrastive_run.validate_audit` (r. 94-99) e `contrastive_benchmark_readout.read_preflight` (r. 327-334) propagano qualunque ValueError; inoltre `config_sha256` dell'audit storico `ab50bc11...` != hash canonico del config di calibrazione nuovo `7471e608...` (100/125): secondo strato indipendente di rifiuto. Quindi nessun effetto di sicurezza.

## Passo 3 — mutazioni (clone tmp/p7f4v1_clone a 438c8d65; 3 file: historical_calibration, source_compat, historical_callsites; M0 46 passed)
| Mutazione | Motivo del rifiuto | Esito dei 3 file |
|---|---|---|
| Ma1 protocol del manifest HEAD alterato | wrong manifest protocol | 46 passed (non rilevata) |
| Ma2 manifest troncato | JSONDecodeError (sottoclasse di ValueError) | 46 passed (non rilevata) |
| Ma3 manifest cancellato | unreadable file | 46 passed (non rilevata) |
| Mc tolta r.162 | current source digest differs (r.164) | 2 failed, solo sintetici |
| Mb tolte r.162 e r.164 | ACCETTATO | 4 failed, incluso il test di r.90 |
| F0 fix con match, nessuna mutazione | inventory/pins | 46 passed (fix valido) |
| F1/F2 fix + Ma1/Ma2 | protocol/JSON | 1 failed: il test di r.90 |
| F3 fix + Mc | digest | 3 failed (r.90 + 2 sintetici) |
- Clone ripristinato pulito dopo ogni mutazione (`git status --short` vuoto).
- Sfumatura: «file illeggibile» vale per il manifest (o un sorgente dell'inventario); un audit .gz mancante fallisce fuori da `pytest.raises` (lettura a r.92), quindi il test darebbe errore, non passerebbe.

## Esito
- NON confutato. Il rilievo e' riprodotto: il test di r.90-94 passa anche quando il rifiuto non dipende dai sorgenti nuovi (manifest alterato/troncato/cancellato) e nessun altro test dei 3 file se ne accorge. Il fix proposto e' corretto e passa a 438c8d65.
- Gravita' reale: NOTA. Il percorso e' fail-closed (ogni ValueError = audit non accettato) e c'e' il secondo strato `config_sha256`; un aggiramento completo (Mb) e' gia' rilevato anche senza match. Nessun effetto su ricetta, evidenze, confronti o tetti.
- Comandi: `tmp/p7f4v1_scripts/repro.py <clone> <dir>` e `tmp/p7f4v1_scripts/mutate.py` (conservati, 12 KB); clone e basetemp cancellati.
