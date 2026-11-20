# R5-9 scettico 1 (rilievo di r5-test): KeyError make-job senza audit

Avvio: 2026-11-20 05:00:21. Clone usa e getta: /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp/sk1_r59_work/clone a 0efa4d45 (sorgenti identici a fbfde797).

## Riproduzione indipendente (clone usa e getta, CLI reale)
Script: `rev_runtime/tmp/sk1_r59_work/scripts/repro_r59.py` (commit di review fittizio solo nel clone, payload con digest calcolati dal codice: source 40d39f56..., science b03c0bc6..., preflight e4b39591...). bind-review rc 0, check-binding rc 0, poi make-job:
- CALIB evidenza `{}` / solo `inputs`; CUDA `{}`; READOUT `{}`: rc=1, nessun file `--out`, `KeyError: 'audit'` in make_job, release.py:241 (`checked_pin(job["audit"])`).
- CALIB con `audit` pinnato ma senza `audit_sacct`: rc=1, `KeyError: 'audit_sacct'` (release.py:242): stesso difetto, non citato dal rilievo.
- CALIB con `"audit": null`: `ValueError: complete artifact path and SHA256 required` (checked_pin, :42).
- CALIB con `"audit_sacct": null`: `ValueError: audit accounting required before release` (:158): il messaggio esplicito previsto esiste ma e' raggiungibile solo con chiave presente a null; con chiave assente lo precede il KeyError.
- Controllo: AUDIT con `inputs`: rc=0, job scritto.

## Motivi cercati per confutare (nessuno regge)
- `main()` (:254-285) non intercetta eccezioni; nessun altro chiamante di make_job oltre CLI e test.
- Prontuario :140-146 dice che la calibrazione richiede audit e accounting, ma non nomina le chiavi `audit`/`audit_sacct` dell'evidenza; la risposta 20261119 non promette errori espliciti.
- Test: test_contrastive_release.py:154-157 prova solo `validate_audit_accounting(audit, None)`; nessun test di make_job per cella non audit senza audit. Il runtime ha invece un controllo esplicito (contrastive_run.py:193-194).

## Gravita'
Fallisce chiuso (rc 1, nessun job scritto, nessuna evidenza accettata, nessun tetto toccato): non BLOCCANTE ne' MINORE. Solo chiarezza del messaggio: **NOTA confermata**. Rilievo NON confutato.
Fine: 2026-11-20 05:03:18. Clone cancellato; restano script e log (pochi KB) in tmp/sk1_r59_work. Copia di sola lettura invariata a 0efa4d45.
