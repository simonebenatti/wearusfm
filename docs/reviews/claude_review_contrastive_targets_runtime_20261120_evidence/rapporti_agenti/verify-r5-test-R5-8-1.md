# Verifica scettica R5-8 (scettico n. 1)

Commit 0efa4d45 (codice fbfde797). Rilievo: ricevuta di parita' a zero punti, il readout solleva invece di INCONCLUSIVE.

## Progressi
- Letti scripts/contrastive_target_parity.py (finish 303-333, measure 246-301, observe_control 336-361), scripts/contrastive_benchmark_readout.py:201-209 e 503-521, scripts/contrastive_run.py:355-371, run.py:1314 e 1629-1633.

## Riproduzione indipendente (fatta)
Script mio: /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp/sk1_r58/scripts/zero_points.py
(fixture `readout_fixture` dei test, vero `ControlParityObserver.finish` senza `measure`, vero `read_preflight`; summary del C0 con initial_sha256/provenance come il trainer, `source_digest` fissato al source della coorte).
Comando, dalla radice della copia: `PYTHONPATH=src:tests/cpu:scripts python -B .../zero_points.py <dir nuova>` (thread a 1). Esito:
- fixture a due punti: `PASSED_FOR_REVIEW` (controllo positivo).
- finish a zero punti: ricevuta `status INCONCLUSIVE`, `inconclusive ["both prescribed real control states required"]`, `config_sha256 null`; source/summary/initial/provenance legati; readout `ValueError: zero-weight parity receipt is not bound to the full control run`.
- stessa ricevuta con il SOLO `config_sha256` canonico: `INCONCLUSIVE_REQUIRES_REVIEW`, motivo `production zero-weight comparison lacks required profile/support`, training_approved False. Causa unica: parity.py:244/254/311 + readout.py:511-514.
- `_run_evidence` reale su un C0 fermato a 127 passi: `ValueError: incomplete/unmatched benchmark C0` (readout.py:205-209), prima della sezione di parita' (464-465 < 503).
pytest mirato (11 passati): missing_or_failed_parity (4), parity_receipt_is_bound (5), gate_requires_both, observer_does_not_instrument.

## Motivi cercati per confutare
- Il docstring di observe_control (parity.py:338) e' rispettato dalla RICEVUTA: a zero punti `validate_parity_report` (229-230) da' INCONCLUSIVE. Il raise e' del readout, che per prontuario (docs/contrastive_targets_execution_20261119.md:42) esige che il rapporto leghi il config canonico; il test 172-178 vuole proprio l'errore con config_sha256 diverso.
- Percorso non raggiungibile in produzione: con contrastive presente `use_terms` e' vero (run.py:1314), prepare fissa lo step (contrastive_objectives.py:208) e measure scatta ai passi 127/255 (parity.py:248) o solleva; un'eccezione dentro lo with salta finish (nessuna ricevuta). Una ricevuta a zero punti nasce solo da un run uscito normalmente prima dell'update 128 (limite di tempo o failure numerica): contrastive_run.py:370 lo segna incompleto e il readout lo rifiuta a 205-209 prima di leggere la ricevuta (prontuario 184: «La lettura richiede tutte le sette esecuzioni complete»).
- Nessun esito sbagliato: errore e INCONCLUSIVE bloccano entrambi; nessuna evidenza accettata, nessun tetto superato.
- Resta un'incoerenza reale ma solo documentale: la ricevuta a zero punti si dichiara INCONCLUSIVE ma non lega il config, quindi non e' leggibile come tale; il prontuario non dichiara questo caso.

## Esito
NON confutato nel fatto (riprodotto). Gravita' confermata NOTA; non sale a MINORE perche' irraggiungibile con evidenze di produzione e comunque bloccante.
Non verificato: run reale 4 GPU/NCCL e finish con ctx.active (barrier); R.train reale fino a 128 sul Mac.
