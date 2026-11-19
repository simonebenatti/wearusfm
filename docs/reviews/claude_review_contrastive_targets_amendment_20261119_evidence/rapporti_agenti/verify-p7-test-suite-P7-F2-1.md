# Verifica scettica n. 1 del rilievo P7-F2 (testo `interpretation` del readout non protetto)

Scettico indipendente, 19/11/2026. Copia a 438c8d65, clone temporaneo proprio per la mutazione.

## Progressi
- avvio: lettura di scripts/contrastive_benchmark_readout.py:95-111 e tests/cpu/test_contrastive_benchmark_readout.py:79-85 a 438c8d65.
- diff della consegna e388a1de..0efdc29c sul readout: una sola riga, la stringa `interpretation` (riga 111); il test del readout NON e' toccato dalla consegna (git diff --stat: nessuna riga per tests/cpu/test_contrastive_benchmark_readout.py).
- `descriptive_only: True` e l'assenza di `passed` erano gia' presenti a e388a1de: l'unico cambiamento del readout nella consegna e' il testo.
- clone temporaneo tmp/v-p7f2-1-mut (git clone --shared, checkout detached 438c8d65); import verificati sul clone (wearusfm e contrastive_benchmark_readout da tmp/v-p7f2-1-mut).
- M0 (senza mutazione), 9 file diretti: 173 passed in 16.76s.
- M6: riga 111 riportata al testo di e388a1de; dopo la mutazione `diff` fra il file e `git show e388a1de:scripts/contrastive_benchmark_readout.py` e' vuoto (blob 90945903): la mutazione annulla per intero la modifica del readout della consegna.
- M6, 9 file diretti: 173 passed in 14.06s. RIPRODOTTO.
- M6, 5 file indiretti (audit, cuda_check, rank_exit, common_checkpoint, common_state_cuda): 32 passed.
- M6, restanti 8 file con "contrastive" (callsite_oracle, compare, core, diagnostics, memory, runtime, arms_collaudo, wsd_chain): 229 passed in 88.91s.
- Nessun test cita 'interpretation' del readout, 'Only update 1', 'need not match', 'RMS100/fast125' (grep su tests/); i run_frozen dei test contrastivi usano solo SHA storici fissi (8905e346..., FROZEN_READER), quindi non vedono il working tree.
- Controllo meccanico fuori dai test: source_digest() include scripts/contrastive_*.py (contrastive_prepare.py:192-201). Clone pulito: 643a8937... = preflight_plan.source_sha256; clone mutato: 335496e9... (diverso).
- contrastive_job.build_command (contrastive_job.py:62-67) rifiuta un job con source_sha256 diverso da source_digest(): sonda con job readout pinnato a 643a8937 -> clone pulito supera il controllo di sorgente (si ferma dopo, su review inesistente, FileNotFoundError); clone mutato -> "ValueError wrong mode, release, source or allocation". Una regressione DOPO il pin della release non arriva all'esecuzione; una regressione PRIMA del pin la ferma solo la review del pacchetto ("single package review and source pins" in preflight_plan.required_before_first_submission).
- Nessun test lega il pin 643a8937 del preflight_plan al codice (grep "643a8937" in tests/: vuoto): la suite resta verde anche col digest cambiato.
- Prova puntuale sul clone mutato: test_first_eight_historical_comparison_is_descriptive_without_a_new_gate 1 passed; initial_calibration_description sul fixture del test restituisce descriptive_only True, nessun 'passed', interpretation = "Only update 1 may match exactly; ...".
- Storia: a e388a1de, 0efdc29c e 438c8d65 nessun test ha mai asserito il testo (git grep su tests/: solo la chiamata alla riga 83). La consegna non ha tolto una copertura: non c'e' mai stata. Le stringhe 'interpretation' testate altrove (pose_e3_b, pose_b19618_report) sono etichette calcolate dai dati, non costanti.
- Uso del campo: nessun codice lo legge; authenticate_preflight (readout.py:401-412) lo confronta solo con la propria ricomputazione (stesso codice).
- Portata della regressione: il testo vecchio e' modale ("may match"), non afferma una corrispondenza. Con 100/125 le famiglie log_env, band_shape, ms_slow e lo stream non dipendono dalle due finestre (anchor_targets.py:143-175; pretraining_loader.py:625-643, maschere e finestre prima dei target): al passo 1 restano in linea di principio uguagliabili; log_rms, ms_fast e grad_norm no. Il testo vecchio sarebbe incompleto/fuorviante, non falso (lettura mia, stream non verificato su dati reali).
- La lettura scientifica corretta e' registrata indipendentemente nell'emendamento (docs/contrastive_targets_amendment_20261118.md:46-47) e nel piano (target_recipe_amendment, calibration.historical_multiplier_interpretation).

## Esito
- Fatto riprodotto: nessun test protegge la stringa `interpretation` (173 + 32 + 229 test verdi con la riga 111 riportata ai byte di e388a1de). Non confutato sul fatto.
- Gravita': NOTA, non MINORE. Il campo e' documentazione costante, non letto da codice; le proprieta' sostanziali (descriptive_only, nessun gate) sono gia' coperte; dopo il pin della release una regressione e' fermata dal controllo di sorgente del dispatcher; la conseguenza sarebbe una frase incompleta, non l'affermazione di una corrispondenza esatta.
- Correzione suggerita (non bloccante, nessun nuovo giro): sottostringhe 'RMS100/fast125' e 'even update 1' nel test esistente.
- pulizia: rimossi tmp/v-p7f2-1-mut e i basetemp v-p7f2-1-bt-*; copia di sola lettura pulita a 438c8d65.
