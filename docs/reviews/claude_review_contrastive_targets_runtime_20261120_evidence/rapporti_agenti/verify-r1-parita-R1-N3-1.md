# Verifica scettica n.1 di R1-N3 (r1-parita)

Avvio: HEAD copia 0efa4d45; diff fbfde797..0efa4d45 solo delivery_pin.json (codice identico).

## Riproduzione (comandi miei)

- `pytest tests/cpu/test_contrastive_target_parity.py` sulla copia: 29 passed in 4,49 s (copia rimasta pulita).
- Script `rev_runtime/tmp/vr1n3_scripts/repro_r1n3.py` (assess_point/validate_parity_report reali, replica di readout.py:517-521):
  - after_rng tolto (una sola casella) -> point FAILED, errors=['backward RNG consumption differs CNb'], within_and_between_byte_identical=True, readout "gradients/invariants not byte-identical".
  - live_state_restored tolto -> FAILED 'live control state was not restored'; global_batch_size tolto -> FAILED 'production global batch differs from 32'.
  - Confronto: flags, nce_eligible, memory assenti o rank UNAVAILABLE -> INCONCLUSIVE (profilo/supporto/memoria come da contratto).
  - Parte B: _measure_case reale su CPU, riga 280 fatta passare, riga 121 reale: measure PROPAGA NumericProfileUnavailable ("required CUDA bf16 profile is unavailable"), 0 punti; dentro observe_control l'eccezione salta finish(): nessun zero_weight_parity.json.
- Meccanica del rilievo quindi RIPRODOTTA. Resta da stabilire se e' un difetto raggiungibile.

## Raggiungibilita' e contratto (perche' non e' un difetto)

Parte A (campi assenti -> FAILED):
- L'osservatore scrive sempre i tre campi: global_batch_size nella riga (parity.py:273) e measure solleva prima se il batch non e' 32 (264-265); after_rng sempre con 4 chiavi (140); live_state_restored nel finally (294), e se False measure solleva (300-301) prima di finish(), quindi nessun rapporto su disco. Stessa sorgente per osservatore e lettore (readout.py:511-514 lega source/config/summary/init/provenance). Solo un rapporto malformato o alterato arriva a 169-172/192-194.
- Contratto: prontuario:40, risposta N-3 e plan.json:395 elencano come inconclusi solo punto, supporto e profilo; assess_point li tratta cosi' (verificato). Gli invarianti sono "obbligatori"/"mandatory" (prontuario:46, plan.json:401); il test test_missing_separate_rng_records_is_not_accepted fissa FAILED come scelta voluta.
- Conseguenza identica: preflight_plan.json:332 "Failed or inconclusive gate stops descendants; no retry..."; authenticate_preflight (readout.py:531) e contrastive_run.py:119-121 accettano solo PASSED_FOR_REVIEW.
- Attribuzione: la ricevuta contiene l'assessment completo (readout.py:516): errori specifici e within_and_between_byte_identical=True (riprodotto). L'etichetta dice "gradients/invariants". Resta solo la dicitura "differs"/"not restored" per campi assenti: cosmetica.

Parte B (profilo assente nel probe -> run interrotto):
- In produzione la riga 121 non puo' fallire se passano 329-330 e 340 di contrastive_run.py e la 280: stesse condizioni CUDA/bf16; flag immutati (git grep di enable_*_sdp/allow_tf32/set_float32_matmul_precision/cudnn.* in src e nei runner: nessun risultato); l'unica condizione in piu' e' l'autocast, creato esplicitamente alla 119 con bf16; torch lo disattiva solo senza CUDA e con bf16 non supportato solleva RuntimeError (autocast_mode.py locale 2.14, righe 282-296; logica analoga nella 2.2).
- Per progetto, sollevare NumericProfileUnavailable e' l'esito "inconcluso" a livello di run: contrastive_run.py:340 interrompe il run senza profilo e il test test_unavailable_numeric_api_is_inconclusive_not_attribute_error si aspetta l'eccezione. _measure_case interrompe anche per altre anomalie (129-135). Le condizioni sono globali e uguali su tutti i rank, quindi l'interruzione e' simmetrica. La correzione proposta, una cattura per rank dentro il probe dopo _ddp_wrap, richiederebbe un all_gather come alla 283, altrimenti rischia collettive asimmetriche.
- Fail-closed in ogni caso: senza parita' la lettura non produce PASSED_FOR_REVIEW.

## Esito
CONFUTATO come difetto: meccanica vera, ma percorsi non raggiungibili in produzione, coerenti con piani e prontuario e con conseguenze identiche all'esito inconcluso. Se lo si conserva, al massimo NOTA (dicitura dei messaggi).
Non verificato: comportamento effettivo su CUDA/torch 2.2 di Leonardo (niente ssh); per le API 2.2 ho solo letto il file rev_runtime/leonardo_torch_api.out.txt (non mio) e il test simulato.
