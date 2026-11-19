# Verifica scettica n. 2 di P3-M1 (comparatore finale senza legame a coorte/ricetta)

Inizio: 2026-11-19. Copia: clone @ 438c8d65 (sola lettura). Script miei: tmp/v2_p3m1/scripts/mixed_manifest.py, uscita tmp/v2_p3m1/mixed_manifest.out.txt.

## Riproduzione indipendente (fatti)
- `grep -n "rms_window|ms_fast|target_version|amendment|cohort|config" scripts/contrastive_compare.py` -> exit 1 (nessuna occorrenza). Confermato.
- `git diff --quiet e388a1de 438c8d65 -- scripts/contrastive_compare.py tests/cpu/test_contrastive_compare.py` -> exit 0: comparatore e test INVARIATI nella consegna e dopo; ultimo commit c680e95d (16/11).
- Caso A (celle posa annotate C0 25/50/v1, CN/CR 100/125/v3, manifest v2 semplice): ACCETTATO, protocol v2, decision_valid=True; l'uscita non contiene l'identita' dell'emendamento. Confermato.
- Caso B (manifest con chiavi in piu' amendment/cohort/configs, config C0 volutamente 25/50): ACCETTATO; le chiavi extra sono tollerate ed ECHEGGIATE in input_pins (amendment presente nell'uscita) ma non verificate.
- Caso C (cella posa con arm interno "T100-G" sotto la chiave C0): RIFIUTATO "incomplete/mixed-dose/nonteacher pose cell".
- Caso D (righe audit con arm "T100-G" sotto C0): RIFIUTATO "audit identity differs".
- Caso E (C0 e CN con lo stesso checkpoint sha): ACCETTATO (limite gia' noto: ingressi scambiati ma coerenti).

## Motivi per cui non e' un difetto della consegna (confutazione)
1. Contratto dichiarato del comparatore, non un'omissione:
   - docs/contrastive_comparison_protocol_20261116.md:166 «Il comparatore verifica OOF appaiati, supporto posa e checkpoint comuni ... Provenienza/leakage degli estrattori sono prerequisiti distinti.»
   - stesso documento :178, «Dodici estrazioni ... Checkpoint/pin, adattatori verificati e benchmark inferenza» come requisito prima di quella fase.
   - docs/reviews/contrastive_review_response_20261116.md:45 «Il comparatore resta uno strumento di lettura; non sostituisce la validazione futura dell'intera catena degli input.»
   - Review Claude del 16/11 (git show 0ab9d985:docs/reviews/claude_review_contrastive_comparison_20261116.md, chiusura di I-9 e righe ~612-613): limite gia' registrato, «il comparatore accetta ingressi troncati o scambiati purché coerenti», «da chiudere prima della valutazione»; prerequisito «catena di valutazione per C0/CN/CR (estrattori, sonde, costruttore del manifest con cardinalità e pin)». P3-M1 e' un caso particolare di questo limite (vedi caso E: stesso checkpoint per due bracci accettato), su un file invariato.
2. Identita' separata voluta: emendamento righe 29-31 (protocollo statistico v2 invariato, ricetta con identita' `contrastive-target-windows-20261118-v1`); plan.json:324 `amendment`. L'uscita v2 e' coerente col disegno; caso B: amendment e pin di cohort/config messi nel manifest futuro sono tollerati ed echeggiati in input_pins senza toccare il codice.
3. Percorso non in produzione: nessun estrattore/lettore posa/costruttore di manifest contrastivo esiste (ls scripts | grep contrast: audit, benchmark_readout, compare, cuda_check, job, prepare, rank_exit, run, source_compat, split_benchmark); plan.json:293 release=null, executable=false; emendamento righe 76-77: i tetti «Non comprendono training scientifici ... o valutazioni downstream»; nessun codice legge l'uscita del comparatore o il suo protocol id (git grep).
4. Controlli altrove, verificati:
   - training: contrastive_run.validate_recipe (config == calibrated_configs[arm], costruiti da build_review_configs 100/125/v3) e validate_cohort (bundle con esattamente i tre bracci di una sola ricevuta autenticata, source_sha256 == source_digest()).
   - coorti storiche: training_approved=False e source pin ab2d6460/46938f31/6b421b75 != source_digest corrente 643a8937: non addestrabili col codice corrente; nessuna release di training ne' training_approved=true nel repo; C0.19618.config storico (campi assenti, v1) esiste solo come config.
   - artefatti T100-G esistenti portano arm "T100-G": il comparatore li rifiuta sotto C0 (casi C e D). Lo scenario «T100-G etichettato C0» richiede un errore di un estrattore futuro, cioe' proprio il prerequisito di provenienza dichiarato.
5. Suite del comparatore sul clone: 32 passed (tests/cpu/test_contrastive_compare.py, 1,50 s); clone pulito dopo.

## Esito
refuted = true come difetto MINORE della consegna. I fatti si riproducono, ma descrivono il contratto dichiarato di un file invariato, su un percorso non ancora in produzione, gia' registrato come prerequisito della catena di valutazione futura. Gravita' reale: NOTA. Va portata nei requisiti della release di valutazione: il costruttore del manifest, o un validatore, lega cohort, config e checkpoint a 100/125/v3 e scrive l'amendment nel manifest, quindi in input_pins. Nessun nuovo giro sul preflight.

## Non verificato
- Checkpoint fuori dal repo (Leonardo): niente ssh per regola; mi baso sui registri del repo (nessuna release di training).
- Il comportamento del futuro estrattore e del costruttore del manifest: non esistono ancora.
