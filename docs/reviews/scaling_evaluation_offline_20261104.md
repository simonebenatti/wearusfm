# Validatori ed estrazioni scaling — consegna offline, 04/11/2026

Stato: implementati `scripts/scaling_evaluation.py` e i test CPU. Nessuna lettura
del cluster, estrazione, sottomissione o nuovo risultato scientifico. Il collaudo
approvato rimane congelato e non usa questo script.

## Contratto implementato

- Due bracci espliciti, `T100-G-30M` e `T100-G-300M`, entrambi teacher a 5.000:
  D384 / D896 nella posa e pool768 / pool1792 nella classificazione. La guardia
  pubblica verifica l'intero preset LADDER oltre a GELU, finestra100, passo25,
  slot64 e testa di posa. I wrapper della campagna tokenizer restano invariati.
- Il controllo mantiene la ricevuta originale `tokenizer-training-20261103-v1`,
  braccio `T100-G`, codice `afb644e` e SHA del checkpoint storico. Il candidato
  richiede `scaling-training-20261104-v1`, braccio300M e il contratto sorgenti
  scientifiche esatto. Configurazione, dose, teacher e gain vengono verificati;
  non si calibra nulla sui segnali di valutazione.
- Il piano delle estrazioni è vincolato al confronto, alle quattro sorgenti di
  training e alle configurazioni complete del contratto `scaling_train`.
  Non basta che due configurazioni abbiano nominalmente le dimensioni attese.
- Un pin separato verifica cinque primitive di estrazione/validazione contro
  `f2061b8bc91a6355213793a9719121135ad48fa2`, codice della cache di posa T100-G5000.
  Gli stessi cinque file sono identici in `b315dad`, riferimento delle estrazioni
  finali di classificazione. Il contratto training copre inoltre tutti gli `src`.
- Posa: target, maschere, metadati, deviazioni, varianza e supporto centrale
  identici alla cache di riferimento; latenti FP16, predizioni FP32, finitezza
  su tutto il tempo reale, anche dove la posa manca. Solo il padding è escluso.
- Classificazione: righe, etichette anche testuali EPN, utenti, ordine e fold
  identici; split ufficiali esclusi, pool finito FP32. Il blocco backbone precede
  quello locale a entrambe le larghezze, verificato sul vero `pool_p1p2`.
- Il driver riusa i comandi dimensionali `pose_eval.py` e `channel_readout.py`.
  La ricevuta completa viene scritta per ultima, dopo hash, supporti, sorgenti
  e output verificati. Un errore nel secondo split lascia la fase incompleta.

Il pin `summary_path` del 30M deve indicare lo snapshot della fase:
`$WORK/wearusfm_runs/results/tokenizer_training_20261103/T100-G/5000/summary.json`.
Il summary nella cartella scratch del run è mobile e oggi descrive 19.618:
non va usato per attestare la fase a 5.000. Nessun pin del futuro checkpoint300M
è stato inventato e nessun piano operativo di estrazione è stato pubblicato.

## Verifica

Comando locale, nel worktree scaling:

```bash
PYTHONPATH=src:tests/cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ~/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider \
  tests/cpu/test_scaling_evaluation.py tests/cpu/test_tokenizer_evaluation.py
```

Esito: **123 test passati in 7,56 secondi**, di cui 88 nuovi. Le regressioni
D384 confrontano gli output completi dei due validatori; D896 conserva le stesse
impronte di righe e fold. Test negativi coprono ricetta, dose, completamento,
checkpoint, provenienza, target, supporto, precisioni, etichette e deriva dei pin.

Il ripristino del teacher e del gain usa un checkpoint reale di un modello piccolo
su CPU; la guardia dei preset reali D384/D896 è testata separatamente. Questa suite
non costituisce una misura di memoria, velocità o correttezza GPU del 300M.

## Lavoro ancora necessario prima del training scientifico

Questa consegna non rende completa la catena di valutazione: mancano i controller
scaling della ridge e delle sonde convergenti, il controllo secondario S5 con
proiezioni fisse, i benchmark CPU e il piano di risorse operativo. I controller
storici di ridge/sonde non accettano questi nuovi bracci e non vanno aggirati.
Il wrapper del training deve continuare a richiedere una readiness distinta.

L'integrità di AdamW/RNG è verificata secondo i controlli strutturali storici;
non è una certificazione della futura ripresa a19.618, perché l'associazione
completa fra momenti e parametri richiede un controllo dedicato. L'estrazione
verifica invece il caricamento esatto e completo del teacher.
