# Review richiesta a Claude: T100-LQ, senza deploy o job

Simone vuole soluzione → tua review → convergenza → deploy a cura di GPT.
Lavora sul ramo origin/codex/tokenizer-logcontrast-20261104, in un tuo checkout.
Il riferimento è il commit di consegna indicato nel messaggio di GPT.
Non modificare i checkout delle altre linee; nessuna sottomissione.

Leggi prima docs/tokenizer_logsquare_protocol_20261104.md e
docs/reviews/tokenizer_logsquare_proposal_20261103.md, incluso emendamento8ee6603.
Esamina il diff rispetto a8ee6603:

1. patch_tokens.py e fm.py: ramo W_log·log(0.001+x²) affiancato al lineare,
   W_log zero, identità completa dell'inizializzazione condivisa e RNG,
   FP32 del log sotto AMP, caricamento stretto legacy e nuovo.
2. run.py e train_jepa.py: default disabilitato, config/resume coerenti,
   combinazione GELU+log rifiutata. Nessuna modifica della loss.
3. logsquare_preflight.py e launcher: due veri update del trainer sul medesimo
   stream, hash storico dell'inizializzazione, supporti/RNG, finitezza,
   diagnostica train nativa200Hz/2000Hz, no checkpoint, input/code pin e no retry.
4. Verifica indipendente dei momenti, maschere/bordi e derivata.
   La memoria riportata comprende due modelli residenti; i tempi separano
   la diagnostica. Valuta eventuali buchi nei gate CUDA/DDP non provati sul Mac.
5. Funzione iniziale identica non significa primo update condiviso identico:
   il gradiente nuovo contribuisce al clipping globale. È esplicitato e
   lasciato invariato. Segnala se richiede un'altra diagnosi prima del training.
6. PrimarioT100-L; secondariT25-L/T100-G; stesse metriche/split/dose.
   Un seme, sviluppo riusato, nessun energy claim o successo globale del FM.
   Controlla costi distinti e gate prima del training.
7. Contrastiva resta successiva e separata: commenta le scelte da fissare,
   senza implementarla né lanciare una ricerca di iperparametri.

Test:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  ~/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider \
  tests/cpu/test_tokenizer_logsquare.py tests/cpu/test_logsquare_preflight.py \
  tests/cpu/test_tokenizer_activation.py tests/cpu/test_tokenizer_preflight.py \
  tests/cpu/test_tokenizer_comparison.py tests/cpu/test_tokenizer_train.py
```

Restituisci APPROVATO / APPROVATO CON CORREZIONI / DA RIVEDERE, separando
difetti dimostrati, limiti e suggerimenti. Per ogni difetto cita file/riga e
una riproduzione o ragionamento verificabile. Distingui approvazione del
collaudo preparatorio da quella del training: orchestrazione/pin scientifici
saranno completati dopo review e misure del collaudo, prima del lancio.
Puoi committare la review sul tuo ramo; GPT integrerà le correzioni concordate.
