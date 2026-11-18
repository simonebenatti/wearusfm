# Coorte contrastiva RMS100 / fast125 — preparazione del 18/11

Stato **REVIEW_REQUIRED**, nessun job e nessun nuovo valore di λ misurato.

- `plan.json`: emendamento della ricetta, politica scientifica e limiti di interpretazione.
- `review_configs/`: C0/CN/CR con pesi non ancora calibrati e config della calibrazione.
- `preflight_plan.json`: 13 allocazioni proposte, massimo 14 GPUh / 137,066667 coreh; nessuna release.
- `local_validation.json`: controlli dei config, costi, pin e chiusura delle release.
- `full_suite/`: suite CPU completa: 3,723 casi distinti passati (ricevuta `verification.json`; recupero di 263 casi in 13 file dopo disco pieno).
- `contrastive_suite_initial.log`: primo tentativo nella sandbox, fallimenti multiprocesso
  e un test storico poi adattato al suo reader congelato; non è il verdetto finale.

Gli artefatti precedenti sotto `contrastive_comparison_20261116` restano immutati.
Le nuove λ richiedono audit e calibrazione reali sul train; i valori vecchi non sono riciclati.
I diagnostici da stato comune sono stati integrati preservando i commit e i loro pin:
**non eseguirli da questo nuovo checkout come sostituto di quello che hanno già congelato**.
Il passaggio di ricetta non sana i precedenti fallimenti bf16/ripresa.

Documento: `docs/contrastive_targets_amendment_20261118.md`.

Codice della suite: `65bc60222e364b09c1e5a54ea3b83a3eb050fa49`.
Il successivo merge di main `afb89b45` e la consegna delle prove aggiungono solo documenti
e ricevute: sorgenti e test restano identici a quelli verificati.
