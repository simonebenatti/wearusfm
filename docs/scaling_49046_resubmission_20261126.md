# Dose 49.046: reinvio del secondo segmento del 300M (26/11/2026)

## Fatto
65188321 (`SCALING-49046-train-49046-20261118`) risulta `CANCELLED` il 23/11 alle 18:05:58, mai partito
(`evaluation/evidence/status/train-final_cancelled_20261126.txt`). Il 26/11 Simone, alla domanda «Preparo il reinvio del segmento 2
con un ID nuovo?», ha risposto «Sì, prepara». Questa risposta copre la preparazione; l'invio non e' ancora stato fatto.

## Che cosa cambia
Solo l'ID logico: `SCALING-49046-train-49046-r2-20261126`. Restano uguali piano `training_49046.json`, checkout di codice
`87da975b`, predecessore 64837401, 10 ore, 4 GPU, 480G, cartella d'uscita (il tentativo annullato non l'ha creata) e tetti di
risorse (il job annullato non ha consumato ore; la sua riga resta nel ledger).

## Come
I file dell'operatore devono restare identici al commit del codice (`scaling_operations49046.attestation`), quindi non sono
modificati. `results/infra/scaling_49046_20261118/operations/resubmit_train_final.py` sostituisce l'ID in memoria ed esegue il
controller congelato con gli stessi argomenti. Prima di `gate train-final` e `lines train-final` rilegge `sacct` di 65188321 e si
ferma se la riga non e' «cancellato senza essere partito». Si rifiuta di girare se non e' identico alla versione committata.

**Da `gate train-final` fino a `compare` si usa l'adattatore al posto di `scripts/scaling_plans49046.py`**: gli stadi successivi
cercano il job di training nel ledger.

```bash
~/.venvs/wearusfm-tok/bin/python -B results/infra/scaling_49046_20261118/operations/resubmit_train_final.py gate train-final
```

```bash
~/.venvs/wearusfm-tok/bin/python -B results/infra/scaling_49046_20261118/operations/resubmit_train_final.py lines train-final
```

Test: `tests/cpu/test_scaling_resubmit_train_final49046.py`. Nessuna review di GPT su questa modifica al 26/11.
