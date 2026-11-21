# Verifica avversaria R5-I-1 (timeout di fase non registrato)

Esito: il rilievo REGGE nel fatto, gravita' ridimensionata a MINORE.

## Eseguito
`agents/verifica-r5-pacchetto-operativo-R5-I-1-2/repro.py` importa il vero `checked_command` (scripts/contrastive_split_benchmark.py:24) e la tupla dell'except copiata da scripts/diagnostics/contrastive_cn_restart.py:174. Il figlio dorme 30 s, il deadline e' 1 s. Output:
- `propagated: TimeoutExpired caught_by_run_except: None elapsed 1.0s`
- `MRO: ['TimeoutExpired', 'SubprocessError', 'Exception', 'BaseException', 'object']`
- `pre-phase path is TimeoutError (caught)`: solo il caso "deadline gia' scaduto prima della fase" (riga 28) e' coperto.

## Dedotto dal codice
- Con un timeout durante una gamba o durante il confronto, `execution_failure.json` non viene scritto. Il job esce con un traceback in `.err` (exec python3 nello sbatch, set -e) e SLURM lo segna FAILED.
- Nessun consumatore legge `execution_failure.json` (grep su scripts/, results/, docs/: compare solo in contrastive_cn_restart.py). Il protocollo (doc riga 21) dice solo che l'errore "viene raccolto separatamente".
- Non c'e' un'attribuzione sbagliata: senza `report.json` non si puo' leggere SUPPORTED o un altro esito. Il compare scrive il report una sola volta alla fine (compare.py:319). Se viene ucciso durante la scrittura, il file resta troncato o assente, non un esito valido. L'esito pratico (prova inconclusiva per un timeout) dipende dal timeout, non da questo difetto. La causa resta nel log `.err` e in `sacct`.
- Il difetto quindi non rende INCONCLUSIVE un run che altrimenti riuscirebbe e non sbaglia la classificazione: non e' bloccante. Lo classifico minore e non importante, perche' perde solo un documento riassuntivo ridondante rispetto al traceback e allo stato SLURM.

## Correzione minima
In contrastive_cn_restart.py:174 aggiungere `subprocess.SubprocessError` (o `subprocess.TimeoutExpired`) alla tupla. Facoltativo: un test con un comando che dorme e un deadline breve.
