# Verifica avversaria R5-I-1 (timeout di fase non registrato)

Esito: il rilievo REGGE. Gravita' corretta: **minore** (non importante, non bloccante).

## Eseguito
- MRO in Python 3.11: `TimeoutExpired -> SubprocessError -> Exception`; `issubclass(TimeoutExpired, (RuntimeError, OSError, ValueError, TimeoutError))` = False.
- `agents/verifica-r5-pacchetto-operativo-R5-I-1-1/demo.py`: importa il vero `checked_command` da `scripts/contrastive_split_benchmark.py`, figlio che dorme 30 s, deadline 1 s, avvolto nella stessa tupla di `run()` (contrastive_cn_restart.py:174). Output: `ESCAPED run() except: TimeoutExpired`.

## Dal codice
- contrastive_split_benchmark.py:32-33: `TimeoutError` solo se il deadline e' gia' passato prima della fase; durante la fase `child.wait(timeout=remaining)` solleva `TimeoutExpired`, che il blocco `except BaseException` rilancia dopo aver ucciso il gruppo.
- contrastive_cn_restart.py:174: la tupla non lo include, quindi niente `execution_failure.json`; il traceback va nel .err, exit 1 (sacct FAILED, non TIMEOUT, perche' il deadline precede di 25 s il limite SLURM).
- Nessun altro file consuma `execution_failure.json` (git grep: un solo riferimento, la scrittura).

## Perche' non e' bloccante
- Non rende INCONCLUSIVE un run che non lo sarebbe gia': il timeout e' esso stesso l'esito operativo; il traceback nomina `TimeoutExpired`.
- Non fa attribuire male l'esito: senza `report.json` non c'e' alcuno status scientifico leggibile; `allocation.json` e i file delle gambe restano su WORK. Il protocollo (doc riga 33) tratta gia' il timeout come «incertezza operativa da documentare».
- Perdita: solo il documento strutturato con la causa; la causa resta ricostruibile dal log.

## Correzione minima
Riga 174: aggiungere `subprocess.TimeoutExpired` (o `subprocess.SubprocessError`) alla tupla. Test opzionale con figlio dormiente e deadline breve.
