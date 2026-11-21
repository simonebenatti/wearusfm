# Verifica avversaria di I-2 (verifica-r7-critico-I-2-1)

Esito: il rilievo regge solo in parte. Lo declasso a **minore** (procedurale). Non è bloccante.

## Che cosa ho controllato (lettura del codice, nessuna esecuzione GPU)
- `scripts/diagnostics/contrastive_cn_restart_compare.py:307-320`: il confronto è uno script autonomo con argomenti CLI obbligatori (`--ordinary --warm --historical-ordinary --historical-continuous --report`). Un'eccezione (righe 317-318) produce un report con `INCONCLUSIVE` e `reason` con tipo ed errore: è chiaro che l'esito viene dal confronto e non dalla misura.
- Il file non legge `os.environ`, `SLURM`, CUDA né `allocation.json` (grep vuoto). È CPU e lavora sui soli file: si può rieseguire fuori dall'allocazione GPU.
- `contrastive_cn_restart.py:133-136`: solo la catena orchestrata fissa `root/report.json`, scritto con `open("x")` (`contrastive_target_resume.write`, riga 52). Per rieseguire basta passare un altro `--report` a compare.py, senza toccare il codice. La parte del rilievo che chiede `--compare-only` dentro `--package` non è quindi necessaria per recuperare l'esito.
- Conservazione dei dati: `plan.output` sta su `$WORK` (`/leonardo_work/.../cn_restart_probe_20261121`), non su SCRATCH. Né `cn_restart_probe.sbatch`, né `contrastive_cn_restart_leg.py`, né compare cancellano o spostano file: grep di `rm`, `unlink`, `rmtree`, `mv` vuoto. I dati grezzi restano, e anche i riferimenti storici puntati dal piano.
- `run()` (righe 171-177): se compare esce con 0 e scrive un report INCONCLUSIVE, non si scrive `execution_failure.json`. Nessun doppio esito contraddittorio.

## Dove il rilievo regge
- In `docs/decisioni.md` (ultima voce) e in `docs/contrastive_cn_restart_probe_20261121.md` non c'è una regola per un INCONCLUSIVE dovuto al solo codice di lettura. Un grep di «riesegu», «a posteriori» e «post hoc» nella nuova voce non trova nulla. Correggere compare.py dopo la misura e rieseguirlo cambierebbe il codice di lettura, non i criteri. Senza una voce registrata prima, però, andrebbe dichiarato come deroga (vedi la regola sui limiti registrati del 21/11).

## Perché non è bloccante né importante
- Non rende il run INCONCLUSIVE: lo rendono così B-1 e B-2, già contati come bloccanti a sé e da correggere prima del lancio. I-2 riguarda solo il recupero dopo un difetto non ancora visto.
- Il recupero non richiede nuove GPU né nuovo codice d'orchestrazione: dati su WORK, compare in CLI, report su un percorso nuovo.
- Nessun controesempio in cui l'esito venga attribuito male: il report INCONCLUSIVE porta tipo ed errore dell'eccezione.

## Correzione minima proposta
Aggiungere una frase alla voce di decisioni.md (o al protocollo) prima del lancio. Un INCONCLUSIVE causato da un errore del solo confronto, a gambe GPU complete, si chiude rieseguendo su CPU (`lrd_all_serial`) compare.py corretto e rivisto. Si usano gli stessi dati grezzi, con le impronte già in `allocation.json` e nelle ricevute. I criteri restano invariati, `--report` va su un percorso nuovo e non si usano nuove GPU. Non serve una modalità `--compare-only` in `--package`.
