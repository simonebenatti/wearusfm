# Controprova del rilievo I-1

Esito: il fatto è vero, ma la gravità "importante" non regge. Gravità corretta: minore.

## Il fatto è vero
- contrastive_cn_restart_compare.py, righe 205-223: confronto byte per byte di trace-{r}.jsonl e numeric_profile.rank{r}.json fra WORK e archivio. Riga 264: lo stesso per metrics.jsonl. Riga 356: lo stesso per trace-{r}.ddp.jsonl.
- contrastive_cn_restart.py, commands() righe 105-137: con --validate-only controlla solo i pin dei tre checkpoint.
- Le righe 403-404 di main() trasformano un fallimento di require in status INCONCLUSIVE. Il job finisce quindi COMPLETED con report.json.
- La lettura del 27/11 (stato_pre_lancio) ha guardato solo i checkpoint (ls -l, con data 21/11).

## Perché la conseguenza pratica è remota
- Archivio CN: le 13 righe CN/(trace|metrics|numeric) di benchmark_execution_20261121/remote_sha256.txt (sha calcolati su WORK alla raccolta) passano `shasum -c` sulle copie nell'archivio di release: tutte OK.
- Archivio CN_A: le righe CN_A/(trace|numeric) di reloads_execution_20261121/CN_A_remote_integrity_and_log.txt passano `shasum -c` sulle copie nell'archivio: tutte OK.
- Quindi archivio e WORK erano identici alla raccolta del 21/11. Perché oggi differiscano, qualcosa dovrebbe aver riscritto file di run conclusi in wearusfm_runs. Nessun job previsto lo fa: le due gambe scrivono in una cartella nuova, e il blocco "root.exists()" lo impone. I checkpoint vicini sono ancora con data 21/11 e saranno riletti per sha alla validazione.
- Anche nel caso improbabile, le gambe GPU restano complete su WORK. L'operatore raccoglie e riferisce in ogni caso (passi 6-8), quindi il suo comportamento non cambia.

## Correzione
Il controllo sha256 proposto è economico: pochi MB, sola lettura, sul nodo di login. Si può aggiungere al passo 3, oppure scrivere nella sezione "Che cosa non si può verificare prima" che i file storici non-.pt su WORK sono stati verificati solo alla raccolta del 21/11. Non è un prerequisito.

## Non verificato
Lo stato attuale dei file su WORK: per divieto non ho usato ssh.
