# Controprova di R3 (ricevute di validazione e checkout fuori dalla raccolta)

Esito: il fatto regge, ma la gravità va abbassata a «minore".

Verificato:
- Sequenza, righe 172-185: il passo 7 non elenca validate_only.out/err, il log campaign_checkout_<jobid>.out, l'uscita della quota né l'ls-remote. L'accounting sacct invece copre "i due job", quindi anche il checkout (riga 183-184).
- Il passo 5 (righe 120-121) salva già stdout e stderr della validazione in ~/wearusfm_local/cn_restart545_20261127, una cartella durevole fuori dallo scratchpad. Così la richiesta del protocollo (riga 45, "trattiene stdout, stderr e ricevute") è soddisfatta.
- Il protocollo (riga 45) chiede di archiviare esplicitamente solo i log SLURM del job GPU con SHA e accounting, e quelli il passo 7 li contiene. Per la quota (riga 43) chiede "Conservare l'output", senza dire di metterlo nel repo.
- Gli state.json del submitter restano comunque in ~/wearusfm_local/jobs.

Conseguenze: nessun passo di lancio cambia, e nessun ID viene bruciato. Nessuna regola o autorizzazione viene violata. L'unico effetto è un archivio nel repo meno completo dei precedenti della linea 3. Resta un piccolo rischio pratico: per la quota la frase "Conservare l'uscita" non indica un percorso, quindi l'operatore potrebbe non salvarla. È un'imprecisione che non cambia l'esito, quindi è "minore". La correzione proposta (una voce "ricevute operative" al passo 7, con un percorso preciso) resta utile e costa poco.
