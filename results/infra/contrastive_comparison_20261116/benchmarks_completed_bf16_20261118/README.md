# Evidenze dei cinque benchmark bf16

Cinque job conclusi al passo 576; accounting in `accounting.txt`.
`remote_hashes.txt` e `verified_files.json` attestano i 25 file raccolti byte per byte.
Le metriche sono archiviate come `metrics.jsonl.gz`: decomprimerle in una cartella
locale di analisi per riutilizzare i lettori; SHA e byte nel manifest si riferiscono
agli originali non compressi. Nessun segnale grezzo e nessun checkpoint sono nel repo.
I checkpoint rimangono nei percorsi WORK originali, per la lettura CPU ufficiale.

`metadata_readout.json` è preliminare: rapporti delle norme, tempi e parità dei primi
256 passi. Non verifica i checkpoint e non certifica le riprese. La parità grad_norm
è negativa; gli altri risultati non eliminano quel controllo fallito.
