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

La lettura finale è `preflight_readout.json` (SHA256
`deb6b8178362e23512f105d95b3859c28945d9ffa5356a653f1c8cad67c929fc`):
`FAILED_REQUIRES_REVIEW`, `training_approved=false`. Il log del job CPU 64775571
è `contrastive_cpu_64775571.out` (SHA256
`ec58c42a9e66dd6234a759060392cfe97659fe9f769ee06d48c03892b82e9a35`).
L'uscita FAILED 2:0 è il comportamento previsto del lettore per un rapporto
negativo completo, non un errore di esecuzione.
