# Controprova di R1 (billing e saldo netto non controllati)

Esito: il fatto regge, la gravita' va abbassata a **minore**.

## Fatto verificato
- Protocollo (worktree di release, docs/contrastive_cn_restart_probe_20261121.md, paragrafo «Usare il submitter e il ledger comuni...»): chiede di verificare «saldo netto degli impegni, billing corrente, quota...».
- docs/contrastive_targets_execution_20261119.md riga 95: «Il launcher controlla il billing reale prima dell'invio».
- Sequenza, passo 3 (righe 87-93) e passo 6: nessuna lettura di saldo/billing (grep "billing|saldo" nella sequenza: zero righe). Il brief, passo 3, elenca solo cartelle, quota, coda, prenotazioni.
- La sequenza dichiara (passo 4) che rispetto al protocollo cambiano solo `--agente claude` e il submitter diretto: l'omissione non e' dichiarata.
- Le letture del 27/11 (stato_pre_lancio, reservation_coda) non contengono saldo o billing.

## Perche' non e' «importante»
- Margini: richiesta 1,0 GPU-ora e 8,27 ore locali contro residuo 12,26 e 114,0 sul tetto firmato. Anche con un billing diverso dalle CPU (es. pesato sulle GPU) il tetto non verrebbe superato in questa prova.
- Un saldo di progetto esaurito non brucerebbe l'ID in modo silenzioso: il job resterebbe in attesa con motivo leggibile da squeue, che la sequenza fa gia' leggere.
- Il controllo a posteriori del billing e' coperto in pratica: il passo 7 del brief raccoglie l'accounting di SLURM, da cui AllocTRES e' ricavabile.
- Quindi seguendo i testi alla lettera il lancio non fallisce e non esce dall'autorizzazione; resta una deviazione non dichiarata dai controlli del protocollo.

## Correzione
Aggiungere al passo 3 `saldo -b` (conservato con la quota) e, dopo l'invio, AllocTRES/billing da `sacct -X -j <jobid> -o JobID,AllocTRES%60`; oppure dichiarare l'omissione accanto alle altre differenze dal protocollo.
