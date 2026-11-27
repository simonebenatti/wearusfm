# Controprova R0: prenotazione finta 'ambiguo' nel ledger vero

Esito: il rilievo regge. Gravita': bloccante.

## Fatti verificati (27/11, 15:37)
- `~/wearusfm_local/jobs/ledger.tsv` contiene due righe per CONTRAST-CHECKOUT-CN-RESTART545-20261121: 15:33:41 'prenotato' e 15:33:42 'ambiguo', agente claude, sha b58a66f6.
- In `~/wearusfm_local/jobs/CONTRAST-CHECKOUT-CN-RESTART545-20261121/state.json`: stato 'ambiguo', job_slurm null, stdout 'finto-stdout\n', stderr 'finto-stderr\n'. E' una prova a secco eseguita sul registro vero (default di `--registro`, submit_job.py riga 316), non quella su prova_a_secco_invii/registro.
- submit_job.py righe 153-161: con stato 'ambiguo' reserve() esce con "gia' prenotato ... usare --riconcilia" senza sbatch.
- reconcile() (righe 253-290): risposta vuota di sacct/squeue porta ad 'assente' solo con due controlli validi (il primo >= 10 min dopo il tentativo, l'ultimo >= 10 min dopo il primo); poi serve --dopo-assenza-verificata.

## Tentativo di confutazione (lente delle conseguenze)
- L'ID su Slurm non e' bruciato: l'ssh era finto, nessun sbatch vero. Ma il blocco resta.
- Passo 1 della sequenza (riga 72) e brief (riga 15) chiedono `grep -c` = 0: l'operatore leggera' 2 e, seguendo i testi, si ferma. Se ignorasse il controllo, il passo 4 fallirebbe con SystemExit. In entrambi i casi il lancio non parte.
- Esiste una via documentata (riconciliazione reale e --dopo-assenza-verificata, almeno 20 minuti), ma nessuno dei due testi la descrive e richiederebbe all'operatore di toccare il registro con passaggi non previsti.
- Anche la frase della sequenza "Il ledger vero non e' stato toccato" (riga 43) e' ora falsa.

Quindi: non e' confutabile, e seguendo i testi alla lettera il lancio fallisce. Bloccante.
