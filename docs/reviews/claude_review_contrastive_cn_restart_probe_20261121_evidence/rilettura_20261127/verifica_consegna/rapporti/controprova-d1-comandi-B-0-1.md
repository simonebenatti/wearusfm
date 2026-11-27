# Controprova B-0 — regge, bloccante

Verificato in sola lettura:
- ~/wearusfm_local/jobs/CONTRAST-CHECKOUT-CN-RESTART545-20261121/state.json esiste (creato 27/11 15:33:41), stato "ambiguo", job_slurm null, stdout "finto-stdout\n", codice_sha b58a66f6..., storia prenotato -> in_sottomissione -> ambiguo. Nessun operazione.lock rimasto.
- ledger.tsv righe 1891 (prenotato, 15:33:41) e 1892 (ambiguo, 15:33:42) con quell'ID; il file ha 1892 righe.
- agents2/d1-comandi/finto_ssh_bin/ssh e' lo script finto (scrive argomenti/stdin, stampa finto-stdout, exit 0); ssh_argomenti.txt = "2 / leonardo / bash -s": la chiamata non e' arrivata a Leonardo.
- Worktree di release pulito (git status --short vuoto).
- Sequenza riga 49 e 101 usano proprio questo ID; riga 72: `grep -c "CN-RESTART545" ledger.tsv  # deve essere 0` -> oggi darebbe 2, quindi il passo 1 si ferma.
- scripts/submit_job.py righe 149-160: con cartella esistente e stato 'ambiguo' -> SystemExit "gia' prenotato, stato 'ambiguo'. Non si risottomette: usare --riconcilia."; righe 16-18: 'assente' solo dopo due controlli vuoti distanziati 10 minuti, poi --dopo-assenza-verificata, percorso non previsto dalla sequenza (riga 159 dice solo riconciliare, mai rilanciare).

Gravita': bloccante (seguendo i testi alla lettera il lancio si ferma; serve una decisione di Simone sulla pulizia o sul percorso di riconciliazione).
