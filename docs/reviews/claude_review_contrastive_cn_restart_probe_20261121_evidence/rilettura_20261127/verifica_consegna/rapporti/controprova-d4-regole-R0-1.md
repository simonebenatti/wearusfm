# Controprova di R0: regge, bloccante

Ho provato a confutare il rilievo e non ci sono riuscito: i fatti sono veri.

- Ho letto `~/wearusfm_local/jobs/ledger.tsv` alle 15:37:50 del 27/11. Le righe 1891 e 1892 riguardano `CONTRAST-CHECKOUT-CN-RESTART545-20261121`:
  - la riga 1891 è `prenotato`, delle 15:33:41;
  - la riga 1892 è `ambiguo`, delle 15:33:42;
  - in entrambe l'agente è claude e lo SHA è b58a66f6.
- Lo `state.json` di `~/wearusfm_local/jobs/CONTRAST-CHECKOUT-CN-RESTART545-20261121/` ha lo stato `ambiguo`, `job_slurm` null, stdout "finto-stdout\n" e stderr "finto-stderr\n". Si tratta quindi di un'esecuzione con l'ssh finto, ma sul registro vero.
- Nel worktree di release, `scripts/submit_job.py` alle righe 153-160 rifiuta un ID nello stato `ambiguo` e chiede di usare `--riconcilia`. Alle righe 287-291, `reconcile()` porta lo stato ad `assente` solo dopo due controlli vuoti validi a distanza di almeno 10 minuti. Poi la riga 312 chiede `--dopo-assenza-verificata`.
- Il registro predefinito è proprio `~/wearusfm_local/jobs` (righe 47 e 316).
- La sequenza usa quell'ID (righe 49 e 101). Il controllo alla riga 72 chiede 0 righe, ma oggi ne trova 2. Il passo 1 si ferma quindi subito. Se l'operatore lo saltasse, l'invio verrebbe rifiutato.
- L'ID del job GPU `CONTRAST-CN-RESTART545-20261121` contiene la sottostringa `CN-RESTART545`, quindi conta anche lui nel grep. Però nel ledger non ha righe sue.

Sulla gravità: è bloccante. Seguendo i testi alla lettera, il lancio non parte. Un operatore con poco contesto potrebbe anche usare `--riconcilia` con l'ssh vero, ripulire la voce senza il via di Simone o cambiare ID. Il job su Slurm non è mai partito, quindi il tentativo non è bruciato. Ma per procedere serve una decisione di Simone sul registro.

Non ho verificato chi abbia eseguito il submitter sul registro vero.
