# Controprova B-0 — regge, bloccante

Verificato in sola lettura:
- ~/wearusfm_local/jobs/ledger.tsv righe 1891 (prenotato, 15:33:41) e 1892 (ambiguo, 15:33:42) per CONTRAST-CHECKOUT-CN-RESTART545-20261121, codice b58a66f6.
- ~/wearusfm_local/jobs/CONTRAST-CHECKOUT-CN-RESTART545-20261121/state.json: stato "ambiguo", job_slurm null, stdout "finto-stdout", stderr "finto-stderr".
- scripts/submit_job.py righe 140-160 (worktree di release): se la cartella esiste e lo stato e' "ambiguo", reserve() esce con «gia' prenotato, stato 'ambiguo'. Non si risottomette: usare --riconcilia».
- Sequenza riga 72: `grep -c "CN-RESTART545" ledger.tsv  # deve essere 0` -> oggi darebbe 2; riga 101 usa lo stesso ID; riga 159 prevede --riconcilia solo per un'uscita ambigua dell'invio vero.

Tentativo di confutazione: il guardiano impedisce un doppio invio, quindi nessun job reale parte per errore. Ma l'esito resta: alla lettera l'operatore si ferma al passo 1 (o, se prosegue, il passo 4 viene rifiutato) e l'ID del checkout e' inutilizzabile senza una pulizia o una riconciliazione fino ad "assente" + --dopo-assenza-verificata, percorso non descritto. Il lancio non puo' avvenire seguendo i testi: bloccante. Va risolto (decisione di Simone) prima della consegna.
