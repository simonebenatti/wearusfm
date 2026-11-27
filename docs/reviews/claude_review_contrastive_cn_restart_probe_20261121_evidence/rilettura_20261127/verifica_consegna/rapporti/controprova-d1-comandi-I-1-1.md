# Controprova del rilievo I-1 (grep -c del passo 1)

Esito: il rilievo REGGE. Gravità corretta: importante.

## Verifiche fatte
- Sequenza, riga 72: `grep -c "CN-RESTART545" ~/wearusfm_local/jobs/ledger.tsv   # deve essere 0`. Nessuna nota sul codice d'uscita, nessun `|| true`. Il blocco del passo 1 è un unico blocco bash e il grep è l'ultima riga: incollato in una sola chiamata, l'uscita del blocco è quella del grep.
- Prova locale su un file senza corrispondenze (scratchpad/g.txt): `grep -c` stampa `0` e `exit=1`. Il comportamento affermato è vero.
- Brief, riga 26: «Se un controllo non torna o un comando dà un errore inatteso: fermati e riferisci». Un modello piccolo che vede «Exit code 1» può fermarsi.
- Seconda parte del rilievo: `scripts/submit_job.py` righe 141-161 (worktree di release). `reserve` fa `mkdir`; se la cartella esiste già, legge lo stato e rifiuta (stato «sconosciuto» se manca state.json) con «Usare un ID nuovo». Quindi anche una cartella `~/wearusfm_local/jobs/<ID>` senza riga nel ledger blocca l'invio: il controllo sul solo ledger non basta.

## Perché non è bloccante
Il guasto è un arresto spurio prima di qualsiasi invio: nessun ID viene bruciato e nessuna regola violata. Ma può far fallire la consegna, quindi è un rischio concreto di errore dell'operatore (importante), non solo un'imprecisione.

## Nota di contesto (non un rilievo nuovo)
La lettura di oggi del ledger dà 2 righe con CN-RESTART545 (exit 0) ed esiste la cartella `~/wearusfm_local/jobs/CONTRAST-CHECKOUT-CN-RESTART545-20261121`: è l'«incidente» citato dal rilievo. Non cambia la verità del fatto.
