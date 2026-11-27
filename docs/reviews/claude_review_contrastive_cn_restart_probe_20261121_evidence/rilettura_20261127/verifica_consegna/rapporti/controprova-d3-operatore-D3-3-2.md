# Controprova D3-3 (attesa, coda lunga, certificato)

Esito: i fatti sono veri, ma le conseguenze sono piccole. Gravità corretta: minore, non importante.

Fatti verificati:
- Sequenza, riga 158: squeue/sacct compaiono solo al passo 6. Al passo 4 (righe 109-111) c'è solo «Atteso COMPLETED 0:0 in meno di un minuto».
- Nessun criterio per un job che resta PENDING. Nel brief, riga 31, c'è la scadenza del certificato senza alcuna istruzione.
- reservation_coda_20261127.out.txt, riga 14: «39 PENDING» nella coda dbg.

Perché non è «importante»:
- Lettura dello stato del checkout: per leggere COMPLETED l'operatore deve comunque fare una lettura, e quella del passo 6 (squeue, poi sacct) è lì, due passi dopo. Il CLAUDE.md, che tutte le chat del progetto caricano, dà lo stesso comando, `ssh leonardo squeue -h -j ID` e poi `sacct -X -j ID`. Non c'è un modo plausibile di bruciare un ID.
- Coda lunga senza criterio: se il job resta PENDING, chi non ha istruzioni aspetta e basta. Il brief (riga 26) e la sequenza (riga 20) vietano di improvvisare, cancellare o rilanciare. Il protocollo chiede una manutenzione generale assente (sequenza, righe 37-38; passo 3) e nessun limite temporale. Non c'è quindi un errore che quel criterio eviterebbe.
- Certificato: alla scadenza i job continuano a girare su SLURM. Si perde solo l'accesso dal Mac, quindi la raccolta slitta. Un invio con risposta ambigua per ssh caduto ha già la sua procedura, il riconcilia (riga 159). La memoria di progetto, condivisa dalle chat sullo stesso progetto, ha la voce «Certificato ssh di Leonardo (step)» con il rinnovo di Simone. Il lancio cade nel pomeriggio del 27/11 e la scadenza è il 28/11 alle 02:34, con un margine di molte ore per job da 4 e 15 minuti.
- Il criterio proposto («non inviare a meno di 60 minuti dalla scadenza») sarebbe utile, ma è una prudenza in più, non la correzione di un errore che cambia l'esito.

Correzione accettabile, facoltativa: una riga al passo 4 che rimandi alla lettura del passo 6, e una riga nel brief: «se il certificato scade, chiedi il rinnovo a Simone; non cancellare i job».
