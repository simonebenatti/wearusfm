# Controprova del rilievo D3-3 (lente dei fatti)

Esito: i fatti reggono; gravità corretta «minore» (non «importante").

## Fatti verificati
- Sequenza (docs/contrastive_cn_restart_probe_launch_20261127.md): `squeue`/`sacct` compaiono solo alla riga 158 (passo 6) e alla riga 183 (accounting della raccolta). Il passo 4 (righe 108-111) dice solo «Atteso: COMPLETED 0:0 in meno di un minuto», senza comando di stato né modo di attendere.
- Nessuna occorrenza di «attesa», «PENDING», «certificato», «scadenza», «step ssh» o di un intervallo di lettura nella sequenza; nel brief solo la riga 31 «Il certificato ssh alle 15:05 era valido fino al 28/11 alle 02:34», senza istruzione.
- reservation_coda_20261127.out.txt: ora 27/11/2026 15:10:08, «stima coda dbg»: 2 COMPLETING, 39 PENDING, 75 RUNNING. Confermato.
- Il protocollo di GPT (riga 43) parla di attesa in coda solo per la finestra di manutenzione del 23/11, ormai passata.

## Perché la gravità scende
- Nessuna delle omissioni fa fallire il lancio o brucia un ID: un job PENDING o RUNNING non consuma il tentativo; la scadenza del certificato blocca solo le letture e la raccolta, non il job sul cluster.
- La sequenza vieta già rilanci e cancellazioni (brief righe 25-29: «Nessuna modifica a file remoti, nessuna cancellazione»), e prescrive «fermati e riferisci» per ogni controllo che non torna (brief riga 26). Questa regola generale copre in modo prudente sia la coda lunga sia il certificato scaduto.
- Il comando di stato esiste (riga 158) ed è riusabile per il checkout; la finestra fino alle 02:34 (circa 11 ore) è ampia rispetto a 4 + 15 minuti di allocazione.
- Resta un rischio pratico (tempo perso, operatore incerto su come attendere senza sleep), utile da chiarire in una riga, ma non un rischio concreto di azione sbagliata: minore.

## Non verificato
- Lo stato attuale della coda e del certificato (divieto di ssh).
