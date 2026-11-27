# Controprova di R3 (controlli di hook, configurazione e riferimento prima del push)

Esito: il fatto regge in parte, le conseguenze no. Gravità corretta: minore.

Che cosa è vero
- La lettura delle 15:08 (leonardo/stato_pre_lancio_20261127.out.txt) mostra solo gli 8 riferimenti deploy più recenti: non prova che deploy/cn-restart545-20261121 manchi.
- Il passo 3 viene dopo il push e non rilegge hook, configurazione e riferimento.

Perché non cambia l'esito
- Il push (passo 2) viene prima di qualunque invio. Se fallisce, l'operatore si ferma (sequenza riga 85, brief punto 2) e nessun ID viene consumato.
- Se il riferimento esistesse già su un altro commit, ci sono due casi. Se quel commit non è antenato, git rifiuta il push: l'operatore si ferma e nessun ID è perso. Se è antenato, il push avanza il riferimento. È innocuo, perché nessun checkout usa quel nome: campaign_checkout.sbatch righe 30-45 lavora solo con lo SHA (`worktree add --detach "$DEST" "$SHA"`). In ogni caso il `ls-remote` della riga 82 verifica lo SHA.
- receive.denyCurrentBranch=updateInstead tocca solo il ramo estratto (main), non un riferimento deploy nuovo. La pulizia del canonico quindi non conta per questo push.
- Alle 15:08 c'erano già il controllo di hook e configurazione («hook e configurazione di ricezione»: updateInstead, nessun hook). La regola di CLAUDE.md parla del «primo push di deploy»: risulta coperta dalla lettura dello stesso giorno. Che compaia un hook nel giro di poche ore non è un rischio concreto.

Resta solo un'imprecisione: la riga 38 dice «vanno rilette al momento del lancio», ma sul riferimento non è previsto un controllo prima del push. Si corregge aggiungendo, prima della riga 81, `git ls-remote leonardo refs/heads/deploy/cn-restart545-20261121` (risultato atteso: vuoto).

Osservazione fuori perimetro, da girare al chiamante
Il ledger vero ha già due righe per CONTRAST-CHECKOUT-CN-RESTART545-20261121 (riga 1891 «prenotato», riga 1892 «ambiguo», 27/11 alle 15:33). Il passo 1 della sequenza (grep deve dare 0) non passerebbe più.
