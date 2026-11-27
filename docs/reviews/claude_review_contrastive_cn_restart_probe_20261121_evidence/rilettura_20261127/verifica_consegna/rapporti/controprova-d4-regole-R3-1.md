# Controprova di R3 (lente dei fatti)

Esito: R3 regge solo in parte. La gravità corretta è «minore», non «importante».

## Che cosa è falso nel rilievo
- Il rilievo dice che manca un controllo di hook e configurazione. Il controllo c'è stato, lo stesso giorno: la lettura delle 15:08 (leonardo/stato_pre_lancio_20261127.out.txt, righe 20-23 e 36-38, script stato_pre_lancio.sh) mostra il canonico su main a 8dad7e82, uno status vuoto («(fine status)»), nessun hook oltre ai file di esempio (la lista è vuota prima di `updateInstead`) e receive.denyCurrentBranch=updateInstead. La sequenza ne riporta l'esito (riga 35-36: «pulito, senza hook propri»). La regola di CLAUDE.md («Prima del primo push di deploy verificare ref/configurazione/hook») è quindi rispettata per configurazione e hook.

## Che cosa è vero
- L'assenza del riferimento `refs/heads/deploy/cn-restart545-20261121` non è dimostrata. Lo script usa `for-each-ref --sort=-creatordate ... | head -8`, e le righe 28-35 contengono 7 riferimenti del 21/11. Un riferimento con quel nome che puntasse a un commit del 21/11 o più vecchio potrebbe stare oltre l'ottava riga. Il commit 8c72655c è assente (riga 26), quindi il riferimento non può puntare a 8c72655c né a b58a66f6. Potrebbe però puntare a un loro antenato, per esempio 9d34dc69 del 21/11.
- La riga 38 della sequenza dice «Vanno rilette al momento del lancio», ma il passo 3, che fa le riletture, viene dopo il push del passo 2 e non comprende hook, configurazione e riferimento.

## Perché la gravità è minore
- Se il riferimento esiste e non è un antenato, il push senza force viene rifiutato. Il passo 2 (riga 85) e il brief (punto 2) dicono di fermarsi, e nessun ID di job è stato ancora consumato.
- Se il riferimento esiste ed è un antenato, il push avanza il riferimento senza errori. Nessun checkout lo usa: quello di campagna è assente (riga 6) e il checkout viene creato da CAMPAIGN_SHA, uno SHA completo. Il passo 2 controlla poi lo SHA del riferimento con ls-remote.
- Il push di un riferimento deploy non tocca il working tree del canonico (updateInstead vale solo per il ramo estratto, main).
- Nessuno dei casi brucia un tentativo o viola un'autorizzazione.

## Correzione (facoltativa)
Prima del push, aggiungere `git ls-remote leonardo refs/heads/deploy/cn-restart545-20261121` con risultato atteso vuoto. Così si chiude anche l'unico punto non dimostrato. Non serve rileggere hook e configurazione.

## Non verificato
Non ho verificato l'esistenza reale del riferimento sul cluster: ssh e ls-remote erano vietati.
