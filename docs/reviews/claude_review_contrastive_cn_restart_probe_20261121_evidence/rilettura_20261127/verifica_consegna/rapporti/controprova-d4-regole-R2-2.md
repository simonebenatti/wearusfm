# Controprova R2 (dove committa la chat 0)

Esito: il fatto regge, la gravita' va abbassata a **minore**.

Verifiche fatte:
- Sequenza righe 190-191: unico riferimento al ramo, «Percorso proposto per la raccolta, sul ramo `worktree-claude-training-engine`». Riga 196: «commit e push» senza worktree. Brief riga 22: «commit e push» senza ramo ne' worktree (il ramo compare solo alla riga 12, come posizione della sequenza).
- `git worktree list`: `worktree-claude-training-engine` e' estratto solo in /Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-training-engine (76f5d0e7); il canonico e' su main (c47e6557). Git non permette di estrarre lo stesso ramo in un secondo worktree, quindi chi segue la sequenza alla lettera scrive nel worktree del revisore.
- `git status --porcelain` in quel worktree: non tracciati la sequenza e la cartella prova_a_secco_invii (prima della consegna il revisore dovra' committare almeno la sequenza per @SEQ_SHA@).

Tentativo di confutazione (conseguenze):
- Il passo 8 viene dopo la fine del job GPU. Non tocca invii, ID, risorse o esito: nessun tentativo bruciato, il lancio non fallisce.
- La sequenza, che il brief chiede di leggere per intero, nomina un ramo di Claude, cosi' il commit su main non e' l'esito di una lettura letterale. Il rischio su main viene solo dall'omissione nel brief.
- Il revisore dichiara chiusa la sua parte («Non serve che tu mi risponda»). Due writer concorrenti si avrebbero solo se il revisore, o un'altra chat Claude che usa lo stesso ramo per il registro, scrivesse in quel worktree nello stesso momento. E' possibile ma non dimostrato.
- Un commit nel posto sbagliato si recupera (cherry-pick o revert sul ramo locale) prima del push.

Quindi il rilievo non regge come «importante»: non cambia l'esito del lancio e l'errore si recupera. L'ambiguita' resta reale e conviene una riga esplicita: dove committare e il divieto di main. Gravita': minore.
