# Controprova R2 (quota senza impegni concorrenti)

Esito: il fatto regge, la gravita' va abbassata a "minore".

Verificato:
- Protocollo (worktree di release), docs/contrastive_cn_restart_probe_20261121.md riga 43: chiede "quota e scritture concorrenti" e "almeno 4 GB per questa prova dopo gli impegni concorrenti".
- Sequenza, riga 90: "almeno 4 GB oltre al checkout"; riga 91: solo la "Coda dell'utente"; riga 147: "rileggere la quota". Brief, riga 17: "almeno 4 GB liberi oltre al checkout". Quindi la sottrazione degli impegni concorrenti manca davvero.
- stato_pre_lancio_20261127.out.txt righe 45-47: 991G usati su 1T, cioe' circa 33 GB liberi; coda dell'utente: 65361388 SCALING PENDING, 10 h.

Conseguenze: oggi il margine e' circa 8 volte i 4 GB richiesti (vettori della prova circa 1,83 GB, protocollo riga 39). Anche togliendo qualche checkpoint del job di scaling, se partisse proprio nei 15 minuti, la decisione dell'operatore non cambierebbe: invierebbe comunque. La coda dell'account mostrerebbe i job degli altri utenti, ma non quanto scrivono, e quindi non darebbe un numero da sottrarre. Il rilievo conta solo con un margine stretto, cioe' meno di circa 10 GB liberi. In quel caso la frase "dopo gli impegni concorrenti" farebbe fermare l'operatore.
Correzione minima: riportare nel passo 3 la formula del protocollo e una soglia pratica (per esempio: fermarsi e riferire se i GB liberi sono sotto circa 10).
